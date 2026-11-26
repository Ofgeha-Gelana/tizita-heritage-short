"""Kaggle kernel: Wan 2.2 TI2V-5B image-to-video for a batch of shots, on 2x T4.

Reads jobs.json from the attached dataset and writes clips/<id>_<key>.mp4 plus
results.json to /kaggle/working. Results are merged after every finished job, so a
timeout keeps finished work.

Design (v3, after two OOM runs; FAILURES.md F8):
  1. Encode phase (parent): load only the UMT5 text encoder in fp32, split across both
     T4s with device_map="auto", encode every prompt plus the negative prompt, save
     the embeddings, and free the encoder. fp16 T5 is prone to overflow; fp32 does
     not fit one T4 (~19 GB) but fits two.
  2. Worker phase: one process per GPU (CUDA_VISIBLE_DEVICES=i) takes jobs[i::2]. It
     loads the pipeline WITHOUT the text encoder, so the fp16 transformer plus the fp32
     VAE sit on the GPU with no CPU offload and get the remaining memory for activations.
     Memory-efficient attention only (the math SDPA kernel is O(n^2)).
  3. OOM ladder per job: frames = min(requested, max_frames) -> 49 -> sequential CPU
     offload. The failed attempt's memory is released before the retry: in run 2 the
     live exception traceback kept 14.4 GB allocated and every rung failed at once.
  4. A wall-clock budget stops new jobs, so outputs come back before the deadline;
     the pipeline falls back to Ken Burns for anything not finished.
"""
import gc
import glob
import json
import os
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/clips", exist_ok=True)
WORKER = os.environ.get("WAN_WORKER")
LOG = open(f"{WORK}/wan{'_w' + WORKER if WORKER is not None else ''}.log", "a")
T_START = float(os.environ.get("WAN_T_START", time.time()))


def log(msg):
    tag = f"w{WORKER}" if WORKER is not None else "main"
    line = f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def sh(cmd):
    log("$ " + cmd)
    subprocess.run(cmd, shell=True, check=True)


jobs_file = sorted(glob.glob("/kaggle/input/**/jobs.json", recursive=True))
if not jobs_file:
    raise SystemExit("jobs.json not found under /kaggle/input")
IN_DIR = os.path.dirname(jobs_file[0])
spec = json.load(open(jobs_file[0]))
JOBS = spec["jobs"]
SET = spec["settings"]
MODEL = SET.get("model", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
FPS = SET.get("fps", 24)
MAX_FRAMES = int(SET.get("max_frames", 81))
BUDGET_S = float(SET.get("budget_s", 4.5 * 3600))
EMB = f"{WORK}/prompt_embeds.pt"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# --------------------------------------------------------------------------- encode
def t5_encode_all():
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    te = UMT5EncoderModel.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=torch.float32,
                                          device_map="auto")
    te.eval()
    first = next(te.parameters()).device

    def enc(text):
        # Same as diffusers WanPipeline._get_t5_prompt_embeds: max 512 tokens, the
        # padding positions zeroed.
        x = tok([text], padding="max_length", max_length=512, truncation=True, add_special_tokens=True,
                return_attention_mask=True, return_tensors="pt")
        ids, mask = x.input_ids.to(first), x.attention_mask.to(first)
        n = int(mask.gt(0).sum())
        with torch.no_grad():
            h = te(ids, mask).last_hidden_state[0].float().cpu()
        return torch.cat([h[:n], h.new_zeros(512 - n, h.size(1))])[None]

    out = {}
    for j in JOBS:
        out[j["id"]] = {"pos": enc(j["prompt"]), "neg": enc(j.get("negative_prompt") or "")}
    torch.save(out, EMB)
    log(f"encoded {len(out)} prompts in {time.time() - t0:.0f}s (fp32 T5 on {torch.cuda.device_count()} GPUs)")
    del te
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- worker
def worker(idx, n_workers):
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from diffusers.utils import export_to_video, load_image

    torch.backends.cuda.enable_flash_sdp(False)   # not on sm75
    torch.backends.cuda.enable_math_sdp(False)    # O(n^2) memory
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    GPU = torch.cuda.get_device_name(0)
    DTYPE = torch.float16  # T4 has no fast bf16
    embeds = torch.load(EMB)
    res_path = f"{WORK}/results_w{idx}.json"
    results = json.load(open(res_path))["results"] if os.path.exists(res_path) else []
    done = {(r["id"], r["key"]) for r in results if r.get("status") == "ok"}
    state = {"pipe": None, "mode": None}

    def load(mode):
        state["pipe"] = None
        gc.collect()
        torch.cuda.empty_cache()
        vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, text_encoder=None, tokenizer=None,
                                                       torch_dtype=DTYPE)
        if mode == "gpu":
            pipe.to("cuda")
        else:
            pipe.enable_sequential_cpu_offload()
        try:
            pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001
            pass
        state["pipe"], state["mode"] = pipe, mode
        log(f"loaded pipeline ({mode}); allocated {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def attempt(job, frames, out_path):
        pipe = state["pipe"]
        img = load_image(os.path.join(IN_DIR, job["image"])).resize((job["width"], job["height"]))
        e = embeds[job["id"]]
        g = torch.Generator(device="cpu").manual_seed(job["seed"])
        video = pipe(image=img, prompt_embeds=e["pos"].to("cuda", DTYPE),
                     negative_prompt_embeds=e["neg"].to("cuda", DTYPE),
                     height=job["height"], width=job["width"], num_frames=frames,
                     num_inference_steps=job["steps"], guidance_scale=job["guidance_scale"],
                     generator=g).frames[0]
        export_to_video(video, out_path, fps=FPS)

    mine = JOBS[idx::n_workers]
    log(f"{GPU}: {len(mine)} jobs: {[j['id'] for j in mine]}")
    load("gpu")
    for job in mine:
        if (job["id"], job["key"]) in done:
            continue
        if time.time() - T_START > BUDGET_S:
            log(f"{job['id']}: skipped, budget of {BUDGET_S / 3600:.1f} h used")
            results.append({"id": job["id"], "key": job["key"], "status": "skipped_budget"})
            json.dump({"results": results}, open(res_path, "w"), indent=2)
            continue
        fname = f"{job['id']}_{job['key']}.mp4"
        out_path = f"{WORK}/clips/{fname}"
        rec = {**{k: job[k] for k in ("id", "key", "seed", "width", "height", "steps", "guidance_scale",
                                      "quality")},
               "file": fname, "gpu_name": GPU, "requested_frames": job["num_frames"]}
        cap = min(job["num_frames"], MAX_FRAMES)
        ladder = [(cap, "gpu"), (49, "gpu"), (cap, "sequential"), (49, "sequential")]
        t0 = time.time()
        ok, errors = False, []
        for frames, mode in ladder:
            if mode != state["mode"]:
                load(mode)
            torch.cuda.reset_peak_memory_stats()
            err = oom = None
            try:
                attempt(job, frames, out_path)
                ok = True
            except Exception as e:  # noqa: BLE001
                err = f"{e!r}"[:400]
                oom = isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()
                log(f"{job['id']}: {frames}f/{mode} failed: {err}\n{traceback.format_exc()[-1500:]}")
            # The exception (and its traceback holding activations) is gone here.
            gc.collect()
            torch.cuda.empty_cache()
            if ok:
                rec.update(status="ok", num_frames=frames, backend=f"diffusers-i2v-fp16-{mode}-memeff-t5fp32")
                break
            errors.append(f"{frames}f/{mode}: {err}")
            if not oom:
                break
        if not ok:
            rec.update(status="error", error=" | ".join(errors))
        wall = time.time() - t0
        rec.update(wall_s=round(wall, 1), gpu_s=round(wall, 1),
                   peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
        log(f"{job['id']}: {rec['status']} in {wall:.0f}s (peak {rec['peak_vram_gb']} GB)")
        results.append(rec)
        json.dump({"results": results}, open(res_path, "w"), indent=2)
    log("worker done")


# --------------------------------------------------------------------------- main
def merge():
    allr = []
    for f in sorted(glob.glob(f"{WORK}/results_w*.json")):
        try:
            allr += json.load(open(f))["results"]
        except Exception:  # noqa: BLE001
            pass
    json.dump({"results": allr}, open(f"{WORK}/results.json", "w"), indent=2)
    return allr


if os.environ.get("WAN_ENCODE"):
    t5_encode_all()
    sys.exit(0)
if WORKER is not None:
    worker(int(WORKER), int(os.environ["WAN_WORKERS"]))
    sys.exit(0)

log(f"{len(JOBS)} jobs from {jobs_file[0]}; max_frames={MAX_FRAMES}, budget={BUDGET_S / 3600:.1f} h")
sh(f"{sys.executable} -m pip install -q -U git+https://github.com/huggingface/diffusers transformers accelerate ftfy imageio imageio-ffmpeg sentencepiece")
# The parent never touches CUDA, so it holds no context memory on either GPU.
gpus = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                      text=True).stdout.split("\n")
gpus = [g.strip() for g in gpus if g.strip()]
n_gpu = len(gpus)
log(f"GPUs: {gpus}")
sh(f"{sys.executable} -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')\"")
subprocess.run([sys.executable, os.path.abspath(sys.argv[0])], env={**os.environ, "WAN_ENCODE": "1"}, check=True)

n_workers = max(1, n_gpu)
procs = []
for i in range(n_workers):
    env = {**os.environ, "WAN_WORKER": str(i), "WAN_WORKERS": str(n_workers), "CUDA_VISIBLE_DEVICES": str(i),
           "WAN_T_START": str(T_START)}
    procs.append(subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0])], env=env))
    log(f"started worker {i}")
while any(p.poll() is None for p in procs):
    time.sleep(30)
    merge()
allr = merge()
log(f"all workers exited {[p.returncode for p in procs]}; ok={sum(r.get('status') == 'ok' for r in allr)}/{len(JOBS)}")
