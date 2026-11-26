"""Kaggle kernel: Wan 2.2 TI2V-5B image-to-video for a batch of shots.

Reads jobs.json from the attached dataset, writes clips/<id>_<key>.mp4 and
results.json to /kaggle/working after EVERY job, so a timeout keeps finished work.

Backends, tried in order and recorded per job:
  1. diffusers WanImageToVideoPipeline with Wan-AI/Wan2.2-TI2V-5B-Diffusers
     (fp16 transformer for the T4, fp32 VAE, memory-efficient attention), walking an
     OOM ladder: 121 -> 81 -> 49 frames, then sequential CPU offload,
  2. the official Wan2.2 repo `generate.py --task ti2v-5B` with offload flags.
"""
import glob
import json
import os
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/clips", exist_ok=True)
LOG = open(f"{WORK}/wan.log", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
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
MODEL = spec["settings"].get("model", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
FPS = spec["settings"].get("fps", 24)
log(f"{len(JOBS)} jobs from {jobs_file[0]}")

results_path = f"{WORK}/results.json"
results = json.load(open(results_path))["results"] if os.path.exists(results_path) else []
done = {(r["id"], r["key"]) for r in results if r.get("status") == "ok"}


def save():
    json.dump({"results": results}, open(results_path, "w"), indent=2)


sh(f"{sys.executable} -m pip install -q -U git+https://github.com/huggingface/diffusers transformers accelerate ftfy imageio imageio-ffmpeg sentencepiece")

# Run 1 (FAILURES.md F8): fp16 on a 15 GB T4 hit CUDA OOM on the first shot. ~48k video
# tokens at 121 frames x 480x832; PyTorch's "math" attention materialises the full
# attention matrix. Fragmentation-friendly allocator + memory-efficient attention only.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
log(f"GPU: {GPU}, count={torch.cuda.device_count()}")
# T4 (sm75) has no fast bf16; use fp16 there.
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and "T4" not in GPU else torch.float16
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)      # not on sm75
    torch.backends.cuda.enable_math_sdp(False)       # O(n^2) memory: the OOM in run 1
    torch.backends.cuda.enable_mem_efficient_sdp(True)

pipe = None
backend = None
offload = "model"
# OOM ladder: (num_frames cap, offload mode). The first rung that works is kept for
# the remaining jobs, so later shots do not pay for the same OOM again.
LADDER = [(121, "model"), (81, "model"), (49, "model"), (49, "sequential")]
rung = 0


def load_diffusers(mode):
    global pipe, backend, offload
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    pipe = None
    torch.cuda.empty_cache()
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=DTYPE)
    if mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001
        pass
    offload = mode
    backend = f"diffusers-i2v-{str(DTYPE).split('.')[-1]}-{mode}offload-memeff"
    log(f"loaded {backend}")


def run_diffusers(job, out_path, frames):
    from diffusers.utils import export_to_video, load_image
    img = load_image(os.path.join(IN_DIR, job["image"])).resize((job["width"], job["height"]))
    g = torch.Generator(device="cpu").manual_seed(job["seed"])
    out = pipe(image=img, prompt=job["prompt"], negative_prompt=job.get("negative_prompt") or None,
               height=job["height"], width=job["width"], num_frames=frames,
               num_inference_steps=job["steps"], guidance_scale=job["guidance_scale"], generator=g).frames[0]
    export_to_video(out, out_path, fps=FPS)


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def run_official(job, out_path, frames):
    """Last resort. Run 1: requirements.txt failed on flash_attn and generate.py then
    died on `import decord`; install only what ti2v needs."""
    global backend
    if not os.path.exists("/kaggle/working/Wan2.2"):
        sh("git clone --depth 1 https://github.com/Wan-Video/Wan2.2 /kaggle/working/Wan2.2")
        sh(f"{sys.executable} -m pip install -q decord easydict einops dashscope 'opencv-python-headless' || true")
        sh("huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /kaggle/working/Wan2.2-TI2V-5B")
    backend = "official-generate.py"
    size = f"{job['width']}*{job['height']}"
    sh(f"cd /kaggle/working/Wan2.2 && {sys.executable} generate.py --task ti2v-5B --size {size} "
       f"--ckpt_dir /kaggle/working/Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu "
       f"--frame_num {frames} --sample_steps {job['steps']} --base_seed {job['seed']} "
       f"--image {os.path.join(IN_DIR, job['image'])} --prompt {json.dumps(job['prompt'])} "
       f"--save_file {out_path}")


def generate(job, out_path):
    """Walk the OOM ladder from the current rung. Returns the frame count produced."""
    global rung
    while rung < len(LADDER):
        cap, mode = LADDER[rung]
        frames = min(job["num_frames"], cap)
        try:
            if pipe is None or offload != mode:
                load_diffusers(mode)
            torch.cuda.reset_peak_memory_stats()
            run_diffusers(job, out_path, frames)
            return frames
        except Exception as e:  # noqa: BLE001
            log(f"{job['id']}: rung {rung} ({frames} frames, {mode} offload) failed: {e!r}")
            log(traceback.format_exc()[-2500:])
            torch.cuda.empty_cache()
            if not is_oom(e):
                break
            rung += 1
    frames = min(job["num_frames"], 49)
    log(f"{job['id']}: diffusers exhausted; trying official repo at {frames} frames")
    run_official(job, out_path, frames)
    return frames


for job in JOBS:
    if (job["id"], job["key"]) in done:
        log(f"{job['id']}: already done")
        continue
    fname = f"{job['id']}_{job['key']}.mp4"
    out_path = f"{WORK}/clips/{fname}"
    rec = {**{k: job[k] for k in ("id", "key", "seed", "width", "height", "num_frames", "steps",
                                  "guidance_scale", "quality")}, "file": fname, "gpu_name": GPU,
           "requested_frames": job["num_frames"]}
    t0 = time.time()
    try:
        frames = generate(job, out_path)
        rec.update(status="ok", backend=backend, num_frames=frames, ladder_rung=rung)
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", backend=backend, error=f"{e!r}\n{traceback.format_exc()[-1500:]}")
        log(f"{job['id']}: FAILED {e!r}")
    wall = time.time() - t0
    rec.update(wall_s=round(wall, 1), gpu_s=round(wall, 1),
               peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None)
    log(f"{job['id']}: {rec['status']} in {wall:.0f}s")
    results.append(rec)
    save()

log("all jobs processed")
