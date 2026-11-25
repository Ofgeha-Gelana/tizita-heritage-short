"""Kaggle kernel: Wan 2.2 TI2V-5B image-to-video for a batch of shots.

Reads jobs.json from the attached dataset, writes clips/<id>_<key>.mp4 and
results.json to /kaggle/working after EVERY job, so a timeout keeps finished work.

Backends, tried in order and recorded per job:
  1. diffusers WanImageToVideoPipeline with Wan-AI/Wan2.2-TI2V-5B-Diffusers
     (fp16 transformer for the T4, fp32 VAE, model CPU offload),
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

import torch  # noqa: E402

GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
log(f"GPU: {GPU}, count={torch.cuda.device_count()}")
# T4 (sm75) has no fast bf16; use fp16 there.
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and "T4" not in GPU else torch.float16

pipe = None
backend = None


def load_diffusers():
    global pipe, backend
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=DTYPE)
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_tiling()
    except Exception:  # noqa: BLE001
        pass
    backend = f"diffusers-i2v-{str(DTYPE).split('.')[-1]}"
    log(f"loaded {backend}")


def run_diffusers(job, out_path):
    from diffusers.utils import export_to_video, load_image
    img = load_image(os.path.join(IN_DIR, job["image"])).resize((job["width"], job["height"]))
    g = torch.Generator(device="cpu").manual_seed(job["seed"])
    frames = pipe(image=img, prompt=job["prompt"], negative_prompt=job.get("negative_prompt") or None,
                  height=job["height"], width=job["width"], num_frames=job["num_frames"],
                  num_inference_steps=job["steps"], guidance_scale=job["guidance_scale"], generator=g).frames[0]
    export_to_video(frames, out_path, fps=FPS)


def run_official(job, out_path):
    global backend
    if not os.path.exists("/kaggle/working/Wan2.2"):
        sh("git clone --depth 1 https://github.com/Wan-Video/Wan2.2 /kaggle/working/Wan2.2")
        sh(f"{sys.executable} -m pip install -q -r /kaggle/working/Wan2.2/requirements.txt || true")
        sh("huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir /kaggle/working/Wan2.2-TI2V-5B")
    backend = "official-generate.py"
    size = f"{job['width']}*{job['height']}"
    sh(f"cd /kaggle/working/Wan2.2 && {sys.executable} generate.py --task ti2v-5B --size {size} "
       f"--ckpt_dir /kaggle/working/Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu "
       f"--frame_num {job['num_frames']} --sample_steps {job['steps']} --base_seed {job['seed']} "
       f"--image {os.path.join(IN_DIR, job['image'])} --prompt {json.dumps(job['prompt'])} "
       f"--save_file {out_path}")


for job in JOBS:
    if (job["id"], job["key"]) in done:
        log(f"{job['id']}: already done")
        continue
    fname = f"{job['id']}_{job['key']}.mp4"
    out_path = f"{WORK}/clips/{fname}"
    rec = {**{k: job[k] for k in ("id", "key", "seed", "width", "height", "num_frames", "steps",
                                  "guidance_scale", "quality")}, "file": fname, "gpu_name": GPU}
    t0 = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
        try:
            if pipe is None and backend is None:
                load_diffusers()
            if pipe is not None:
                run_diffusers(job, out_path)
            else:
                run_official(job, out_path)
        except Exception as e:  # noqa: BLE001
            log(f"{job['id']}: diffusers path failed: {e!r}; trying official repo")
            pipe = None
            torch.cuda.empty_cache()
            run_official(job, out_path)
        rec.update(status="ok", backend=backend)
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
