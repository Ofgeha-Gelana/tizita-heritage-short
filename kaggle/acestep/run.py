"""Kaggle kernel: ACE-Step 1.5 instrumental score candidates.

One job per seed. Uses the documented Python API (docs/en/INFERENCE.md):
    generate_music(dit_handler, llm_handler, GenerationParams, GenerationConfig, save_dir)
Writes music/<id>_<key>.wav and results.json after every job.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

WORK = "/kaggle/working"
os.makedirs(f"{WORK}/music", exist_ok=True)
LOG = open(f"{WORK}/acestep.log", "a")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def sh(cmd):
    log("$ " + cmd)
    subprocess.run(cmd, shell=True, check=True)


jobs_file = sorted(glob.glob("/kaggle/input/**/jobs.json", recursive=True))[0]
spec = json.load(open(jobs_file))
JOBS = spec["jobs"]
results_path = f"{WORK}/results.json"
results = []

REPO = "/kaggle/working/ACE-Step-1.5"
if not os.environ.get("ACE_INNER"):
    # Run 1 (FAILURES.md F9): Kaggle's image is Python 3.13, ACE-Step 1.5 needs <3.13, and
    # its nano-vllm dependency only has a prebuilt flash-attn wheel for 3.11. So follow the
    # README (`uv sync`) with a uv-managed 3.11, then re-run this script inside that venv.
    if not os.path.exists(REPO):
        sh(f"git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5 {REPO}")
    sh(f"{sys.executable} -m pip install -q uv")
    sh(f"cd {REPO} && {sys.executable} -m uv sync --python 3.11")
    venv_py = f"{REPO}/.venv/bin/python"
    log(f"re-running inside {venv_py}")
    env = {**os.environ, "ACE_INNER": "1"}
    sys.stdout.flush()
    LOG.close()
    os.execve(venv_py, [venv_py, os.path.abspath(sys.argv[0])], env)
sys.path.insert(0, REPO)

import torch  # noqa: E402
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
log(f"GPU: {GPU}")

t_load = time.time()
dit = AceStepHandler()
dit.initialize_service(project_root=REPO, config_path="acestep-v15-turbo", device="cuda")
llm = None
use_lm = spec["settings"].get("use_lm", True)
if use_lm:
    try:
        from acestep.llm_inference import LLMHandler
        llm = LLMHandler()
        # flash-attn (used by the vllm backend) needs sm80+; a T4 is sm75, so use the
        # transformers backend there. Elsewhere try vllm first, then pt, then no planner.
        for backend in (("pt",) if "T4" in GPU else ("vllm", "pt")):
            try:
                llm.initialize(checkpoint_dir=os.path.join(REPO, "checkpoints"), lm_model_path="acestep-5Hz-lm-0.6B",
                               backend=backend, device="cuda")
                log(f"LM planner loaded with backend={backend}")
                break
            except Exception as e:  # noqa: BLE001
                log(f"LM backend {backend} failed: {e!r}")
        else:
            llm = None
    except Exception as e:  # noqa: BLE001
        log(f"LM planner unavailable: {e!r}")
        llm = None
log(f"models loaded in {time.time() - t_load:.0f}s (planner={'on' if llm else 'off'})")

for job in JOBS:
    fname = f"{job['id']}_{job['key']}.wav"
    rec = {"id": job["id"], "key": job["key"], "seed": job["seed"], "file": fname, "gpu_name": GPU,
           "planner": bool(llm)}
    t0 = time.time()
    try:
        torch.cuda.reset_peak_memory_stats()
        params = GenerationParams(caption=job["caption"], duration=float(job["duration_s"]), bpm=job["bpm"],
                                  keyscale=job["key_scale"], seed=job["seed"], instrumental=True,
                                  inference_steps=8, shift=3.0, task_type="text2music", thinking=bool(llm))
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[job["seed"]], audio_format="wav")
        tmp = f"{WORK}/tmp_{job['id']}"
        res = generate_music(dit, llm, params, config, save_dir=tmp)
        if not res.success:
            raise RuntimeError(f"generate_music failed: {res.error or res.status_message}")
        shutil.move(res.audios[0]["path"], f"{WORK}/music/{fname}")
        shutil.rmtree(tmp, ignore_errors=True)
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001
        rec.update(status="error", error=f"{e!r}\n{traceback.format_exc()[-1500:]}")
        log(f"{job['id']}: FAILED {e!r}")
    wall = time.time() - t0
    rec.update(wall_s=round(wall, 1), gpu_s=round(wall, 1),
               peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    log(f"{job['id']}: {rec['status']} in {wall:.0f}s")
    results.append(rec)
    json.dump({"results": results}, open(results_path, "w"), indent=2)

log("done")
