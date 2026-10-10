"""Run any experiment script on Modal (big CPU machine or A100 GPU) with our code, splits and data attached.

Usage (from the code folder):
  .venv/bin/modal run cloud/modal_run.py --where cpu32 --script scripts/b01_cv.py
  .venv/bin/modal run cloud/modal_run.py --where a100  --script scripts/a02_features.py --args "--model vit_large"

How results come home: scripts write their run folder to /data/runs on the Modal volume "ai4s-data"; this
launcher returns every new run folder (logs, settings, predictions, model files) and unpacks it into
code/runs/ on the Mac, then adds the runs to code/runs/registry.csv. Heavy files also stay on the volume for
reuse by later Modal jobs. Package versions are pinned to code/requirements-lock-mac.txt.
"""
import csv
import io
import json
import tarfile
from pathlib import Path

import modal

CODE = Path(__file__).resolve().parents[1]
# Same versions as code/requirements-lock-mac.txt, written out here because this file is also read on the cloud machine
PKGS = ['numpy==2.5.3', 'pandas==3.0.6', 'scipy==1.18.1', 'scikit-learn==1.9.1', 'scikit-image==0.26.0',
        'tifffile==2026.9.20', 'pillow==12.3.0', 'openpyxl==3.1.5', 'matplotlib==3.11.2', 'pyarrow==25.0.1',
        'torch==2.14.1', 'torchvision==0.29.1', 'timm==1.0.30', 'gymnasium==1.4.0', 'stable-baselines3==2.9.0', 'psutil']
HEAVY = ["**/*.npz", "**/*.npy", "**/*.pt", "**/*.pth", "**/*.parquet", "**/*.zip", "**/*.png", "**/*.log", "*.log"]

app = modal.App("ai4s-run")
vol = modal.Volume.from_name("ai4s-data", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install(*PKGS)
         .add_local_dir(str(CODE / "src"), "/root/code/src")
         .add_local_dir(str(CODE / "scripts"), "/root/code/scripts")
         .add_local_dir(str(CODE / "splits"), "/root/code/splits")
         .add_local_dir(str(CODE / "runs"), "/root/code/runs", ignore=HEAVY))


def _run(script, args, workers):
    import os
    import subprocess
    import sys
    import time
    os.makedirs("/data/runs", exist_ok=True)
    vol.reload()
    before = {d for d in os.listdir("/data/runs") if os.path.isdir(f"/data/runs/{d}")}
    env = dict(os.environ, AI4S_DATA="/data", AI4S_RUNS="/data/runs", AI4S_WORKERS=str(workers), PYTHONUNBUFFERED="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, f"/root/code/{script}", *args.split()], env=env, cwd="/root/code",
                       capture_output=True, text=True)
    vol.commit()
    new = sorted({d for d in os.listdir("/data/runs") if os.path.isdir(f"/data/runs/{d}")} - before)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for d in new:
            tar.add(f"/data/runs/{d}", arcname=d)
    return {"returncode": p.returncode, "stdout_tail": p.stdout[-15000:], "stderr_tail": p.stderr[-15000:],
            "seconds": round(time.time() - t0, 1), "new_runs": new, "tar": buf.getvalue()}


@app.function(image=image, volumes={"/data": vol}, cpu=32, memory=65536, timeout=6 * 3600)
def cpu32(script: str, args: str = ""):
    return _run(script, args, 32)


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=32768, timeout=6 * 3600)
def cpu8(script: str, args: str = ""):
    return _run(script, args, 8)


@app.function(image=image, volumes={"/data": vol}, gpu="A100-40GB", cpu=8, memory=32768, timeout=6 * 3600)
def a100(script: str, args: str = ""):
    return _run(script, args, 8)


@app.local_entrypoint()
def main(script: str, where: str = "cpu8", args: str = ""):
    fn = {"cpu32": cpu32, "cpu8": cpu8, "a100": a100}[where]
    res = fn.remote(script, args)
    runs = CODE / "runs"
    with tarfile.open(fileobj=io.BytesIO(res.pop("tar")), mode="r:gz") as tar:
        tar.extractall(runs, filter="data")
    reg = runs / "registry.csv"
    rows = list(csv.DictReader(open(reg))) if reg.exists() else []
    known = {r["run_id"] for r in rows}
    for d in res["new_runs"]:
        if d in known:
            continue
        t = json.load(open(runs / d / "timing.json")) if (runs / d / "timing.json").exists() else {}
        env = json.load(open(runs / d / "env.json")) if (runs / d / "env.json").exists() else {}
        parts = d.split("_", 2)
        rows.append({"run_id": d, "plan_id": parts[1] if len(parts) > 1 else "", "name": parts[2] if len(parts) > 2 else d,
                     "status": "measured" if res["returncode"] == 0 and t else "failed",
                     "started": t.get("started", ""), "finished": t.get("finished", ""), "seconds": t.get("wall_seconds", ""),
                     "machine": f"modal:{where}:" + "+".join(env.get("cuda_devices", []) or ["cpu"]),
                     "notes": f"run on Modal ({where}) via cloud/modal_run.py"})
    if rows:
        with open(reg, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps({k: v for k, v in res.items() if k not in ("stdout_tail", "stderr_tail")}, indent=1))
    print("---- stdout (tail) ----\n" + res["stdout_tail"][-4000:])
    if res["returncode"] != 0:
        print("---- stderr (tail) ----\n" + res["stderr_tail"][-6000:])
