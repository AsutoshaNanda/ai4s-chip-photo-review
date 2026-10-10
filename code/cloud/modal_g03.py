"""G03 speed test on Modal GPUs (runs only after the user has signed up and run `modal setup`).

Usage (from the code folder):  .venv/bin/modal run cloud/modal_g03.py --gpu T4
Each call runs scripts/g03_speed_test.py on one GPU type and saves the JSON into runs/g03_modal_<gpu>.json.
Modal bills per second (GPU + CPU + memory); this job is short. The cost shown in the Modal dashboard is
recorded next to the result.
"""
import json
from pathlib import Path

import modal

CODE = Path(__file__).resolve().parents[1]
app = modal.App("ai4s-g03-speed")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "timm", "stable-baselines3", "gymnasium", "psutil")
         .add_local_file(str(CODE / "scripts" / "g03_speed_test.py"), "/root/g03_speed_test.py"))


def _run():
    import os
    import subprocess
    env = dict(os.environ, G03_OUT="/tmp")
    subprocess.run(["python", "/root/g03_speed_test.py"], env=env, check=False)
    out = [p for p in os.listdir("/tmp") if p.startswith("g03_results_")]
    return json.load(open(os.path.join("/tmp", out[0]))) if out else {"error": "no result file"}


@app.function(image=image, gpu="T4", cpu=4, memory=16384, timeout=3600)
def run_t4():
    return _run()


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=3600)
def run_l4():
    return _run()


@app.function(image=image, gpu="A10", cpu=4, memory=16384, timeout=3600)
def run_a10():
    return _run()


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=32768, timeout=3600)
def run_a100():
    return _run()


@app.local_entrypoint()
def main(gpu: str = "T4"):
    fn = {"T4": run_t4, "L4": run_l4, "A10": run_a10, "A100": run_a100}[gpu]
    res = fn.remote()
    out = CODE / "runs" / f"g03_modal_{gpu}.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)
