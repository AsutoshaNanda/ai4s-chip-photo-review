"""Kaggle runner with parallel lanes (one per GPU, or CPU lanes). Filled in by make_jobs2.py.
Code: private dataset itsasup/ai4s-code. Public data fetched here. Outputs: /kaggle/working/code/runs."""
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

LANES = [(0, ['scripts/c03_fetch.py', 'scripts/c03_prep.py --depth_rule input_only', 'scripts/c03_translate.py --slices c03_slices_inputonly.npz', 'scripts/c03_prep.py --depth_rule target', 'scripts/c03_translate.py --slices c03_slices.npz'])]
FETCH = ['pip:czifile==2026.8.16,imagecodecs==2026.8.16']
W = Path("/kaggle/working")
CODE = W / "code"
DATA = Path("/tmp/ai4s/data")
BBBC038_SHA256 = "dcb6edc2690f137406638b2309581a71522c4dff19157d118453b448dcddcb68"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def code():
    hits = list(Path("/kaggle/input").rglob("runlog.py"))
    if hits:
        shutil.copytree(hits[0].parents[2], CODE, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(next(Path("/kaggle/input").rglob("code.zip"))) as z:
            z.extractall(W)
    log(f"code ready: {sorted(p.name for p in CODE.iterdir())}")


def fetch():
    for item in FETCH:
        if item.startswith("pip:"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *item[4:].split(",")])
            log(f"installed {item[4:]}")
        elif item == "bbbc039":
            d = DATA / "bbbc039"
            d.mkdir(parents=True, exist_ok=True)
            for z in ("images.zip", "masks.zip", "metadata.zip"):
                urllib.request.urlretrieve(f"https://data.broadinstitute.org/bbbc/BBBC039/{z}", d / z)
            log("BBBC039 (CC0) downloaded")
        elif item == "bbbc038":
            d = DATA / "bbbc038"
            d.mkdir(parents=True, exist_ok=True)
            z = d / "stage1_train.zip"
            urllib.request.urlretrieve("https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip", z)
            ok = hashlib.sha256(z.read_bytes()).hexdigest() == BBBC038_SHA256
            log(f"BBBC038 stage1_train downloaded, SHA256 ok={ok}")
            if not ok:
                raise SystemExit("BBBC038 checksum mismatch")
            with zipfile.ZipFile(z) as zz:
                zz.extractall(d / "stage1_train")


def lane(i, gpu, jobs):
    env = dict(os.environ, AI4S_DATA=str(DATA), AI4S_RUNS=str(CODE / "runs"), PYTHONUNBUFFERED="1")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = " ; ".join(f"echo '=== {j}' ; {sys.executable} {j}" for j in jobs)      # ';' : one failure does not stop the lane
    return subprocess.Popen(["bash", "-c", cmd], cwd=CODE, env=env, stdout=open(W / f"lane{i}.log", "w"), stderr=subprocess.STDOUT)


def main():
    code()
    before = {d.name for d in (CODE / "runs").iterdir()} if (CODE / "runs").exists() else set()
    fetch()
    t0 = time.time()
    ps = [(i, lane(i, gpu, jobs)) for i, (gpu, jobs) in enumerate(LANES)]
    for i, p in ps:
        rc = p.wait()
        log(f"lane{i} finished with exit code {rc} after {time.time() - t0:.0f} s")
        print(open(W / f"lane{i}.log").read()[-6000:])
    shutil.rmtree(DATA, ignore_errors=True)
    for d in (CODE / "runs").iterdir():                  # keep only NEW run folders in the output (small downloads)
        if d.name in before:
            shutil.rmtree(d, ignore_errors=True) if d.is_dir() else d.unlink()
    for sub in ("src", "scripts", "splits"):
        shutil.rmtree(CODE / sub, ignore_errors=True)
    log(f"kept new runs: {sorted(p.name for p in (CODE / 'runs').iterdir())}")


if __name__ == "__main__":
    main()
