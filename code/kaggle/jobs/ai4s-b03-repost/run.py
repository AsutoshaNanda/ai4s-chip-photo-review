"""Generic Kaggle runner (the Mac must not run compute; Modal is at the user's money line).
Filled in by make_jobs.py: JOB (command after `python`), FETCH (public data to download first).
Our code comes from the private dataset itsasup/ai4s-code; outputs land in /kaggle/working/code/runs."""
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

JOB = "scripts/b_repost.py --exp B03"
FETCH = "bbbc039"
W = Path("/kaggle/working")
CODE = W / "code"
DATA = Path("/tmp/ai4s/data")


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
    if "bbbc039" in FETCH:
        d = DATA / "bbbc039"
        d.mkdir(parents=True, exist_ok=True)
        for z in ("images.zip", "masks.zip", "metadata.zip"):
            urllib.request.urlretrieve(f"https://data.broadinstitute.org/bbbc/BBBC039/{z}", d / z)
        log("BBBC039 (CC0) downloaded")


def main():
    code()
    fetch()
    env = dict(os.environ, AI4S_DATA=str(DATA), AI4S_RUNS=str(CODE / "runs"), PYTHONUNBUFFERED="1")
    t0 = time.time()
    rc = subprocess.call([sys.executable, *JOB.split()], cwd=CODE, env=env)
    log(f"job '{JOB}' exit {rc} after {time.time() - t0:.0f} s")
    shutil.rmtree(DATA, ignore_errors=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
