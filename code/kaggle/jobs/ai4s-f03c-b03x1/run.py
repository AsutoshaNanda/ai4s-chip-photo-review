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

LANES = [(None, ['scripts/b_repost.py --exp B03 --seed 1 --thr_extra 0.96,0.97,0.98,0.99,0.995,0.999'])]
FETCH = ['bbbc039', 'venv:requirements']
W = Path("/kaggle/working")
CODE = W / "code"
DATA = Path("/tmp/ai4s/data")
VENV = Path("/tmp/ai4s/venv")
PIN_CHECK = ("import importlib.metadata as m,sys\nbad=[]\nfor p in sys.argv[1:]:\n    n,_,v=p.partition('==')\n"
             "    got=m.version(n)\n    print(n,got)\n    if v and got.split('+')[0]!=v: bad.append((n,v,got))\n"
             "sys.exit(str(bad) if bad else 0)\n")
PY = sys.executable                     # becomes the virtual environment's python after a "venv:requirements" fetch
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


def venv():
    """Fresh virtual environment with EXACTLY the versions declared in code/requirements.txt (lines above the RL-only marker).
    Every pin is checked after installation; a mismatch stops the job so nothing is reported as 'pinned' when it is not."""
    global PY
    r = subprocess.run([sys.executable, "-m", "venv", str(VENV)], capture_output=True, text=True)
    if r.returncode:                                   # Kaggle's python has no ensurepip: use the virtualenv package instead
        log(f"python -m venv failed ({r.stderr.strip()[-200:]}); using virtualenv")
        shutil.rmtree(VENV, ignore_errors=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])
        subprocess.check_call([sys.executable, "-m", "virtualenv", "-q", str(VENV)])
    py = str(VENV / "bin" / "python")
    pins = []
    for line in (CODE / "requirements.txt").read_text().splitlines():
        if line.startswith("# only for the control-simulator"):
            break
        if line.strip() and not line.startswith("#"):
            pins.append(line.strip())
    subprocess.check_call([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", *pins])
    out = subprocess.run([py, "-c", PIN_CHECK, *pins], capture_output=True, text=True)
    (W / "venv_versions.txt").write_text(out.stdout + out.stderr)
    log(f"virtual environment versions: {' '.join(out.stdout.split())}")
    if out.returncode:
        raise SystemExit(f"pinned versions not reproduced: {out.stderr.strip()[-300:]}")
    PY = py


def fetch():
    for item in FETCH:
        if item == "venv:requirements":
            venv()
        elif item.startswith("pip:"):
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
    cmd = " ; ".join(f"echo '=== {j}' ; {PY} {j}" for j in jobs)      # ';' : one failure does not stop the lane
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
