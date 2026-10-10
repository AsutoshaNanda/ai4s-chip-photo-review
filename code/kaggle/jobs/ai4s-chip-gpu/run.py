"""Chip-photo GPU work on Kaggle's free 2x T4 (Modal stopped; user decision 6 Oct 2026: rest on Kaggle, nothing skipped).

1. code bundle (private dataset itsasup/ai4s-code) -> /kaggle/working/code
2. public OOC chip archive from Zenodo (provider MD5 checked), same grey long-side-1036 arrays as before, practice only
3. step 1: GPU0 A02 light-retraining timing on T4 (plan: measure before scaling up) | GPU1 A05 changed-copy features
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

OOC_URL = "https://zenodo.org/api/records/10203721/files/OOC_image_dataset.zip/content"
OOC_MD5, OOC_SIZE = "8f7e058996203d48eb03b2d86c0a2e4d", 6710767405
LONG_SIDE = 1036
W = Path("/kaggle/working")
CODE = W / "code"
DATA = Path("/tmp/ai4s/data")
ZIP = Path("/tmp/ai4s/OOC_image_dataset.zip")
LANES = {0: ["scripts/a02_light_retrain.py --mode timing"],
         1: ["scripts/a05_confirm.py --part features"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bundle():
    hits = list(Path("/kaggle/input").rglob("runlog.py"))
    if hits:
        shutil.copytree(hits[0].parents[2], CODE, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(next(Path("/kaggle/input").rglob("code.zip"))) as z:
            z.extractall(W)
    log(f"code ready: {sorted(p.name for p in CODE.iterdir())}")


def download():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 12
    chunk = (OOC_SIZE + n - 1) // n
    procs = [subprocess.Popen(["curl", "-sS", "--retry", "10", "-r", f"{i * chunk}-{min(OOC_SIZE - 1, (i + 1) * chunk - 1)}",
                               "-o", f"/tmp/ai4s/part{i}", OOC_URL]) for i in range(n)]
    for p in procs:
        p.wait()
    md5 = hashlib.md5()
    with open(ZIP, "wb") as f:
        for i in range(n):
            with open(f"/tmp/ai4s/part{i}", "rb") as g:
                for b in iter(lambda: g.read(1 << 24), b""):
                    f.write(b)
                    md5.update(b)
            os.remove(f"/tmp/ai4s/part{i}")
    ok = md5.hexdigest() == OOC_MD5
    log(f"zenodo download {ZIP.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s, MD5 ok={ok}")
    if not ok:
        raise SystemExit("MD5 mismatch")


def prep_one(member):
    z = zipfile.ZipFile(ZIP)
    im = Image.open(io.BytesIO(z.read(member)))
    im.load()
    w, h = im.size
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    s = LONG_SIDE / max(w, h)
    g = np.asarray(Image.fromarray(g).resize((round(w * s), round(h * s)), Image.BILINEAR), dtype=np.uint8)
    name = member.rsplit("/", 1)[1].rsplit(".", 1)[0]
    np.save(DATA / "ooc" / f"gray{LONG_SIDE}" / f"{name}.npy", g)
    return name


def prep():
    t0 = time.time()
    (DATA / "ooc" / f"gray{LONG_SIDE}").mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(CODE / "splits" / "ooc_splits.csv", dtype={"image_id": str})
    want = set(sp.loc[sp.role == "practice", "image_id"])
    members = [m for m in zipfile.ZipFile(ZIP).namelist()
               if m.lower().endswith(".png") and m.rsplit("/", 1)[1].rsplit(".", 1)[0] in want]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        done = list(ex.map(prep_one, members, chunksize=16))
    log(f"prepared {len(done)} practice photos (of {len(want)} listed) in {time.time() - t0:.0f} s")
    ZIP.unlink()


def lane(gpu, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AI4S_DATA=str(DATA), AI4S_RUNS=str(CODE / "runs"), PYTHONUNBUFFERED="1")
    cmd = " ; ".join(f"{sys.executable} {j}" for j in jobs)          # ';' so one failure does not stop the next job
    return subprocess.Popen(["bash", "-c", cmd], cwd=CODE, env=env, stdout=open(W / f"gpu{gpu}.log", "w"), stderr=subprocess.STDOUT)


def main():
    bundle()
    download()
    prep()
    t0 = time.time()
    ps = {g: lane(g, jobs) for g, jobs in LANES.items()}
    for g, p in ps.items():
        rc = p.wait()
        log(f"gpu{g} finished with exit code {rc} after {time.time() - t0:.0f} s")
        print(open(W / f"gpu{g}.log").read()[-4000:])
    shutil.rmtree(DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
