"""A12 remaining folds on Kaggle's free 2x T4 (Modal stopped at the user's $20 line, 6 Oct 2026).

1. copy our code bundle (private dataset itsasup/ai4s-a12-bundle) to /kaggle/working/code
2. download the public OOC chip archive from Zenodo (parallel byte ranges), check the provider's MD5
3. make the same 8-bit grey, long-side-1036 photo arrays as cloud/modal_data.py, practice photos only
4. run a12_heatmaps.py: GPU0 fold 1 (both hiding styles; restart of the fold cut on Modal) then fold 2;
   GPU1 folds 3 and 4; folds 2-4 median fill only (user's budget choice, recorded)
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bundle():
    found = [p for p in Path("/kaggle/input").rglob("a12_heatmaps.py")]
    if found:
        src = found[0].parent.parent
        shutil.copytree(src, CODE, dirs_exist_ok=True)
    else:                                            # bundle kept as code.zip
        src = next(Path("/kaggle/input").rglob("code.zip"))
        with zipfile.ZipFile(src) as z:
            z.extractall(W)
        if not (CODE / "scripts").exists():
            inner = next(W.rglob("a12_heatmaps.py")).parent.parent
            shutil.copytree(inner, CODE, dirs_exist_ok=True)
    log(f"code copied from {src}")


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


def run_gpu(gpu, jobs):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), AI4S_DATA=str(DATA), AI4S_RUNS=str(CODE / "runs"), PYTHONUNBUFFERED="1")
    cmd = " && ".join(f"{sys.executable} scripts/a12_heatmaps.py {j}" for j in jobs)
    return subprocess.Popen(["bash", "-c", cmd], cwd=CODE, env=env, stdout=open(W / f"gpu{gpu}.log", "w"), stderr=subprocess.STDOUT)


def main():
    bundle()
    download()
    prep()
    t0 = time.time()
    p0 = run_gpu(0, ["--folds 1 --fills median,blur", "--folds 2 --fills median"])
    p1 = run_gpu(1, ["--folds 3 --fills median", "--folds 4 --fills median"])
    for name, p in (("gpu0", p0), ("gpu1", p1)):
        rc = p.wait()
        log(f"{name} finished with exit code {rc} after {time.time() - t0:.0f} s")
        print(open(W / f"{name}.log").read()[-3000:])
    shutil.rmtree(DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
