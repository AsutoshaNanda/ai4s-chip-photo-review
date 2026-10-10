"""Chip-photo GPU work on Kaggle's free 2x T4 (Modal stopped; user decision 6 Oct 2026: rest on Kaggle, nothing skipped).

1. code bundle (private dataset itsasup/ai4s-code) -> /kaggle/working/code
2. public OOC chip archive from Zenodo (provider MD5 checked), same grey long-side-1036 arrays as before, practice only
demo
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



import pandas as pd

import base64
import zlib

import requests


class RemoteFile:
    """Seekable read-only file over HTTP range requests, so zipfile can pull single photos out of the 6.7 GB archive."""
    def __init__(self, url, size):
        self.url, self.size, self.pos, self.s = url, size, 0, requests.Session()

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        for attempt in range(6):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, timeout=120)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError("range request failed")
        self.pos += len(r.content)
        return r.content


G01_CLEAN_CSV = "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"      # per_photo.csv of the clean rebuild (camera format of every photo), embedded so no bundle upload is needed


def write_format_table():
    d = CODE / "runs" / "20261007-150641_G01_ooc_check_clean"
    d.mkdir(parents=True, exist_ok=True)
    (d / "per_photo.csv").write_bytes(zlib.decompress(base64.b64decode(G01_CLEAN_CSV)))

DEMO = W / "demo_photos"


def extract_demo():
    sp = pd.read_csv(CODE / "splits" / "ooc_splits.csv", dtype={"image_id": str, "group": str})
    fin = sp[sp.role == "final_test"]
    rng = __import__("numpy").random.default_rng(20261007)
    pick = fin.sample(12, random_state=20261007)                      # 12 random locked-final photos (fixed seed, no cherry-picking)
    hdr = pd.read_csv(CODE / "runs" / sorted(p.name for p in (CODE / "runs").iterdir() if "G01_ooc_check_clean" in p.name)[-1] / "per_photo.csv")
    fmt = hdr.assign(key=hdr["mode"] + "/" + hdr["width"].astype(str) + "x" + hdr["height"].astype(str)).set_index("member")["key"]
    z = zipfile.ZipFile(RemoteFile(OOC_URL, OOC_SIZE))
    members = {m.rsplit("/", 1)[1].rsplit(".", 1)[0]: m for m in z.namelist() if m.lower().endswith(".png")}
    extra = []
    for _, r in sp.iterrows():                                        # all unseen-format final photos + the first two small-format development photos
        k = fmt.get(r["member"], "")
        if (r.role == "final_test" and k not in ("L/2056x1542", "RGB/2048x1536")) or (r.role == "practice" and k == "RGB/640x480" and len([e for e in extra if e[1] == "small"]) < 2):
            extra.append((r.image_id, "unseen" if r.role == "final_test" else "small"))
    chosen = pd.concat([pick[["image_id", "label_bad", "role", "cell_type"]], sp[sp.image_id.isin([e[0] for e in extra])][["image_id", "label_bad", "role", "cell_type"]]]).drop_duplicates("image_id")
    DEMO.mkdir(parents=True, exist_ok=True)
    for _, r in chosen.iterrows():
        (DEMO / f"{r.image_id}.png").write_bytes(z.read(members[r.image_id]))
    chosen.to_csv(W / "demo_labels.csv", index=False)
    log(f"extracted {len(chosen)} demo photos")


def prep_clean_formats():
    # the shipped model needs the development format table and the freshly computed development features (from the F03 clean rebuild)
    pass


def run(cmd):
    env = dict(os.environ, AI4S_DATA=str(DATA), AI4S_RUNS=str(CODE / "runs"), PYTHONUNBUFFERED="1")
    p = subprocess.run([sys.executable, *cmd], cwd=CODE, env=env, capture_output=True, text=True)
    print(p.stdout[-5000:], p.stderr[-2000:], flush=True)
    if p.returncode:
        raise SystemExit(f"{cmd} failed")


def main():
    bundle()
    write_format_table()
    extract_demo()
    run(["scripts/train_final_model.py", "--out", str(W)])
    shutil.copy(W / "model_x01_3x3.json", W / "model_x01_3x3.json.bak")
    run(["inference.py", "--photos", str(DEMO), "--out", str(W / "demo_ranking.csv"), "--sheet", str(W / "demo_sheet.png"), "--model", str(W / "model_x01_3x3.joblib")])
    lab = pd.read_csv(W / "demo_labels.csv", dtype={"image_id": str}).set_index("image_id")
    r = pd.read_csv(W / "demo_ranking.csv")
    r["image_id"] = r["photo"].str.rsplit("/", n=1).str[1].str.replace(".png", "", regex=False)
    r = r.join(lab, on="image_id")
    r.to_csv(W / "demo_ranking_with_labels.csv", index=False)
    print(r.to_string(index=False))
    for sub in ("code",):
        shutil.rmtree(CODE / "runs", ignore_errors=True)


if __name__ == "__main__":
    main()
