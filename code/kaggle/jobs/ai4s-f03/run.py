"""F03 clean rebuild on a Kaggle GPU: fresh virtual environment with the pinned package versions, public inputs only
(Zenodo OOC archive with checksum, BBBC039), our code + splits + frozen config from the code dataset, and NO saved features,
models or run folders. Rebuilds the frozen chip-photo pipeline end to end (photo preparation -> features -> classifier ->
final evaluation) and retrains ONE nucleus fold from scratch. Every step is timed; failures are recorded, not hidden.
Comparison with the saved development/final records is done afterwards by scripts/f03_compare.py (differences reported, no pass mark).
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

W = Path("/kaggle/working")
CODE = W / "code"
DATA = Path("/tmp/ai4s/data")
ZIP = Path("/tmp/ai4s/OOC_image_dataset.zip")
VENV = Path("/tmp/f03env")
OOC_URL = "https://zenodo.org/api/records/10203721/files/OOC_image_dataset.zip/content"
OOC_MD5, OOC_SIZE = "8f7e058996203d48eb03b2d86c0a2e4d", 6710767405
PKGS = ['numpy==2.5.3', 'pandas==3.0.6', 'scipy==1.18.1', 'scikit-learn==1.9.1', 'scikit-image==0.26.0', 'tifffile==2026.9.20',
        'pillow==12.3.0', 'openpyxl==3.1.5', 'matplotlib==3.11.2', 'pyarrow==25.0.1', 'torch==2.14.1', 'torchvision==0.29.1',
        'timm==1.0.30', 'psutil']
STEPS = []


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def step(name, fn):
    t0 = time.time()
    try:
        out = fn()
        STEPS.append({"step": name, "seconds": round(time.time() - t0, 1), "ok": True, "note": out if isinstance(out, str) else ""})
    except BaseException as e:                                     # noqa: BLE001 - record and continue/stop explicitly
        STEPS.append({"step": name, "seconds": round(time.time() - t0, 1), "ok": False, "error": f"{type(e).__name__}: {e}"})
        log(f"STEP FAILED {name}: {e}")
        raise
    finally:
        (W / "f03_steps.json").write_text(json.dumps(STEPS, indent=1))
        log(f"step {name}: {STEPS[-1]}")


def code():
    hits = list(Path("/kaggle/input").rglob("runlog.py"))
    src = hits[0].parents[2]
    for d in ("src", "scripts", "splits"):
        shutil.copytree(src / d, CODE / d, dirs_exist_ok=True)
    shutil.copy(src / "final_config.json", CODE / "final_config.json")
    (CODE / "runs").mkdir(exist_ok=True)                           # EMPTY: no saved results are used
    return f"code copied: {sorted(p.name for p in CODE.iterdir())}; runs/ empty"


def venv():
    r = subprocess.run([sys.executable, "-m", "venv", str(VENV)], capture_output=True, text=True)
    if r.returncode:                                   # Kaggle's python has no ensurepip: use the virtualenv package instead (recorded)
        log(f"python -m venv failed ({r.stderr.strip()[-200:]}); using virtualenv")
        shutil.rmtree(VENV, ignore_errors=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])
        subprocess.check_call([sys.executable, "-m", "virtualenv", "-q", str(VENV)])
    py = str(VENV / "bin" / "python")
    subprocess.check_call([py, "-m", "pip", "install", "-q", *PKGS])
    out = subprocess.run([py, "-c", "import numpy,pandas,sklearn,skimage,torch,timm;print(numpy.__version__,pandas.__version__,sklearn.__version__,skimage.__version__,torch.__version__,timm.__version__)"], capture_output=True, text=True)
    return f"installed; versions: {out.stdout.strip()} {out.stderr.strip()[:200]}"


def download():
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    n = 12
    chunk = (OOC_SIZE + n - 1) // n
    procs = [subprocess.Popen(["curl", "-sS", "--retry", "10", "-r", f"{i * chunk}-{min(OOC_SIZE - 1, (i + 1) * chunk - 1)}", "-o", f"/tmp/ai4s/part{i}", OOC_URL]) for i in range(n)]
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
    if md5.hexdigest() != OOC_MD5:
        raise SystemExit("MD5 mismatch")
    return f"{ZIP.stat().st_size / 1e9:.2f} GB, MD5 ok"


def prep_one(member):
    import numpy as np
    from PIL import Image
    z = zipfile.ZipFile(ZIP)
    im = Image.open(io.BytesIO(z.read(member)))
    im.load()
    w, h = im.size
    g = np.asarray(im.convert("L"), dtype=np.uint8)
    s = 1036 / max(w, h)
    g = np.asarray(Image.fromarray(g).resize((round(w * s), round(h * s)), Image.BILINEAR), dtype=np.uint8)
    name = member.rsplit("/", 1)[1].rsplit(".", 1)[0]
    np.save(DATA / "ooc" / "gray1036" / f"{name}.npy", g)
    return name, im.mode, w, h, member


def prep():
    import pandas as pd
    (DATA / "ooc" / "gray1036").mkdir(parents=True, exist_ok=True)
    sp = pd.read_csv(CODE / "splits" / "ooc_splits.csv", dtype={"image_id": str})
    want = set(sp["image_id"])
    members = [m for m in zipfile.ZipFile(ZIP).namelist() if m.lower().endswith(".png") and m.rsplit("/", 1)[1].rsplit(".", 1)[0] in want]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        done = list(ex.map(prep_one, members, chunksize=16))
    # minimal "G01" table rebuilt from the file headers: only the columns the later steps use (member, mode, width, height)
    d = pd.DataFrame([{"member": m, "mode": mode, "width": w, "height": h} for _, mode, w, h, m in done])
    run = CODE / "runs" / time.strftime("%Y%m%d-%H%M%S_G01_ooc_check_clean")
    run.mkdir(parents=True)
    d.to_csv(run / "per_photo.csv", index=False)
    return f"prepared {len(done)} photos of {len(want)} listed (practice + final); format table rebuilt from file headers"


def sub(py, args, cwd=CODE):
    env = dict(os.environ, AI4S_DATA=str(DATA), AI4S_RUNS=str(CODE / "runs"), PYTHONUNBUFFERED="1")
    p = subprocess.run([py, *args], cwd=cwd, env=env, capture_output=True, text=True)
    print(p.stdout[-6000:], p.stderr[-3000:], flush=True)
    if p.returncode:
        raise RuntimeError(f"exit {p.returncode}: {p.stderr[-800:]}")
    return p.stdout[-300:]


def main():
    step("code", code)
    step("venv", venv)
    py = str(VENV / "bin" / "python")
    step("download", download)
    step("prep", prep)
    step("chip_features", lambda: sub(py, ["scripts/f01_final.py", "--part", "chip_features", "--roles", "final_test,practice", "--tag", "clean"]))
    feat = sorted((CODE / "runs").glob("*F03_chip_features_clean"))[-1]
    step("chip_eval", lambda: sub(py, ["scripts/f01_final.py", "--part", "chip_eval", "--fresh", str(feat), "--tag", "clean"]))

    step("practice_check", lambda: sub(py, ["scripts/f03_practice_check.py"]))

    def nucleus():
        d = DATA / "bbbc039"
        d.mkdir(parents=True, exist_ok=True)
        for z in ("images.zip", "masks.zip", "metadata.zip"):
            urllib.request.urlretrieve(f"https://data.broadinstitute.org/bbbc/BBBC039/{z}", d / z)
        return sub(py, ["scripts/b_unet.py", "--target", "boundary", "--seed", "0", "--folds", "0"])
    step("nucleus_fold0_retrain", nucleus)
    shutil.rmtree(DATA, ignore_errors=True)
    new = sorted(p.name for p in (CODE / "runs").iterdir())
    for sub_ in ("src", "scripts", "splits"):
        shutil.rmtree(CODE / sub_, ignore_errors=True)
    log(f"outputs: {new}")


if __name__ == "__main__":
    main()
