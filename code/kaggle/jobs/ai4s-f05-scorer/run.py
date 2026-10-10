"""F05 (8 Oct 2026): rebuild the shipped chip-photo scorer under the versions declared in requirements.txt, then test a fresh installation.

Free Kaggle CPU kernel (Linux, internet on). Steps, each logged to /kaggle/working:
  1. fresh virtual environment + the literal README command `pip install -r requirements.txt`; every pin compared with what got installed
  2. control: the OLD scorer (saved under scikit-learn 1.6.1) must raise when scikit-learn version warnings are turned into errors
  3. scripts/train_final_model.py (development photos only; the locked final photos are never used) -> new model + json
  4. the new model must load with that same strict setting
  5. clean copy of the package with NO runs/ folder + the new model; the 17 demo photos are pulled from the public Zenodo archive by file name
  6. `python inference.py --photos ... --out ... --sheet ...` on CPU with the same strict setting
Nothing here selects anything with the final photos."""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

W = Path("/kaggle/working")
T = Path("/tmp/ai4s_f05")
VENV = T / "venv"
OOC_URL = "https://zenodo.org/api/records/10203721/files/OOC_image_dataset.zip/content"
OOC_SIZE = 6710767405
STRICT = "error::sklearn.exceptions.InconsistentVersionWarning"
REPORT = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()), "steps": {}, "checks": {}}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(name, cmd, cwd, env=None):
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=dict(os.environ, **(env or {})), capture_output=True, text=True)
    dt = time.time() - t0
    (W / f"{name}.log").write_text(f"$ {' '.join(map(str, cmd))}\n[exit {p.returncode}, {dt:.1f} s]\n--- stdout ---\n{p.stdout}\n--- stderr ---\n{p.stderr}")
    REPORT["steps"][name] = {"cmd": " ".join(map(str, cmd)), "exit": p.returncode, "seconds": round(dt, 1)}
    log(f"{name}: exit {p.returncode} in {dt:.0f} s")
    return p


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


def make_venv():
    r = subprocess.run([sys.executable, "-m", "venv", str(VENV)], capture_output=True, text=True)
    if r.returncode:                                    # Kaggle's python has no ensurepip: use the virtualenv package instead
        shutil.rmtree(VENV, ignore_errors=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "virtualenv"])
        subprocess.check_call([sys.executable, "-m", "virtualenv", "-q", str(VENV)])
    return VENV / "bin" / "python"


PIN_CHECK = """import importlib.metadata as m, json, sys
out = {}
for line in open('requirements.txt').read().splitlines():
    line = line.split('#')[0].strip()
    if not line:
        continue
    name, _, want = line.partition('==')
    try:
        got = m.version(name)
    except Exception as e:
        got = 'MISSING ' + type(e).__name__
    out[name] = {'want': want or None, 'got': got, 'ok': (not want) or got.split('+')[0] == want}
print(json.dumps(out, indent=1))
"""


def extract_demo(tree, dest):
    sp = pd.read_csv(tree / "splits" / "ooc_splits.csv", dtype={"image_id": str, "group": str})
    fin = sp[sp.role == "final_test"]
    pick = fin.sample(12, random_state=20261007)          # same 12 random locked-final photos as the earlier demo (fixed seed, no cherry-picking)
    runs = sorted(p for p in (tree / "runs").iterdir() if "G01_ooc_check_clean" in p.name)
    hdr = pd.read_csv(runs[-1] / "per_photo.csv")
    fmt = hdr.assign(key=hdr["mode"] + "/" + hdr["width"].astype(str) + "x" + hdr["height"].astype(str)).set_index("member")["key"]
    z = zipfile.ZipFile(RemoteFile(OOC_URL, OOC_SIZE))
    members = {m.rsplit("/", 1)[1].rsplit(".", 1)[0]: m for m in z.namelist() if m.lower().endswith(".png")}
    extra = []
    for _, r in sp.iterrows():                            # all unseen-format final photos + the first two small-format development photos
        k = fmt.get(r["member"], "")
        if (r.role == "final_test" and k not in ("L/2056x1542", "RGB/2048x1536")) or (r.role == "practice" and k == "RGB/640x480" and len([e for e in extra if e[1] == "small"]) < 2):
            extra.append((r.image_id, "unseen" if r.role == "final_test" else "small"))
    cols = ["image_id", "label_bad", "role", "cell_type"]
    chosen = pd.concat([pick[cols], sp[sp.image_id.isin([e[0] for e in extra])][cols]]).drop_duplicates("image_id")
    dest.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for _, r in chosen.iterrows():
        data = z.read(members[r.image_id])
        (dest / f"{r.image_id}.png").write_bytes(data)
        hashes[r.image_id] = hashlib.sha256(data).hexdigest()
    chosen.to_csv(W / "demo_labels.csv", index=False)
    REPORT["demo_photo_sha256"] = hashes
    log(f"extracted {len(chosen)} demo photos")


def main():
    T.mkdir(parents=True, exist_ok=True)
    inp = next(Path("/kaggle/input").rglob("inference.py")).parent
    ds = inp.parent
    tree = T / "tree"
    shutil.copytree(inp, tree)
    shutil.copytree(ds / "train_inputs" / "runs", tree / "runs")
    REPORT["platform"] = platform.platform()
    REPORT["system_python"] = sys.version
    REPORT["cpu_count"] = os.cpu_count()
    REPORT["input_sha256"] = {f: sha(tree / f) for f in ("requirements.txt", "inference.py", "scripts/train_final_model.py", "scripts/a_views.py", "model_x01_3x3.joblib", "model_x01_3x3.json")}

    # 1. fresh environment, README command
    py = make_venv()
    env = {"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_PROGRESS_BAR": "off"}
    p = run("1_pip_install", [VENV / "bin" / "pip", "install", "-r", "requirements.txt"], tree, env)
    run("1_pip_check", [VENV / "bin" / "pip", "check"], tree)
    p2 = run("1_pin_check", [py, "-c", PIN_CHECK], tree)
    run("1_pip_freeze", [VENV / "bin" / "pip", "freeze"], tree)
    try:
        pins = json.loads(p2.stdout)
        REPORT["pins"] = pins
        REPORT["checks"]["all_pins_match"] = all(v["ok"] for v in pins.values())
    except Exception as e:
        REPORT["checks"]["all_pins_match"] = False
        REPORT["pin_error"] = repr(e)
    if p.returncode:
        raise SystemExit("pip install failed")
    run("1_torch_device", [py, "-c", "import torch; print('cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"], tree)

    # 2. control: the old scorer must be rejected by the strict setting
    c = run("2_control_old_model", [py, "-W", STRICT, "-c", "import joblib; joblib.load('model_x01_3x3.joblib')"], tree)
    REPORT["checks"]["old_model_rejected_by_strict_setting"] = c.returncode != 0 and "InconsistentVersionWarning" in c.stderr

    # 3. rebuild (development photos only)
    (W / "new_model").mkdir(exist_ok=True)
    t = run("3_train", [py, "scripts/train_final_model.py", "--out", W / "new_model"], tree, {"PYTHONUNBUFFERED": "1"})
    if t.returncode:
        raise SystemExit("training failed")

    # 4. new model loads strictly
    n = run("4_load_new_model_strict", [py, "-W", STRICT, "-c", "import joblib, sys; m = joblib.load(sys.argv[1]); print(type(m).__name__, 'loaded')", W / "new_model" / "model_x01_3x3.joblib"], tree)
    REPORT["checks"]["new_model_loads_strict"] = n.returncode == 0

    # 5. clean package copy without runs/
    fresh = T / "fresh_pkg"
    shutil.copytree(inp, fresh)
    assert not (fresh / "runs").exists()
    for f in ("model_x01_3x3.joblib", "model_x01_3x3.json"):
        shutil.copy(W / "new_model" / f, fresh / f)
    REPORT["fresh_package_has_runs_folder"] = (fresh / "runs").exists()
    REPORT["fresh_package_files"] = sorted(str(p.relative_to(fresh)) for p in fresh.rglob("*") if p.is_file())
    demo = T / "demo_photos"
    extract_demo(tree, demo)

    # 6. README command on CPU, strict
    i = run("6_inference_cpu", [py, "-W", STRICT, "inference.py", "--photos", demo, "--out", W / "ranking_fresh.csv", "--sheet", W / "sheet_fresh.png"], fresh, {"PYTHONUNBUFFERED": "1"})
    REPORT["checks"]["inference_exit_zero"] = i.returncode == 0
    if i.returncode == 0:
        r = pd.read_csv(W / "ranking_fresh.csv")
        REPORT["inference_routes"] = r["route"].value_counts().to_dict()


try:
    main()
except BaseException as e:                                # noqa: BLE001 - always write the report
    REPORT["error"] = repr(e)
    log(f"STOPPED: {e!r}")
finally:
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    for f in ("model_x01_3x3.joblib", "model_x01_3x3.json"):
        if (W / "new_model" / f).exists():
            REPORT.setdefault("new_model_sha256", {})[f] = sha(W / "new_model" / f)
    (W / "job_report.json").write_text(json.dumps(REPORT, indent=1))
    print(json.dumps({k: v for k, v in REPORT.items() if k not in ("pins", "demo_photo_sha256", "fresh_package_files")}, indent=1))
