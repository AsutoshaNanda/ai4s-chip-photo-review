"""F06 (8 Oct 2026): fresh-installation test of the package exactly as it now stands in code/ (rebuilt scorer included).

Free Kaggle CPU kernel (Linux, internet on). F05 already rebuilt the scorer; its '-W error::sklearn.exceptions...' flag was silently
IGNORED by Python ("Invalid -W option ignored"), which the F05 control step revealed. Here the strict setting is applied inside Python
(warnings.simplefilter('error', InconsistentVersionWarning) around runpy), and a positive control proves it can fail.
  1. clean copy of the package with NO runs/ folder; fresh virtual environment; the literal README command `pip install -r requirements.txt`
  2. every pin compared with what got installed; `pip check`
  3. control: the OLD scorer (scikit-learn 1.6.1) must raise under the strict harness; the package's scorer must load under it
  4. the 17 demo photos are pulled from the public Zenodo archive by file name (same selection as the earlier demo)
  5. README command `python inference.py --photos ... --out ... --sheet ...` on CPU, literally; then once more under the strict harness
Nothing here trains anything or touches the locked final photos' labels beyond the fixed demo selection."""
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


HARNESS = """import runpy, sys, warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.simplefilter('error', InconsistentVersionWarning)      # any scikit-learn version mismatch stops the run
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
"""
LOADER = """import sys, joblib
m = joblib.load(sys.argv[1])
print(type(m).__name__, 'loaded')
"""


def main():
    T.mkdir(parents=True, exist_ok=True)
    inp = next(Path("/kaggle/input").rglob("inference.py")).parent
    ds = inp.parent
    fresh = T / "fresh_pkg"
    shutil.copytree(inp, fresh)
    REPORT["fresh_package_has_runs_folder"] = (fresh / "runs").exists()
    assert not (fresh / "runs").exists()
    REPORT["platform"], REPORT["system_python"], REPORT["cpu_count"] = platform.platform(), sys.version, os.cpu_count()
    REPORT["package_sha256"] = {f: sha(fresh / f) for f in ("requirements.txt", "inference.py", "README.md", "model_x01_3x3.joblib", "model_x01_3x3.json", "scripts/a_views.py", "scripts/train_final_model.py")}
    REPORT["package_files"] = sorted(str(p.relative_to(fresh)) for p in fresh.rglob("*") if p.is_file())
    (T / "strict_run.py").write_text(HARNESS)
    (T / "load_model.py").write_text(LOADER)

    py = make_venv()
    env = {"PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_PROGRESS_BAR": "off"}
    p = run("1_pip_install", [VENV / "bin" / "pip", "install", "-r", "requirements.txt"], fresh, env)
    run("1_pip_check", [VENV / "bin" / "pip", "check"], fresh)
    p2 = run("1_pin_check", [py, "-c", PIN_CHECK], fresh)
    run("1_pip_freeze", [VENV / "bin" / "pip", "freeze"], fresh)
    try:
        pins = json.loads(p2.stdout)
        REPORT["pins"] = pins
        REPORT["checks"]["all_pins_match"] = all(v["ok"] for v in pins.values())
    except Exception as e:
        REPORT["checks"]["all_pins_match"] = False
        REPORT["pin_error"] = repr(e)
    if p.returncode:
        raise SystemExit("pip install failed")
    run("1_torch_device", [py, "-c", "import torch; print('cuda', torch.cuda.is_available(), 'threads', torch.get_num_threads())"], fresh)

    old = next(ds.rglob("old_model/model_x01_3x3.joblib"))
    c = run("3_control_old_model_strict", [py, T / "strict_run.py", T / "load_model.py", old], fresh)
    REPORT["checks"]["control_old_model_rejected_by_strict_harness"] = c.returncode != 0 and "InconsistentVersionWarning" in c.stderr
    n = run("3_package_model_strict", [py, T / "strict_run.py", T / "load_model.py", fresh / "model_x01_3x3.joblib"], fresh)
    REPORT["checks"]["package_model_loads_strict"] = n.returncode == 0 and "loaded" in n.stdout

    demo = T / "demo_photos"
    tree = T / "tree"
    (tree / "runs").mkdir(parents=True)
    shutil.copytree(inp / "splits", tree / "splits")
    shutil.copytree(next(ds.rglob("extra/runs")), tree / "runs", dirs_exist_ok=True)
    extract_demo(tree, demo)

    i = run("5_inference_cpu_readme_command", [py, "inference.py", "--photos", demo, "--out", W / "ranking_readme.csv", "--sheet", W / "sheet_readme.png"], fresh, {"PYTHONUNBUFFERED": "1"})
    REPORT["checks"]["readme_command_exit_zero"] = i.returncode == 0
    s = run("5_inference_cpu_strict", [py, T / "strict_run.py", "inference.py", "--photos", demo, "--out", W / "ranking_strict.csv", "--sheet", W / "sheet_strict.png"], fresh, {"PYTHONUNBUFFERED": "1"})
    REPORT["checks"]["strict_inference_exit_zero"] = s.returncode == 0
    for tag, ok in (("readme", i.returncode == 0), ("strict", s.returncode == 0)):
        if ok:
            REPORT[f"routes_{tag}"] = pd.read_csv(W / f"ranking_{tag}.csv")["route"].value_counts().to_dict()
    REPORT["version_warning_text_in_readme_run_stderr"] = "InconsistentVersionWarning" in i.stderr


try:
    main()
except BaseException as e:                                # noqa: BLE001 - always write the report
    REPORT["error"] = repr(e)
    log(f"STOPPED: {e!r}")
finally:
    REPORT["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    (W / "job_report.json").write_text(json.dumps(REPORT, indent=1))
    print(json.dumps({k: v for k, v in REPORT.items() if k not in ("pins", "demo_photo_sha256", "package_files")}, indent=1))
