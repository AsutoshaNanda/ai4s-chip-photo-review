"""Copy code, splits and every saved run folder into kaggle/bundle/code and publish it as a new version of the
private dataset itsasup/ai4s-code (used by every Kaggle job). Usage: python kaggle/make_bundle.py "message" """
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
OUT = HERE / "bundle"
shutil.rmtree(OUT, ignore_errors=True)
SLIM = "--slim" in sys.argv
# run folders the remaining jobs read (A02/A03/X01 features and predictions, G01 formats, B01 settings, B02/B03 fold models and v2 settings)
KEEP = ("F03_chip_features", "A02_", "A03_", "X01_whole_plus_patches", "G01_ooc_check", "B01_cv", "B02_unet", "B03_unet", "repost_v2", "G02_", "A01_simple", "A05_confirm_fit")
SKIP = ("20261006-100315",)          # B02 smoke-test run (500 steps), not a real model set


def runs_ignore(src, names):
    base = Path(src)
    if base.name != "runs" or not SLIM:
        return shutil.ignore_patterns("__pycache__", "*.pyc", "_incomplete")(src, names)
    return [n for n in names if n == "_incomplete" or n.endswith(".pyc") or not (any(k in n for k in KEEP) and not any(x in n for x in SKIP)) and not n.endswith((".csv", ".json"))]


for d in ("src", "scripts", "splits", "runs"):
    shutil.copytree(CODE / d, OUT / "code" / d, ignore=runs_ignore if d == "runs" else shutil.ignore_patterns("__pycache__", "*.pyc", "_incomplete"))
for f in ("final_config.json", "inference.py", "requirements.txt"):
    shutil.copy(CODE / f, OUT / "code" / f)
(OUT / "dataset-metadata.json").write_text(json.dumps({"title": "ai4s-code", "id": "itsasup/ai4s-code", "licenses": [{"name": "other"}]}))
msg = sys.argv[1] if len(sys.argv) > 1 else "update"
subprocess.check_call(["kaggle", "datasets", "version", "-p", str(OUT), "-m", msg, "--dir-mode", "zip"])
