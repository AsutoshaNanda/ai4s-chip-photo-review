"""Input gate (improvements.md: "explicit unsupported inputs"; section 5: colour images fail; X01: small photos).

Decides, BEFORE any model runs, whether an input is handled automatically or sent to a person. The rule is explicit
and every decision it makes is logged; nothing is silently scored. No number is invented here: the routing follows only
what improvements.md names and what is present in the development data.

Chip photos  (key = mode/width x height):
  thin-support format : the format improvements.md names as failing for patches, "87 small 640x480 photos" with only six
                        bad ones (THIN_FORMATS). Routed by the frozen small_rule: "manual_review" (improvements.md
                        fallback, default) or "whole_image" (whole-photo model; 0.741 on those photos in X01).
  other format seen in the development data : automatic.
  format never seen in the development data : manual review.
Nuclei images:
  grey single-channel : automatic.
  colour (R and B channels differ, same test as B07): manual review (detection falls to F1@0.75 = 0.000 in B07).
Self-check (run as a script): applies the gate to every chip photo (development and locked final membership; no results
involved) and to the BBBC038 outside images, and prints support per format (photos, bad, groups). No model is run.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from ai4s.paths import DATA_ROOT, SPLITS_ROOT
from ai4s.runlog import Run, latest_run

THIN_FORMATS = ["RGB/640x480"]      # named in improvements.md section 4


def seen_formats():
    fmt = pd.read_csv(latest_run("G01_ooc_check") / "per_photo.csv", usecols=["member", "mode", "width", "height"])
    fmt["photo_format"] = fmt["mode"] + "/" + fmt["width"].astype(str) + "x" + fmt["height"].astype(str)
    sp = pd.read_csv(SPLITS_ROOT / "ooc_splits.csv", dtype={"image_id": str, "group": str}).merge(fmt[["member", "photo_format"]], on="member")
    dev = sp[sp["role"] == "practice"]
    support = dev.groupby("photo_format").agg(photos=("label_bad", "size"), bad=("label_bad", "sum"), groups=("group", "nunique"))
    return sp, set(support.index), support


def chip_route(photo_format, seen, small_rule="manual_review"):
    if photo_format in THIN_FORMATS:
        return small_rule                      # "manual_review" or "whole_image"
    if photo_format in seen:
        return "automatic"
    return "manual_review"


def nucleus_route(img):
    img = np.asarray(img)
    colour = bool(img.ndim == 3 and np.abs(img[..., 0].astype(int) - img[..., 2].astype(int)).mean() > 0)
    return "manual_review" if colour else "automatic"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small_rule", default="manual_review", choices=["manual_review", "whole_image"])
    a = ap.parse_args()
    run = Run("G02", "input_gate_selfcheck", config={"small_rule": a.small_rule, "thin_formats": THIN_FORMATS})
    sp, seen, support = seen_formats()
    run.log("support per format in the development data:\n" + support.to_string())
    sp["route"] = [chip_route(f, seen, a.small_rule) for f in sp["photo_format"]]
    tab = sp.groupby(["role", "photo_format", "route"]).size().rename("photos").reset_index()
    run.log("chip photos by role, format and route:\n" + tab.to_string(index=False))
    run.save_df("chip_routes.csv", tab)
    res = {"chip": tab.to_dict("records"), "support": support.reset_index().to_dict("records")}
    root = DATA_ROOT / "bbbc038" / "stage1_train"
    if root.exists():
        from PIL import Image
        rows = []
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            im = np.asarray(Image.open(d / "images" / f"{d.name}.png"))
            rows.append({"image_id": d.name, "route": nucleus_route(im)})
        t = pd.DataFrame(rows).groupby("route").size().to_dict()
        run.log(f"BBBC038 outside images by route: {t}")
        res["bbbc038"] = t
    else:
        run.log("BBBC038 not on this machine: nucleus colour check not run here")
    run.save_json("results.json", res)
    run.finish("measured", notes="input gate self-check; no model run")


if __name__ == "__main__":
    main()
