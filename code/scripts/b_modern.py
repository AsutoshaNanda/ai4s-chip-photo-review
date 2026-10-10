"""B02 modern-model comparison (improvements.md section 5, item 6; plan 11: StarDist, Cellpose-SAM or InstanSeg,
whichever passes the licence check and the "has it already seen our test images?" check).

Checks done before this run (recorded in the run config):
  StarDist 2D_versatile_fluo: trained on a DSB2018 (BBBC038) subset; our G01 check found copies of BBBC039 photos
      inside BBBC038 -> likely exposure to our evaluation images.
  Cellpose-SAM: training data licensed CC-BY-NC (non-commercial) -> fails the licence check.
  InstanSeg: code and model weights Apache-2.0 -> passes the licence check; its training datasets are listed in the
      README inside the model download -> read and saved here BEFORE results are used; mentions of BBBC039 or
      DSB2018/BBBC038 mark the result as a "possible prior exposure" row (reported separately, not a clean comparison).
Model: InstanSeg "single_channel_nuclei" (one fluorescence nuclear channel; BBBC039 is Hoechst nuclear stain).
Same 150 practice images, same 5 plate-grouped folds as B02/B03. Its one free setting (pixel size passed to the model)
is chosen per outer fold on the same inner validation fold as b_unet.py, two objectives kept (lowest count error;
best outline score over all overlap levels). A choice on the edge of the searched sizes extends the search.
Measured: count error, outline F1 at every level, joined/split/missed/false objects, time per image, GPU memory.
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from ai4s import bbbc039 as B
from ai4s import metrics_seg as S
from ai4s.paths import SPLITS_ROOT
from ai4s.runlog import Run

MODEL = "single_channel_nuclei"
START = [0.125, 0.177, 0.25, 0.354, 0.5, 0.707, 1.0, 1.414, 2.0, 2.828]     # microns per pixel, geometric steps of sqrt(2)
EXPOSURE_WORDS = ["BBBC039", "BBBC038", "DSB", "Data Science Bowl", "2018 Data", "U2OS", "U-2 OS"]


def main():
    try:
        import instanseg  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "instanseg-torch"])
    import torch
    from instanseg import InstanSeg
    run = Run("B02", "modern_instanseg", config={"model": MODEL, "start_pixel_sizes_um": START, "checks": {
        "stardist_2D_versatile_fluo": "DSB2018 subset; BBBC038 contains copies of BBBC039 photos (G01) -> likely exposure",
        "cellpose_sam": "training data CC-BY-NC -> fails licence check",
        "instanseg": "Apache-2.0 code and weights -> passes licence; training data read from model README below"}})
    import instanseg as _pkg
    run.log(f"instanseg version {getattr(_pkg, '__version__', 'unknown')}")
    model = InstanSeg(MODEL, verbosity=0)
    # ---- exposure check: find and save the model's README / rdf, search for our datasets
    texts = {}
    roots = {Path.home(), Path(_pkg.__file__).parent, Path("/tmp"), Path.cwd()}
    for root in roots:
        for f in list(root.rglob(f"*{MODEL}*/**/*.md")) + list(root.rglob(f"*{MODEL}*/**/*.yaml")) + list(root.rglob(f"*{MODEL}*/*.md")) + list(root.rglob(f"*{MODEL}*/*.yaml")):
            if f.is_file() and f.stat().st_size < 2_000_000:
                texts[str(f)] = f.read_text(errors="ignore")
    for name, t in texts.items():
        Path(run.path("model_docs")).mkdir(parents=True, exist_ok=True)
        Path(run.path("model_docs"), re.sub(r"[^A-Za-z0-9._-]", "_", name)[-120:]).write_text(t)
    hits = {w: [n for n, t in texts.items() if w.lower() in t.lower()] for w in EXPOSURE_WORDS}
    found_docs = len(texts) > 0
    exposure = "unknown (model documents not found)" if not found_docs else \
        ("possible prior exposure: " + ", ".join(w for w, h in hits.items() if h) if any(hits.values()) else "none of the checked dataset names listed")
    run.log(f"model documents found: {len(texts)}; exposure check: {exposure}")
    run.save_json("exposure_check.json", {"documents": list(texts), "word_hits": hits, "verdict": exposure})

    sp = pd.read_csv(SPLITS_ROOT / "bbbc039_splits.csv", dtype={"plate": str})
    prac = sp[sp.role == "practice"].reset_index(drop=True)
    imgs = {s: B.read_image(s).astype(np.float32) for s in prac["stem"]}
    refs = {s: B.read_labels(s) for s in prac["stem"]}
    cache, timing = {}, []

    def seg(stem, px):
        key = (stem, round(px, 6))
        if key not in cache:
            x = imgs[stem]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = None
            for arr in (x, x[..., None], x[None]):
                try:
                    out, _ = model.eval_small_image(arr, px)
                    break
                except Exception as e:                                   # noqa: BLE001 - input layout differs by version
                    err = e
            if out is None:
                raise err
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timing.append(time.perf_counter() - t0)
            lab = np.asarray(out.squeeze().cpu().numpy() if hasattr(out, "cpu") else out).squeeze()
            if lab.ndim == 3:
                lab = lab[0]
            cache[key] = lab.astype(np.int32)
        return cache[key]

    def score(stems, px):
        rs = [S.score_image(refs[s], seg(s, px)) for s in stems]
        sm = S.summarize(rs)
        return sm["count_mae"], float(np.mean([sm[f"mean_image_f1@{lv:.2f}"] for lv in S.DEFAULT_IOU_LEVELS]))

    folds = sorted(prac["cv_fold"].unique())
    oof = {"best_count": [], "best_outline": []}
    picks = {}
    for k in folds:
        inner = folds[(folds.index(k) + 1) % len(folds)]
        va = prac.loc[prac["cv_fold"] == inner, "stem"].tolist()
        sizes, table = sorted(START), {}
        while True:
            for px in sizes:
                if px not in table:
                    table[px] = score(va, px)
            bc = min(sizes, key=lambda p: (table[p][0], -table[p][1]))
            bo = min(sizes, key=lambda p: (-table[p][1], table[p][0]))
            edge = [p for p in (bc, bo) if p in (sizes[0], sizes[-1])]
            if not edge:
                break
            new = [sizes[0] / 2 ** 0.5] if sizes[0] in edge else []
            new += [sizes[-1] * 2 ** 0.5] if sizes[-1] in edge else []
            new = [p for p in new if 0.01 < p < 20 and p not in table]
            if not new:
                break
            run.log(f"fold {k}: choice on edge of {sizes[0]:.3f}..{sizes[-1]:.3f} um; adding {new}")
            sizes = sorted(sizes + new)
        picks[int(k)] = {"best_count": bc, "best_outline": bo, "inner_fold": int(inner),
                         "inner_table": {f"{p:.4f}": {"count_mae": v[0], "mean_f1_levels": v[1]} for p, v in table.items()}}
        for name, px in (("best_count", bc), ("best_outline", bo)):
            for stem in prac.loc[prac["cv_fold"] == k, "stem"]:
                r = S.score_image(refs[stem], seg(stem, px))
                r.update({"stem": stem, "plate": prac.set_index("stem").loc[stem, "plate"], "cv_fold": int(k), "pixel_size_um": px})
                oof[name].append(r)
        run.log(f"fold {k}: pixel size best count {bc:.3f} um, best outline {bo:.3f} um")
    results = {"exposure": exposure, "picks": picks,
               "seconds_per_image_median": float(np.median(timing)), "images_timed": len(timing),
               "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
    for name, rows in oof.items():
        df = pd.DataFrame(rows)
        run.save_df(f"oof_per_image_{name}.csv", df.drop(columns=[c for c in df.columns if c.startswith(("tp@", "fp@", "fn@"))]))
        aa = S.summarize(rows)
        results[name] = aa
        run.log(f"InstanSeg {name}: count MAE {aa['count_mae']:.2f} (bias {aa['count_bias']:.2f}); pooled F1 0.25/0.50/0.75/0.90 "
                f"{aa['pooled_f1@0.25']:.3f}/{aa['pooled_f1@0.50']:.3f}/{aa['pooled_f1@0.75']:.3f}/{aa['pooled_f1@0.90']:.3f}; "
                f"joined {aa['joined_ref']} split {aa['split_pred']} missed {aa['missed']} false {aa['false']}")
    run.log(f"time per image (median) {results['seconds_per_image_median']:.3f} s; peak GPU memory {results['peak_gpu_gb']}")
    run.save_json("results.json", results)
    run.finish("measured", notes=f"InstanSeg {MODEL} on BBBC039 practice folds; exposure: {exposure}")


if __name__ == "__main__":
    main()
