"""B07: do the nucleus methods work, unchanged, on a different nucleus dataset (BBBC038 stage-1 training, CC0)?

Images that are copies of BBBC039 photos are removed first, using the G01 overlap run: the copies are the images
above the largest gap in the sorted best-match scores (rule recorded in that run; 43 images on 6 Oct).
Every method is applied exactly as chosen on BBBC039 practice folds, once per fold (5 versions), so the spread
over the 5 versions is reported too:
  B01  threshold + watershed with each fold's chosen settings (best-count and best-outline)
  B02  fold U-Net (nucleus / background) + that fold's chosen clean-up settings
  B03  fold U-Net (inside / edge / background) + that fold's chosen clean-up settings
Images: grey = the mean of the colour channels; no other change (the methods were trained on bright nuclei on a
dark background; stained tissue photos are reported as their own group, not adapted).
Scores per image as in B01-B03; summaries per image group (grey vs colour, image size).
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ai4s import metrics_seg as S
from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run, latest_run, n_workers

ROOT = DATA_ROOT / "bbbc038" / "stage1_train"
RUNS = Path(__file__).resolve().parents[1] / "runs"


def read(iid):
    im = np.asarray(Image.open(ROOT / iid / "images" / f"{iid}.png"))
    colour = bool(im.ndim == 3 and np.abs(im[..., 0].astype(int) - im[..., 2].astype(int)).mean() > 0)
    g = im[..., :3].mean(-1) if im.ndim == 3 else im.astype(np.float32)
    lab = np.zeros(g.shape, np.int32)
    for k, p in enumerate(sorted((ROOT / iid / "masks").glob("*.png")), 1):
        lab[np.asarray(Image.open(p)) > 0] = k
    return g.astype(np.float32), lab, colour


def duplicates():
    d = pd.read_csv(latest_run("G01_bbbc038_check") / "per_image_overlap.csv")
    s = d["best"].dropna().sort_values(ascending=False).to_numpy()
    i = int(np.argmin(np.diff(s)))
    return set(d.loc[d["best"] >= s[i], "image_id"]), float(s[i]), float(s[i + 1])


@torch.no_grad()
def prob_any_size(m, x, dev):
    h, w = x.shape
    H, W = -(-h // 16) * 16, -(-w // 16) * 16
    xp = np.pad(x, ((0, H - h), (0, W - w)), mode="reflect")
    p = torch.softmax(m(torch.from_numpy(xp[None, None]).to(dev)).float(), 1)[0, :, :h, :w]
    return p.cpu().numpy()


def b01_one(args):
    import b01_threshold_watershed as T
    iid, settings = args
    g, lab, _ = read(iid)
    return [(name, k, S.score_image(lab, T.segment(g, **p))) for (name, k), p in settings.items()]


OPTS = {"seed": 0, "posts": "orig"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--posts", default="orig", choices=["orig", "v2"], help="v2 = settings from the wider v2 clean-up search")
    ap.add_argument("--skip_b01", action="store_true", help="B01 has no v2 settings; its outside results are unchanged")
    a = ap.parse_args()
    OPTS.update(seed=a.seed, posts=a.posts)
    import b_unet as U
    from g03_speed_test import UNet
    dup, above, below = duplicates()
    ids = sorted(p.name for p in ROOT.iterdir() if p.is_dir() and p.name not in dup)
    run = Run("B07", "bbbc038_outside" + ("_posts_v2" if a.posts == "v2" else "") + (f"_seed{a.seed}" if a.seed else ""), config={"seed": a.seed, "posts": a.posts,"images": len(ids), "removed_copies": len(dup),
                                                "copy_rule": f"largest gap in best-match scores: {above:.4f} vs {below:.4f}"})
    run.log(f"{len(ids)} BBBC038 images after removing {len(dup)} copies of BBBC039 photos (gap {above:.4f} -> {below:.4f})")
    info = {iid: read(iid)[2] for iid in ids}
    rows = []
    b01 = latest_run("B01_cv_threshold_watershed")
    chosen = json.load(open(b01 / "chosen_settings_per_fold.json"))
    settings = {(f"B01_{name}", int(k)): p for k, d in chosen.items() for name, p in d.items()}
    if not a.skip_b01:
        with ProcessPoolExecutor(max_workers=n_workers()) as ex:
            for iid, res in zip(ids, ex.map(b01_one, [(i, settings) for i in ids], chunksize=4)):
                for name, k, r in res:
                    rows.append({"image_id": iid, "method": name, "fold": k, **r})
        run.log("B01 done")
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    for exp, target, n_cls in (("B02", "fg", 2), ("B03", "boundary", 3)):
        from b_posts import fold_models
        parts = fold_models(exp, target, OPTS["seed"], OPTS["posts"])
        if sorted(parts) != [0, 1, 2, 3, 4]:
            run.log(f"{exp}: fold runs incomplete ({sorted(parts)}); skipped in this run, re-run when complete")
            continue
        for k, (d, post) in parts.items():
            m = UNet(out_ch=n_cls)
            m.load_state_dict(torch.load(d / f"model_fold{k}.pt", map_location="cpu"))
            m = m.eval().to(dev)
            for iid in ids:
                g, lab, _ = read(iid)
                p = prob_any_size(m, U.norm(g), dev)
                for name in ("best_count", "best_outline"):
                    s = post[name]
                    r = S.score_image(lab, U.to_objects(p, target, s["thr"], int(s["min_distance"]), int(s["min_size"])))
                    rows.append({"image_id": iid, "method": f"{exp}_{name}", "fold": k, **r})
            run.log(f"{exp} fold {k} done")
    D = pd.DataFrame(rows)
    D["colour_image"] = D["image_id"].map(info)
    run.save_df("per_image.csv", D.drop(columns=[c for c in D.columns if c.startswith(("tp@", "fp@", "fn@"))]))
    out = {}
    for (meth, grp), d in D.groupby(["method", "colour_image"]):
        per_fold = [S.summarize(x.to_dict("records")) for _, x in d.groupby("fold")]
        key = f"{meth}|{'colour' if grp else 'grey'}"
        out[key] = {m_: [float(np.mean([p[m_] for p in per_fold])), float(np.std([p[m_] for p in per_fold]))]
                    for m_ in ("count_mae", "count_bias", "pooled_f1@0.50", "pooled_f1@0.75")}
        out[key]["images"] = int(d["image_id"].nunique())
        run.log(f"{key:28s} images {out[key]['images']:4d} | count MAE {out[key]['count_mae'][0]:7.2f} (sd over folds {out[key]['count_mae'][1]:.2f}) "
                f"bias {out[key]['count_bias'][0]:+7.2f} | F1@0.50 {out[key]['pooled_f1@0.50'][0]:.3f} F1@0.75 {out[key]['pooled_f1@0.75'][0]:.3f}")
    run.save_json("results.json", out)
    run.finish("measured", notes=f"B07 on {len(ids)} BBBC038 images (copies removed); methods unchanged")


if __name__ == "__main__":
    main()
