"""C03 data prep (Allen DIC -> lamin B1). Channel identity checked by eye (run C03_allen_channels, 6 Oct):
ch6 = 63X DIC (input), ch3 = EGFP lamin B1 nuclear envelope (target); Hoechst channels carry no signal.

Per stack: z* = the depth slice where the lamin rings are clearest (largest 99.9th-percentile minus median of the
real lamin slice; the same plane is used for every method, it only picks WHERE to compare). Slices z*-2..z*+2
(5 planes, recorded choice) are kept, reduced 2x in x/y by averaging (0.34 um pixels), cropped to a multiple of 8.
Reference nuclei = lamin rings -> Gaussian smoothing (sigma 1 px) -> Otsu threshold -> fill holes -> drop
pieces touching nothing but the ring (holes not filled: objects whose filled area equals ring area) -> label.
The same measurement function is later applied to predicted lamin images.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import czifile
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run

D = DATA_ROOT / "allen" / "dic_lamin_b1" / "dic_lamin_b1"
OUT = DATA_ROOT / "allen" / "c03_slices.npz"
HALF = 2


def reduce2(a):
    h, w = (a.shape[-2] // 2) * 2, (a.shape[-1] // 2) * 2
    a = a[..., :h, :w]
    return a.reshape(*a.shape[:-2], h // 2, 2, w // 2, 2).mean((-3, -1))


def nuclei_from_lamin(img):
    """Shared measurement: smooth -> Otsu -> fill holes -> keep only filled regions (enclosed by a ring) -> label."""
    g = ndi.gaussian_filter(img.astype(np.float32), 1)
    ring = g > threshold_otsu(g)
    filled = ndi.binary_fill_holes(ring)
    inside = filled & ~ring
    lab, n = ndi.label(filled)
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[inside])] = True
    keep[0] = False
    lab = np.where(keep[lab], lab, 0)
    lab, _ = ndi.label(lab > 0)
    return lab.astype(np.int32)


def main():
    run = Run("C03", "allen_prep", config={"input_channel": 6, "target_channel": 3, "half_band": HALF, "xy_reduce": 2})
    files = sorted(D.glob("*.czi"))
    dic, lam, lab, meta = [], [], [], []
    for f in files:
        with czifile.CziFile(f) as c:
            a = list(c.scenes.values())[0].asarray()
        L, X = a[3].astype(np.float32), a[6].astype(np.float32)
        contrast = [np.percentile(L[z], 99.9) - np.median(L[z]) for z in range(L.shape[0])]
        zs = int(np.argmax(contrast))
        start = min(max(zs - HALF, 0), L.shape[0] - (2 * HALF + 1))   # same number of planes for every stack
        zr = range(start, start + 2 * HALF + 1)
        x = reduce2(X[list(zr)])
        y = reduce2(L[list(zr)])
        H, W = (x.shape[-2] // 8) * 8, (x.shape[-1] // 8) * 8
        x, y = x[..., :H, :W], y[..., :H, :W]
        dic.append(x)
        lam.append(y)
        lab.append(np.stack([nuclei_from_lamin(s) for s in y]))
        well = f.name.split("-")[-1].replace(".czi", "")
        meta.append((f.name, well, zs - start, len(zr)))
        run.log(f"{f.name}: z* {zs}, planes {len(zr)}, nuclei at z* {int(lab[-1][zs - start].max())}")
    np.savez_compressed(OUT, dic=np.stack(dic).astype(np.float32), lamin=np.stack(lam).astype(np.float32),
                        labels=np.stack(lab), files=np.array([m[0] for m in meta]), wells=np.array([m[1] for m in meta]),
                        zstar_index=np.array([m[2] for m in meta]))
    run.log(f"saved {OUT} ({OUT.stat().st_size / 1e9:.2f} GB); wells {dict(zip(*np.unique([m[1] for m in meta], return_counts=True)))}")
    run.finish("measured", notes="C03 slices prepared")


if __name__ == "__main__":
    main()
