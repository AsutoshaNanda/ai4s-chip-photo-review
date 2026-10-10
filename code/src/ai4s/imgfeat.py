"""Simple photo measurements for chip photos (A01 and later reuse): brightness, contrast, sharpness/focus,
texture and spectrum. Computed at the native size and at a fixed long side (so photos of different sizes are
comparable); both are kept, nothing is pre-selected. Photo width/height are recorded but NOT used as features
(they describe how the photo was taken, a possible shortcut)."""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import graycomatrix, graycoprops

FIXED_LONG_SIDE = 512


def gray(arr):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def resize_long(g, long_side):
    h, w = g.shape
    s = long_side / max(h, w)
    im = Image.fromarray(g.astype(np.float32), mode="F")
    return np.asarray(im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR), dtype=np.float32)


def focus_measures(g, tag):
    lap = ndi.laplace(g)
    sx, sy = ndi.sobel(g, 0), ndi.sobel(g, 1)
    grad = np.hypot(sx, sy)
    brenner = np.mean((g[:, 2:] - g[:, :-2]) ** 2)
    f = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    tot = f.sum() + 1e-12
    out = {f"{tag}_lap_var": float(lap.var()), f"{tag}_tenengrad": float(np.mean(grad ** 2)),
           f"{tag}_grad_mean": float(grad.mean()), f"{tag}_grad_p90": float(np.percentile(grad, 90)),
           f"{tag}_brenner": float(brenner)}
    for lo in (0.25, 0.5, 0.75):
        out[f"{tag}_spec_energy_above_{lo}"] = float(f[r > lo].sum() / tot)
    return out


def texture(g8):
    glcm = graycomatrix(g8, distances=[1, 3, 9], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64,
                        symmetric=True, normed=True)
    out = {}
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        v = graycoprops(glcm, prop)  # distances x angles
        for i, d in enumerate([1, 3, 9]):
            out[f"glcm_{prop}_d{d}"] = float(v[i].mean())
    return out


def measure(arr):
    g = gray(arr)
    p = np.percentile(g, [1, 5, 25, 50, 75, 95, 99])
    hist, _ = np.histogram(g, bins=256, range=(float(g.min()), float(g.max()) + 1e-6))
    pr = hist / max(hist.sum(), 1)
    ent = float(-(pr[pr > 0] * np.log2(pr[pr > 0])).sum())
    out = {"mean": float(g.mean()), "std": float(g.std()), "p1": p[0], "p5": p[1], "p25": p[2], "p50": p[3],
           "p75": p[4], "p95": p[5], "p99": p[6], "range_p99_p1": p[6] - p[0], "range_p95_p5": p[5] - p[1],
           "cv": float(g.std() / (abs(g.mean()) + 1e-6)), "entropy": ent,
           "skew": float(((g - g.mean()) ** 3).mean() / (g.std() ** 3 + 1e-12))}
    out.update(focus_measures(g, "native"))
    small = resize_long(g, FIXED_LONG_SIDE)
    out.update(focus_measures(small, "fixed"))
    lo, hi = np.percentile(small, [0.5, 99.5])
    g8 = np.clip((small - lo) / max(hi - lo, 1e-6) * 63, 0, 63).astype(np.uint8)
    out.update(texture(g8))
    return {k: float(v) for k, v in out.items()}
