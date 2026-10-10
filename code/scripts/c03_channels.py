"""C03 check: which array channel holds which signal? Saves, for 2 stacks, every channel's middle slice and its
brightest-over-depth projection (1-99.8 percentile scaled), plus per-channel 99.9th percentiles over all stacks'
middle slices."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import czifile
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run

D = DATA_ROOT / "allen" / "dic_lamin_b1" / "dic_lamin_b1"


def sc(a):
    lo, hi = np.percentile(a, [1, 99.8])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    run = Run("C03", "allen_channels", config={})
    files = sorted(D.glob("*.czi"))
    names = None
    stats = []
    for i, f in enumerate(files):
        with czifile.CziFile(f) as c:
            im = list(c.scenes.values())[0]
            names = list(im.channels) if names is None else names
            a = im.asarray()                        # C, Z, Y, X
        z = a.shape[1] // 2
        stats.append([float(np.percentile(a[k, z], 99.9)) - float(np.median(a[k, z])) for k in range(a.shape[0])])
        if i < 2:
            fig, ax = plt.subplots(2, a.shape[0], figsize=(3 * a.shape[0], 5))
            for k in range(a.shape[0]):
                ax[0, k].imshow(sc(a[k, z].astype(float)), cmap="gray")
                ax[0, k].set_title(f"ch{k} {names[k]}\nmiddle z", fontsize=7)
                ax[1, k].imshow(sc(a[k].max(0).astype(float)), cmap="gray")
                ax[1, k].set_title("max over z", fontsize=7)
                for q in ax[:, k]:
                    q.axis("off")
            fig.tight_layout()
            fig.savefig(run.path(f"channels_{i}.png"), dpi=60)
            plt.close(fig)
    S = np.array(stats)
    run.log("channel names (metadata order): " + str(names))
    run.log("99.9th percentile minus median at middle z, mean over stacks: " +
            str({f"ch{k}": round(float(S[:, k].mean()), 1) for k in range(S.shape[1])}))
    run.finish("measured", notes="C03 channel identity check")


if __name__ == "__main__":
    main()
