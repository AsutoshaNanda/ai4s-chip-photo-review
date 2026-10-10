"""C03 G01-style check of the Allen DIC / lamin B1 stacks: dimensions, channels, pixel sizes, value ranges, and the
grouping hidden in the file names (session _1/_2, scene, position P#, well B0#). No training here."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import czifile
import numpy as np
import pandas as pd

from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run

D = DATA_ROOT / "allen" / "dic_lamin_b1" / "dic_lamin_b1"


def main():
    run = Run("C03", "allen_inspect", config={"dir": str(D)})
    rows = []
    files = sorted(D.glob("*.czi"))
    for i, f in enumerate(files):
        m = re.search(r"_(\d)-Scene-(\d+)-P(\d+)-(B\d+)", f.name)
        with czifile.CziFile(f) as c:
            sc = list(c.scenes.values())
            im = sc[0]
            r = {"file": f.name, "session": m.group(1), "scene": int(m.group(2)), "position": int(m.group(3)), "well": m.group(4),
                 "n_scenes": len(sc), "dims": str(im.dims), "shape": str(im.shape), "dtype": str(im.dtype),
                 "channels": str(getattr(im, "channels", None)), "scales": str(im.coord_scales), "units": str(im.coord_units)}
            if i == 0:
                a = im.asarray()
                cax = list(im.dims).index("C") if "C" in im.dims else None
                if cax is not None:
                    for k in range(a.shape[cax]):
                        v = np.take(a, k, axis=cax)
                        r[f"ch{k}_p1_p50_p99"] = str(np.percentile(v, [1, 50, 99]).round(1).tolist())
            rows.append(r)
        run.log(f"{f.name}: {rows[-1]['dims']} {rows[-1]['shape']} {rows[-1]['channels']}")
    T = pd.DataFrame(rows)
    run.save_df("files.csv", T)
    run.log(f"first file details: {T.iloc[0].dropna().to_dict()}")
    run.log("groups: " + str(T.groupby(["session", "well"]).size().to_dict()))
    run.finish("measured", notes="C03 Allen stacks inspected")


if __name__ == "__main__":
    main()
