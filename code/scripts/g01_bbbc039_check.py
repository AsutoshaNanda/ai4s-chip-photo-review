"""G01 for BBBC039: are the files what the provider says, and do we read the nucleus masks correctly?

Checks (all measured, nothing assumed):
- zip fingerprints (MD5) and file counts per split; no name shared between splits
- every image opens; shape and bit depth; every mask matches its image size
- mask decoding: provider method (first channel -> connected components) compared with alternatives:
  4- vs 8-neighbour connectivity, and the number of distinct colours inside each component
  (if one component holds several colours, the provider method would merge touching nuclei)
- overlay pictures of a few training images for a by-eye check
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from skimage.segmentation import find_boundaries

from ai4s import bbbc039 as B
from ai4s.paths import BBBC039_DIR
from ai4s.runlog import Run, file_md5


def main():
    run = Run("G01", "bbbc039_check", config={"data_dir": str(BBBC039_DIR)})
    run.save_json("zip_md5.json", {z: file_md5(BBBC039_DIR / z) for z in ["images.zip", "masks.zip", "metadata.zip"]})

    splits = B.split_lists()
    plates = B.plates()
    names = {s: set(v) for s, v in splits.items()}
    overlap = {f"{a}&{b}": len(names[a] & names[b]) for a in B.SPLITS for b in B.SPLITS if a < b}
    run.log(f"split sizes: { {s: len(v) for s, v in splits.items()} }, overlaps: {overlap}")

    rows = []
    for split, stems in splits.items():
        for stem in stems:
            img = B.read_image(stem)
            raw = B.read_mask_raw(stem)
            lab8 = B.decode_mask(raw, connectivity=2)
            lab4 = B.decode_mask(raw, connectivity=1)
            merged8 = B.decode_mask_merged(raw, connectivity=2)
            # distinct non-black colours per 8-connected component
            if raw.ndim == 3:
                flat = raw.reshape(-1, raw.shape[-1])
                colour_id = np.unique(flat, axis=0, return_inverse=True)[1].reshape(raw.shape[:2])
            else:
                colour_id = raw
            multi_colour_components = 0
            for lab_id in range(1, lab8.max() + 1):
                if len(np.unique(colour_id[lab8 == lab_id])) > 1:
                    multi_colour_components += 1
            ch_vals = {c: sorted(np.unique(raw[..., c]).tolist())[:6] for c in range(raw.shape[-1])} if raw.ndim == 3 else {}
            rows.append({
                "split": split, "stem": stem, "plate": plates.get(stem, ""),
                "img_shape": "x".join(map(str, img.shape)), "img_dtype": str(img.dtype),
                "img_min": int(img.min()), "img_max": int(img.max()),
                "mask_shape": "x".join(map(str, raw.shape)), "mask_dtype": str(raw.dtype),
                "size_match": raw.shape[:2] == img.shape[:2],
                "n_objects_8conn": int(lab8.max()), "n_objects_4conn": int(lab4.max()),
                "n_objects_if_values_merged": int(merged8.max()),
                "multi_colour_components": multi_colour_components,
                "mask_channel_values_first6": str(ch_vals),
            })
    df = pd.DataFrame(rows)
    run.save_df("per_image_checks.csv", df)

    summary = {
        "split_sizes": {s: len(v) for s, v in splits.items()},
        "split_overlaps": overlap,
        "images_opened": int(len(df)),
        "image_shapes": df["img_shape"].value_counts().to_dict(),
        "image_dtypes": df["img_dtype"].value_counts().to_dict(),
        "mask_shapes": df["mask_shape"].value_counts().to_dict(),
        "all_sizes_match": bool(df["size_match"].all()),
        "objects_8conn_total_by_split": df.groupby("split")["n_objects_8conn"].sum().to_dict(),
        "images_where_4conn_differs_from_8conn": int((df["n_objects_4conn"] != df["n_objects_8conn"]).sum()),
        "objects_4conn_total": int(df["n_objects_4conn"].sum()),
        "objects_8conn_total": int(df["n_objects_8conn"].sum()),
        "objects_if_values_merged_total": int(df["n_objects_if_values_merged"].sum()),
        "images_with_multi_colour_components": int((df["multi_colour_components"] > 0).sum()),
        "total_multi_colour_components": int(df["multi_colour_components"].sum()),
        "plates": df["plate"].value_counts().to_dict(),
        "plates_per_split": df.groupby("split")["plate"].nunique().to_dict(),
        "plates_shared_between_splits": {
            f"{a}&{b}": len(set(df[df.split == a].plate) & set(df[df.split == b].plate))
            for a in B.SPLITS for b in B.SPLITS if a < b},
    }
    run.save_json("summary.json", summary)
    run.log(f"summary: {summary}")

    # overlays for a by-eye check (training images only)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for stem in splits["training"][:6]:
            img = B.read_image(stem).astype(np.float32)
            lo, hi = np.percentile(img, [1, 99.8])
            disp = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
            lab = B.read_labels(stem)
            rgb = np.dstack([disp] * 3)
            rgb[find_boundaries(lab, mode="outer")] = [1, 0.2, 0.2]
            plt.figure(figsize=(7, 5.2))
            plt.imshow(rgb)
            plt.title(f"{stem[:40]}... objects={lab.max()}")
            plt.axis("off")
            plt.savefig(run.path(f"overlays/{stem}.png"), dpi=110, bbox_inches="tight")
            plt.close()
    except Exception as e:  # overlays are a convenience; record failure rather than hide it
        run.log(f"overlay failed: {e!r}")

    run.finish("measured", notes="G01 BBBC039 file and mask-decoding checks")


if __name__ == "__main__":
    main()
