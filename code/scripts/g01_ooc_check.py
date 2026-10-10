"""G01 for the chip-photo (OOC) dataset: is it what the provider says, and what exactly is in it?

Measured (nothing assumed):
- archive MD5 against the provider's value
- spreadsheet <-> archive: ids both ways, label agreement (folder vs decision), cell type agreement, split counts
- every distinct flow-rate / seeding-density text and whether it parsed
- every photo decoded: mode, size, bit depth, whether RGB channels are identical, brightness stats, decode time
- exact duplicates (identical pixels) and near-duplicates: full nearest-neighbour distance distribution
  (DCT hash and tiny-thumbnail distance), the closest pairs, and whether they cross published splits / prefixes.
  No similarity cut-off is assumed; only identical hashes are called "hash-identical".
- composition of every filename prefix (unconfirmed grouping clue)
- a contact sheet of training photos for a by-eye look
"""
import hashlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from ai4s import ooc
from ai4s.paths import OOC_DIR
from ai4s.runlog import Run, file_md5

PROVIDER_ZIP_MD5 = "8f7e058996203d48eb03b2d86c0a2e4d"


def _to_gray_float(arr):
    a = arr.astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    return a


def measure(member):
    t0 = time.perf_counter()
    im = ooc.read_image(member)
    arr = np.asarray(im)
    t_decode = time.perf_counter() - t0
    row = {"member": member, "mode": im.mode, "width": im.size[0], "height": im.size[1],
           "dtype": str(arr.dtype), "channels": 1 if arr.ndim == 2 else arr.shape[2], "decode_seconds": t_decode}
    if arr.ndim == 3 and arr.shape[2] >= 3:
        row["rgb_channels_identical"] = bool(np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]))
    else:
        row["rgb_channels_identical"] = None
    row["pixel_md5"] = hashlib.md5(arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()).hexdigest()
    g = _to_gray_float(arr)
    p1, p50, p99 = np.percentile(g, [1, 50, 99])
    row.update({"gray_min": float(g.min()), "gray_max": float(g.max()), "gray_mean": float(g.mean()),
                "gray_std": float(g.std()), "gray_p1": float(p1), "gray_p50": float(p50), "gray_p99": float(p99)})
    lo, hi = g.min(), g.max()
    g8 = ((g - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    small = np.asarray(Image.fromarray(g8).resize((32, 32), Image.BILINEAR), dtype=np.float32)
    d = dctn(small, norm="ortho")[:8, :8].ravel()
    bits = d > np.median(d[1:])
    row["dct_hash"] = int(np.packbits(bits).view(">u8")[0])
    thumb = np.asarray(Image.fromarray(g8).resize((16, 16), Image.BILINEAR), dtype=np.float32).ravel()
    thumb = (thumb - thumb.mean()) / (np.linalg.norm(thumb - thumb.mean()) + 1e-6)
    row["thumb"] = thumb.astype(np.float32).tobytes().hex()
    return row


def main():
    run = Run("G01", "ooc_check", config={"data_dir": str(OOC_DIR), "provider_zip_md5": PROVIDER_ZIP_MD5})
    zpath = OOC_DIR / ooc.ZIP_NAME
    md5 = file_md5(zpath)
    run.log(f"zip md5 {md5}; provider {PROVIDER_ZIP_MD5}; match={md5 == PROVIDER_ZIP_MD5}")
    sheet_md5 = file_md5(OOC_DIR / ooc.SHEET_NAME)

    t = ooc.table()
    meta_only = t[t["_merge"] == "left_only"]["image_id"].tolist()
    zip_only = t[t["_merge"] == "right_only"]["image_id"].tolist()
    both = t[t["_merge"] == "both"].copy()
    both["folder_label_bad"] = (both["folder_label"] == "bad").astype(int)
    label_mismatch = both[both["folder_label_bad"] != both["label_bad"]]["image_id"].tolist()
    ct_mismatch = both[both["cell_type_folder"] != both["cell_type"]]["image_id"].tolist()

    recon = {
        "zip_md5": md5, "zip_md5_matches_provider": md5 == PROVIDER_ZIP_MD5, "sheet_md5": sheet_md5,
        "spreadsheet_rows_with_id": int((t["_merge"] != "right_only").sum()),
        "archive_pngs": int((t["_merge"] != "left_only").sum()),
        "ids_only_in_spreadsheet": meta_only, "ids_only_in_archive": zip_only,
        "folder_vs_decision_label_mismatches": label_mismatch, "cell_type_mismatches": ct_mismatch,
        "published_split_counts": both["published_split"].value_counts().to_dict(),
        "label_counts_bad1_good0": both["label_bad"].value_counts().to_dict(),
        "flow_rate_texts": {str(k): int(v) for k, v in both["flow_rate_raw"].value_counts(dropna=False).items()},
        "flow_rate_unparsed_nonmissing": both[both["flow_rate_raw"].notna() & both["flow_rate_ul_per_min"].isna()]["flow_rate_raw"].astype(str).unique().tolist(),
        "flow_rate_parsed_values": {str(k): int(v) for k, v in both["flow_rate_ul_per_min"].value_counts(dropna=False).items()},
        "seeding_density_unparsed_nonmissing": both[both["seeding_density_raw"].notna() & both["seeding_density"].isna()]["seeding_density_raw"].astype(str).unique().tolist(),
        "missing_counts": both[["seeding_density", "hours_after_seeding", "day", "flow_rate_ul_per_min"]].isna().sum().to_dict(),
        "day_bin_folders": both["day_bin"].value_counts().to_dict(),
    }
    run.save_json("reconciliation.json", recon)
    run.log(f"reconciliation: ids only in sheet={len(meta_only)}, only in archive={len(zip_only)}, "
            f"label mismatches={len(label_mismatch)}, cell type mismatches={len(ct_mismatch)}")

    workers = max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(measure, both["member"].tolist(), chunksize=16))
    run.log(f"decoded {len(rows)} photos with {workers} processes in {time.time() - t0:.1f} s")
    m = pd.DataFrame(rows)
    df = both.merge(m, on="member")

    img_summary = {
        "decode_wall_seconds": round(time.time() - t0, 1), "workers": workers,
        "decode_seconds_per_photo_median": float(df["decode_seconds"].median()),
        "modes": df["mode"].value_counts().to_dict(), "dtypes": df["dtype"].value_counts().to_dict(),
        "sizes_WxH": (df["width"].astype(str) + "x" + df["height"].astype(str)).value_counts().to_dict(),
        "rgb_channels_identical": df["rgb_channels_identical"].value_counts(dropna=False).astype(int).to_dict(),
        "gray_mean_by_label": df.groupby("label_bad")["gray_mean"].describe().round(2).to_dict(),
    }

    # exact duplicates
    dup = df.groupby("pixel_md5")["image_id"].apply(list)
    exact_groups = [g for g in dup.tolist() if len(g) > 1]
    img_summary["exact_pixel_duplicate_groups"] = exact_groups

    # near-duplicates: DCT-hash Hamming distance and thumbnail distance, nearest neighbour per photo
    h = df["dct_hash"].to_numpy(dtype=np.uint64)
    ham = np.bitwise_count(h[:, None] ^ h[None, :]).astype(np.int16)
    np.fill_diagonal(ham, 99)
    th = np.stack([np.frombuffer(bytes.fromhex(x), dtype=np.float32) for x in df["thumb"]])
    sq = (th ** 2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * th @ th.T, 0))
    np.fill_diagonal(dist, np.inf)
    nn_h = ham.argmin(1)
    nn_t = dist.argmin(1)
    df["nn_hash_dist"] = ham[np.arange(len(df)), nn_h]
    df["nn_hash_id"] = df["image_id"].to_numpy()[nn_h]
    df["nn_thumb_dist"] = dist[np.arange(len(df)), nn_t]
    df["nn_thumb_id"] = df["image_id"].to_numpy()[nn_t]
    split_arr, pref_arr = df["published_split"].to_numpy(), df["prefix"].to_numpy()
    df["nn_thumb_other_split"] = split_arr[nn_t] != split_arr
    df["nn_thumb_other_prefix"] = pref_arr[nn_t] != pref_arr
    df["nn_hash_other_split"] = split_arr[nn_h] != split_arr
    iu = np.triu_indices(len(df), 1)
    hash_identical = [(df["image_id"].iat[i], df["image_id"].iat[j]) for i, j in zip(*np.where(np.triu(ham == 0, 1)))]
    img_summary["hash_identical_pairs"] = len(hash_identical)
    img_summary["hash_identical_pairs_list"] = hash_identical[:200]
    img_summary["nn_hash_distance_distribution"] = df["nn_hash_dist"].value_counts().sort_index().to_dict()
    img_summary["nn_thumb_distance_quantiles"] = df["nn_thumb_dist"].quantile([0, .001, .01, .05, .25, .5, .75, 1]).round(4).to_dict()
    img_summary["all_pairs_thumb_distance_quantiles"] = pd.Series(dist[iu]).quantile([0, .0001, .001, .01, .5]).round(4).to_dict()
    img_summary["share_nn_in_other_published_split"] = float(df["nn_thumb_other_split"].mean())
    img_summary["share_nn_in_other_prefix"] = float(df["nn_thumb_other_prefix"].mean())
    closest = df.sort_values("nn_thumb_dist").head(60)[
        ["image_id", "nn_thumb_id", "nn_thumb_dist", "published_split", "prefix", "nn_thumb_other_split", "nn_thumb_other_prefix",
         "nn_hash_dist", "label_bad", "cell_type"]]
    run.save_df("closest_pairs_by_thumbnail.csv", closest)

    # prefix composition (unconfirmed grouping clue)
    pref = df.groupby("prefix").agg(
        n=("image_id", "size"), cell_types=("cell_type", lambda s: "|".join(sorted(set(s)))),
        n_bad=("label_bad", "sum"), splits=("published_split", lambda s: "|".join(sorted(set(s)))),
        day_bins=("day_bin", lambda s: "|".join(sorted(set(s)))), days=("day", lambda s: "|".join(map(str, sorted(set(s))))),
        sizes=("width", lambda s: "|".join(map(str, sorted(set(s))))))
    pref["n_good"] = pref["n"] - pref["n_bad"]
    run.save_df("prefix_composition.csv", pref.reset_index())
    img_summary["prefixes"] = int(len(pref))
    img_summary["prefixes_with_more_than_one_cell_type"] = int((pref["cell_types"].str.contains(r"\|")).sum())
    img_summary["prefixes_in_more_than_one_published_split"] = int((pref["splits"].str.contains(r"\|")).sum())

    keep = [c for c in df.columns if c not in ("thumb", "_merge")]
    run.save_df("per_photo.csv", df[keep])
    run.save_json("summary.json", img_summary)
    run.log({k: v for k, v in img_summary.items() if k not in ("hash_identical_pairs_list", "exact_pixel_duplicate_groups")})

    # contact sheet: training photos only, one good and one bad per cell type
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tr = df[df["published_split"] == "train"]
        cts = sorted(tr["cell_type"].unique())
        fig, axes = plt.subplots(2, len(cts), figsize=(3.2 * len(cts), 6.6))
        rng = np.random.default_rng(0)
        for j, ct in enumerate(cts):
            for i, lab in enumerate([0, 1]):
                sub = tr[(tr["cell_type"] == ct) & (tr["label_bad"] == lab)]
                ax = axes[i, j]
                ax.axis("off")
                if len(sub):
                    r = sub.iloc[rng.integers(len(sub))]
                    a = _to_gray_float(np.asarray(ooc.read_image(r["member"])))
                    ax.imshow(a, cmap="gray")
                    ax.set_title(f"{ct} {'BAD' if lab else 'good'}\n{r['image_id']}", fontsize=8)
        plt.tight_layout()
        plt.savefig(run.path("contact_sheet_train.png"), dpi=90)
        plt.close()
    except Exception as e:
        run.log(f"contact sheet failed: {e!r}")

    run.finish("measured", notes="G01 OOC archive/metadata/pixel checks")


if __name__ == "__main__":
    main()
