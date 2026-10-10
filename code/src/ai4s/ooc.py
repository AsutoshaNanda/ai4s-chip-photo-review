"""Organ-on-a-chip (OOC) brightfield image dataset reader (Zenodo record 10203721).

Photos are read straight from the provider's zip (no unpacking; saves ~6.7 GB of disk).
Archive layout (from the saved file list): OOC_image_dataset/{train|val|test}/{good|bad}/cell_type_X/{day bin}/{imageID}.png
The folder names reveal the answer, so paths must never be given to a model.

Label convention for all scoring: bad = 1 (spreadsheet decision 2), good = 0 (decision 1).
"""
import io
import re
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image

from .paths import OOC_DIR

ZIP_NAME = "OOC_image_dataset.zip"
SHEET_NAME = "OOC_datasheet.xlsx"


def load_metadata(root=OOC_DIR):
    """Spreadsheet rows with an image ID, with tidy column names. Raw text values are kept next to parsed ones."""
    df = pd.read_excel(root / SHEET_NAME, sheet_name="Main")
    df = df[df["imageID"].notna()].copy()
    df = df.rename(columns={
        "imageID": "image_id", "cell type": "cell_type", "seeding density, cells/ml": "seeding_density_raw",
        "time after seeding, h": "hours_after_seeding", "day": "day",
        "Decision 1/2 (good/bad)": "decision", "flow rate": "flow_rate_raw"})
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["label_bad"] = (df["decision"] == 2).astype(int)
    df["seeding_density"] = df["seeding_density_raw"].map(_parse_number)
    df["flow_rate_ul_per_min"] = df["flow_rate_raw"].map(_parse_flow)
    df["prefix"] = df["image_id"].str.split("_").str[0]
    return df.reset_index(drop=True)


def _parse_number(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_flow(v):
    """Flow text such as '2uL/min', '2 uL/min', '2ul/min' -> 2.0 (microlitres per minute).
    Only the uL/min unit is present in the measured values (G01 records every distinct text)."""
    if pd.isna(v):
        return np.nan
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*u[lL]/min\s*", str(v))
    return float(m.group(1)) if m else np.nan


@lru_cache(maxsize=2)
def _zip(path):
    return zipfile.ZipFile(path)


def zip_index(root=OOC_DIR):
    """One row per PNG in the archive: member path, published split, folder label, cell type folder, day bin, image id."""
    rows = []
    for info in _zip(root / ZIP_NAME).infolist():
        p = info.filename
        if not p.lower().endswith(".png"):
            continue
        parts = p.split("/")
        _, split, folder_label, ct_folder, day_bin, fname = parts
        rows.append({"member": p, "published_split": split, "folder_label": folder_label,
                     "cell_type_folder": ct_folder.replace("cell_type_", ""), "day_bin": day_bin,
                     "image_id": fname.rsplit(".", 1)[0], "zip_crc32": format(info.CRC, "08x"),
                     "bytes": info.file_size})
    return pd.DataFrame(rows)


def table(root=OOC_DIR):
    """Spreadsheet joined to the archive index on image id (G01 checks that every row matches)."""
    meta = load_metadata(root)
    idx = zip_index(root)
    return meta.merge(idx, on="image_id", how="outer", indicator=True)


def read_image(member, root=OOC_DIR):
    """Decoded photo exactly as stored (mode and bit depth are measured in G01, not assumed)."""
    with _zip(root / ZIP_NAME).open(member) as f:
        im = Image.open(io.BytesIO(f.read()))
        im.load()
    return im
