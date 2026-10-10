"""BBBC039 (nuclei of U2OS cells, CC0) reader. Images and masks are read straight from the provider's zip files.

Provider facts used here (measured on 5 Oct 2026, see runs/*G01_bbbc039*):
- metadata/training.txt, validation.txt, test.txt list mask file names (.png); images use the same stem with .tif.
- metadata/filenames_and_plates.csv maps each file to its BBBC022 plate number (useful as a grouping unit).
- Provider-linked decoding: keep the mask's first channel, then label connected components.
"""
import io
import zipfile
from functools import lru_cache

import numpy as np
import tifffile
from PIL import Image
from skimage.measure import label as cc_label

from .paths import BBBC039_DIR

SPLITS = ("training", "validation", "test")


def _metadata_text(name, root):
    """Read a provider metadata file from the unpacked folder, or straight from metadata.zip."""
    p = root / "metadata" / name
    if p.exists():
        return p.read_text()
    return _zip(root / "metadata.zip").read(f"metadata/{name}").decode()


def split_lists(root=BBBC039_DIR):
    """Return {split: [stem, ...]} using the provider's own lists."""
    out = {}
    for s in SPLITS:
        names = _metadata_text(f"{s}.txt", root).split()
        out[s] = [n.rsplit(".", 1)[0] for n in names]
    return out


def plates(root=BBBC039_DIR):
    """Return {stem: plate}. The provider CSV uses old Mac line endings (CR), so split on any newline."""
    text = _metadata_text("filenames_and_plates.csv", root)
    out = {}
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        name, plate = line.split(",")
        out[name.rsplit(".", 1)[0]] = plate
    return out


@lru_cache(maxsize=2)
def _zip(path):
    return zipfile.ZipFile(path)


def read_image(stem, root=BBBC039_DIR):
    """16-bit fluorescence image (DNA channel) as stored by the provider."""
    with _zip(root / "images.zip").open(f"images/{stem}.tif") as f:
        return tifffile.imread(io.BytesIO(f.read()))


def read_mask_raw(stem, root=BBBC039_DIR):
    with _zip(root / "masks.zip").open(f"masks/{stem}.png") as f:
        return np.array(Image.open(io.BytesIO(f.read())))


def decode_mask(raw, connectivity=2):
    """Provider method: first channel -> connected components -> one integer id per nucleus (0 = background).

    The first channel holds 0 (background), 1 or 2. Touching nuclei carry different values (1 vs 2), so
    components must be found per value (skimage's label does this when given the raw values). Turning the
    channel into a yes/no image first would merge touching nuclei: G01 on 5 Oct measured that mistake.
    """
    first = raw[..., 0] if raw.ndim == 3 else raw
    return cc_label(first, background=0, connectivity=connectivity).astype(np.int32)


def decode_mask_merged(raw, connectivity=2):
    """The wrong (yes/no) reading, kept only so G01 can measure how many nuclei it would merge."""
    first = raw[..., 0] if raw.ndim == 3 else raw
    return cc_label(first > 0, connectivity=connectivity).astype(np.int32)


def read_labels(stem, root=BBBC039_DIR, connectivity=2):
    return decode_mask(read_mask_raw(stem, root), connectivity=connectivity)
