"""Put the datasets on Modal storage and prepare the chip photos there (no work on the Mac).

Run:  .venv/bin/modal run cloud/modal_data.py
Steps (all inside Modal, CPU only):
  1. download the chip archive from Zenodo with parallel byte ranges, check MD5 against the provider
  2. download BBBC039 (images, masks, metadata)
  3. decode every chip photo, convert to one format (8-bit grayscale; G01 found 4 formats that track the label),
     resize to a long side of 1036 px (2 x 518, the big model's native input) and store as .npy, plus a table
     of what was done to each photo (original mode/size kept for within-format reporting)
Volume: "ai4s-data" (Modal storage; free up to 1 TiB/month per Modal's pricing page).
"""
import modal

app = modal.App("ai4s-data")
vol = modal.Volume.from_name("ai4s-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").apt_install("curl").pip_install("numpy", "pillow", "pandas", "openpyxl")

OOC_URL = "https://zenodo.org/api/records/10203721/files/OOC_image_dataset.zip/content"
OOC_SHEET = "https://zenodo.org/api/records/10203721/files/OOC_datasheet.xlsx/content"
OOC_MD5, OOC_SIZE = "8f7e058996203d48eb03b2d86c0a2e4d", 6710767405
BBBC = "https://data.broadinstitute.org/bbbc/BBBC039/"
LONG_SIDE = 1036


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16384, timeout=4 * 3600)
def download():
    import hashlib
    import os
    import subprocess
    import time
    os.makedirs("/data/ooc", exist_ok=True)
    os.makedirs("/data/bbbc039", exist_ok=True)
    t0 = time.time()
    out = "/data/ooc/OOC_image_dataset.zip"
    if not (os.path.exists(out) and os.path.getsize(out) == OOC_SIZE):
        n = 12
        chunk = (OOC_SIZE + n - 1) // n
        procs = []
        for i in range(n):
            s, e = i * chunk, min(OOC_SIZE - 1, (i + 1) * chunk - 1)
            procs.append(subprocess.Popen(["curl", "-sS", "--retry", "10", "-r", f"{s}-{e}", "-o", f"/tmp/part{i}", OOC_URL]))
        for p in procs:
            p.wait()
        with open(out, "wb") as f:
            for i in range(n):
                with open(f"/tmp/part{i}", "rb") as g:
                    while True:
                        b = g.read(1 << 24)
                        if not b:
                            break
                        f.write(b)
    md5 = hashlib.md5()
    with open(out, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            md5.update(b)
    ok = md5.hexdigest() == OOC_MD5
    subprocess.run(["curl", "-sSL", "-o", "/data/ooc/OOC_datasheet.xlsx", OOC_SHEET], check=True)
    for z in ["images.zip", "masks.zip", "metadata.zip"]:
        subprocess.run(["curl", "-sSL", "-o", f"/data/bbbc039/{z}", BBBC + z], check=True)
    vol.commit()
    return {"ooc_md5_ok": ok, "ooc_md5": md5.hexdigest(), "seconds": round(time.time() - t0, 1)}


def _prep_one(member):
    import io
    import zipfile
    import numpy as np
    from PIL import Image
    z = zipfile.ZipFile("/data/ooc/OOC_image_dataset.zip")
    im = Image.open(io.BytesIO(z.read(member)))
    im.load()
    mode, (w, h) = im.mode, im.size
    g = np.asarray(im.convert("L"), dtype=np.uint8)          # one format for every photo
    s = LONG_SIDE / max(w, h)
    g = np.asarray(Image.fromarray(g).resize((round(w * s), round(h * s)), Image.BILINEAR), dtype=np.uint8)
    name = member.rsplit("/", 1)[1].rsplit(".", 1)[0]
    np.save(f"/data/ooc/gray{LONG_SIDE}/{name}.npy", g)
    return {"image_id": name, "member": member, "orig_mode": mode, "orig_w": w, "orig_h": h, "new_h": g.shape[0], "new_w": g.shape[1]}


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=32768, timeout=2 * 3600)
def prepare():
    import os
    import time
    import zipfile
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    t0 = time.time()
    os.makedirs(f"/data/ooc/gray{LONG_SIDE}", exist_ok=True)
    members = [m for m in zipfile.ZipFile("/data/ooc/OOC_image_dataset.zip").namelist() if m.lower().endswith(".png")]
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(_prep_one, members, chunksize=16))
    pd.DataFrame(rows).to_csv(f"/data/ooc/gray{LONG_SIDE}_index.csv", index=False)
    vol.commit()
    return {"photos": len(rows), "seconds": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    print("download:", download.remote())
    print("prepare:", prepare.remote())
