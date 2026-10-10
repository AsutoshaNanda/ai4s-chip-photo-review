"""C03 data: stream the Allen label-free DIC / nuclear-envelope archive (dic_lamin_b1.tar.gz, 11.0 GB) straight
into storage, unpacking on the way (the compressed file is never stored). Terms: allencell.org/terms-of-use
(research use; no re-sharing). Writes a file list with sizes and a SHA-256 of the whole stream.
"""
import hashlib
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai4s.paths import DATA_ROOT
from ai4s.runlog import Run

URL = "https://s3-us-west-2.amazonaws.com/downloads.allencell.org/publication-data/label-free-prediction/dic_lamin_b1.tar.gz"


class Hashing:
    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, k=-1):
        b = self.f.read(k)
        self.h.update(b)
        self.n += len(b)
        return b


def main():
    out = DATA_ROOT / "allen" / "dic_lamin_b1"
    out.mkdir(parents=True, exist_ok=True)
    run = Run("C03", "allen_fetch", config={"url": URL, "dest": str(out), "terms": "https://www.allencell.org/terms-of-use.html"})
    t0 = time.time()
    rows = []
    with urllib.request.urlopen(URL) as r:
        src = Hashing(r)
        with tarfile.open(fileobj=src, mode="r|gz") as tf:
            for m in tf:
                if m.isfile():
                    tf.extract(m, out, filter="data")
                    rows.append(f"{m.name}\t{m.size}")
                    if len(rows) % 200 == 0:
                        run.log(f"{len(rows)} files, {src.n / 1e9:.2f} GB read, {time.time() - t0:.0f} s")
    (Path(run.path("file_list.tsv"))).write_text("name\tbytes\n" + "\n".join(rows))
    run.save_json("stream.json", {"bytes": src.n, "sha256": src.h.hexdigest(), "files": len(rows), "seconds": time.time() - t0})
    run.log(f"done: {len(rows)} files, {src.n / 1e9:.2f} GB compressed, {time.time() - t0:.0f} s")
    run.finish("measured", notes="Allen dic_lamin_b1 streamed + unpacked to storage; not re-shared")


if __name__ == "__main__":
    main()
