"""F04 (improvements.md): number provenance check for a results table (companion to f04_trace.py, which only checks paths).
For every table row it collects the numbers stored in the files and run folders named in that row's last cell (JSON values, and
CSV cells of small tables) and checks that each decimal number written in the row's Result and Had-to-beat cells exists there at the
precision written (percentages also as fractions). Signed gaps and numbers inside backticks are skipped (derived / paths).
A number that is NOT found is a lead to read by hand (it may be a derived figure such as a mean, ratio or difference); a number
that IS found only shows it exists somewhere in the named files. No model is run.
Usage: python scripts/f04_numbers.py DOC.md [--section "SECTION 2"] [--integers]   (--integers checks whole numbers of 2 or more digits instead)"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "code" / "runs"
CSV_MAX_ROWS = 80
NUM = re.compile(r"(?<![\w.@])([+−-]?)(\d+\.\d+|\d+(?=%))(%?)")
INT = re.compile(r"(?<![\w.@/×-])(\d{1,3}(?:,\d{3})+|\d{2,5})(?![\w.%×/-])")


def in_order(name, parts):
    i = 0
    for p in parts:
        j = name.find(p.strip("/"), i)
        if j < 0:
            return False
        i = j + len(p.strip("/"))
    return True


def resolve(q, names):
    q = q.strip().rstrip("/")
    if q.startswith(("code/", "Research_and_Experiments/")) and "…" not in q:
        return [ROOT / q] if (ROOT / q).exists() else []
    q = re.sub(r"…\d$", "", q)
    file_part = None
    if re.search(r"/[^/]+\.(csv|json|md|txt)$", q) and "…" in q:
        q, file_part = q.rsplit("/", 1)
    parts = [x for x in q.replace("code/runs/", "").split("…") if x]
    found = [RUNS / n for n in sorted(names) if in_order(n, parts)]
    return [f / file_part for f in found if (f / file_part).exists()] if file_part else found


def flat(o, out):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        out.append(float(o))
    elif isinstance(o, dict):
        for v in o.values():
            flat(v, out)
    elif isinstance(o, list):
        for v in o:
            flat(v, out)


def stored_numbers(paths):
    vals = []
    for p in paths:
        files = [p] if p.is_file() else sorted(p.rglob("*")) if p.is_dir() else []
        for f in files:
            try:
                if f.suffix == ".json":
                    flat(json.loads(f.read_text()), vals)
                elif f.suffix == ".csv":
                    d = pd.read_csv(f)
                    if len(d) <= CSV_MAX_ROWS:
                        vals.extend(float(x) for x in d.select_dtypes("number").to_numpy().ravel() if np.isfinite(x))
            except Exception:       # noqa: BLE001 - unreadable file: skip, the row will show missing numbers
                continue
    return np.array(sorted(set(vals)))


def found(arr, token, pct):
    dec = len(token.split(".")[1]) if "." in token else 0
    v = float(token)
    tol = 0.5 * 10 ** -dec + 1e-9
    targets = [v / 100, v] if pct else [v]
    for t, d_tol in zip(targets, [tol / 100, tol] if pct else [tol]):
        i = np.searchsorted(arr, t - d_tol)
        if i < len(arr) and arr[i] <= t + d_tol:
            return True
    return False


def main():
    doc = Path(sys.argv[1])
    sect = sys.argv[sys.argv.index("--section") + 1] if "--section" in sys.argv else None
    names = {p.name for p in RUNS.iterdir()}
    active = sect is None
    ints = "--integers" in sys.argv
    total = missing_total = 0
    for line in doc.read_text().splitlines():
        if line.startswith("# "):
            active = sect is None or line.startswith("# " + sect)
        if not active or not line.startswith("| ") or line.startswith("|---") or line.startswith("| ID"):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        if len(cells) < 4:
            continue
        rid, paths_cell = cells[0], cells[-1]
        files = []
        for p in re.findall(r"`([^`]+)`", paths_cell):
            for q in re.split(r",\s*", p):
                files += resolve(q, names)
        arr = stored_numbers(files)
        body = cells[2:4] if len(cells) >= 7 else cells[1:2]
        miss = []
        for c in body:
            c = re.sub(r"`[^`]*`", "", c)
            if ints:
                for m in INT.finditer(re.sub(r"\d+\.\d+", " ", c)):
                    tok = m.group(1).replace(",", "")
                    total += 1
                    if len(arr) == 0 or not np.any(np.abs(arr - float(tok)) < 1e-9):
                        miss.append(m.group(1))
                continue
            for m in NUM.finditer(c):
                sign, tok, pct = m.groups()
                if sign:
                    continue                                # signed numbers are gaps / differences (derived)
                total += 1
                if len(arr) == 0 or not found(arr, tok, bool(pct)):
                    miss.append(tok + pct)
        missing_total += len(miss)
        print(f"{rid[:22]:22s} files {len(files):2d} stored numbers {len(arr):6d} | not found ({len(miss)}): {' '.join(miss) if miss else '-'}")
    print(f"\nchecked {total} {'whole' if ints else 'unsigned decimal'} numbers; {missing_total} not found in the named files (read those by hand)")


if __name__ == "__main__":
    main()
