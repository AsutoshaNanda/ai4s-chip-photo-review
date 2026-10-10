"""F04 (improvements.md): trace every stored path named in the results table back to a saved run folder, and flag gaps.
Reads Research_and_Experiments/16_Results_vs_Expected_7_Oct.md (or the file given), extracts every `...` code path, and
checks it exists under the project folder. Paths written with a leading "…" refer to a run-folder name inside code/runs and
are looked up by suffix. Reports missing or ambiguous paths and the rows that name no stored path. No model is run."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "code" / "runs"


def _in_order(name, parts):
    i = 0
    for p in parts:
        j = name.find(p.strip("/"), i)
        if j < 0:
            return False
        i = j + len(p.strip("/"))
    return True


def main():
    doc = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "Research_and_Experiments" / "16_Results_vs_Expected_7_Oct.md"
    text = doc.read_text()
    names = {p.name for p in RUNS.iterdir()}
    bad, ok, rows_without = [], 0, []
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        paths = re.findall(r"`([^`]+)`", line)
        paths = [p for p in paths if "/" in p or p.startswith("…")]
        if not paths and re.match(r"\| \**[A-GRX]\d{0,2}", line) and "Stored at" not in line and "---" not in line:
            rows_without.append(line.split("|")[1].strip())
        for p in paths:
            for q in re.split(r",\s*", p):
                q = q.strip().rstrip("/")
                if q.startswith(("code/", "Research_and_Experiments/")) and "…" not in q and re.search(r"\.(csv|json|md|txt|py|yaml)$", q):
                    if (ROOT / q).exists():
                        ok += 1
                    else:
                        bad.append(q)
                    continue
                q2 = re.sub(r"…\d$", "", q)                      # "fold0…4" = folds 0 to 4: check the first
                if re.search(r"/[^/]+\.(csv|json|md|txt)$", q2) and "…" in q2:      # a file inside a (shorthand) run folder
                    folder, fname = q2.rsplit("/", 1)
                    fparts = [x for x in folder.replace("code/runs/", "").split("…") if x]
                    hits = [n for n in names if _in_order(n, fparts) and (RUNS / n / fname).exists()]
                    if hits:
                        ok += 1
                    else:
                        bad.append(q)
                    continue
                parts = [x for x in q2.replace("code/runs/", "").split("…") if x]
                if q.startswith("…") or q.startswith("code/runs/"):
                    found = [n for n in names if _in_order(n, parts)]
                    if found:
                        ok += 1
                    else:
                        bad.append(q)
                elif q.startswith(("code/", "Research_and_Experiments/")):
                    if (ROOT / q).exists():
                        ok += 1
                    else:
                        bad.append(q)
    print(f"checked {doc.name}: {ok} paths found; {len(bad)} not found; rows without any stored path: {rows_without}")
    for b in bad:
        print("  NOT FOUND:", b)


if __name__ == "__main__":
    main()
