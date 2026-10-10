# Chip-photo splits (locked 2026-10-06T00:24:26)

G02 for chip photos: lock the final-test photos BEFORE any training, then build practice (development) folds.

Rules (decided from the G01 evidence, recorded in splits/README_ooc.md):
- Grouping unit = filename prefix (an UNCONFIRMED clue, probably an acquisition date). Photos that are exact
  pixel duplicates or have identical DCT hashes are merged into one group, so copies never sit on both sides.
- Final-test size = the dataset authors' own test share (656 of 3,072 = 21.4%), taken from the source,
  not invented. Whole groups only.
- Selection: StratifiedGroupKFold over (cell type, label) with a recorded seed; among its folds, the one that
  covers both classes in the most cell types is chosen (ties: closest to 21.4%). This uses only labels and
  counts, never model results.
- Practice data = everything else, split into grouped, stratified rotating folds (cross-validation).
The original published split is kept as a column for reference only.

Final-test photos: 675; practice photos: 2397.
SHA-256 of the sorted final-test id list: 85f99f0ff56ae7a425787c56c267ad9c149049e2e9a3e0ec4c2f63905de230f1
The final-test photos are opened once, on Thursday (F01). Do not change this file after training starts.
