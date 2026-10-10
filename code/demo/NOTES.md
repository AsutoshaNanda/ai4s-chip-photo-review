# Demo run (what the files in this folder are)

Command run (on a free Kaggle GPU, after `scripts/train_final_model.py`):
`python inference.py --photos demo_photos --out demo_ranking.csv --sheet demo_sheet.png`

Input: 17 photos from the Organ-on-a-Chip dataset (Zenodo 10203721, CC-BY 4.0; credit the authors). They are **not** included here; they were pulled from the public archive by file name:
* 12 photos drawn at random (fixed seed 20261007) from the 675 locked final-test photos;
* all 3 final-test photos in the unseen portrait format (1536×2048);
* the first 2 development photos in the small 640×480 format.

Output (`demo_ranking_with_labels.csv`; `demo_labels.csv` holds the true labels, which the tool never sees):
* 12 photos scored automatically. The 6 bad photos got ranks 1–6 (scores 0.86, 0.83, 0.78, 0.69, 0.66, 0.22); the 6 good photos got ranks 7–12 (scores 0.11 to 0.01).
* 5 photos were **not scored** and sent to manual review: the two 640×480 photos and the three portrait-format photos.
* `demo_sheet.png`: the top four and bottom four ranked photos with their scores.

**How to read this:** it is 12 random photos, so the clean split is an illustration, not an accuracy figure. The accuracy evidence is the locked final test (672 photos: AUC 0.850 [0.78, 0.92]; paired gain over the whole-image model +0.049 [+0.010, +0.086]). `p_bad` is the raw model score (0 to 1; higher = review first). On the two supported formats its calibration error is 0.02–0.04 (development) and 0.037 (final test), but it is offered as a ranking score, not a validated probability.

**Re-run, 8 October 2026 (rebuilt scorer, fresh install, CPU):** the same 17 photos were scored again with the rebuilt scorer (scikit-learn 1.9.1) on a free Kaggle CPU session, after a fresh `pip install -r requirements.txt`. Routing is unchanged (12 scored, 5 to manual review) and the rank order is identical; scores differ from the files above by at most 0.0012 (rebuilt scorer, CPU instead of GPU). The files above were not regenerated. Details: `../../Research_and_Experiments/Evidence/2026-10-08_scorer_rebuild_and_fresh_install.md`.
