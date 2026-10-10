# F03 remaining reproduction: seed-1 and extended-boundary clean-up under newer package versions, 8 October 2026

**What this checks.** The rest of plan item F03 (rebuild in a clean setup and reproduce the tables): the four clean-up searches that earlier ran only under Kaggle's default packages (B02 seed 1; B03 seed 1; B03 seed 0 and seed 1 with thresholds extended to 0.999). **These runs repeat only the clean-up search:** same saved fold models, same settings grid, same rule (settings picked on an inner check fold, then the held-out fold is scored). No network was trained, nothing was tuned, and no pass mark was set.

## Part A: scikit-image changed to the declared version (done)

Kaggle CPU job `ai4s-f03b-pinned` (four runs sharing one machine). scikit-image **0.26.0** (the version in `code/requirements.txt`) replaced Kaggle's **0.25.2**. Every other library was Kaggle's own version, exactly as in the earlier runs (numpy 2.1.3, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.6.1, torch 2.11.0+cpu, timm 1.0.29, Python 3.13.15). Compared by `code/scripts/f03_pinned_compare.py`:

| Run | Picks in all 5 folds | 49 summary numbers, count setting / outline setting | Per-image tables | Settings-grid tables (5 folds) |
|---|---|---|---|---|
| B02 seed 1 | identical | largest difference 0 / 0 | exactly equal | exactly equal |
| B03 seed 1 | identical | 0 / 0 | exactly equal | exactly equal |
| B03 seed 0, thresholds to 0.999 | identical | 0 / 0 | exactly equal | exactly equal |
| B03 seed 1, thresholds to 0.999 | identical | 0 / 0 | exactly equal | exactly equal |

Headline numbers, unchanged (count error / F1@0.75, count setting then outline setting): B02 seed 1 4.82 / 0.842 and 10.41 / 0.866; B03 seed 1 5.38 / 0.864 and 10.30 / 0.892; B03 seed 0 extended 4.63 / 0.860 and 9.86 / 0.891; B03 seed 1 extended 5.21 / 0.798 and 10.30 / 0.892. Run times with four runs sharing one machine: 6850, 2438, 2671 and 2678 seconds.

## Earlier check with every declared version (seed 0, B03)

`…155516_B03_unet_boundary_repost_v2` ran on a Mac with the declared versions (numpy 2.5.3, scipy 1.18.1, torch 2.14.1, scikit-image 0.26.0, Python 3.12.14); `…161337_B03_unet_boundary_repost_v2` ran on Kaggle's default packages. Same picks in all five folds; count-setting numbers identical; outline-setting numbers differ by at most 0.00006 (largest of 49).

## Part B: every declared version (done)

Four single-run Kaggle kernels (`ai4s-f03c-b02s1`, `…b03s1`, `…b03x0`, `…b03x1`), each building a fresh virtual environment from the pins in `code/requirements.txt` (the lines above the RL-only marker; `joblib` and `psutil` are unpinned) and stopping if any installed version differs from its pin. All four environments came out the same and passed the check: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, scikit-image 0.26.0, tifffile 2026.9.20, pillow 12.3.0, openpyxl 3.1.5, matplotlib 3.11.2, pyarrow 25.0.1, torch 2.14.1 (CUDA build, run on CPU), torchvision 0.29.1, timm 1.0.30, joblib 1.6.0, psutil 7.2.2; Python 3.13.15. (Environment build took 160–247 s.) Kaggle prints "Error in sitecustomize … No module named 'wrapt'" when the new environment starts; it comes from a Kaggle start-up hook, not from our code, and did not stop any run.

| Run | Run time | Result against the earlier run |
|---|---|---|
| B02 seed 1 | 3,208 s | **identical**: same picks in all 5 folds; all 49 + 49 summary numbers, per-image tables and settings-grid tables differ by 0 |
| B03 seed 1 | 791 s | **identical** (same checks) |
| B03 seed 0, thresholds to 0.999 | 1,065 s | **identical** (same checks) |
| B03 seed 1, thresholds to 0.999 | 1,108 s | **identical** (same checks) |

So the four clean-up searches give the same picks and the same numbers under Kaggle's default packages, under the declared scikit-image alone, and under every declared version. The tiny differences (at most 0.00006) in the earlier Mac run did not appear when only the library versions changed on the same kind of machine, which points to the computer type (Apple chip versus Linux) rather than the library versions; that explanation was not tested separately.

## Limits

* Clean-up search only; networks for other training splits were not retrained, so this does not show that every model can be rebuilt from scratch.
* Other nucleus analyses from the first Kaggle days (B05 crowding and tiny-nucleus diagnosis, B06 warning flags, B07 outside images, B04 stress refresh for seed 0) also ran under Kaggle's default scikit-image 0.25.2 and were not repeated; they use the same clean-up code that gave identical results in all eight repeats above.
* One machine type (Kaggle Linux CPU). The only other operating system tested is the Mac run above (seed 0).

## Files

`code/logs/kaggle/f03b_skimage_repeat/` (Part A) and `code/logs/kaggle/f03c_all_versions_repeat/` (Part B): the new run folders, logs and `f03_pinned_comparison.json`; script `code/scripts/f03_pinned_compare.py`; job definitions `code/kaggle/jobs/ai4s-f03b-pinned/` and `code/kaggle/jobs/ai4s-f03c-*/`; earlier runs in `code/runs/` (`…230901_B02_unet_fg_repost_v2_seed1`, `…230901_B03_unet_boundary_repost_v2_seed1`, `…233119_B03_unet_boundary_repost_v2_thr_extended`, `…235533_B03_unet_boundary_repost_v2_seed1_thr_extended`).
