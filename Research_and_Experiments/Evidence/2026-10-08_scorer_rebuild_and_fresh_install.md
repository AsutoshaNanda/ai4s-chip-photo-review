# Scorer rebuild and fresh-install test, 8 October 2026

## Answer

**The photo scorer was rebuilt under the software versions the install file asks for, and it works in a fresh installation. One extra strict check did not complete.** The ranking result is unchanged.

| What | Result |
|---|---|
| Version match | New scorer built and saved with scikit-learn **1.9.1**, the version in `requirements.txt`. The old one was saved with 1.6.1. |
| Does it load cleanly? | Yes. On the Mac, with every warning turned into an error, it loads with no warning. The old scorer prints three version warnings under the same test. |
| Same ranking quality? | Yes. On the 672 automatically scored final photos the AUC is **0.849599**, identical to the clean rebuild (0.849599). Largest score difference from the clean run: **0.00000007**. The old scorer gave 0.849662 (largest difference to the new one 0.0008). |
| Fresh installation | `pip install -r requirements.txt` in a new virtual environment on a free Kaggle CPU session: all 15 pinned versions installed exactly (2 more packages are unpinned), `pip check` clean, package copy had **no `runs/` folder**. |
| Demo photos | The 17 demo photos scored 12 and sent 5 to manual review, same as before. Same rank order. Scores differ from the earlier demo by at most **0.0012** (mean 0.0003). |
| Strict re-test of the final files | **Not completed.** See "What did not work". |

## What was changed

- `code/model_x01_3x3.joblib` and `.json` replaced by the rebuilt scorer. The old pair and the old training script are kept, with hashes, in `superseded_scorer_2026-10-08/`.
- `code/scripts/train_final_model.py`: only the metadata it writes changed. It now records the software versions and the feature run it trained from, and the tile wording is exact ("cut from the grey photo after it is resized to long side 1036 px, each tile fed at its own size rounded to a multiple of 14"). The training steps are unchanged.
- `code/README.md` and `code/demo/NOTES.md`: one added note each about the scorer version and the re-run.
- E08 corrected in `22_Final_Results_Before_and_After.md`, `20_Technical_Report_Draft.md`, `17_Decision_Ledger.md` and `16_Results_vs_Expected_7_Oct.md`: **cause not isolated**. The clean rebuild (0.850 vs 0.810) is now the main comparison.

## E08, in short

The first final run and the clean rebuild differ in the whole-photo baseline (0.801 vs 0.810). Measured differences between the two runs: the first run trained on features made on Modal and scored photos whose features came from Kaggle's default setup (different numpy, torch, timm, scikit-learn), the clean rebuild used one pinned setup; the saved development features differ slightly (mean difference 0.00177, largest 0.201); the baseline's chosen regularisation changed (0.001 to 0.0031623). The earlier claim that clipping explains it is withdrawn: model inputs lie between −2.118 and +2.640, so clipping at ±10 changes nothing. No controlled check has been run, so no cause is claimed.

## How the rebuild was done

Free Kaggle CPU session (Linux, 4 cores), job `itsasup/ai4s-f05-scorer`, files in `2026-10-08_scorer_rebuild_job/`. Steps and measured times: fresh environment and install 1420.6 s, `pip check` clean, training 17.0 s, README-style inference on 17 photos on CPU 402.8 s. Training used only the 2,397 development photos (checked: every training id has role "practice"); the 675 final photos were not used for training or for any choice. The rebuilt scorer's stored settings match the clean run (C 0.001; yes/no cut-off 0.45117, which `inference.py` does not use).

Files that were tested and are identical to what is installed now (SHA-256 checked): `requirements.txt`, `inference.py`, `scripts/a_views.py`, `scripts/train_final_model.py`, and the scorer. Only `README.md` and `demo/NOTES.md` were edited afterwards (text only).

## What did not work

1. **The "treat version warnings as errors" flag was ignored on Kaggle.** Python printed "Invalid -W option ignored". My control step (the old scorer should have been rejected) caught this. So in that job the evidence that no version warning appeared is the absence of warning text: the old scorer printed three such warnings in the same setup, the new scorer's load and the inference run printed none. The strict check itself was done properly on the Mac (inside Python, scikit-learn 1.9.1): new scorer loads, old scorer warns.
2. **The planned strict re-test on Kaggle (`itsasup/ai4s-f06-install-test`) failed to run.** It ended with status ERROR after about 95 minutes (18:44 to 20:19 IST), with an empty log and no output files. The cause is unknown. Not retried yet.
3. **Not tested:** a fresh install on the Mac, Windows, or from the final public link. Compute was kept off the Mac as agreed.

## Smaller notes

- Training prints two scikit-learn FutureWarnings (defaults that change in version 1.10). Harmless with the pinned 1.9.1; do not upgrade scikit-learn without re-checking.
- CPU and GPU give slightly different scores (CPU runs in full precision, the GPU path in half precision): part of the 0.0012 demo difference.
- Still open from the audit's wording list: "native-resolution" is still in the docstrings of `inference.py` and `train_final_model.py`; JUMP's CC0 1.0 wording in `README.md`; the other items in the audit table.

Machine-readable record: `2026-10-08_scorer_rebuild_and_fresh_install.json`.
