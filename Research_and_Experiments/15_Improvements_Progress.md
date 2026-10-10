# Improvements progress (following improvements.md exactly)

Started: **6 October 2026, 21:20 IST**. This file tracks every action in [improvements.md](improvements.md). Nothing is dropped. Each row gets a measured result, a running job, or an exact, evidenced blocker.

## Compute decisions (from the user)

| When (IST) | Decision |
|---|---|
| 6 Oct 21:15 | Use the rest of the free $30 Modal credit only; never real money. B02/B03 confirmation: **1 new training seed** (seed 1, fixed before results). |
| 6 Oct 21:35 | A02 light retraining picks its settings with **4 grouped inner splits** (same as the frozen reference). |
| 6 Oct 22:35 | Modal stopped at **$24.26** used (real cost ran about 1.5 times the job queue's estimate). **All remaining work runs on Kaggle's free T4 GPUs; nothing skipped.** |

Choices made here and why (each traceable to improvements.md or earlier recorded settings):
- A02 "light retraining" = train only the final block (+ final norm + output), as defined in the experiment playbook (A02). The earlier blocks' output is cached. The equality check against the full model had a largest relative difference of 7.3e-3 (bfloat16 rounding).
- Noise training change (B04 follow-up): per-crop noise strength drawn between 0 and **0.1**, which is the largest noise level already recorded in the B04 stress test (`b_unet.STRESS`). No new strength was invented. The test uses the same kind of synthetic noise, so this is **not** a real outside test.
- B03 boundary refinement: thresholds 0.96, 0.97, 0.98, 0.99, 0.995 and 0.999 were added, toward the valid limit of 1.0.
- C03 input-only depth: the sharpest **DIC** plane (variance of the Laplacian, a standard focus measure). The fluorescent target is never used.
- Modern B02 comparison, per plan 11 ("StarDist, Cellpose-SAM or InstanSeg, whichever passes the licence check and the exposure check"):
  - StarDist was trained on part of DSB2018 (BBBC038), which contains copies of BBBC039 photos (G01), so it likely saw our test images.
  - Cellpose-SAM's training data is CC-BY-NC, so it fails the licence check.
  - InstanSeg (Apache-2.0) passes the licence check. Its training-data list is read from the model download inside the run and saved before the result is used.
- R02 confirmation: the frozen R02 protocol on fresh episode seeds 50,000–50,049 (the same count of 50 as R02's test).

## Results so far

| ID | What was run | Measured result | Decision |
|---|---|---|---|
| A03 | Repaired: held-out cell type **and its filename groups** removed from training (run `20261006-212404_A03_unseen_cell_types_groups_excluded`, Modal CPU) | Group overlap after exclusion: 0 for every type. Pooled AUC with groups excluded vs the earlier shared-group rule, on the same photos: whole image **0.756 vs 0.860**; X01 3x3 **0.799 vs 0.886**; X01 5x5 **0.812 vs 0.875**. Per type, whole image (groups excluded): A549 0.768, CACO 0.897, HPMEC 0.836 (only 462 training photos left), HSAEC 0.821, HUVEC 0.981 (test has only 3 good photos), NHBE 0.877. X01 minus whole (groups excluded, pooled): 3x3 **+0.043** [+0.008, +0.083]; 5x5 **+0.056** [+0.002, +0.112]. | **The earlier "no drop on unseen cell types" claim is withdrawn.** Performance drops when groups are kept apart. X01 keeps its advantage over the whole image under the repaired rule. Feature alignment was checked: saved X01 AUCs were reproduced exactly (0.8433 / 0.8751 / 0.8775). |
| X01 | Small-photo handling and format dependence on cached features (run `X01_small_photos_and_formats`, Modal CPU) | Small 640x480 photos (87 photos, **6 bad, 3 groups**): fixed 5x5 **0.529**, size-aware 4x4 **0.693**, small photos sent to the whole-image model **0.741** (the same as whole). The large formats keep the X01 gain (e.g. small-to-whole 5x5: overall 0.879). Format-balanced weighting did **not** help small photos (0.444–0.654). | Evidence is too thin to validate any small-photo rule (6 bad photos). The locked final test has no small photos. Per improvements.md, small-format inputs need **manual review / unsupported scope** unless confirmed. No rule is selected yet; the decision is pending. |
| A02 | Light-retraining timing on A100 (Modal) | 1.60 s per pass (last block), 0.62 s (output layer only); peak GPU memory 8.5 GB; input caching 16 s for 2,397 photos. | Full comparison moved to Kaggle T4 (timing on T4 runs first, as the plan says to measure before scaling up). |
| B03 | New training seed 1, all 5 folds (Modal A100), original clean-up search | Outline F1@0.75 **0.891** (best-outline setting, same as seed 0 v2). Count MAE 6.30 with the *original* narrow search, so this is not yet comparable to v2. | The v2 clean-up search for seed 1 is running on Kaggle. |
| B02 | New training seed 1 (Modal A100) | All 5 fold models were saved; the job was cut off during fold 4's summary (the Mac's connection dropped). | The v2 clean-up search for seed 1 is running on Kaggle (no retraining needed). |

### Results added 7 Oct, 01:45 IST (Kaggle)

| ID | What was run | Measured result | Decision |
|---|---|---|---|
| A02 | Light-retraining timing on the Kaggle T4 (float16) | Last block **13.06 s per pass** (8x slower than A100), output layer only 4.41 s per pass. Peak memory 8.4 GB. Equality with the full model 1.0e-3. | Projected from these measurements, 4 inner splits cost about 2.7 GPU-hours per fold, about 13.5 GPU-hours for all 5 folds (more if the search range extends). It runs after the nucleus GPU job, because Kaggle allows one GPU session at a time. |
| A05 | Changed-copy features for whole, 3x3 and 5x5 (6 changes) on T4 | Done; 6,989 s. | The confirmation fits run as a CPU job (`ai4s-chip-cpu`). |
| B02 | New seed 1, v2 clean-up | Count MAE **4.82** (seed 0: 4.51); outline F1@0.75 **0.866** (seed 0: 0.867). | Outline quality repeats. Count error is slightly worse on the new seed. Count-setting threshold picks touch the edge (0.9) in folds 2 and 4, recorded (improvements.md names the boundary check for B03 only). |
| B03 | New seed 1, v2 clean-up | Count MAE **5.38** (seed 0: 4.68); outline F1@0.75 **0.892** (seed 0: 0.891). | Outline quality repeats. Count error is worse on the new seed. Still no clear counting winner between B02 and B03. |
| B03 | Boundary refinement (thresholds up to 0.999), seeds 0 and 1 | Seed 0: count MAE 4.68 → **4.63**; seed 1: 5.38 → **5.21**. Chosen thresholds now 0.85–0.995, and none sits on the edge. Outline setting unchanged. | Boundary question resolved: a small change, no large hidden gain. min_size = 0 is the valid lower limit (not a search edge). |
| C02 | Repaired (fold 3 done) | Chosen strength 0.1, inside the range, scored on all training batches. Held-out: batch information **0.914** (raw 0.371; chance 0.080); replicate mAP **0.058** (raw 0.007). | Biology retrieval improves, but batch information gets much worse. Report both separately, as improvements.md says. Waiting for the other folds. |
| C02 | Repaired, folds 0, 1 and 2 done | Folds 0/1/2: the chosen strength (1) first sat on the upper edge, so 10 was added, scored worse, and 1 was kept. Biology signal (replicate mAP) raw → sphering: 0.009 → 0.030, 0.001 → 0.041, 0.009 → 0.059. Batch signal: 0.44 → 0.85, 0.40 → 0.84, 0.40 → 0.89 (worse). | The same pattern in all 4 folds so far: biology retrieval up, batch signal much worse. Fold 4 running. |
| R02 | Fresh-condition re-test (first try) | Rule-based controllers re-tested on fresh episodes (e.g. PI mean diameter error 1.44 µm, FF+PI 1.32 µm on TRAIN-range). PPO policies not loaded: Kaggle had unpacked the saved .zip files. | Fixed and rerunning in `ai4s-nuc-cpu2`. |
| B05/B06 | v2 refresh (first try) | Crashed: Kaggle's scikit-image is older than ours. | Rerunning with scikit-image 0.26.0 pinned (`ai4s-nuc-cpu2`). |

Note: the seed-1 v2 clean-up and B03 boundary runs used Kaggle's preinstalled scikit-image (watershed / peak finding). Version effects are checked in F03 (clean rebuild). Repeated on 8 Oct under the pinned scikit-image and under every pinned version: identical (see `Evidence/2026-10-08_F03_package_version_repeat.md`).

### Results added 7 Oct, 11:45 IST (Kaggle)

| ID | What was run | Measured result | Decision |
|---|---|---|---|
| A05 | Confirmation fits (6 changes: flips, turns, brightness x0.8/x1.25 added to training copies), practice folds and the repaired A03 rule | Practice folds, AUC without → with changes: whole 0.8432 → 0.8464; X01 3x3 0.8755 → 0.8727; X01 5x5 0.8775 → 0.8741. Paired difference with changes, pooled, 95% range from resampling filename groups: whole +0.003 [-0.003, +0.012]; 3x3 -0.003 [-0.010, +0.004]; 5x5 -0.003 [-0.010, +0.004]. Under the repaired A03 rule: whole +0.002 [-0.002, +0.007]; 3x3 -0.002 [-0.011, +0.008]; 5x5 +0.001 [-0.006, +0.009]. | **A05 is not confirmed.** The earlier +0.005 does not hold for X01 and all ranges include zero. Not adopted for the chosen candidate. Kept in the record as an unconfirmed small signal on the whole-image model. |
| B04 | Refresh with the retained v2 settings, seed 0 (clean + all recorded stress levels) | Best-outline setting, F1@0.75 clean → noise 0.02 / 0.05 / 0.10: B02 0.865 → 0.858 / 0.787 / **0.254**; B03 0.895 → 0.868 / 0.624 / **0.070**. Blur 3: B02 0.756, B03 0.696. Brightness 0.5: B02 0.856, B03 0.854. Count MAE at noise 0.10: 64 (B02) and 96 (B03) vs about 10 clean. | The noise failure is confirmed for the retained settings. Seed-1 refresh also done (`B04_stress_refresh_*_seed1`). |
| B04 | Planned noise-based training change (per-crop noise up to 0.1, the largest recorded stress level), B02 and B03 seed 0, training-run settings (original clean-up) | F1@0.75 under noise 0.02 / 0.05 / 0.10, unchanged → noise-trained: **B02** 0.853/0.773/0.261 → **0.860/0.859/0.851**; **B03** 0.865/0.626/0.078 → **0.880/0.880/0.872**. Clean outline F1@0.75 (pooled, original settings): B02 0.867 → 0.863, B03 0.891 → 0.885. Costs: blur 3 count error 9.9 → 12.1 (B02) and 17.0 → 20.1 (B03); brightness 0.5 count error 8.9 → 9.8 (B02). B02 count-setting count MAE 5.11 (unchanged training 4.51 with v2; not matched yet). | Large fix for noise at small clean cost. Caveat: the test noise is the same synthetic type as the training noise, so this is not a real-camera test. Matched v2 clean-up and v2 stress for the noise-trained models are running (`ai4s-nuc-cpu3`). |
| B05 | Refresh with v2 settings (seed 0 and seed 1) | Smallest tenth of nuclei found, at overlap 0.50 / 0.75: B01 0.39/0.22; **B02 0.43/0.31**; **B03 0.43/0.26** (count settings). Second tenth: 0.87–0.90 at 0.50. | Tiny nuclei remain the main weakness; confirmed. |
| B05 follow-up | Where tiny nuclei are lost (stage diagnosis, seed 0, v2 settings) | Small nuclei lose in different stages: **B02**: mostly **merged with a neighbour** (612 of 1,814 small nuclei at 0.50 for the count setting, 688 for the outline setting); outline setting also loses 409 to the **minimum-size rule**. **B03 count setting**: 573 never reach the probability threshold (0.85–0.95), 0 removed by size rule. No nucleus is lost in preparation: clipped share 0, but missed nuclei have lower contrast (prepared contrast median 0.14–0.15 vs 0.23–0.27 for found). | Hypotheses to test: B02 marker spacing, B02 outline min-size, B03 threshold. The full sweeps over each stage's recorded range are running (`ai4s-nuc-cpu3`, `b_tiny_sweep.py`). |
| B06 | Refresh with v2 settings, seeds 0 and 1 | B02 count error left, area under the correction curve: random 0.499; B02/B03 disagreement **0.388** (interval versus random [-0.202, -0.009]); blur 0.579; brightness 0.429 [-0.140, +0.005]; perfect order 0.218. | Disagreement between B02 and B03 is a useful warning for B02 (the interval excludes zero). Blur and brightness warnings are not. It cannot catch failures shared by both models. |
| B07 | Refresh with v2 settings, seed 0 and 1 (BBBC038 outside images, grey; colour separate) | Grey images, count MAE / F1@0.75, seed 0: B02 outline **12.9 / 0.502**; B03 outline **9.75 / 0.513**; B02 count **21.3 / 0.409**; B03 count **14.2 / 0.362**. Seed 1: B02 outline 13.8 / 0.501; B03 outline 12.2 / 0.515. Colour images: F1@0.75 = **0.000** for every setting. | Outline settings transfer better than count settings. Count error roughly triples versus inside BBBC039 (do not call it a transfer ratio: different settings and versions). Colour images stay unsupported. |
| B02 modern | InstanSeg (`single_channel_nuclei`, Apache-2.0), same folds, pixel size chosen on the inner fold | Count MAE **3.97** (count setting), F1@0.75 **0.913** (outline setting); F1@0.50 0.950. Joined 666, split 823, missed 544, false 166 (outline setting). 0.105 s per image; 5.3 GB peak GPU memory. Exposure check from the model's own README: **mentions BBBC038 and DSB** (no mention of BBBC039). | Reported in a separate "possible prior exposure" row, not as a clean comparison. B02/B03 trained here: count MAE 4.5–5.4, F1@0.75 0.87–0.89. |
| C02 | Repaired, all 5 folds | Chosen strength 1 on the upper edge in folds 0, 1, 2, 4 (10 was added, scored worse; kept 1) and 0.1 in fold 3. Replicate mAP raw → sphering per fold: 0.009 → 0.030, 0.001 → 0.041, 0.009 → 0.059, 0.007 → 0.058, 0.004 → 0.107. Batch signal raw → sphering: 0.44 → 0.85, 0.40 → 0.84, 0.40 → 0.89, 0.37 → 0.91, 0.40 → 0.81 (higher means more batch information, worse). | Biology retrieval improves in every fold; batch information also rises in every fold, so **batch removal is not demonstrated**. Report both separately. |
| R02 | Fresh-condition confirmation of the frozen protocol (50 new episodes per condition, 5 PPO seeds) | Mean diameter error: PI 1.44 / 2.10 µm, FF+PI 1.32 / 1.95 µm, PPO 1.12 / 1.48 µm (TRAIN-range / UNTOUCHED conditions). PPO minus PI: -0.32 / -0.61 µm, better in 94% / 96% of episodes; minus FF+PI: -0.20 / -0.47 µm, better in 64% / 82%. Out-of-range share: PI 0.053, FF+PI 0.117, PPO 0.003 (TRAIN-range). Worst fresh episode for PPO is sometimes worse than a rule-based one (+0.20 to +0.81 µm). | Confirmed on fresh episodes in the practice simulator. Not a real-chip result. |
| C03 | Input-only depth rerun | **Failed twice before the model started**: the first attempt hit a naming bug in my new option (`c03_prep.py`, fixed), so the second attempt had no data. Rerunning (`ai4s-c03-v3`: input-only depth, then the original depth, both under the same procedure). | Pending. |

## Running / queued on Kaggle (free)

| Kaggle job | Contents |
|---|---|
| `ai4s-chip-gpu` (step 1) | A02 light-retraining timing on T4; A05 changed-copy features (whole + 3x3 + 5x5) |
| `ai4s-nuc-gpu` | B02/B03 noise-training change (seed 0); InstanSeg comparison; B04 refresh (v2, seed 0); B07 refresh (v2, seed 0); tiny-nucleus diagnosis; C03 input-only depth rerun |
| `ai4s-nuc-cpu` | B02/B03 v2 clean-up for seed 1; B03 boundary refinement (seeds 0 and 1); B05/B06 refresh (v2, seed 0); R02 fresh-condition confirmation |
| `ai4s-c02v2-fold0..4` | C02 repaired: every strength on the same full set of training batches; edge extension |

## Still to do after these jobs (none skipped)

- A02 full light retraining (after T4 timing) → combine → A03 rule for retained candidates.
- A05 confirmation fits (CPU), then A08 reliability and A09 review curves for the final candidate. A10 complementarity check from saved failures. A06/A07/A11: check their reopening conditions against the results.
- B: seed-1 B04–B07 refresh; v2 clean-up + B04 stress for the noise-trained models; tiny-nucleus follow-up (change the responsible stage, then retest).
- G01–G03 records; A12 claims removed from the report; report corrections (B05 overlap label, B07 wording, C04 wording, A05 framing).
- F01 (freeze, then the final test once), F02 (workload curves), F03 (clean rebuild), F04 (claims, rights, access, decision record, package). **Submission itself needs the user's go-ahead.**
