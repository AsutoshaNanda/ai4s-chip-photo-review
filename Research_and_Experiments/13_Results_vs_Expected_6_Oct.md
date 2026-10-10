# Results vs what each experiment had to beat (6 Oct 2026, 20:00 IST)

> **For the current combined status, use [the final consolidated file](22_Final_Results_Before_and_After.md#section-4-consolidated-status-of-all-36-original-ids).** It lists every original ID, later work, unfinished tests, proposed next tests and write-up readiness. R02 to R04 now need a corrected evaluation; the old “F01–F04 not started” row below is historical.

> **Superseded in parts on 7 Oct 2026.** Use [16_Results_vs_Expected_7_Oct.md](16_Results_vs_Expected_7_Oct.md) for current numbers. Corrections that apply when quoting this file (from [improvements.md](improvements.md)):
> **A03** "Yes, no drop" is withdrawn (groups were shared; repaired rule: 0.756 whole image, 0.812 X01 5×5) · **A05** not confirmed · **A12** heat maps are not shown as explanations · **B03** count error 4.68 and F1@0.75 0.891 are two different settings, never one configuration · **B04–B07** were first run with the original clean-up settings and are refreshed in file 16 · **B05** "32%" belongs to overlap 0.75; at overlap 0.50 it is 45.2% (original) / 43% (v2) · **B07** "9.6 vs 4.68" is not a direct ratio (different settings and versions) · **C03** depth was chosen using the glowing target, so it is provisional · **C04** means "no suitable dataset found in the recorded search".

**How to read this.** No fixed pass marks were set in advance (user rule), so "had to beat" is the reference
number for the same test: the simpler method, the "without it" version, or chance. "Enough?" is **Yes** only when
the gain is clear (a 95% interval that excludes 0 where one was computed, or a large, consistent gap); otherwise
it says what is still needed. All paths are relative to the project folder `AI4S/`.
Scores: **AUC** = chip-photo ranking grade (0.5 = coin flip, 1 = perfect, higher is better); **count error** =
nuclei per image the count is off by (lower is better); **F1@0.75** = outline match at strict overlap (higher is better).

## Chip-photo quality (A, X)

| ID | Question | Result | Had to beat | Gap (95% interval) | Enough? / next step | Stored at |
|---|---|---|---|---|---|---|
| A01 | Simple methods beat chance? | AUC 0.695 | 0.5 (chance) | +0.195 | Yes (baseline set) | `code/runs/20261006-003232_A01_simple_baselines/` |
| A02 | Big modern model beats small? | AUC 0.843 [0.803, 0.884] | 0.768 (small, ResNet18@448) | +0.075 | Yes; carry ViT-L@518 forward | `code/runs/20261006-012654_A02_frozen_features/` |
| A03 | Works on unseen cell types? | AUC 0.860 [0.819, 0.891] | 0.843 (seen types, A02) | +0.017 | Yes, no drop | `code/runs/20261006-013213_A03_unseen_cell_types/` |
| A04 | Uses the cells, not edges? | Centre-only 0.840; edges-only 0.821–0.834 | 0.843 (whole) | edges keep ~97% | **No**: the signal is photo-wide; check the format shortcut in F01 | `code/runs/20261006-103553_A04_crop_vs_edges/` |
| A05 | Flips/turns/brightness help? | AUC 0.848 (all changes) | 0.843 | +0.005 [+0.001, +0.012] | Too small to matter; don't use | `code/runs/20261006-102436_A05_safe_changes_frozen/` |
| A06 | Class weighting helps? | AUC 0.847 | 0.843 | +0.004 (paired test not run) | No | `code/runs/20261006-013353_A06_class_weighting/` |
| A07 | Sample details add info? | AUC 0.845 | 0.843 | +0.002 | No | `code/runs/20261006-013554_A07_photos_plus_details/` |
| A08 | Confidence correction helps? | AUC 0.829 (Platt), log loss 0.505 | 0.843, log loss 0.486 | −0.014, worse | No; keep raw scores | `code/runs/20261006-013826_A08_calibration/` |
| A09 | Better review order? | review area 0.682 (proposed) | 0.692 (big model alone) | −0.010 [−0.016, −0.002] | No; big model alone is best | `code/runs/20261006-013846_A09_review_order/` |
| A10 | Two models guard each other? | AUC 0.835 (average) | 0.843 | −0.008 | No | `code/runs/20261006-013905_A10_two_models_guard/` |
| A11 | Smart picking saves labels? | AUC 0.810 at 800 photos (uncertainty) | 0.794 (random) | +0.016 (5 folds × 5 repeats, noisy) | Small gain; not needed for the submission | `code/runs/20261006-013926_A11_label_efficiency/` |
| A12 | Heat maps show real reasons? | Hiding "important" squares: −0.05 to +0.43 | random squares: −0.38 to −0.71 | important < random in all 5 folds | **No**: maps unreliable; don't show them as explanations | Modal fold 0: `code/runs/20261006-103711_A12_heatmaps_folds0/`; Kaggle folds 1–4: `code/runs/20261006-160015_A12_heatmaps_folds1/`, `…175753_A12_heatmaps_folds2/`, `…160015_A12_heatmaps_folds3/`, `…170439_A12_heatmaps_folds4/` |
| **X01** | Zoomed patches help? | **AUC 0.877** (whole + 5×5 patches) | 0.843 | **+0.034 [+0.012, +0.056]**; 3×3–6×6 all +0.032–0.034 | **Yes, the best idea so far.** Next: repeat with new random starts (F01); fix small 640×480 photos (0.53 vs 0.74) | `code/runs/20261006-103251_X01_whole_plus_patches/`, `code/runs/20261006-103757_X01_whole_plus_patches_grids4-5-6/` |

## Nucleus counting (B)

| ID | Question | Result | Had to beat | Gap | Enough? / next step | Stored at |
|---|---|---|---|---|---|---|
| B01 | Simple counting baseline | count error 6.73; F1@0.75 0.787 | — (sets the bar) | — | Yes (bar set) | `code/runs/20261006-100550_B01_cv_threshold_watershed/` |
| B02 | AI beats B01? | **count error 4.45**; F1@0.75 0.847 | 6.73 / 0.787 | −2.28 errors (−34%) | Yes; paired interval to add in F01. Wider search gave 4.51 (no change) | `code/runs/20261006-105952_B02_unet_fg_seed0_combined/`, `code/runs/20261006-161349_B02_unet_fg_repost_v2/` |
| B03 | Edge class helps? | count error 4.68; **F1@0.75 0.891** | B02: 4.45 / 0.847 | counting tied; outlines +0.044 | Yes for outlines. Repeated on 2 machines, identical | `code/runs/20261006-103241_B03_unet_boundary_seed0_combined/`, `code/runs/20261006-161337_B03_unet_boundary_repost_v2/` (Mac repeat: `…155516_B03_unet_boundary_repost_v2/`) |
| B04 | Robust to bad images? | strong noise (sd 0.10): F1@0.5 0.36 (B02), 0.14 (B03) | clean: 0.907 / 0.929 | large drop | **No** for strong noise; blur and dim OK | inside the B02/B03 combined folders: `B04_stress_per_image.csv` |
| B05 | Tiny nuclei and crowding | smallest 10% found: 32% (B02) at IoU 0.5 | larger nuclei: 79–95% | big gap | **No**: tiny nuclei are a known weak spot; report it | `code/runs/20261006-110006_B05_crowding_tiny_nuclei/` |
| B06 | Warning flag catches bad counts? | area 0.410 (B02 vs B03 disagreement) | 0.500 (random) | −0.090 [−0.173, −0.002] | Barely; perfect order is 0.232, so it catches little | `code/runs/20261006-110006_B06_warning_flags/` |
| B07 | Works on a new dataset? | grey images: count error 9.6 (B03); colour: F1 ≈ 0 | 4.68 on our data | 2× worse; colour fails | **No**: limit the claim to grey fluorescent nuclei | `code/runs/20261006-110006_B07_bbbc038_outside/`, overlap check `code/runs/20261006-102723_G01_bbbc038_check/` |

## Extra datasets (C)

| ID | Question | Result | Had to beat | Gap | Enough? / next step | Stored at |
|---|---|---|---|---|---|---|
| C01 | Features survive lab batches? | 86.0% correct (new batches) | 97.0% (same batches); chance 0.09% | −11 points; corrections ±0.5 | Partly; may be inflated (provider's model may have seen these batches). Non-commercial licence → not for the submission | `code/runs/20261006-102841_C01_rxrx1_batches/` |
| C02 | Re-scaling removes noise, keeps biology? | repeat-matching 0.049; batch tell-apart 0.95 | raw: 0.006 / 0.40 (chance 0.08) | biology ×8 better, batch noise worse | **Mixed**: no clean win. Optional next: test one standard batch-correction method | `code/runs/20261006-104800_C02_jump_batch_rescaling_fold0/`, `…161337_…fold1/`, `…161328_…fold2/`, `…161338_…fold3/`, `…164500_…fold4/` |
| C03 | Predicted glowing image useful? | route A count error 8.48 | route B (direct) 6.50 | +1.98 worse [+0.84, +3.09] | **No**: measuring directly is better | `code/runs/20261006-162515_C03_dic_to_lamin_v2/` (first unfair version: `…110158_C03_dic_to_lamin/`) |
| C04 | Works on another lab's chips? | no outside data exists | — | — | **Blocked**; unblock only by emailing authors (needs your OK) | `Research_and_Experiments/Evidence/2026-10-06_C04_search_log.md` |

## Control simulator (R)

| ID | Question | Result | Had to beat | Gap | Enough? / next step | Stored at |
|---|---|---|---|---|---|---|
| R01 | Usable environment? | practice simulator reproduces the paper's equations | — | — | Yes (labelled "practice only") | `code/runs/20261006-003501_R01_droplet_model_check/`, `Research_and_Experiments/Evidence/2026-10-05_R01_search_log.md` |
| R02 | Learned controller beats rule-based? | size error 1.44 µm (untouched test) | 1.89 (FF+PI), 2.14 (PI) | −0.45 µm | Yes (on the simulator only) | `code/runs/20261006-011158_R02_pi_vs_ppo/` |
| R03 | Staged training helps? | 1.52 µm | 1.44 (ordinary) | +0.08, worse | No | `code/runs/20261006-012739_R03_staged_vs_ordinary_ppo/` |
| R04 | Extra inputs help? | all inputs 1.44 µm | camera only 1.79 | −0.35 | Yes | `code/runs/20261006-013244_R04_input_sets/` |
| R05 | RL picks labels better? | AUC 0.767 at 800 photos | random 0.794; uncertainty 0.810 | −0.027 vs random | **No** (10× training also no) | 150 rounds: `code/runs/20261006-101832_R05_label_picking_fold0…4/`; 1500 rounds: `code/runs/20261006-101938_R05_label_picking_fold0…4/` |

## Checks (G) and not started (F)

| ID | Result | Enough? / next step | Stored at |
|---|---|---|---|
| G01 | All data checked (MD5 match, decoding fixed, 43 BBBC038 copies found) | Yes | `code/runs/20261005-233524_G01_bbbc039_check/`, `…002338_G01_ooc_check/`, `…102723_G01_bbbc038_check/` |
| G02 | Scoring tests 20/20; final tests locked (50 nucleus images; 675 chip photos) | Yes | `code/runs/20261005-233829_G02_scoring_tests/`, `…000039_G02_bbbc039_splits/`, `…002426_G02_ooc_splits/` |
| G03 | Speed: A100 167 photos/s, Kaggle T4 15.5, Mac 5.9 (big model) | Partly: Vast not measured (not needed now) | `code/runs/g03_kaggle_t4/`, `code/runs/g03_modal_A100.json`, `code/runs/20261006-005904_G03_real_data_pilot/` |
| F01–F04 | Not started | Next: repeat X01 + B02/B03 with new random starts; then **one** locked final test | — |

**Stopped runs** (not results): `code/runs/_incomplete/`. **Run list:** `code/runs/registry.csv`. **Live status:** `Research_and_Experiments/10_All_Experiments_Parallel_Schedule.md`.
