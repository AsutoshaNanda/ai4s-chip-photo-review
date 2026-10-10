# All experiments: priority and parallel schedule

## Live status: updated Tue 6 Oct, 11:15 IST

**Modal stopped at $19.62 of $30 (user's $20 line; billing report lagged).** Remaining GPU work moves to Mac/Kaggle (free).

This follows the approved [simple start plan](11_Simple_Start_Plan_5_Oct.md); target days below replace the older windows.

**Rules for this update:**
- No fixed pass marks: every comparison reports full curves. See D46 in the [decision log](06_Evidence_and_Decisions.md).
- No experiment is skipped.
- Run IDs are listed in `code/runs/registry.csv`.

| ID | Target day | Status | Evidence so far / next step |
|---|---|---|---|
| G01 | Mon night → Tue AM, then per new source | running | BBBC039 and chip photos **measured** (D49, D53, D54: photo format tracks the label). C-source checks pending. |
| G02 | Tue AM | measured | Scoring 20/20 (D50); nucleus final test locked (50 images); chip final test locked (675 photos, 13 groups, D55). |
| G03 | Tue AM–noon | running | Kaggle T4 speed **measured** (`code/runs/g03_kaggle_t4`). Kaggle L4 not available (2 tries ran on T4). Mac, Modal, Vast next, plus the real-data pilot per pipeline (audit addition). |
| A01 | Mon night | measured | AUC 0.695 (photo measurements) vs 0.639 (details) vs 0.576 (format only); within color 2048×1536 photos only 0.506 (D56). Shortcut check done. |
| A02 | Tue–Wed | measured | `20261006-012654_A02_frozen_features` (Modal A100): ViT-L DINOv2 @518 AUC 0.843 [0.803, 0.884] vs ResNet18 @448 0.768, @224 0.717; within format 0.74–0.83. |
| A03, A06, A11 | Wed | measured | A03 unseen cell types: ViT-L@518 AUC 0.860 [0.819, 0.891]. A06 balanced weights: +0.004 (inside noise). A11 label-efficiency curve saved (noisy; ~0.85 at 23 of 39 groups). |
| A04 | Wed | measured | Centre 0.5 crop 0.840 vs whole 0.843; edges-only still 0.82–0.83; crop shifts ±20% 0.83–0.84. Signal is spread over the whole photo (photo-wide quality or format shortcut). |
| A05 | Wed | measured | Safe changes added to training (frozen features). Paired, group-resampled: all changes +0.005 [+0.001, +0.012]; single changes −0.000 to +0.007, mostly crossing 0. Real but tiny; many versions tried. |
| X01 | Wed | measured | **Whole + patches 0.875–0.877 (3×3 to 6×6 grids, plateau) vs whole 0.843; paired gain +0.032 to +0.034, 95% [+0.007, +0.057] (excludes 0 at every grid)**; better within both large formats; worse on small 640×480 photos (0.74 → 0.51–0.54). |
| A07, A08, A09, A10 | Wed–Thu | measured | A07 photo+details 0.845 vs photo 0.843 (no gain). A08 calibration (Platt/isotonic) lowers AUC and raises log loss. A09 review order: ViT-L alone best (0.692). A10 two-model average 0.835, no gain over ViT-L alone. |
| A12 | Wed–Thu | measured (5 folds; fold 0 Modal, folds 1–4 Kaggle) | **Heat maps fail the check.** Hiding the 'most important' squares lowers the bad score LESS than hiding random squares (e.g. 8/64 hidden: −0.05…+0.43 vs −0.38…−0.71, all folds). Maps vs shuffled-label maps: rank similarity ≈0 (varies by fold). Hiding style changes the map (median vs blur fill agree only 0.52–0.64). → Treat maps as unreliable; don't use them as explanations. |
| B01 | Mon night | measured (5 folds) | `20261006-100550`: count MAE 6.73 (bias −1.40), F1@0.75 0.787; most crowded quarter MAE 15.0. Earlier single split: | Run `20261005-234900_B01_threshold_watershed`: on 50 practice-check images, best-count setting off by 7.2 nuclei per image (about 118 per image); best-outline setting has higher F1 at every overlap level but misses small nuclei. Re-run on the 5 plate-grouped folds next. |
| B02 | Tue–Wed | measured (U-Net); ready-made model pending | 5-fold U-Net: count MAE 4.45, F1@0.75 0.847. Wider clean-up search (Kaggle): 4.51 / 0.845 (no change; spacing now chosen inside the range, 8–12). Tied with B03 v2 on counting (4.68); B03 has better outlines (0.891). Ready-made models trained on DSB 2018 have seen 9 of our 50 test images (B07) — audit before use. |
| B03 | Wed | measured | Inside/edge U-Net. First search: MAE 6.10, F1@0.75 0.873 (settings on range edge). **Wider search (Kaggle, same models): MAE 4.68, F1@0.75 0.891**; outline settings now inside the range; count threshold 0.95 still at the top in 2 folds (max possible 1.0). |
| B04 | Wed | measured | Stress (inside B02/B03 runs): blur and dim images cost little; strong noise (sd 0.10) collapses both (B03 F1@0.5 0.14, B02 0.36). |
| B05 | Wed | measured | Smallest 10% of nuclei found only 22–33% (IoU 0.5) by all methods; most crowded quarter: B02 MAE 7.8 vs B01 15.0. Gallery saved. |
| B06 | Wed–Thu | measured | Warning flag for B02 counts: only B02-vs-B03 disagreement beats random (area 0.410 vs 0.500, 95% [−0.173, −0.002]); blur/brightness flags do not; perfect order 0.232. |
| B07 | Wed–Thu | measured | BBBC038 (CC0): 43 images are copies of BBBC039 (9 in our final test) → removed. On 519 grey images: B03 outline best (MAE 9.6, F1@0.5 0.69); B02 best-count overcounts (+43). Colour tissue images: all methods fail (F1 ≈ 0). |
| C01 | Wed | measured | RxRx1 embeddings (CC-BY-NC-SA, research only): siRNA accuracy 86% on new batches vs 97% same batches; U2OS 54%. Simple batch corrections: no change (±0.005). Embedding model may have seen test batches → upper bound. |
| C02 | Wed | measured (5 folds) | JUMP ORF, 12 batches. Sphering learned on training batches: cross-batch replicate mAP 0.006 → 0.049 (higher in every fold) but batch identifiable 0.40 → 0.95 (chance 0.08). Mixed: helps biology retrieval, makes batch stand out more. |
| C03 | Wed–Thu | measured (fair v2) | v1 (Mac) said route A (predict picture → measure) wins 8.5 vs 22.5, but route B had no speck clean-up. **v2 (Kaggle), same clean-up rule for both: route B (direct) 6.50 vs route A 8.48 count MAE; A−B 95% [+0.84, +3.09]** → predicting the picture does not help counting. F1@0.5 equal (0.47). Route B clean-up size at range top in 1 fold. |
| C04 | Wed–Thu | blocked: no data (with evidence) | Search log `Evidence/2026-10-06_C04_search_log.md`: only our own Zenodo set (and a paper re-using it); skin-on-chip lead has no download link and a different label meaning. Unblocking needs the user's OK to contact authors. |
| R01 | Tue | running | Search started (`Evidence/2026-10-05_R01_search_log.md`): best lead is a droplet-size model fitted to real lab data. Otherwise a labelled practice simulator. |
| R02 | Wed | measured | Practice simulator. PPO size error 1.44 µm (untouched test) vs PI 2.14 and FF+PI 1.89; PPO out-of-range share 0.009 vs 0.086/0.144. |
| R05 | Wed | measured | RL label picker loses to random and uncertainty picking at equal photos labelled (e.g. 800 photos: 0.767 vs 0.794 vs 0.810); 10× training no better (learned 'pick big groups'). |
| R03, R04 | Thu | measured | R03: staged training 1.52 vs ordinary 1.44 µm (no gain). R04: all inputs best; camera-only worst (1.79); noise ×4 hurts most. |
| F01 | Thu | planned | Repeats, then one locked final test |
| F02 | Thu | planned | Workload numbers from saved predictions |
| F03 | Fri | planned | Fresh Kaggle rebuild |
| F04 | Fri (Sat buffer) | planned | Claims, links and submission check; user presses Submit |

---

Earlier version below; it is kept for history.

Updated: **3 October 2026, Asia/Kolkata**. The user requested a 2 October start; work on this planning refresh resumed on 3 October. No completed model runs are recorded here. All unfinished setup carries into today; earlier dates are not evidence of completed work.

**All 30 existing experiment and validation plans are included below.** The earlier E0 to E6 list is mapped to those plans. Additional whole-image/local-feature and RL ideas are also scheduled separately. This file replaces the earlier assumption of one experiment at a time and the project-wide limit of two improvements.

This is a proposed schedule, not an execution report. The inspected workspace contains no recorded model runs. Current GPU provider, device count, quota, spending limit, daily work hours, and operator availability are unverified. Earlier interest in free/low-cost services and Colab is not an available resource allocation. No rental or spending is authorized by this update. Completion of all full experiments cannot be promised before data and timing checks. Every item remains listed, including any that cannot yet run.

## How to use the priorities

- **P0:** First claim on available resources: trustworthy data, core comparisons, major failure checks, confirmation, and delivery.
- **P1:** Run beside P0 when dependencies and resources allow. These are scheduled experiments, not a discarded backlog.
- **P2:** Begin the prerequisite check immediately. Run the method when suitable data, a meaningful task, and capacity are established.

Priority controls resource contention. Independent work does not have to wait for every higher-priority experiment to finish. A failed prerequisite must be recorded as blocked or not applicable, not as a completed experiment. A promising result can raise any experiment's priority; contradictory evidence reopens its affected decision immediately.

## Parallel work tracks

| Track | Work | Resource use and sharing |
|---|---|---|
| Shared checks | G01 to G03, file loading, split definitions, source/rights checks, duplicate checks | Mostly local CPU and inspection. Reuse tooling, but define a separate valid data contract for each task. |
| Chip quality | A01 to A12 and X01 | One training slot when available. Image statistics, cached-feature fitting, scoring, and review analysis can overlap training. |
| Nucleus counting | B01 to B07 | Start B01 without waiting for the chip route to fail. B02 uses a separate training slot or the priority queue. Later analyses reuse masks and predictions. |
| Additional datasets and RL | C01 to C04 and R01 to R05 | Research, small downloads, rights checks, and metadata work can overlap both main tracks. Training requires its own passed checks and allocated capacity. |
| Validation and delivery | F01 to F04, demo, report, video | Start setup and drafting early. Protect final confirmation and reproduction capacity for every model or task retained in the submission. |

## Complete coverage: the original 30 plans

The dates are proposed target windows from the 3 October continuation, not measured duration estimates. The deadline last checked live on 2 October is 10 October at 21:29:59 IST. Prerequisites must pass before a dependent run starts, even when both appear on the same date. Read the full protocol and keep/reject rules in [the experiment playbook](04_Experiment_Playbook.md).

| ID | Priority | Experiment or check | Target window | Must be ready first | Output or decision |
|---|---|---|---|---|---|
| G01 | P0 | Usable data, labels, and rights | 3 Oct; each new source before use | Source files and proposed use | Verified input contract or an explicit blocker for each route. |
| G02 | P0 | Separate related samples and check scoring | 3 Oct; each new source before fitting | G01 for the source | Fixed train/development/calibration/final-test lists and checked scoring code. |
| G03 | P0 | Small runnable pilot and timing | 3 Oct; each distinct pipeline | G01/G02 for the pipeline | Save/reload proof, memory, speed, and a measured run-budget estimate. |
| A01 | P0 | Simple image and metadata comparisons | 3 Oct | Chip G01/G02/G03 | Majority, image-statistics, and metadata-only results on identical splits. |
| A02 | P0 | Pretrained features and staged fine-tuning | 3 to 4 Oct, after baseline readiness | A01 reference and valid chip pipeline | Frozen-feature and limited-fine-tuning comparison, with saved predictions. |
| A03 | P0 | Transfer to unseen cell types | 4 to 6 Oct | A02 configuration and valid development groups | Each eligible type, class counts, worst type, and the resulting scope limits. |
| A04 | P0 | Full-image, crop, and context dependence | 4 to 5 Oct | A01/A02 and label-preserving view definitions | Matched view comparisons; evidence about useful signal or shortcuts. |
| A05 | P1 | Label-preserving image changes during training | 4 to 6 Oct | A02 and inspected training examples | A controlled transformation comparison; no blur added with an invalid unchanged label. |
| A06 | P1 | Class weighting or sampling | 4 to 6 Oct | A01/A02 class and subgroup errors | One balancing-policy comparison with both classes and worst types reported. |
| A07 | P1 | Images plus culture information | 4 to 6 Oct | A01/A02 and verified available fields | Image-only, metadata-only, combined, and missing-field results. |
| A08 | P0 | Check and correct probability confidence | 4 to 6 Oct; final candidate before 7 Oct test | Candidate predictions and disjoint calibration groups | Probability-error scores and reliability by group. Remove unsupported confidence claims. |
| A09 | P0 | Review order at equal workload | 4 to 6 Oct; freeze rule before 7 Oct test | A01/A02 predictions and separate rule-selection/evaluation data | Bad samples caught at 10%, 20%, and 30% review, including each supported type. |
| A10 | P1 | Complementary models and failure guard | 5 to 6 Oct | Two eligible methods with saved predictions | Joint failures, simple combination, guard behavior, and runtime. No gain assumed from model count. |
| A11 | P1 | Label efficiency and active selection | 4 to 6 Oct | Fixed A02 setup and a training-only label pool | Nested 25%/50%/100% group subsets, repeated selections, and random versus active selection at equal budget. |
| A12 | P1 | Check visual explanations | 4 to 6 Oct | A02 and a proposed explanation method | Randomized-model and region-removal checks. Remove explanations that mislead. |
| B01 | P0 | Simple nucleus-counting comparison | 3 Oct | Nucleus G01/G02/G03 | Correct decoded masks, threshold/watershed counts, errors, and runtime. |
| B02 | P0 | Small trained model versus modern method | 3 to 4 Oct, after baseline readiness | B01 and model-rights/exposure audit | Same-image comparison, saved masks, counts, and timing. |
| B03 | P1 | Boundary or distance training targets | 4 to 6 Oct | B02, with evidence of merged-object errors | One changed target; merged, fragmented, missed, and count errors. |
| B04 | P1 | Image robustness | 4 to 6 Oct | B02 and documented plausible changes | Clean and each synthetic stress condition separately; one justified training change if needed. |
| B05 | P0 | Difficult objects and groups | 4 to 6 Oct | B01/B02 saved predictions | Dense/sparse and small/large-object results, with a worst-case gallery. |
| B06 | P0 | Error warnings and correction workflow | 4 to 6 Oct; timed use under F02 | B01/B02 predictions and fixed large-error definition | Equal-workload comparison against random and simple warnings; real correction time separately. |
| B07 | P1 | Independent microscopy dataset | Screen 3 to 4 Oct; development test 5 to 6 Oct or reserved final test 7 Oct | B02 plus source-specific checks; select development or final-test role before any use | Per-source errors and warning behavior. If data are not independent, record the limitation. |
| C01 | P1 | Cell features across experimental batches | Screen 3 to 4 Oct; eligible pilot 4 to 6 Oct | Source-specific G01/G02/G03 and a defined biological task | Small feature-table comparison, held-out-batch results, and runtime. |
| C02 | P1 | JUMP normalization and biological retrieval | Screen 3 to 4 Oct; eligible pilot 4 to 6 Oct | Eligible profiles and their processing history; independent of C01 | Raw versus one training-fitted adjustment, with biological signal and technical effects measured separately. |
| C03 | P2 | Brightfield-to-fluorescence prediction | Screen 3 to 4 Oct; eligible pilot 4 to 6 Oct | Real aligned pairs, intended-use rights, separate sample groups, and a downstream reference; screen the Allen lead | Simple prediction model versus direct analysis of the original image; meaningful measurement plus visual errors. |
| C04 | P0 | Independent real-chip validation | Screen 3 to 4 Oct; development check 5 to 6 Oct or reserved final check 7 Oct | Compatible source, labels, rights, and groups; trained A model | External performance and review utility. Choose development or final-test role before use. |
| F01 | P0 | Confirm selected improvements independently | Repeat promising signals 4 to 6 Oct; locked final evaluation 7 Oct | Baseline/candidate artifacts and protected final data | Individual runs, paired differences, worst groups, and uncertainty for every retained claim. |
| F02 | P0 | Practical usefulness | Define 3 Oct; offline checks 4 to 6 Oct; eligible timed use through 7 Oct | Working review/correction flow and appropriate operator for time claims | Measured effort and errors, or an explicitly labeled offline estimate. No outreach is authorized by this plan. |
| F03 | P0 | Clean reproduction | Setup record from 3 Oct; rehearsal 5 to 7 Oct; clean reproduction 8 Oct | Versioned inputs, environment, models, and evaluation entry point | Reproduced results and a record of setup failures and fixes. |
| F04 | P0 | Claims, access, and submission | Rolling from 3 Oct; final package audit and submission 8 to 9 Oct | Evidence ledger, report, code, video, eligibility check | Traceable claims and confirmed submission by the internal 9 October target. |

G01 to G03 are repeated per independent task/source as needed. F01 to F04 cover every method or task included in the final entry. The 30 IDs describe plans, not a promise that exactly 30 training jobs will suffice.

## Original E0 to E6: no work lost

| Earlier experiment | Covered by |
|---|---|
| E0: image/mask loading | G01, G02, B01 |
| E1: simple nucleus count | B01 |
| E2: small U-Net | B02 |
| E3: separate touching nuclei | B03 |
| E4: training flips/rotations | B04, with clean-image validation |
| E5: warnings for bad counts | B05, B06 |
| E6: fresh run and final test | F01, F03 |
| Earlier extra external-data test | B07 |

These are the [original plans](../Resources/06_experiments.md), mapped rather than duplicated as unnecessary extra runs.

## Additional feature and RL experiments

These six entries supplement the original 30. Their results are unmeasured. Research supporting the control direction and its current data limits is in [the RL reference](09_Remaining_Timeline_and_RL_Assessment.md#does-rl-or-multi-stage-ppo-fit).

| ID | Priority | Comparison | Target window and dependency | Keep/reject decision |
|---|---|---|---|---|
| X01 | P1 | Whole-image features versus whole image plus local regions | 4 to 6 Oct after A04 and stable A02; metadata combination only after A07 shows value | Use matched source-image splits and comparable training budgets. Do not assign the image's label blindly to every crop. Add combined features only after their individual value is understood. |
| R01 | P2 | RL feasibility: controllable actions, outcomes, and reliable environment | 3 to 4 Oct alongside all source checks | Require an observable objective, meaningful actions, validated dynamics or usable interaction data, and independent evaluation. Static quality labels alone do not pass. |
| R02 | P2 | Simple feedback controller versus ordinary PPO | 4 to 5 Oct if R01 and pipeline checks pass and measured cost fits confirmation time | Compare tracking error, constraint failures, worst conditions, training interactions, and runtime. Keep a fair reference controller and record all failed runs. |
| R03 | P2 | Ordinary PPO versus multi-stage training | 5 to 6 Oct after R02; matched total training budget | Compare the same total interaction budget: ordinary conditions, then justified noise/dynamics variation. Test untouched conditions. Record each stage; do not credit extra training budget to the staging method. |
| R04 | P2 | Minimal control inputs versus multiple global features | 5 to 6 Oct after R02; independent of R03 | Compare available pressure/flow/image/history inputs under matched budgets. Fit scaling on training only, exclude future information, and test missing/noisy inputs. Combine with R03 only after isolated effects are known. |
| R05 | P2 | RL-based label selection versus random and simple active selection | Screen 3 to 4 Oct; pilot 5 to 6 Oct after A11 controls and a valid selection task | Reveal labels only from the training pool. Match annotation and compute budgets, report selection-run variability, and keep final-test labels hidden. Existing labels do not measure human labeling time. |

R05 is a different task from fluid control and does not depend on R02. A positive result on either task must receive F01 confirmation before becoming a headline claim. If R01 fails, R02 to R04 remain blocked with the exact missing prerequisite; they are not silently removed or called tested.

## Resource-aware execution

Do not assume that two jobs running on one GPU will finish faster. G03 measures whether concurrency fits memory and improves throughput. Start all independent low-cost work that fits available CPU, memory, storage, and network limits.

| Available capacity | Execution policy |
|---|---|
| One GPU plus a local computer | Queue accelerator jobs by priority, while local data checks, small CPU models, saved-prediction analyses, and report work continue. Start A02 and B02 in the same early window, using sequential GPU slots if needed. |
| Two GPUs or independent training sessions | Give A and B one slot each initially. Keep source checks and scoring outside those slots where possible. Reassign freed capacity using actual results. |
| More independent capacity | Start eligible C/R pilots and split A/B comparisons across slots. Preserve confirmation capacity and a shared fixed protocol. |

After the first pilot, write available device-hours, measured run durations, memory limits, and operator review time into the run log. For each job, compare its projected finish with the 6 October candidate-selection target and reserved confirmation/reproduction windows. An infeasible job stays explicitly queued or blocked by capacity; add actual capacity or reduce its exploratory size. Never quietly turn a partial comparison into a completed result.

The initial A02 and B02 model limits remain small pilot comparisons per route. There is **no project-wide limit of two improvement experiments**. Run A05, A06, A07, B03, B04, and other independent changes against their shared route baseline. Only combine supported changes afterward.

A single-factor comparison changes one component at a time; it does not require running only one experiment at a time. Use the current [resource readiness guide](../Resources/19_Resource_Readiness.md) to check the exact input or method first. Shared baseline predictions can be reused only when data splits, seed, preprocessing, model version, and evaluation protocol match. Keep immutable configurations and separate output folders so parallel runs cannot overwrite each other.

## Daily decisions and finish conditions

1. **3 October, carry-over setup first:** Complete any unfinished G01/G02/G03 checks, registration review, source searches, and compute inventory. No earlier planned action is assumed complete. Set up the report and run record alongside this work.
2. **3 October, after prerequisites pass:** Run A01/B01 and start eligible A02/B02 jobs. Continue every C/R prerequisite check independently. Record measured runtime, memory, transfer, and save/reload behavior; if setup does not finish today, keep its dependent jobs visibly pending rather than claim the calendar was met.
3. **4 to 6 October:** Run all eligible independent comparisons, including C/X/R items. Repeat promising results immediately. By the 6 October checkpoint, each item has a real result or a specific unresolved prerequisite/capacity status. A feasibility check alone is not a completed model experiment.
4. **6 October, selection checkpoint:** Freeze the supported submission candidates after development-side confirmation. A branch unable to finish confirmation and reproduction remains listed and uncompleted rather than entering the report as a result.
5. **7 October:** Evaluate frozen choices on the untouched final data and complete eligible practical-use checks. Model choices cannot be tuned against these final outcomes. Unexpected failure reopens the claim.
6. **8 October:** Reproduce every retained result from a clean setup, finish the report and video, and audit access. Drafting and demo construction have already been running alongside experiments.
7. **9 October:** Submit and verify status. **10 October** is the buffer before the live **21:29:59 IST** cutoff. A late change requires unused independent confirmation data, complete reproduction, and an updated decision ledger before submission; otherwise retain it as unfinished work.

Do not compare chip-classification, nucleus-counting, phenotype, and control metrics as though they measure the same task. Select the final story from measured usefulness, technical contribution, reliability, and deliverability. A route with severe unexplained group failure cannot be rescued by averaging its score with another task.

## Status and evidence rules

Use these statuses: **planned, ready, running, measured, confirmed, rejected, blocked by prerequisite, blocked by capacity, or not applicable with evidence**. This planning update launched no runs. None of the 36 entries has a recorded measured or confirmed model result in this workspace. Existing metadata checks remain source/audit evidence, not completed model tests.

At registration, record plan ID, owner/operator, input/model version, split, metric, comparison, machine, estimated cost from G03, and output path. At completion, record the observation, evidence quality, supported/rejected explanation, changed action, and smallest next test. Keep failure outcomes and incomplete jobs visible.

More comparisons increase the chance of an accidental good score. Use development data for exploration, reserve final data from the start, confirm the selected improvements, and report the number of comparisons tried. Repeated seeds test training stability, not biological independence.

Protect the 7 to 9 October confirmation, reproduction, and submission targets when allocating measured run cost. No fixed available hours or fraction of an unknown compute budget is assumed. Stop repeating a failed approach after two controlled unsuccessful repairs; reopen its explanation or try a structurally different method. Evidence, not the planned priority label, decides promotion.
