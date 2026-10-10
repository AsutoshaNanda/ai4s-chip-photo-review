# Experiment run log

**When the project folder was checked on 2 October, no model runs or model outputs were recorded here.** Work elsewhere was not checked. The completed sample-detail checks belong in [the evidence record](06_Evidence_and_Decisions.md), not in a model-performance table. Blank measurements mean not measured.

The [complete schedule for work that can run together](10_All_Experiments_Parallel_Schedule.md) covers all 30 existing plans and six extra feature/RL entries. Updating the schedule started no jobs. Give each actual run a separate ID; one experiment plan can lead to several runs.

## Extra fields when work runs at the same time

Besides the fields below, record priority, status, IDs of runs that must finish first, who is doing the work, computer/session, separate output folder, measured small-trial cost, estimated finish, and any missing requirement or resource. Valid statuses are planned, ready, running, measured, confirmed, rejected, blocked by prerequisite, blocked by capacity, and not applicable with evidence. A prerequisite is something needed first; capacity means the available time and computing resources.

The current GPU provider, device count, remaining allowance, spending budget, daily work hours, and availability of the person doing the work are unverified. Do not record paid or guaranteed free resources as available until checked. A run that has not happened has no measured score.

## Register each run before starting it

| Field | Required entry |
|---|---|
| Run ID and plan ID | Unique run ID; G/A/B/C/F plan from the experiment guide or X/R entry in the complete schedule |
| Date and person responsible | Actual run date and responsible person/agent |
| Idea to test | What the comparison can support or reject |
| Main result and useful margin | One main measurement, whether higher or lower is better, and the minimum useful change |
| Data | Source, version, file-check values called hashes, meaning of the target label, and usage-rights status |
| Independent sample unit | Confirmed meaning of each sample group, or a clearly labeled unconfirmed clue used for grouping |
| Sample-group file list | Exact training/development/calibration/test IDs and the file list's hash; calibration data is used to adjust probability estimates |
| Method/comparison | Simple reference method and candidate; one part changed |
| Model origin | Exact trained-model file ID/hash, training sources, and overlap status |
| Settings | Versions, random seed, training settings, data/image changes, and stopping rule |
| Budget | Computer, speed measured in a small trial, and time/memory limits |
| Saved outputs | Planned locations for predictions, measurements, figures, and logs |

## Fill these in immediately after the run

| Field | Required entry |
|---|---|
| Observation | Actual numbers, failures, and missing outputs |
| Evidence quality | Direct measurement, repeated result, independent test, or an indirect substitute |
| Overall result | Reference method and candidate using identical measurements |
| Result for each group | Cell type, physical group, data conditions, and whether a warning/review rule is on or off, as relevant |
| Worst failures | Sample IDs, images with any outlines drawn over them, and observed effects |
| Uncertainty | Interval for the difference on matching examples, variation by random seed, sample/group counts, and limits |
| Time/cost | Measured, not predicted |
| Decision about the idea | Supported, rejected, or inconclusive |
| Changed action | Keep, reject, limit the claimed use, or investigate |
| Smallest next test | One concrete follow-up, with a reason |
| Record links | Code/settings version and saved predictions/results |

A record of an important result is unfinished until it says what decision changes and what happens next. Add failures and conflicting findings to the main evidence record immediately.

## Comparison list

| Run | Plan | Data division | Method | Main result | Worst group | Time taken | Evidence quality | Decision |
|---|---|---|---|---|---|---|---|---|
| `20261005-233339_G01_bbbc039_check` | G01 | all 200 | yes/no mask reading | **Rejected**: merges touching nuclei (19,565 vs 23,615 objects) | — | 49 s, Mac | Direct | Superseded by the next run |
| `20261005-233524_G01_bbbc039_check` | G01 | all 200 | provider value-aware decoding | 23,615 nuclei; splits 100/50/50; all 20 plates in every split | — | 48 s, Mac | Direct | Use this decoding |
| `20261005-233829_G02_scoring_tests` | G02 | fake answers | scoring tests | 20/20 pass | — | 10 s, Mac | Direct | Scoring approved |
| `20261006-000039_G02_bbbc039_splits` | G02 | provider test locked; 150 practice in 5 plate folds | — | Final test 50 images / 5,720 nuclei | — | 8 s, Mac | Direct | Locked |
| `20261005-234900_B01_threshold_watershed` | B01 | tune 100 training → report 50 validation | threshold + watershed | Count MAE 7.18 (best-count); pooled F1 0.87 at IoU 0.5 (best-count), 0.895 (best-outline) | best-outline misses small nuclei (419 missed) | 24 min grid, Mac 10 processes | Direct, single split | Re-run on 5 folds |
| `g03_kaggle_t4`, `g03_kaggle_l4`, `g03_kaggle_l4_v2`, `g03_modal_T4.json` | G03 | random tensors | speed test | ViT-L@518 frozen 15.5 img/s (T4); PPO faster on CPU | L4 not available on Kaggle | ~5 min each | Direct | Kaggle = 2×T4 lane |
| `*_G01_ooc_check` | G01 | all 3,072 | archive/metadata/pixel checks | MD5 match; 0 mismatches; 4 photo formats; 29 near-identical pairs | format tracks label (D54) | 46 s, Mac | Direct | Normalise format; report within format |
| `*_G02_ooc_splits` | G02 | 675 final test (13 groups) / 2,397 practice (5 folds) | grouped stratified split | Locked with SHA-256 | — | seconds | Direct | Locked |
| `*_A01_simple_baselines` (2nd run) | A01 | 5 practice folds | majority / details / photo measurements / size-normalised / format only | AUC 0.695 (photo measurements) vs 0.639 (details) vs 0.576 (format only) | within color 2048×1536 photos: AUC 0.506; HSAEC bad found 6% | minutes, Mac | Direct, cross-validated | Judge everything within format |

## Suggested prediction records

For category predictions, save image ID, group ID/type, cell type, original training/test assignment, evaluation assignment, true label, predicted chance of bad quality, predicted label, review score, review status, model ID, and prediction time.

For nuclei, save image ID/group, reference and predicted counts, TP/FP/FN under the fixed matching rule (correct matches/false detections/missed objects), count error, warning score, saved location of the predicted object mask, model ID, and prediction time.

Do not save just an average score. Each sample's predictions are needed to compare errors, find methods that catch different failures, and check the final table.

## Final selection record

Chosen task/category: **pending**.

Chosen method: **pending**.

Decisive runs and independently confirmed benefit: **pending**.

Worst failures, tested backup, and claims left out: **pending**.

Resolved usage rights/overlap and a repeat from a clean setup: **pending**.

Reason the choice follows the strongest evidence: **pending**.

Having this blank template is not evidence that a submission is ready.
