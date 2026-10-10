# Finding bad organ-on-a-chip photos efficiently — zoomed patches on a frozen foundation model, validated without shortcuts

**Category (to declare at the top of the Writeup): Model & Algorithm** *(your choice: Tool & Platform also fits the review workflow)*
**Team:** *[to be filled by the team leader: members, roles, backgrounds in AI/CS and biology — the rules give a bonus for cross-disciplinary teams]*
**Status of this draft:** the main image tests, their final evaluation, and a limited clean rebuild are finished. C04 is blocked by missing outside chip data. R02 to R04 were re-scored under a repaired evaluation on 8 October ([review](Evidence/2026-10-07_RL_scoring_review.md), [result](Evidence/2026-10-08_RL_repaired_evaluation.md)). This is a working draft, not a finished submission. Items in *[brackets]* need completion. Use [the consolidated status and readiness check](22_Final_Results_Before_and_After.md#section-7-can-we-write-and-submit-the-write-up).

## 1. Project summary (200–300 words)

Organ-on-a-chip experiments produce photos that people must review for quality. We built a tool that ranks photos from most to least likely to be bad, so a person can check likely problems first. It uses a frozen pretrained image model (DINOv2 ViT-L) on the whole photo plus a 3×3 grid of zoomed tiles, and a simple logistic-regression scorer. On the **672 automatically scored photos out of 675 final photos**, it reaches AUC **0.850** (95% range 0.78–0.92), against **0.80–0.81** for the whole-image model. The paired gain is **+0.04 to +0.05**, with ranges excluding zero in both the first evaluation and a clean rebuild on the same test data. It puts bad photos earlier than random order (review area 0.703 vs 0.499); actual time saved has not been measured.

Absolute ranking scores fall under the harder test that keeps a cell type and its filename groups out of training (AUC 0.80 vs 0.76 for the whole image), while the patch advantage remains. An input check sends small or unfamiliar photo formats and colour nucleus images to manual review.

Alongside, small U-Net models improved nucleus outlines on the final images (F1@0.75 0.90 vs 0.86), with a modest counting gain. Tiny nuclei, strong noise and colour images exposed weaknesses; noise training helped in artificial-noise tests. We retain negative results, including light retraining and failed heat-map explanations. A clean rebuild on a free Kaggle GPU reproduced the chip development scores to four decimals and retrained one nucleus split; four nucleus clean-up searches repeated under the declared package versions gave identical results. Public repository publication remains pending.

## 2. Problem and application

Quality control of chip photos is a bottleneck for chip-based in-vitro studies. Our aim: reduce the number of photos a person must open to catch the bad ones, and say clearly where the tool must not be trusted. We separate four questions that are often blurred: (1) ranking bad photos, (2) counting and outlining nuclei, (3) whether labels or features survive lab-to-lab differences, (4) a control simulator. Only (1) and (2) are the main contribution; (3) and (4) are supporting studies.

## 3. Data, licences and compliance

| Data | Used for | Licence (as read 7 Oct 2026) | Notes |
|---|---|---|---|
| Organ-on-Chip (OOC) image dataset, Zenodo 10203721 | Chip-photo quality (A, X) | CC-BY 4.0 per the Zenodo record; the paper says CC-BY-SA | Conflict unresolved at the source; we credit the authors and treat the stricter reading as binding for shown images. Dataset is not redistributed. 3,072 photos: 2,397 for development, 675 locked as the final test |
| BBBC039 | Nucleus training and validation | CC0 | 150 development images, 50 locked final images |
| BBBC038 (DSB 2018 stage-1 train) | Outside-data test of the nucleus models | CC0 | 43 images that are copies of BBBC039 photos were removed |
| JUMP-CP ORF profiles | Supporting study C02 | open (Cell Painting Gallery); metadata BSD-3 | *[confirm data licence line before naming]* |
| Allen Cell DIC / lamin B1 stacks | Supporting study C03 | research/non-commercial terms, no redistribution | *[team decides whether C03 is shown; no data is included in the repository]* |
| RxRx1 | Supporting study C01 | non-commercial / share-alike | *[research record only; not in the package]* |

No personal or clinical data are used. Details: `Research_and_Experiments/Evidence/2026-10-07_release_rights_check.md`.

## 4. Methods

### 4.1 Chip-photo review aid (main contribution)
* **Photos:** grey, long side 1036 px for the whole-photo view; tiles are cut at native resolution.
* **Features:** frozen DINOv2 ViT-L/14 (Apache-2.0; weights file hash recorded) at long side 518; for each photo the whole-photo feature plus the mean and max over a 3×3 grid of tiles (native resolution).
* **Scorer:** standardised logistic regression; the regularisation strength and the yes/no cut-off are chosen by grouped inner cross-validation on training photos only. No success pass mark was invented after viewing the final results. The shipped tool uses scores for ranking.
* **Selection rule:** written before the light-retraining results were known (`scripts/freeze_choice.py`): start from the whole-image model and move to a more expensive candidate only if the paired gain under the repaired unseen-cell-type test has a 95% range excluding zero. Result: 3×3 tiles (5×5 not clearly better, light retraining clearly worse).
* **Input gate** (`scripts/input_gate.py`): 640×480 photos (87 photos, six bad, three groups) and formats unseen in development go to manual review; colour nucleus images likewise. 3 of the 675 final photos are in an unseen portrait format and were routed to review.

### 4.2 Nucleus analysis
* B01: threshold and watershed baseline. B02: small U-Net (foreground + distance-watershed). B03: U-Net with a nucleus-edge class. Clean-up settings are searched on an inner validation fold only; two objectives are kept separate (count error; outline score) and never quoted as one configuration.
* Plate-grouped five-fold development; the 50 final images and 675 final photos were locked before any training.
* Tested training change for noise: per-crop noise up to the strongest level already recorded in the stress test (0.1).

### 4.3 Supporting studies (not the main claim)
C02 tests batch re-scaling of image profiles; C03 compares predicting a fluorescence image with direct measurement. R01 to R04 use a practice droplet simulator; their control comparisons were re-scored under a repaired evaluation (identical scenarios for every controller). R05 separately tests choosing labels for chip photos and did not help. These supporting studies are distinct from the main image result.

## 5. Validation design (what protects the numbers)
* Final sets locked first (SHA-256 of ids recorded in `splits/*lock.json`); opened once per frozen configuration.
* Folds keep whole filename groups together (a clue to acquisition batches, not verified physical chips). A repaired test also removes the held-out cell type's groups from training; the first version leaked shared groups (1,813 of 2,397 photos) and its "no drop on new cell types" claim is withdrawn.
* Differences are paired on identical photos with 95% ranges from resampling whole groups. No assumed pass marks; full curves (bad photos found vs share reviewed) are saved.
* Results refer to stored runs in `code/runs`. The registry is an index and contains older incomplete entries. `scripts/f04_trace.py` checks that named paths exist; it does not verify every number or every saved file. The later RL scoring review demonstrates why source and calculation checks are also needed.

## 6. Results

### 6.1 Chip photos
| Test | Whole image | X01 3×3 | Paired gain (95% range) |
|---|---|---|---|
| Development, 5 grouped folds (2,397 photos) | AUC 0.843 | **0.875** | +0.032 (+0.008 to +0.057) |
| Repaired unseen-cell-type rule (groups removed) | 0.756 | **0.799** | +0.043 (+0.008 to +0.083) |
| **Locked final test (672 automatic photos), clean rebuild (fresh pinned environment, one consistent pipeline) — main result** | 0.810 [0.74, 0.89] | **0.850 [0.78, 0.92]** | **+0.040 (+0.015 to +0.067)** |
| Locked final test (672 automatic photos), first run (kept as history; mixed software environments) | 0.801 [0.73, 0.89] | **0.850 [0.78, 0.92]** | **+0.049 (+0.010 to +0.086)** |

The two final-test rows differ only in the whole-image baseline (0.801 vs 0.810). The cause is not isolated. What we measured: the first run trained on development features made in one software environment and scored final-photo features made in another, the clean rebuild used one pinned environment throughout, and the baseline's chosen regularisation changed (C 0.001 to 0.0031623). We tested one earlier explanation, clipping of the model inputs, and withdrew it because the inputs lie between −2.118 and +2.640, so clipping at ±10 cannot change them. The candidate is unaffected (0.8497 vs 0.8496). We use the consistent clean-rebuild comparison (0.850 vs 0.810) as the main result and report both.

Review workload (area under the bad-photos-found curve; 0.5 = random): X01 **0.703** on the final set, **0.710** in development, versus 0.609 for a simple photo-statistics order. Raw scores are reasonably reliable on the two large formats (calibration error 0.02–0.04) but not on small photos (0.25); Platt and isotonic corrections made log loss worse, so raw scores are shown. Per cell type X01 is at least as good as the whole image on the final set. HUVEC has only three good photos, so its 1.0 is not meaningful.

### 6.2 Nuclei
* Development (v2 clean-up): B02 count error 4.51, outline F1@0.75 0.867; B03 4.68 / 0.891; B01 6.73 / 0.787. Second training seed: B02 4.82 / 0.866; B03 5.38 / 0.892. No clear counting winner between B02 and B03; both beat B01.
* **Noise:** at noise 0.10 the retained models fall to F1@0.75 0.254 (B02) and 0.070 (B03); noise-based training gives 0.851 and 0.871 at that level with a small clean cost (0.865 → 0.860; 0.895 → 0.882) and worse clean counting (4.49 → 5.27; 4.71 → 6.48). The test noise is of the same synthetic type as the training noise.
* **Tiny nuclei:** only 43% of the smallest tenth are found (overlap 0.50). The minimum-size rule and B03's threshold lose them, but every setting change that recovers them costs overall quality (e.g. B03 threshold 0.40: small-nucleus recall 0.54 but F1@0.75 0.806 instead of 0.872 for the retained setting).
* **Outside data (BBBC038, grey):** outline-setting count error 9.75 (B03) and 12.91 (B02), against 9.86 and 10.01 for the same settings on our data; count-setting error 14.2 and 21.3 against 4.7 and 4.5; colour images fail completely (F1@0.75 = 0.000), so they go to manual review.
* **Shared failures:** B02 and B03 are wrong in the same direction on 83 of 150 images (78% of B02's count error), so their disagreement is only a partial warning.
* **Modern model:** InstanSeg (Apache-2.0) reaches count error 3.97 and F1@0.75 0.913 on our folds, but its documentation names BBBC038/DSB, so it is reported as a separate "possible prior exposure" row, not a clean comparison.
* **Locked final test (50 images, mean over the five fold models; sd over fold models in brackets for counts):** count setting, count error / F1@0.75: **B02 4.31 (0.19) / 0.847**, **B03 4.90 (0.37) / 0.880**, B01 4.93 / 0.803. Outline setting, count error / F1@0.75: B02 9.34 / 0.876, **B03 9.90 / 0.901**, B01 11.36 / 0.855. Second seed: B02 4.36 / 0.844 (count), 9.60 / 0.878 (outline); B03 5.27 / 0.876, 10.18 / 0.899. So on the final images the **outline advantage over B01 is clear** (0.901 vs 0.855 for the outline setting) but the **counting advantage is small** (B02 −0.6 errors, B03 about equal to B01), smaller than in development (6.73 vs 4.5). The noise-trained variant costs clean counting (B02 4.63, B03 5.92). The 50 final images share plates with development images, no interval is computed on 50 images, and this is not a new-lab test. Correction workload (simulated from saved predictions): reviewing the 10 most-disagreeing images of 50 (20%) removes 37% of B02's count error (random order 20%); area under the error-left curve 0.36 vs 0.50 random and 0.24 perfect order.

### 6.3 Supporting studies
C02: biology retrieval improves in every fold (mean 0.006 → 0.059) but batch information also rises (0.40 → 0.86), so batch removal is not shown. C03: measuring nuclei directly beats predicting the fluorescence image under both depth rules (count error 7.25 vs 12.09 with depth chosen from the plain image only); choosing the depth with the fluorescent target made both routes look better. **R02–R04 (repaired evaluation: 50 new identical scenarios, saved policies, five training seeds):** the learned controller's mean diameter error is 1.008 / 1.333 µm (training / harder ranges) vs 1.382 / 2.062 for the simple rule controller and 1.175 / 1.946 for the model-assisted one (paired differences −0.374 [−0.431, −0.312] / −0.729 [−0.816, −0.633] and −0.167 [−0.257, −0.080] / −0.613 [−0.723, −0.503]), with 0.4% / 0.8% of steps outside the operating range vs 6.6–10.5% / 8.4–13.0%; after the first half of each target block the model-assisted controller is better in the training ranges (+0.104 µm for the learned one). Staged training is worse (+0.076 / +0.068 µm); all inputs beat camera-only (−0.394 / −0.405 µm). Simulator only; R02, R03 and R04 share one set of five policies. R05 is separate: curve estimates at 800 photos are 0.767 for RL vs 0.794 for random and 0.810 for uncertainty picking; 10× training gives 0.755, with no improvement.

### 6.4 What did not work (kept in the record)
Light retraining of the last block (0.819 vs 0.843 frozen); training-photo tweaks (no gain for X01); no confirmed benefit from class weighting or sample details; probability corrections; a two-model guard; heat-map explanations (hiding "important" squares changed the score less than random squares in all five folds); RL label picking; fixed-grid patches on small photos (0.53 vs 0.74). Staged RL showed no gain under the original scoring, but that comparison is now provisional pending the shared evaluation repair.

## 7. Reliability and limitations
* Development groups are filename clues, not verified physical chips; the final chip set shares no photos with development but may share acquisition conditions.
* The final nucleus images share plates with development images; this is not a new-lab test. The final chip set contains no small-format photos, so the small-photo problem is unresolved and handled by the gate.
* Results are on one chip dataset and one fluorescence dataset; transfer to other labs is untested.
* Noise robustness is shown on synthetic noise only. Tiny nuclei remain a documented weakness. Colour nucleus images are unsupported.
* Control studies run on a practice simulator, not a real chip. R02 to R04 were re-scored under the repaired protocol ([result](Evidence/2026-10-08_RL_repaired_evaluation.md)); they remain simulator results with 50 scenarios and five training seeds. C03 reference masks come from fluorescence, not hand counts.
* The review-workload curves are simulated from saved predictions; they are **not** measured time savings.
* Several early nucleus analyses ran under Kaggle's preinstalled scikit-image 0.25.2 instead of our pinned 0.26.0. The seed-1 clean-up and the B03 boundary extension were repeated under the pinned scikit-image and again under every pinned version: all four gave identical results (difference 0). The B04 seed-0 refresh, B05, B06 and B07 analyses were not repeated; they use the same clean-up code.

## 8. Reproduction
A clean rebuild on a free Kaggle T4 (`kaggle/jobs/ai4s-f03`, 8 steps, 42 min in total: environment 4 min, data download 10 min, photo preparation 2 min, features 14 min, fits and evaluation 2 min, one nucleus fold retrained in 10 min) installed pinned packages in a fresh environment, downloaded the public data with checksum, recomputed all features from scratch and reproduced: development 5-fold AUC 0.84327 / 0.87512 (records 0.8433 / 0.8751); final-test X01 3×3 AUC 0.8496 (first run 0.8497); a nucleus fold retrained from scratch (F1@0.75 0.875 / 0.892 vs 0.877 / 0.892; GPU training is not bit-identical). Four nucleus clean-up searches (B02 seed 1; B03 seed 1; B03 seed 0 and seed 1 with thresholds to 0.999) were repeated in a fresh environment with every pinned version (`kaggle/jobs/ai4s-f03c-*`, 13–54 min each plus about 3–4 min to build the environment) and were identical to the earlier runs (`scripts/f03_pinned_compare.py`). *[exact command list for the README to be added]* Entry points: `scripts/f01_final.py` (final evaluation), `scripts/freeze_choice.py`, `scripts/a03_groups.py`, `scripts/x01_small.py`, `scripts/b_unet.py` + `scripts/b_repost.py` (nuclei), `kaggle/jobs/ai4s-f03` (clean rebuild on a free Kaggle GPU). Pinned package versions: `code/requirements-lock-mac.txt` and `cloud/modal_run.py`.

## 9. External resources and licences
DINOv2 ViT-L/14 weights (Apache-2.0); timm, PyTorch, scikit-learn, scikit-image, NumPy, pandas, SciPy, Stable-Baselines3, Gymnasium (permissive); InstanSeg `single_channel_nuclei` (Apache-2.0, comparison only). StarDist and Cellpose-SAM were considered and **not** used (training-data exposure; non-commercial training data). Data licences: Section 3. The public code will carry an OSI-approved licence *[team to choose]*.

## 10. Prior work and what is new
Existing before this project: the published datasets, DINOv2, the U-Net design, watershed clean-up, InstanSeg. New here: the zoomed-tile review aid with its selection rule; the repaired group-aware validation (and the withdrawal of the earlier unseen-cell-type claim); the input gate; the noise-training fix and tiny-nucleus stage diagnosis; the shared-failure analysis; the record of negative results.
