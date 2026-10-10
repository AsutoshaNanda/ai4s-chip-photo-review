# Results review and next steps

Reviewed 6 October 2026. Scope: the results report, saved development predictions, experiment code, split metadata and original plan. No new models were trained and no final-test images or predictions were opened.

**Recommendation: keep X01 as the provisional lead for chip-photo review. Keep B02 and B03 as separate nucleus-analysis candidates. Fix the evaluation gaps and known failures before F01, then finish F02-F04.**

Confidence is high in the file-based corrections below. Confidence in which model will win a fresh final test remains provisional.

## Final verdict after checking the two screenshots

**Yes, improve the entry before submission. First repair validity, then run focused improvement tests on the strongest routes, while protecting final confirmation and the working demo.** A higher development score alone does not establish a stronger entry. No reliable probability of winning can be calculated from these results.

The screenshots were treated as claims to check, not instructions to execute. Source inspection confirms the following distinctions:

| Item | Verdict | Action |
|---|---|---|
| C03 depth selection | A validity limitation, not merely an untuned setting. `c03_prep.py` selects the central depth from the real fluorescent target and then selects all five retained planes around it. This affects both the central-plane and all-planes reports. Equal treatment of both routes does not make the selection usable when fluorescence is unavailable. | Rebuild selection using the plain input image alone or a fixed acquisition rule established without held-out target images, then rerun both routes. Until then, label the comparison as target-assisted plane selection with automatically generated reference masks. Do not call it independent proof of true nucleus counts or a deployable stain-free workflow. |
| C02 batch selection | The inner search uses only four training batches, with `rng.permutation(tr_batches)[:4]` called again for each candidate strength. Candidate strengths can therefore be evaluated on different batch subsets. This is an uneven tuning comparison; it is not, by itself, held-out-test leakage. | Use the same complete set of eligible training batches for each candidate strength. Save the selected settings and rerun the held-out-batch evaluation. Keep biology retrieval and batch information as separate outcomes. |
| Fixed learning rate, crop, batch size and stopping patience | These are recorded baseline choices, not established best settings. Fixed settings alone do not invalidate a controlled comparison. Exhaustively tuning everything is neither possible nor required. | Test sensitivity only where it can change the final choice. Choose settings inside training/development groups, use matching examples and comparable resources, and report the search limits. |
| B03 threshold at the search boundary | The best tried value is not necessarily the best feasible value. A boundary choice also does not prove that a better value exists beyond it. | Extend/refine the training-only cleanup search toward the valid probability limit. Watch missed and merged nuclei as well as count error. Refresh all downstream checks if settings change. |
| C03 direct-route cleanup boundary | Worth checking if C03 is retained, but depth-selection validity comes first. | After rebuilding valid inputs, extend the training-derived cleanup range if the selected setting still sits at its boundary. |
| Big-model retraining | Light retraining was already promised in A02, contrary to the screenshot heading saying these ideas were not planned. It remains unfinished and could change the main model choice. | Promote one controlled frozen-versus-light-retraining comparison after the split repair. Use the existing large backbone before adding an even larger one. Record whether retraining improves worst formats and review workload, not just overall AUC. Keep the frozen model as the control. |
| Higher resolution or another model | A testable extension, not a guaranteed gain. X01 supports investigating useful local detail; it does not prove that a larger network or full retraining will help. | First test image-size-aware patches on the existing model. Evaluate a higher-resolution or larger-model run if measured resources and prior results justify it without displacing final confirmation. |
| Paired uncertainty checks | Required for decisive promotion/rejection claims. B02/B03 count comparisons were added in this review. | Check remaining decision-relevant comparisons on matching examples with group-level uncertainty. It is unnecessary to rerun every rejected route merely to fill a table. |
| Modal spending statement | The screenshots report a costing error. This follow-up did not independently reconcile invoices or establish its cause. | Before any further paid runs, reconcile actual usage and all charged resources, then set an enforceable budget with allowance for billing delay. No spending is authorized by this review. |

### Recommended improvement tests

These are proposals, not completed results or guaranteed gains. They revise the earlier ordering below by moving A02 light retraining ahead of secondary C/R expansion.

1. **Main route:** fix A03 group separation, then compare the frozen X01 control with light retraining and image-size-aware patch handling. Test each change separately before testing a combination. Preserve A05 as a positive but unconfirmed follow-up. Use patches based on available image detail, with a whole-image candidate for small inputs. Do not infer validated fallback behavior from the examples already used to design it.
2. **Camera shortcut check:** the existing pipeline already converts images to grey, which does not eliminate camera/resolution cues. Compare methods within matched formats and excluded acquisition groups. Test format-balanced training only within training groups, while preserving true quality cues. Lower within-format scores alone do not prove shortcut use; differences in case difficulty can also contribute.
3. **Nucleus route:** confirm B02/B03 using exact settings, address plausible noise and tiny-object misses, check the threshold boundary, and retain the planned modern-model comparison only with verified training-exposure and rights status. Keep B01 as an independently structured reference because its outside-data results remain useful.
4. **Secondary research:** repair C02 and C03 before making stronger claims about them. Keep them on the inventory, with provisional or pending status until repaired. R02/R04 remain positive simulator results; failed R03/R05 versions do not deserve priority over better-supported improvements without a new diagnosis.
5. **Submission contribution:** make the chip-photo review workflow the main story, conditional on successful confirmation. Demonstrate how many bad images are found at each review workload, show failures and supported input limits, and make the result reproducible. Nucleus analysis and simulator control may remain separate supporting studies; do not pretend they were validated together on real chips. Keep the tested unreliable heat maps out of explanatory claims.

The official judging allocates 30% to impact, 30% to technical approach and innovation, 20% to results and validation, 10% to reproducibility and 10% to presentation. This supports reserving effort for a useful, demonstrable and reproducible workflow as well as model improvements. These weights do not supply a formula for predicting winning odds. [Official judging criteria](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview#evaluation).

## 1. Are all experiments present?

All **36 experiment IDs** are represented in [the results report](13_Results_vs_Expected_6_Oct.md). All 38 explicit run/file references checked in that report exist.

- **31 IDs have measured parts.** This includes setup checks, failed approaches and simulator-only results. It does not mean 31 successful or fully completed protocols.
- **C04 is blocked:** the recorded search found no suitable downloadable outside chip-quality dataset. That does not prove no such dataset exists.
- **F01-F04 are unfinished.** There is no completed final confirmation, final-candidate workload study, clean reproduction or final claims check.
- Some earlier substeps remain pending: A02's light fine-tuning comparison, B02's eligible modern ready-made model comparison, and B04's proposed training response to the observed noise failure. Keep them visible rather than calling their entire original protocols complete.
- B04-B07 describe the original B02/B03 cleanup settings. The later `repost_v2` runs changed those settings. Their safety and outside-data results must be refreshed for any retained v2 configuration.

The [audit JSON](Evidence/2026-10-06_results_review_audit.json) records the checks, source-file hashes, recalculated numbers and evidence-to-decision ledger. These are checks of existing development results, not new independent model experiments.

## 2. Corrections that change the next decision

### A03 does not yet establish group-independent transfer

The code excludes the held-out cell type, but keeps other cell types from the same filename groups in training. **1,813 of 2,397 evaluated photos belong to groups also present in training.** Every held-out cell type has this overlap. All HUVEC and NHBE evaluation photos have it.

This is direct evidence of shared proxy groups, not proof that the photos show the same physical chips. The filename groups themselves are unconfirmed acquisition clues.

**Action:** reopen the "Yes, no drop" conclusion. Rerun A03 excluding both the held-out cell type and every group used to evaluate it. Report each type and the remaining training support. Compare on matching held-out images. Do this for the retained X01 candidate as well as the whole-image reference. Its 0.860 score cannot establish better transfer than A02's 0.843 under a different splitting rule.

Source: [A03 implementation](../code/scripts/a_suite.py), function `a03`; [split definitions](../code/splits/README_ooc.md).

### X01 has a real development signal and a serious input limitation

AUC is a ranking score: 0.5 is chance and 1 is perfect. I independently recalculated the scores and paired uncertainty ranges from the saved predictions. The ranges below use 2,000 resamples of whole filename groups.

| Comparison | AUC change | Exploratory 95% range |
|---|---:|---:|
| X01 3x3 minus whole image | +0.0318 | +0.0076 to +0.0567 |
| X01 5x5 minus whole image | +0.0342 | +0.0127 to +0.0591 |
| X01 5x5 minus X01 3x3 | +0.0024 | -0.0095 to +0.0171 |
| A05 transformations minus whole image | +0.0052 | +0.0003 to +0.0117 |

X01 improves both large-photo formats. However, on the **87 small 640x480 photos**, 5x5 drops to **0.527**, versus **0.741** for whole-image features. There are only **six bad photos** in this subgroup, so the estimate is fragile. The paired development difference is nevertheless negative: -0.214, range -0.375 to -0.040.

**The locked 675-photo final test contains no small-format photos.** Passing it cannot resolve this failure. Keep the locked membership unchanged and state this coverage limit.

**Action:** compare 3x3 and 5x5 using worst-format performance, full review curves and measured running time. The evidence does not establish that 5x5 is better. Test a whole-image fallback or a patch rule that accounts for image size using development data only. Do not call that fallback validated just because it is plausible or improves the already-inspected examples. If confirmation is insufficient, send small-format inputs for manual review and limit the supported scope.

Possible explanation to test: the fixed grid produces patches with much less image information on small photos. This is a hypothesis, not a demonstrated cause.

### B03's best counting and best outlines are different settings

F1 measures how well detected nuclei match the reference shapes. Higher is better. The following table uses the saved pooled F1 at overlap 0.75, consistently across rows.

| Method and exact setting | Count error per image | F1 at overlap 0.75 |
|---|---:|---:|
| B01, original best-count setting | 6.73 | 0.787 |
| B02, original best-count setting | 4.45 | 0.847 |
| B02 v2, best-count setting | 4.51 | 0.845 |
| B03 v2, best-count setting | 4.68 | 0.872 |
| B03 v2, best-outline setting | 9.86 | 0.891 |

**4.68 count error and 0.891 outline F1 are not one operating setting.** Also, comparing 0.891 with B02's 0.847 compares different objectives. For outline-selected settings, B02 v2 reaches 0.867; for count-selected settings, B03 v2 reaches 0.872.

New paired checks, resampling the 20 plates, support the count improvements over B01:

- B02 minus B01: **-2.28 nuclei/image**, 95% range **-4.68 to -0.52**.
- B03 v2 minus B01: **-2.05**, range **-4.49 to -0.30**.
- B03 v2 minus B02: **+0.23**, range **-0.49 to +0.87**. A clear counting winner between them is not established.

These checks reuse one training seed. Repeating cleanup on two computers checks reproducibility, not independent training. The reported headline B03 metrics agree across machines, but "identical" is too strong: pooled F1 at overlap 0.95 differs slightly, 0.420040 versus 0.419982.

**Action:** keep B02 and B03 for confirmation, label the count and outline modes, and repeat B04-B07 with the exact retained settings. Do not promote a broad nucleus-counting claim from the clean-data average.

### B05 and B07 need corrected interpretation

- **B05:** the original B02 best-count method found **45.2%** of the smallest tenth at overlap **0.50**. **32.2%** is the result at **0.75**. The report labels the latter with the wrong overlap. Tiny nuclei remain a weakness.
- **B07:** the original B02 count setting reaches **48.84** count error on outside grey images. B03's original outline setting reaches **9.56**, and B01's outline setting reaches **11.16**. This reverses the clean-benchmark preference. B01 outline F1 at overlap 0.50 is **0.716**, versus **0.693** for B03 outline. Preserve these distinct alternatives while testing the retained configurations.
- The report's "9.6 versus 4.68, twice worse" comparison mixes an original outside-data outline setting with a later internal count setting. Use matching configurations before claiming a transfer ratio.
- Colour-image F1 is near zero. Restrict the supported input type or require manual handling until a separately validated method exists. B02/B03 disagreement alone is not an established protection against their shared failures.

Sources: [B05 saved results](../code/runs/20261006-110006_B05_crowding_tiny_nuclei/results.json), [B07 saved results](../code/runs/20261006-110006_B07_bbbc038_outside/results.json), [B03 v2 saved results](../code/runs/20261006-161337_B03_unet_boundary_repost_v2/results.json).

## 3. What to keep, pause and retain as separate research

| Route | Decision and smallest useful next action |
|---|---|
| X01 | Provisional main chip-quality candidate. Repair transfer evaluation, test small-photo handling, then confirm against A02. |
| A02 whole image | Keep the reference and fallback candidate. Shared backbone with X01 means it is not automatically an independent safeguard. |
| B02 and B03 | Keep both for nucleus analysis. Confirm training stability and exact-setting stress/outside results before choosing a default. |
| B01 | Preserve the simple independent comparison. Its outside-data outline performance remains useful. |
| A05 | Small positive signal, not automatically worthless. Keep as an unconfirmed follow-up after validity fixes. Training-only transformations do not add an inference-time transformation step. Many variants were tried, so confirmation matters. |
| A06, A07, A08, A09, A10 | Current versions do not justify promotion. Recheck only the necessary calibration/review behavior for the final X01 candidate; these earlier conclusions were based on A02. An AUC score is not a guarantee that a displayed probability is accurate. |
| A11 | Retain the noisy label-efficiency result as research. It is not needed to deliver the present review tool. |
| A12 | Omit the tested heat maps as explanations. Their failure does not show that every possible explanation method fails. Show real failure examples instead. |
| C01 | Research-only evidence with uncertain prior model exposure. Do not infer categorical competition ineligibility merely from the licence abbreviation; exact-use compatibility remains unverified. Do not make the submission depend on unresolved rights. |
| C02 | Preserve the mixed result as provisional. Repair the inconsistent inner batch subsets before relying on the selected setting. An alternative correction method remains a secondary experiment. |
| C03 | Direct measurement wins in the recorded comparison, but all retained planes were selected using the fluorescent target. Reopen the deployment claim and rerun both routes with input-only selection before generalizing the preference. Do not promote predicted fluorescence as an established counting improvement. |
| C04 | Keep blocked with its search record. No claim of independent-lab chip validation. Author contact requires explicit user instruction. |
| R01, R02, R04 | Preserve the positive simulator evidence separately. R02 already has five training seeds. Real-chip control remains untested; final simulation claims need a frozen protocol, worst-case reporting and fresh confirmation conditions if development choices used the existing test results. |
| R03, R05 | Stop promoting the tested staged-training and RL label-picking versions. Retain their negative results and controls. Do not generalize their failure to all RL. |

Chip-photo review, fluorescent nucleus analysis and simulator control remain separate tasks. Their scores cannot be combined into one validated system score.

## 4. Order of work from here

These are proposed dates, not measured duration promises. The required checks take priority over the calendar.

1. **6-7 October: fix the decision-changing gaps.** Rerun A03 with groups excluded, test X01's small-photo handling, and refresh B04-B07 for retained settings. For nucleus noise failure, test the already-planned training change based on plausible noise, against unchanged training with matched resources. Check clean and every stress level separately. Examine tiny-object misses before proposing another targeted training change.
2. **7-8 October: test focused improvements, then confirm the shortlist on development data.** Following the screenshot review, prioritize the already-planned A02 light-retraining comparison after the split repair. Compare against the frozen control on matching groups. For B02/B03, use newly chosen training seeds with fixed splits and matching references. X01 uses frozen features and a largely deterministic classifier, so merely changing a random-start seed is weak confirmation. Reuse cached features for controlled grouped development resplits or nested comparisons while keeping the final-test membership fixed. These repeats still do not create independent outside data. Select using full curves, subgroup failures and measured cost. Keep the eligible modern B02 comparison visible. If measured resources cannot fit an experiment alongside required checks, record it as pending rather than completed or disproven.
3. **Before opening final tests: freeze the complete rules.** Record exact weights, settings, preprocessing, supported formats, fallback/manual-review handling, probability display and review order. Write a decision ledger connecting the strongest positive and negative evidence to the choice. If unresolved failure contradicts the proposed claim, narrow the claim or withhold that route.
4. **F01 and F02: run the fixed final evaluation once, then report practical workload.** Use the 675 reserved chip photos and 50 reserved nucleus images. Report per-type/format and worst-case results, including groups too small to judge. The nucleus final images share plates with development images; this is not a new-plate or new-lab test. Measure full review/correction curves against simple alternatives. Claims about time saved require an actual timed user study. If final results disappoint, report them; tuning on that same final test would require a new independent test later.
5. **By 9 October if ready: finish F03 and F04.** Reproduce the frozen result from a fresh accessible environment, prepare the actual demo video, public code and report, verify source/model rights and access, and finish the existing Kaggle draft. Keep all unsupported or incomplete routes clearly labelled. Final submission remains a separate user action.

I verified the official page in the live browser on 6 October: the deadline is **10 October 2026, 21:29:59 IST**, and the account shows an in-progress draft. The required package includes a Kaggle Writeup, public demo video, public code and technical report. Judging includes impact and method quality as well as measured results, so confirmation and a working research workflow deserve time now. [Official overview and requirements](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview).

## 5. Limits of this review

The new uncertainty ranges describe the existing development data after multiple approaches were explored. They are not corrected for selecting the best of many trials, proof on new laboratories, or a forecast of winning. No GPU rental, outreach, model training, publication or submission was performed. The earlier experiment inventory was carried forward from the prior planning record and verified against the current files; older notes saying no models had run are superseded by the inspected artifacts.
