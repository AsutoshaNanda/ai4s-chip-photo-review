# How to test fairly and make the final choice

This is a proposed test plan, not a record of completed model tests. Fix the choices specific to each task before running comparisons.

**Update for experiments running together, 2 October:** Use [the full schedule](10_All_Experiments_Parallel_Schedule.md) for all 30 plans and extra feature/RL comparisons. Each task keeps its own fixed rules for using data. People or agents working at the same time share unchanged versions of the sample lists and settings, and save outputs separately. Record how many early comparisons were tried, set aside final-test data before starting, and independently confirm chosen improvements. Running work together does not relax any of the test requirements below.

## Keep three kinds of confidence separate

| Kind | Meaning | Present status |
|---|---|---|
| Source confidence | We checked what a source or file says | High for live rules and direct sample-detail counts; other access limits are in the library |
| Experiment value | The test can change an important decision | High for sample independence, simple comparisons, failing groups, and repeating the work |
| Confidence in an improvement | Our method actually helps the intended task | Not measured until fair comparison runs and confirmation exist |

Do not give a percentage chance of winning. We have no tested model for predicting judging outcomes. An experiment can be useful even when it rejects the preferred method.

## Fix the rules for using data

1. Define the target answer, which inputs would be available during real use, the physical sample being counted as one unit, and the exact claim.
2. List IDs, labels, duplicates, image-collection groups, data/model-file versions, and usage rights.
3. Set aside the main final-test data before training. Do not use its predictions or errors to choose the model.
4. Use only development groups to choose the model design, training-image changes, crops, early stopping, and cutoff values. Development data is the data allowed for choosing settings.
5. Learn probability adjustments on separate groups, or use a recorded group-aware cross-fitting process that rotates groups used to learn and check the adjustment. Do not measure success on the same data used to learn the adjustment.
6. Record every change learned from data: filling gaps, rescaling measurements, choosing features, reducing the number of features, and any index used to find reference examples. Learn these only from training data, unless clearly reporting a different transductive test plan, where test inputs are used while learning even though their labels are hidden.

Reading public labels to check data divisions is different from choosing settings based on test predictions. Once the division is fixed, do not repeatedly inspect final-test errors to improve the model.

### Option A: real-chip quality

The original image folders do not prove that the chips are independent. Check actual chip/run IDs if available. If valid groups are found, define training/development/test divisions that keep each group together before any model training. Keep the original folder assignments as sample information.

If only filename beginnings are available, use them for a **proxy-group sensitivity test**, meaning a test of how results change when samples are grouped by an unconfirmed clue. Clearly limit the claim. Grouping by an unchecked prefix does not establish performance on new chips or labs. Do not rename prefixes as chip IDs.

Choose one main final-test plan. Do not train another model using the published data division if that would include images reserved for the main grouped final test, then use that model's results to choose the candidate. All comparisons used to develop the model, including optional checks of random-image versus grouped divisions, must stay within the main development data. A paper's score can be kept as a background reference that is not directly comparable.

Keeping groups independent and ensuring the categories are represented matters more than exact split sizes. Start with roughly 70/15/15 only if group counts allow it; otherwise record a justified design that keeps groups separate. Never break groups to hit exact percentages. Check that training includes both target labels, and report validation groups that lack the examples needed for a score.

### Option B: nucleus analysis

Use BBBC039's published training/test groups. Keep all pieces cut from an original image and all changed training versions of it together. When repeatedly sampling data to estimate uncertainty, keep together the largest known shared-source groups. If only image identity is known, state that limit.

An already-trained model that may have seen the test images is not a clean independent comparison. Confirm the images were excluded, use a fresh suitable test set, or report the result separately as an early investigation. BBBC038 must pass both dataset-overlap and earlier-model-training checks before being called independent.

## Measurements for option A

In scoring code, map bad to positive label 1, even though the spreadsheet records bad as decision 2. Check this mapping against the archive folders.

| Measurement | Meaning and use |
|---|---|
| Balanced accuracy | Average of the share of bad examples correctly found and the share of good examples correctly found; report both shares too |
| Bad-class average precision | Summarizes how well the ranking finds bad samples; also report the share of samples that are bad |
| False-pass rate | Bad samples predicted as good divided by all bad samples, using a cutoff chosen on development data |
| Brier score | Average squared difference between the predicted chance of bad quality and the actual 0/1 label |
| Log loss | A penalty for poor probability predictions, especially confidently wrong ones; lower is better |
| Queue catch rate | Bad samples in the review list divided by all bad samples |
| Residual bad rate | Bad samples outside the review list divided by all samples outside that list |
| Running time | Measured seconds/image for data preparation and prediction, on a named computer |

Always show the counts of correct and wrong predictions in each category, plus sample/group counts. If a group has no positive or no negative examples, mark the affected measurement unavailable and explain why. Never hide a required missing or failing group inside an average.

### Comparing the same amount of review work

Use review of 20% of images as the proposed main workload, with 10% and 30% as extra comparisons. These are design choices, not organizer rules or scientifically established best values.

Rank the same test images with each method, select the same number for review, and use a fixed rule to break ties. This ranks a batch using scores for incoming images whose true labels are not available to the ranking rule. If the product instead needs a fixed probability cutoff, choose it on calibration/development data, keep it fixed, and report the actual share of test images it selects.

Compare random review, an image-quality rule, the predicted chance of bad quality, and the proposed improved score. Report how random-review results vary. An uncertainty-based score must beat the simple bad-probability review order to justify the extra complexity.

Also report category-prediction errors on all inputs. Do not advertise a low error rate among unreviewed images after sending almost every image for review. Analysis of a review list does not establish human review speed without a timed study.

## Measurements for option B

- **Count MAE:** the average absolute difference between predicted and reference object counts; ignore the sign of each difference.
- **Signed count bias:** average predicted count minus reference count; shows whether the method tends to count too few or too many.
- **Object F1 at IoU 0.5:** match each predicted object to at most one reference object, allowing only pairs that meet the overlap cutoff. Use a fixed matching procedure that makes the greatest possible number of matches, breaking ties by total overlap. F1 = `2 TP / (2 TP + FP + FN)`. TP means correct matches, FP means false detections, and FN means missed objects.
- **Mean per-image F1 and pooled object F1:** report the average of image-level scores separately from a score calculated using all objects together; they give different weight to images.
- **Joined/split and missed-object errors:** inspect how objects fail, not only the overlap of all pixels marked as objects.
- **Seconds/image and memory:** use the original image size and name the hardware.

IoU is the shared area divided by the total combined area, counting the shared area once. The 0.5 cutoff is our proposed scoring choice, not an AI4S judging measure. Fix it before comparisons. If both prediction and reference contain no objects, define object F1 as 1; if only one is empty, define it as 0. Check these rules in the scoring code.

Do not divide count error by the reference count when that count is zero. For warning tests, a proposed large-error rule is `absolute error >= max(5 nuclei, 10% of reference count)`. Check with the intended user that this is practically meaningful, and fix it before testing. Until then, it is a proposed, untested definition, not a known user requirement.

## Comparisons on matching examples and earlier work

When comparing ways of dividing data, distinguish changes in training overlap from changes in which examples are tested. Where possible, plan a comparison using the same images within development data, show how many images it uses, and report tests on different sample populations separately. The inspected [participant audit](https://www.kaggle.com/code/loxigicck/audit-acquisition-groups-in-organ-on-chip-images) is earlier related work, not a result we repeated independently. It does not establish actual physical-chip identities. Do not expose the reserved final-test data while repeating a published data split.

## Show how uncertain the results are

- Compare the candidate and simple reference method on the same examples, using the same data division and scoring code. This is a paired comparison.
- Use bootstrap sampling: repeatedly resample whole confirmed independent groups, keeping every group's members together. A proposed setting is 2,000 resamples with a recorded random seed and a 95% uncertainty interval for the difference on matching examples.
- If independence is supported only for images or unconfirmed filename-prefix groups, state that the interval shares this limit. Many nuclei do not mean equally many independent experiments.
- Report each training seed and the difference it gives on matching examples. Repeating random seeds on one test set does not create independent biological experiments.
- Small groups give uncertain estimates. Show sample counts and uncertainty intervals; do not turn a wide interval into high confidence.
- Choose one main comparison in advance for each decision stage. Trying many comparisons and then reporting a narrow interval for a chosen one is not confirmation. Confirm the selected candidate on fresh held-out groups or the fixed final test.

## Rules for accepting an improvement or stopping

The practical margins below are **suggested decision cutoffs from the earlier plan that have not been validated**. They are not expected improvements, official criteria, or known user requirements. Establish a justified rule for the specific task before using a margin to accept a method. If no practical margin is established, report the measured difference and uncertainty without claiming the benefit is large enough in practice. Never choose or move a cutoff after seeing final-test results.

| Decision | Proposed minimum practical benefit | Other required conditions |
|---|---|---|
| Better chip category model | At least 2 percentage points higher balanced accuracy | Evidence on matching examples supports improvement; no more than 2 points loss in the share of bad examples found for any required cell type with enough data |
| Better review order | At least 5 percentage points more bad cases caught at 20% review | Supported improvement on matching examples; no severely failing subgroup; same inputs and workload |
| Better nucleus method | At least 10% relative reduction in count MAE | Supported improvement on matching examples; object F1 and required groups do not get meaningfully worse |
| Useful simpler method | Similar task performance within a tolerance chosen in advance, with meaningfully lower measured effort/running time | No hidden loss in a required group; no claim of higher accuracy |
| Better probability adjustment | Better probability-error scores on held-out data and a useful check of confidence accuracy | No claim that rescaling alone improved two-category ranking; performance on changed data reported |

Do not apply a relative-improvement rule when the reference method already has zero error. If the sample is too small to show a clear improvement, call the result inconclusive. The size of a practical benefit and the uncertainty around it answer different questions; report both.

For every selected method, show results by cell type, image-collection group where known, unchanged/altered data conditions, and with/without the review-warning rule. Include the worst examples. A required group at or near zero means reconsidering selection, whatever the overall average. Test a backup or warning rule on held-out failures; simply proposing one is not enough.

If two fair, controlled attempts do not solve an option's main limitation, stop making surface changes. Recheck whether the data and target are valid, or try a method that works in a different way. Keep the simple reference method and all negative results.

## Final decision record

Complete this before submitting. All fields stay pending until experiments run.

| Item | Required evidence | Current status |
|---|---|---|
| Chosen task and category | One real user need, exact outcome to measure, and honest limits | Conditional proposal only |
| Chosen method | Saved predictions from the candidate and strongest simple reference method | Unmeasured |
| Decisive improvement | Result on matching examples, useful size of benefit, and independent confirmation | Unmeasured |
| Handling the worst cases | Failures for each group/condition, with a tested warning rule or a smaller claimed use | Unmeasured |
| New contribution | A comparison showing what our new work adds | Proposed |
| Rights and overlap | Exact inputs/model files and recorded resolutions | OOC license disagreement and model checks remain open |
| Repeating the work | Record of a clean run using accessible inputs | Not run |
| Conflicts with evidence | Explanation of any stronger evidence against the choice and how it was resolved | No model choice made |

If the chosen action conflicts with the strongest evidence, do not submit that claim until stronger evidence resolves the conflict. A deadline is a reason to reduce the work, not leave out results that disagree.
