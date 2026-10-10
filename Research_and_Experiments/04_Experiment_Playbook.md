# Guide to running experiments and checking results

Status as of **2 October 2026: 30 plans; no model runs recorded in the checked project folder**. G01 includes completed checks of sample details, but its full data/usage-rights check is not complete. Use [the run log](08_Run_Log.md) for measurements.

**High priority does not mean a positive result is guaranteed.** Confidence below refers to how useful the test is to run. No proposed performance improvement has been measured. A negative result can prevent a false claim.

Sources S01 to S31 are in the earlier [research library](02_Research_Library.md). Dated new sources are listed there and in [resource readiness](../Resources/19_Resource_Readiness.md). Measurements, tests that must stay unchanged, and shared rules for accepting an improvement are explained in [the test plan](05_Validation_and_Selection.md).

The limits below are proposed budgets for first small trials. They are not measured run times or a limit on the number of experiments in the project. G = shared checks; A = chip quality; B = nucleus analysis; C = extra work if requirements pass; F = confirmation and submission. **All 30 plans are scheduled in [the plan for work that can run together](10_All_Experiments_Parallel_Schedule.md).** Start independent options and comparisons together after their own checks pass. Use the resulting evidence to choose what the final project adds. Extra tests of image features and reinforcement learning (RL, learning from rewards) are defined in that schedule.

**Start-date update:** Use the current 3 October schedule for target dates. The numerical limits below say when to stop a trial; they do not establish available resources or predict run times. Set actual resource budgets from G03 measurements before starting.

## Shared checks before model work

### G01: Verify usable data and rights

- **Priority and confidence:** Required; very useful for deciding what to do. Sources S02 to S04, S08, S15, S17.
- **Question/comparison:** Do the downloaded files match provider records, and can the planned inputs and outputs be used and shared under the known terms?
- **Steps:** Record versions and file-check values, match image IDs and labels, open a sample that represents the training data only, check exact model/data terms, and record missing fields. Resolve the OOC license disagreement. Matching the sample details to the archive file list is already recorded; actual image condition and usage rights are still unresolved.
- **Record:** Files that fail, label meanings, exceptions, planned sharing, and exact permissions still unresolved.
- **Decision/limit:** Stop recommending the model if its target is unsupported or final sharing cannot be justified. Review unmet requirements after one focused workday; other ready options continue alongside it. Record exactly what this option needs before it can continue.

### G02: Fix sample groups, training/test lists, and scoring

- **Priority and confidence:** Required; very useful for deciding what to do. Sources S28, S30 and the direct OOC data check.
- **Question/comparison:** Are test examples kept separate from model training and the choice of method?
- **Steps:** Verify chip/run groups where possible; keep related images and pieces cut from them together. If grouping by filename beginnings, call this an unconfirmed clue, or proxy. Set aside one main final-test group before training and do not change it. Check identical and nearly identical images. Test the scoring code with perfect predictions, all-wrong/all-empty outputs, wrongly joined/split objects, and one-category groups as appropriate.
- **Record:** Sample IDs in each data group, number of groups, category counts in each repeated-test division, duplicates left out, and cases where scoring needs special handling.
- **Decision/limit:** Do not claim performance on new samples until there is sound evidence for which samples are independent. Missing chip IDs mean a smaller claim, not invented IDs. Limit the first implementation to half a workday.

### G03: Measure the smallest system that runs

- **Priority and confidence:** Required; very useful for deciding what to do. Sources S01, S17, S27.
- **Question/comparison:** Can the chosen method run and reload using the available resources?
- **Steps:** Use 16 training examples, one short training pass where needed, then save/reload predictions. Measure data-preparation time and prediction time separately on the actual computer.
- **Record:** Software/model-file versions, memory, seconds/image, loading failures, output format, and available computing resources.
- **Decision/limit:** Allow one hour of setup before choosing a simpler supported option. Do not claim a full-run cost before measuring this.

## Option A: review real-chip image quality

### A01: Establish simple methods and possible shortcuts for comparison

- **Priority and confidence:** First model test; very useful for deciding what to do. Sources S04 to S06, S28.
- **Idea to test:** Image information adds value beyond category frequency, cell type, and details of how images were taken.
- **Steps:** On the same data divisions, compare always predicting the most common training category, logistic regression using permitted sample details, and logistic regression using simple image measurements. Logistic regression is a simple model that predicts a category probability. Image measurements can include focus, brightness, contrast, and texture. Leave out IDs, paths, and answer fields; learn all data-preparation settings inside each training division.
- **Measure:** Accuracy averaged across good/bad categories, average precision for ranking bad images, tables of correct and wrong predictions, the share of each cell type correctly found, and running time.
- **Keep/reject/limit:** Keep all results as named comparisons. Strong results from sample details trigger a shortcut check, not automatic rejection of the image model. One CPU work session.

### A02: Test one small already-trained image category model

- **Priority and confidence:** Main test; moderate confidence in an improvement. Sources S04, S27.
- **Idea to test:** A model whose image-feature extractor stays fixed, with a simple linear category predictor on top, beats A01. Limited further training may help. An extractor turns images into useful numerical features.
- **Steps:** Choose one small model design, record its model-file source and image-preparation steps, and compare fixed image features with further training of only the final block. Use equal data and limited setting searches. Use the error calculation and label mappings checked in G02. Never give the model paths that reveal labels.
- **Measure:** A01 measurements, log loss (a penalty for poor probability predictions), memory, and prediction time on the same groups kept out of training.
- **Keep/reject/limit:** Use the shared rule for accepting improvements. If it loses, keep the stronger simple method and investigate one specific failure. Limit early trials to two accelerator-hours, meaning hours on a GPU or similar processor, after G03. This is a spending limit, not a speed prediction.

### A03: Test on cell types absent from training

- **Priority and confidence:** Main test if claiming performance on new cell types; very useful, outcome uncertain. Sources S05, S21.
- **Idea to test:** Performance remains useful on a cell type that was not used for training.
- **Steps:** Within data used to develop the model, leave out one cell type at a time when labels/groups allow it. Never train on the main final-test data. Compare with validation that holds out sample groups but uses cell types seen in training. Keep fitting of probability adjustments separate too.
- **Measure:** For each held-out type, record category counts, the share of bad examples found, the share wrongly passed as good, how accurate confidence estimates are, and the worst type.
- **Keep/reject/limit:** A cell type with collapsed performance rules out broad claims about new cell types. Limit the claimed use or test a warning/review rule that catches the problem. Try two representative types first; test all suitable types before making a claim covering all six.

### A04: Test image cropping and use of surrounding areas

- **Priority and confidence:** Main test; very useful for finding why something happens. Sources S04, S06; S25 provides a related example.
- **Idea to test:** The model uses information that stays useful under the intended changes in image-taking conditions.
- **Steps:** Compare the full image, its center crop, and a view showing only surroundings chosen before testing. Use the same source images/groups and similar training budgets. Test small crop-position changes. Check whether removed surroundings contain real information needed for the quality label.
- **Measure:** Differences between methods on matching examples, errors for each group, and examples whose cropped view no longer supports the original label.
- **Keep/reject/limit:** Channel-wall defects and image-taking problems may be valid reasons for a bad label. Depending on surroundings alone does not prove a shortcut. Keep a view only if testing on separate groups and the task meaning support it. One small comparison of three views.

### A05: Test image changes that keep the same valid label

- **Priority and confidence:** Depends on earlier checks; moderately useful for decisions, improvement uncertain. Sources S04, S21.
- **Idea to test:** A small set of justified image changes improves performance on new conditions without changing the correct quality labels.
- **Steps:** Compare the fixed A02 setup with and without flips/rotations or small brightness changes, after checking on training samples that these changes make sense for the biology and imaging. Change one type of training-image alteration at a time.
- **Measure:** Results on separate validation groups, the worst cell type, and performance on unchanged images.
- **Keep/reject/limit:** Blur or image defects can define the bad category, so adding blur while keeping a good label may be wrong. Use these changes only with a sound rule for the label. At most two versions.

### A06: Test balancing the training categories

- **Priority and confidence:** Depends on a failing subgroup; moderately useful. Direct data check plus S28.
- **Idea to test:** Giving some categories more weight or changing how training examples are sampled improves the weak category without hiding another failure.
- **Steps:** Compare one weighting rule with the unchanged training-error calculation. Do not change weights, repeat extra minority examples, and introduce a new model design in the same test. Use the same group divisions and random seeds, the numbers that control repeatable random choices.
- **Measure:** The share correctly found in both categories for every cell type, accuracy averaged across categories, and accuracy of probability estimates.
- **Keep/reject/limit:** Reject an improvement that harms a required group or produces confidently wrong answers. HUVEC has few good examples, so report uncertainty. One version, with confirmation only if promising.

### A07: Test whether sample details help in a fair test

- **Priority and confidence:** Optional; moderately useful for finding problems, improvement uncertain. Sources S28, S30 and direct data check.
- **Idea to test:** Available cell-growing details improve image predictions across separate sample groups.
- **Steps:** Compare images alone, sample details alone, and a small model that combines them. Learn how to fill missing values and convert inputs only within training divisions. Include markers showing which values are missing. Keep original flow-rate text and record how its units are read. Also test when optional sample details are absent.
- **Measure:** Results for each cell type and sample group, performance loss when details are missing, and accuracy of probability estimates.
- **Keep/reject/limit:** Reject improvements that disappear on separate-group tests or need details unavailable during real use. Do not infer that flow causes the outcome. One combined-model version.

### A08: Check and adjust probability estimates

- **Priority and confidence:** Main test if probabilities are shown; very useful to test, moderate expectation of better probability estimates. Sources S20, S21, S29.
- **Idea to test:** Learning a probability adjustment on separate data improves probability accuracy.
- **Steps:** Compare original probabilities with temperature scaling, a simple confidence adjustment, or one other suitable small adjustment model. Learn it only from separate calibration groups. Test it on different groups; record changes in cell type separately.
- **Measure:** Brier score and log loss, two probability-error measures; predicted-confidence ranges with sample counts; and results when data conditions change.
- **Keep/reject/limit:** Keep only an improvement backed by evidence. Remove claims that a number means probability of being correct if unsupported. For two-category models, temperature scaling does not improve ranking when reviewing a fixed share of images. Try one adjustment before more complex ones.

### A09: Test the actual review queue

- **Priority and confidence:** Main test of what the project adds; very useful for decisions, benefit uncertain. Sources S06, S19, S21.
- **Idea to test:** The proposed image-review order catches more bad images for the same review effort.
- **Steps:** Compare random review, focus/brightness rules, reviewing the highest predicted bad probabilities first, and one proposed score that also uses uncertainty or disagreement between methods. Fix that score using development data. Review 10%, 20%, and 30% of images, with 20% the main comparison unless a user requirement changes it before testing.
- **Measure:** The share of all bad images reviewed, bad images missed, the bad-image rate among unreviewed images, and results for each cell type. Also keep category-prediction results for the full set.
- **Keep/reject/limit:** Keep the rule only if evidence shows it beats the strongest simple ordering and no group has a severe failure. Reviewing everything does not reduce workload. One comparison of review lists using saved predictions at the same workload.

### A10: Test methods that catch different errors and a warning rule

- **Priority and confidence:** Only after two useful methods exist; moderately useful, improvement uncertain. Sources S19, S22.
- **Idea to test:** The methods make different enough errors that combining them or adding a review-warning rule helps.
- **Steps:** Save a table showing both models' errors on the same examples. On separate validation groups, compare the best single method, an equal-weight average, and one fixed review rule. Develop rules without feedback from the final test.
- **Measure:** Bad cases both methods miss, failures by cell type, usefulness of the review order, probability accuracy, and doubled running time.
- **Keep/reject/limit:** Reject methods that fail together or a rule that merely hides their failures. Keep an independently confirmed alternative when it catches different failures. One two-model comparison, not a search through many model combinations.

### A11: Test how many labeled examples are needed

- **Priority and confidence:** Optional; moderately useful. Source S13 as a related example, S28 for dividing data.
- **Idea to test:** A smaller labeled training set that represents the data keeps useful performance.
- **Steps:** Use 25%, 50%, and 100% training subsets, choosing whole groups and keeping each smaller subset inside the larger one. Keep validation fixed and repeat subset selection. If testing active selection, where a method chooses which samples to label, compare it with random selection using the same label budget. Request labels only from the training pool.
- **Measure:** Performance against the actual number of images/groups, variation between runs, and training time.
- **Keep/reject/limit:** Existing labels can imitate having limited labels available, but do not measure a person's labeling time. Do not claim labeling savings without a real effort study. Three subset sizes, one model design.

### A12: Check visual explanations

- **Priority and confidence:** Only if explanations are included; very useful to check, uncertain whether explanations reflect the actual prediction. Sources S01, S06.
- **Idea to test:** The highlighted explanation regions respond meaningfully to the trained model's prediction.
- **Steps:** Compare explanations from the trained model with those from a model whose learned values are randomized, and carefully hide chosen image regions. Use several good, bad, and incorrectly predicted cases. Check whether cropping or hiding a region creates a new image defect.
- **Measure:** Changes in predictions and explanations, and disagreement with visible features relevant to the task.
- **Keep/reject/limit:** Treat colored explanation maps as inspection aids. They do not prove biological cause and effect. Remove misleading explanations. Limit this to one focused review session.

## Option B: nucleus counting and review

### B01: Establish a correct simple counting method

- **Priority and confidence:** Required for B; very useful for decisions. Sources S08, S31.
- **Idea to test:** A clear brightness-cutoff/watershed process gives a valid reference method. Watershed helps separate touching objects.
- **Steps:** Read the colors for separate objects correctly; check training images with outlines drawn over them; keep the provider's training/test groups and keep all pieces of an image together. Choose the brightness cutoff, spacing between object markers, and size filters using validation only. Restore cutout positions and score images at their original size.
- **Measure:** Count MAE (average absolute count error), object F1 (balance of missed and false objects) at IoU 0.5 (required area overlap), whether counts are too high or low on average, crowded-image errors, and seconds/image.
- **Keep/reject/limit:** Keep the result even if it is weak. Fix broken scoring before any AI comparison. One CPU work session.

### B02: Compare a small trained model with a modern method

- **Priority and confidence:** Main test for B; very useful comparison, winner uncertain. Sources S10 to S18.
- **Idea to test:** AI gives useful improvements in object outlines and counts over B01.
- **Steps:** Compare a small U-Net image model trained from the beginning with at most one current method that passes the checks, such as StarDist or a checked general-purpose model. Use matching validation data and record prior exposure to labels/training images. Put results with unknown earlier training overlap in a separate early-test row.
- **Measure:** B01 measurements, objects wrongly joined or split, prediction time, and memory.
- **Keep/reject/limit:** Using an established model alone is not a new contribution. Choose using the shared acceptance rule. Limit the first comparison to two accelerator-hours after timing the small trial.

### B03: Teach object edges or distances to separate touching nuclei

- **Priority and confidence:** Only after wrongly joined objects are seen; moderate confidence that this test will be useful. Sources S10, S11.
- **Idea to test:** Teaching a target that distinguishes separate objects reduces wrongly joined nuclei.
- **Steps:** Keep the small network, data division, and training budget fixed. Compare learning only which pixels belong to objects with learning one boundary or distance target. Choose the object-separation settings on validation data.
- **Measure:** Object F1, numbers of joined/split objects, whether counts tend to be too high or low, and images with few versus many nuclei.
- **Keep/reject/limit:** Reject fewer joined objects if it also causes worse splitting or count errors. Change one training target.

### B04: Test how well the model handles changed images

- **Priority and confidence:** Depends on earlier checks; very useful for finding problems, improved handling of image changes uncertain. Sources S14, S21.
- **Idea to test:** Performance stays useful under plausible changes in how images are taken.
- **Steps:** Test fixed, recorded levels of brightness changes, random image noise, and blur on validation data as artificial difficult conditions. Keep this separate from comparisons that alter training images. Move object masks to match any geometric image changes and keep the unchanged-image reference.
- **Measure:** Unchanged images and every difficulty level separately; worst count error and object F1.
- **Keep/reject/limit:** Artificial image damage is not a real outside dataset. Reject a claim of handling changes if quality on unchanged images collapses. At most three realistic difficulty levels for each chosen type of change.

### B05: Check difficult objects and groups

- **Priority and confidence:** Main test for B; very useful for decisions. Sources S08, S10, S11.
- **Idea to test:** An overall improvement still holds for crowded images and small objects.
- **Steps:** Use training and development reference answers to define groups by object density and size. Keep those boundaries fixed. Review every method's missed, joined, wrongly split, and falsely detected objects using the same matching rule.
- **Measure:** Scores for each group, image counts, whether counts are too high or low on average, and images showing the worst cases.
- **Keep/reject/limit:** Reconsider the choice if a required group scores near zero, even if the overall average is strong. One analysis of saved predictions.

### B06: Test failure warnings or correction effort

- **Priority and confidence:** Main test of what B adds; very useful, benefit uncertain. Sources S13, S18, S19.
- **Idea to test:** Disagreement between methods finds bad counts better than simple image-quality rules.
- **Steps:** At the same share of images reviewed, compare random review, blur/brightness warnings, disagreement in counts, and disagreement in object masks. Define a large error using development data and do not change the definition afterward. If adding corrections made with user input, measure actual corrections and time separately.
- **Measure:** Large errors caught/missed, count error among unreviewed images, error across the whole set, and actual correction effort if collected.
- **Keep/reject/limit:** Methods can agree on most pixels while missing individual objects. Keep only a rule that passes separate confirmation. One session to analyze predictions; budget any human study separately.

### B07: Test on an independent outside image dataset

- **Priority and confidence:** Required for claims about other datasets; very useful for decisions. Sources S08 to S10, S21.
- **Idea to test:** The selected method still works on suitable new microscope-image data.
- **Steps:** Check BBBC038 or another matching dataset for identical/nearly identical images and prior model-training exposure. Fix settings before this test; report outside-data results separately. If overlap is known, the affected result cannot be claimed as an independent test.
- **Measure:** Every method, set of data conditions, subgroup, and worst case; also how well warnings work.
- **Keep/reject/limit:** Limit the claimed use if outside-data failures remain. Do not claim independent testing if overlap cannot be resolved. Try one dataset once the main result is stable.

## Extra options: check early, run once requirements pass

### C01: Test cell measurements on lab batches kept out of training

- **Priority and confidence:** Source check scheduled alongside other work, then a small trial if checks pass; moderate evidence for the method, project value uncertain. Source S23.
- **Question/comparison:** Can small, ready-made image measurements distinguish experimental changes across lab batches?
- **Steps:** Check how the measurements were made and their usage rights. Compare finding the nearest matching example with a simple category model on experiments kept out of training. When choosing among fixed categories, test labels need matching categories in training. Experimental changes absent from training need a separately defined matching task or outcome to predict.
- **Measure:** Accuracy or matching performance across lab batches, results for each cell type, how easy it is to predict the batch, and running time.
- **Decision/limit:** Start only if it serves the selected project and passes data checks. Use a small measurement-table trial; do not download the full image collection by default.

### C02: Adjust JUMP measurements and find biological matches

- **Priority and confidence:** Source check scheduled alongside other work, then a small trial if checks pass; moderately useful. Sources S24, S25, S30.
- **Question/comparison:** Does one way of rescaling measurements reduce lab-process differences without removing biological information?
- **Steps:** Use a small, recorded subset. Compare raw measurements, or suitable measurements with minimal processing, with one rescaling method learned from training data. Keep finding repeated experiments across batches separate from predicting an outcome for new compounds. Separately report any transductive use, meaning the method uses test inputs while learning even though their labels are hidden.
- **Measure:** Matching repeated experiments across batches and a separately justified biological outcome. Report both, not one combined score.
- **Decision/limit:** Making batches mix better is not enough by itself. One limited subset and one rescaling method.

### C03: Predict fluorescence using real matching image pairs

- **Priority and confidence:** Early image-pair check, then a scheduled small trial if it passes; low confidence that it can be done at present. Source S26.
- **Question/comparison:** Does predicted fluorescence preserve a specified measurement that will be made from the output?
- **Steps:** Require real, aligned input/answer image pairs and separate training/test groups at sample level. Check the newly listed [Allen training data](../Resources/19_Resource_Readiness.md) or another verified paired source, including intended-use and sharing rights. Compare a simple image-to-image prediction model with the best method that makes the same final measurement directly from the unchanged input. Check alignment before training. Do not replace measured reference answers with generated images.
- **Measure:** Structure/count or another justified biological outcome, plus image-similarity scores; show failures.
- **Decision/limit:** Do not start without paired data. Generated images that look similar do not prove biological correctness. Limit the data search before committing to training.

### C04: Test a matching outcome on independent real-chip data

- **Priority and confidence:** Early high-priority source check, then independent testing when suitable data is obtained; very useful for checking results, but access is uncertain. Sources S03, S07.
- **Question/comparison:** Does a suitable quality-review method work on chip data from an independent source?
- **Steps:** Check downloadable images, reference labels, physical sample groups, rights, and whether the measured outcome matches our task. Keep our model unchanged. If the outcome changes, such as outlining melanin regions, report it as a separate task with its own simple comparison method.
- **Measure:** Errors for each source, usefulness of the review order, probability accuracy, and all failed groups.
- **Decision/limit:** A paper saying data is available does not mean an outside-data test is complete. Do not delay required submission work for data we cannot access.

## Confirming results and finishing the submission

### F01: Independently confirm the candidate

- **Priority and confidence:** Required; very useful. Sources S21, S28, S30.
- **Question/comparison:** Can the chosen improvement be repeated, and does it hold up beyond random variation that may have affected selection?
- **Steps:** Repeat the candidate and reference method with at least three random seeds chosen in advance when affordable, using the same fixed data divisions. Look for a separate image-collection group/source where available. Fix all choices before the final test. Repeating the same images checks that the code and random training choices give stable results; it does not establish performance on outside data.
- **Measure:** Differences on matching examples, uncertainty ranges, each seed/group, and worst failures.
- **Decision/limit:** If the uncertainty range stays wide, report it and prefer the simpler supported claim. Reserve computer time for this before adding another candidate.

### F02: Measure practical usefulness

- **Priority and confidence:** Required for claims about time/effort; very useful, benefit uncertain. Sources S01, S06, S13, S18.
- **Question/comparison:** Does the tool reduce actual review or correction work while keeping errors acceptable?
- **Steps:** If a suitable researcher is available, choose cases in advance and compare ordinary review with tool-assisted review. Balance which method is used first, or use separate matched sets of cases. Record the person's experience and the limits of this small study. Do not contact anyone without permission.
- **Measure:** Time, corrections, missed errors, and problems using the tool. If nobody takes part, report only the review-workload results calculated from saved predictions.
- **Decision/limit:** Do not turn a simulated share of images reviewed into claimed time or cost savings. One small recorded study, not an unsupported estimate of impact.

### F03: Repeat the work from a clean setup

- **Priority and confidence:** Required; very useful. Source S01.
- **Question/comparison:** Can the documented starting command recreate the saved outputs without depending on unrecorded files or session state?
- **Steps:** Start a fresh software setup/session using the declared public inputs and exact model version. Recreate result tables and example outputs, then compare with the saved record using defined acceptable differences.
- **Measure:** Setup failures, required resources, measured running time, numerical differences, and missing files.
- **Decision/limit:** Fix failures before submitting, or remove the unsupported claim that the work can be repeated. Reserve one full work session.

### F04: Final claim and submission check

- **Priority and confidence:** Required; very useful. Sources S01, S02.
- **Question/comparison:** Does every submitted claim match saved evidence, and can others access the submission?
- **Steps:** Trace each main number to a run and saved prediction file. Check claims about what we added, failure cases, usage rights, links while signed out, video length, and the report/code starting point. Write the final decision record before submitting.
- **Measure:** Missing or conflicting evidence, inaccessible links, unproven claims, and remaining entry requirements.
- **Decision/limit:** Correct the submission before sending it. A draft or completed experiment is not a submitted Writeup. Keep the final day as extra time for fixes.
