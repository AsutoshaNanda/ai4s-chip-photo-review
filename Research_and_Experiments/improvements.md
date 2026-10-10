# Improvements from all experiment results

Prepared: **6 October 2026, IST**.

**Final recommendation:** keep X01 as the leading chip-photo candidate, repair the testing gaps, test light retraining of the big image model, and strengthen handling of small photos. Keep B02 and B03 as separate nucleus-analysis candidates. Finish confirmation, practical-use checks and reproduction before submission.

This is an improvement plan based on the reviewed results and both screenshots. **Creating this file did not run any new experiments.** Proposed changes below are untested unless explicitly described as already measured.

## 1. Current position

- All **36 experiment IDs** are accounted for below.
- **31 IDs have measured parts**, including setup checks, unsuccessful methods and simulator results. This does not mean every original step is finished.
- **C04 is blocked** because the recorded search found no suitable outside chip-quality dataset.
- **F01-F04 remain unfinished** in the reviewed records.
- The improvement sizes and chances of winning are unknown. Choose the final method from valid measurements, including its weakest groups.

Priority meanings:

- **P0:** correctness, serious failures, confirmation and delivery. These receive first use of available resources.
- **P1:** focused improvements that could change the chosen method.
- **P2:** secondary research or extensions after the required work has enough resources.
- **Blocked:** a named prerequisite is missing. This is not a failed model result.

Lower priority does not remove an experiment from the record. Preserve useful alternatives and all negative results.

## 2. Start with these actions

| Order | Action | Why it matters | Evidence needed to finish |
|---|---|---|---|
| 1 | Repair A03's training/test separation and correct the misleading result labels. | Shared groups and mixed settings weaken the current conclusions. | Group-overlap checks, corrected comparisons and saved predictions. |
| 2 | Test light retraining of the existing big image model against its frozen version. | This was already part of A02 and remains untested. | Matching development groups, measured resources, per-format results and review curves. |
| 3 | Improve X01's treatment of small photos and test camera-format dependence. | The overall improvement hides a serious small-photo weakness. | Each format reported separately, a tested handling rule and explicit unsupported inputs. |
| 4 | Confirm B02/B03, address noise and tiny nuclei, and check B03's search boundary. | The clean-image winner may fail badly on different images. | Exact settings, new training seeds, refreshed B04-B07 and matched comparisons. |
| 5 | Repair C02/C03 before making stronger claims about those routes. | Their current selection procedures limit what the results establish. | Matched training-batch comparisons for C02; input-only depth selection for C03. |
| 6 | Freeze the selected methods, run the reserved final tests once, then complete F02-F04. | A useful, reproducible and clearly demonstrated result is essential to the entry. | Final decision record, workload results, clean rebuild, public-access checks and submission package. |

C02/C03 repairs are required for stronger claims about those routes. They need not delay the main chip-photo route if they remain clearly labelled supporting investigations.

## 3. Fix these testing and reporting problems

### A03: keep evaluation groups out of training

**Verified observation:** A03 leaves out the cell type, but keeps other cell types from the same filename groups in training. This affects **1,813 of 2,397 evaluated photos**. Filename groups are only unconfirmed clues about image collection, not verified physical chip IDs.

**Decision:** reopen the claim that performance does not fall on unseen cell types under independent-group evaluation.

**Next test:** exclude both the held-out cell type and every filename group used to evaluate it. Compare the whole-image reference and retained X01 candidate on matching held-out photos. Report every type, its class counts and the training data left after exclusions. Mark unsupported comparisons as unavailable.

### C03: choose depth without looking at the answer

**Verified observation:** `c03_prep.py` chooses the central depth using the real fluorescent target. It then keeps five planes around that choice. Therefore, both the central-plane and all-planes results depend on target-assisted selection.

**Decision:** retain the recorded comparison with this limitation. It is not proof that an input-only, stain-free workflow works. The reference nucleus masks are automatically generated from fluorescence, not independent manual counting labels.

**Next test:** choose depth from the plain input image alone, or use a fixed acquisition rule established without held-out target images. Rebuild the data and rerun both routes with the same selection and cleanup-selection procedure. Check direct-route cleanup range boundaries afterwards. If this cannot be completed, keep the result explicitly provisional and restrict the claim.

### C02: compare settings on the same training batches

**Verified observation:** the inner search chooses four training batches again for each candidate strength. Different settings can therefore be evaluated on different batch subsets.

**Decision:** treat the selected-strength result as provisional. This shortcut alone is not evidence that final-test labels leaked into training.

**Next test:** evaluate every strength on the same complete set of eligible training batches. Save the comparisons and selected setting, then rerun the held-out evaluation. Report biological repeat matching and batch information separately. Better biology retrieval does not establish successful batch removal.

### B03: separate counting settings from outline settings

**Verified results:**

| Exact configuration | Mean count error per image, lower is better | Outline F1 at overlap 0.75, higher is better |
|---|---:|---:|
| B01, original counting setting | 6.73 | 0.787 |
| B02, original counting setting | 4.45 | 0.847 |
| B02 v2, counting setting | 4.51 | 0.845 |
| B03 v2, counting setting | 4.68 | 0.872 |
| B03 v2, outline setting | 9.86 | 0.891 |

F1 measures how well detected objects match the reference objects. The overlap level controls how closely their shapes must agree.

**Decision:** do not describe count error 4.68 and outline F1 0.891 as one configuration. For an outline-selected comparison, B02 v2 reaches F1 0.867. Compare matching objectives and report both counting and outline quality for each setting.

The saved-error checks support B02 and B03 improving on B01. They do not establish a clear counting winner between B02 and B03. Repeating cleanup on two computers does not replace independent training repeats.

### Other corrections to carry into the report

- **B05:** original B02 counting settings found **45.2%** of the smallest tenth of nuclei at overlap **0.50**. **32.2%** belongs to overlap **0.75**. The weakness remains, but the original summary used the wrong overlap label.
- **B04-B07:** the previous stress, small-object, warning and outside-data checks used the original settings. Refresh them for any retained `repost_v2` configuration.
- **B07:** do not describe outside error 9.6 versus internal error 4.68 as a direct transfer ratio. Those values use different settings and versions.
- **C04:** write "no suitable dataset found in the recorded search", not "no outside data exists".
- **A05:** a small positive development gain is not automatically worthless. Confirmation and practical cost should decide whether to use it.
- **A12:** the tested heat maps failed their checks. Remove their explanatory claims, without claiming that every explanation method fails.

## 4. Main chip-photo improvement tests

### Test light retraining of the big model

**Hypothesis:** allowing the existing large image model to learn from the chip photos may improve useful image features.

1. Keep the current frozen model and X01 outputs as references. Frozen means the image model itself is not trained on these photos.
2. After repairing the evaluation design, compare frozen versus light retraining on matching development groups.
3. Choose learning speed, stopping and other training settings using training/development data only. Measure running time and memory before scaling up.
4. Report overall ranking, each cell type, each format, bad-photo catch curves and the weakest groups.
5. Test retraining and patch changes separately before combining them. Keep only independently confirmed useful changes.

This is unfinished A02 work, not a guaranteed improvement. X01's result supports testing local detail; it does not prove that retraining or a larger network will help.

### Make patches fit the photo's resolution

**Verified observation:** whole-image plus 5x5 patches reaches AUC **0.877**, versus **0.843** for the whole image. AUC measures how well bad photos are ranked above good ones. On the **87 small 640x480 photos**, the scores are **0.527 versus 0.741**. Only six photos in this subgroup are bad.

**Hypothesis:** forcing every photo into the same grid leaves too little useful detail in patches from small photos.

**Proposed comparison:** fixed grids versus patches sized according to available image detail, with a whole-image candidate for small photos. Select any routing rule within development data and confirm it before relying on it. If support is too weak, require manual review for that input format.

The current evidence does not clearly distinguish 3x3 from 5x5 overall. Choose using subgroup failures, review workload and measured running time as well as the average score.

**Important limit:** the locked 675-photo final test contains no small-format photos. Keep its membership unchanged. Passing it cannot establish that the small-photo problem is solved.

### Test camera-format dependence directly

The current pipeline already converts photos to grey. That does not remove all resolution and camera cues.

- Compare results within each format while keeping related groups apart.
- Test format-balanced sampling within training groups if it is supported by the available class counts.
- Preserve actual biological and image-quality information. Do not erase detail merely to make photos look uniform.
- Report the full-data result and each format separately. A lower within-format score alone does not prove a shortcut; case difficulty may differ.

### Keep A05 available for confirmation

The reviewed development gain is approximately **+0.005 AUC**, with a paired exploratory interval above zero. Many variants were tried, so this is not final confirmation.

Test the promising training changes after the core validity checks. Training-only transformations do not require extra transformations during prediction. Judge their value by confirmed benefit, weakest-group behavior and training cost.

## 5. Nucleus-analysis improvement tests

1. **Confirm B02 and B03:** use newly chosen training seeds, unchanged development splits and matching reference methods. Report each repeat, not only the best run.
2. **Check B03's boundary:** its counting threshold reached 0.95 in two folds. Refine the search toward the valid probability limit using training-side validation only. A boundary result does not prove a better setting exists. Track misses and merged objects as well as counting error.
3. **Address noise:** first test the exact retained settings under the existing stress curve. Then compare unchanged training against a training change based on plausible image noise. Report clean performance and every noise level separately.
4. **Diagnose tiny-object misses:** inspect whether small nuclei disappear during preprocessing, model prediction or cleanup. Change the responsible stage and test that explanation. Do not assume a larger crop or more training will fix it.
5. **Refresh warnings and outside tests:** rerun B04-B07 whenever weights or cleanup settings change. Model disagreement is not an established guard against failures shared by both models.
6. **Complete the modern-model comparison:** preserve the planned eligible ready-made model comparison. Check the exact model's licence and training-data exposure first. A model that has seen evaluation images cannot be treated as a clean independent comparison on those images.

Preserve B01 as a simple, independently structured alternative. On the recorded outside grey images, the original B02 counting method has error **48.84**, versus **9.56** for B03's original outline setting and **11.16** for B01's outline setting. The preferred method therefore changes with the data and objective. These outside scores do not validate the later v2 settings.

Colour-image detection falls near zero in the recorded outside test. Limit supported inputs or require manual handling until a separate method and its handling rule have been validated.

## 6. Complete experiment-by-experiment action list

Status here describes the reviewed evidence. An action in the last column remains pending unless explicitly stated as already done.

| ID | Priority | Current position | Improvement or next decision |
|---|---|---|---|
| G01 | P0 | Data checks exist; rights and prior model exposure require source-specific care. | Preserve duplicate checks, source records and exact model-file versions. Resolve intended-use issues before release. |
| G02 | P0 | Scoring tests passed and final memberships are locked; A03 needs repair. | Keep final memberships unchanged, check group exclusion, and state physical-sample and plate-independence limits. |
| G03 | P0 | Speed measurements exist for several machines. | Measure each newly changed pipeline before estimating cost. Reconcile actual billing before more paid work. |
| A01 | P0 reference | Simple methods provide a baseline; format dependence was observed. | Keep the reference and within-format results for all main comparisons. |
| A02 | P1, high | Large frozen features beat the small model; light retraining is unfinished. | Run the controlled frozen-versus-light-retraining comparison after fixing the evaluation design. |
| A03 | P0 | Cell types were excluded but shared filename groups remained. | Exclude both the evaluated type and its groups; repeat for retained candidates. |
| A04 | P0 diagnostic | Centre and edges retain substantial signal. | Test acquisition/format dependence without assuming that edge performance proves a shortcut. |
| A05 | P1 | Small positive development signal. | Confirm useful training changes and test combinations only after their separate effects are understood. |
| A06 | P2 | Small unconfirmed weighting gain. | Reconsider only if class-specific failures justify it; use a paired comparison and report every class/type. |
| A07 | P2 | Added sample details did not show a clear gain. | Keep the image-only reference; reopen only with a concrete reason and available-at-use metadata. |
| A08 | P0 for final claims | The tested probability adjustments worsened A02. | Check reliability for the final candidate. Do not advertise raw ranking scores as validated probabilities. |
| A09 | P0 for usefulness | The proposed ordering lost to A02 alone. | Recalculate full review curves for the final X01/retrained candidate against simple ordering and random review. |
| A10 | P1 only with complementary evidence | The tested model average lost. | Test a guard or combination only if saved failures show useful complementarity; validate the complete rule. |
| A11 | P2 | Noisy label-efficiency gain; classifier strength and starting groups were fixed. | Retain as research. If revisited, test these choices inside training data while matching label and compute budgets. |
| A12 | P0 reporting; P2 new method | The tested heat maps failed validation. | Remove them as explanations. A different explanation method needs fresh checks; show real examples and failures meanwhile. |
| X01 | P1, main candidate | Clear overall development gain; serious small-photo weakness. | Compare size-aware patches, whole-image handling and 3x3/5x5 candidates, then confirm the complete chosen rule. |
| B01 | P0 reference | Useful simple baseline, including outside-data strengths. | Preserve it for clean, stress and outside-data comparisons. |
| B02 | P1 | Good internal counts; outside-data weakness; modern-model comparison unfinished. | Repeat training, diagnose noise/tiny-object failures, and compare an eligible modern method. |
| B03 | P1 | Strong outlines; count and outline settings differ. | Extend the boundary search, retain explicit operating settings and confirm with new training seeds. |
| B04 | P0 if B is retained | Strong noise causes severe failure. | Refresh stress curves for final settings and test the planned justified training change. |
| B05 | P0 if B is retained | Tiny nuclei are missed; the summary mislabelled the overlap level. | Correct 45.2% at 0.50 versus 32.2% at 0.75; diagnose and retest the exact retained settings. |
| B06 | P0 for warning claims | Disagreement has limited predictive usefulness. | Refresh warnings and full correction-workload curves; test shared failures and manual-review behavior. |
| B07 | P0 for transfer claims | Performance changes sharply outside BBBC039; colour images fail. | Test unchanged retained settings, preserve alternatives and limit unsupported input types. |
| C01 | P2 | Positive batch-transfer result with possible earlier model exposure and unresolved intended-use compatibility. | Keep the caveats. Resolve exposure and rights before making it essential to the submission. |
| C02 | P2; correction required for stronger claims | Mixed biology/batch outcome; uneven inner batch subsets. | Compare strengths on all the same training batches, then rerun. Test a different correction method only afterwards. |
| C03 | P2; correction required for stronger claims | Direct route wins on target-selected slices. | Rebuild input-only selection, rerun both routes and check cleanup boundaries. Keep reference-mask limitations explicit. |
| C04 | Blocked | No suitable independent chip dataset found in the recorded search. | Preserve the search record. Matching labels and genuinely independent data are required to unblock it; outreach requires instruction. |
| R01 | P0 for retained simulator claims | A practice simulator based on published fitted equations exists. | Keep checks and assumptions documented; do not equate the simulator with a validated real-chip controller. |
| R02 | P2; positive separate route | Learned control beat the recorded rule-based controls across five training seeds. | Confirm a frozen protocol on fresh conditions where needed, including worst errors and violations. |
| R03 | P2; tested version not promoted | Staged training was worse. | Preserve the negative result. Retry only with a new, testable diagnosis and matching resources. |
| R04 | P2; positive separate route | Extra inputs helped in simulation. | Keep missing/noisy-input results; ensure any retained control claim includes those limits. |
| R05 | P2; tested version not promoted | Label-picking RL lost; ten times the training did not fix it. | Do not prioritize more rounds alone. Reopen only with a different explanation and fair label/compute controls. |
| F01 | P0 | Unfinished. | Confirm selected improvements, freeze all choices and perform one final evaluation on the reserved data. |
| F02 | P0 | Unfinished for final candidates. | Measure review/correction workload. Actual time-saving claims require a timed user study. |
| F03 | P0 | Unfinished. | Rebuild the frozen pipeline in a clean, accessible environment and reproduce its tables and example outputs. |
| F04 | P0 | Unfinished. | Trace claims to saved evidence, verify rights and public access, finish the decision record and submission package. |

## 7. How to search settings fairly

Fixed learning rates, crop sizes, batch sizes, stress strengths and stopping patience are baseline choices. They are not proven optimal, and they do not automatically invalidate a comparison.

- Search the settings most likely to change a decision. Do not spend the remaining resources on every possible combination.
- Use the same eligible training/development examples for candidate settings and comparable training resources.
- Choose ranges from data, valid parameter limits, learning curves and measured resource constraints. Explain any remaining limits.
- Extend a range when a boundary result warrants it. Do not assume extension must improve performance.
- Distinguish synthetic stress settings from measured real-world image conditions.
- For A04/A05/A12, additional crop, brightness or grid values are useful only if they resolve an open question. More values alone do not repair the failed heat-map method.
- For C02, compare all strengths on the same batches. For C03, fix input selection before tuning cleanup further.
- Use paired checks for decisive comparisons. Paired means the methods are compared on the same examples.
- Repeatedly finding the best setting on the same development data can produce a lucky result. Keep independent confirmation and the locked final test protected.
- Do not invent a minimum improvement or pass mark. Record the full curves, uncertainty, failure patterns and practical trade-offs.

## 8. Confirmation and submission

### Before the final test

- [ ] A03's affected transfer claims have been corrected and retested or explicitly restricted.
- [ ] Retained C02/C03 claims match the repaired protocol or clearly state the unresolved limitations.
- [ ] Every selected model has exact weights, preprocessing and settings recorded.
- [ ] Small-photo and other unsupported-input handling is fixed and tested, or the supported scope is narrowed.
- [ ] B04-B07 match the exact retained B02/B03 settings.
- [ ] Decisive comparisons include matching examples, group-level uncertainty and weakest-group results.
- [ ] New training repeats use recorded seeds. Frozen-feature repeats are not presented as independent training evidence merely because a seed changed.
- [ ] All design choices, displayed confidence, review ordering and operating settings are frozen.
- [ ] A final decision ledger explains why the chosen action follows the strongest positive and negative evidence.

The chip final set has **675 photos** and the nucleus final set has **50 images**. The chip set does not cover the small-photo format. Nucleus final images share plates with development images. These are limits on the claims, not reasons to change the locked membership after seeing model results.

If the final result is disappointing, report it. Tuning on that final set would require a new independent test for a fresh final claim.

### Make the useful workflow the main contribution

Proposed main story: **a chip-photo review tool that finds bad photos efficiently, reports its supported input types and handles unsupported inputs explicitly.** Its value must be demonstrated, not assumed.

Show how many bad images are found across the full range of review effort. Include actual inputs, outputs, difficult cases and a reproducible starting command. Use a timed study only if claiming actual time savings. Keep nucleus analysis and simulator control separate unless their integration has itself been tested.

The official judging weights, checked during the 6 October review, are impact **30%**, technical approach and innovation **30%**, results and validation **20%**, reproduction **10%**, and presentation **10%**. These support reserving effort for usefulness, confirmation and delivery alongside model improvements. [Official criteria](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview#evaluation).

### Proposed timing

These are targets, not measured duration promises. Actual running time, available compute and remaining budget decide what fits.

| Target | Work |
|---|---|
| 6-7 October | Correct the testing/reporting issues and prepare focused main-route improvements. Start report/demo preparation. |
| 7-8 October | Run justified improvement comparisons, confirm promising results and decide the supported scope. |
| 8 October, only when ready | Freeze choices and run the reserved final evaluation once. |
| 9 October | Complete clean reproduction, workload reporting, demo, code, report and final checks. Aim to submit when ready. |
| 10 October | Buffer for required fixes before the verified cutoff. |

The deadline last verified in the live browser on **6 October** was **10 October 2026 at 21:29:59 IST**. Recheck before the final submission action. [Official overview](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview).

The screenshot's claim that an incorrect Modal price caused overspending has not been independently reconciled with billing in this review. Before any new paid work, check actual usage and every charged resource, and establish a budget control that accounts for billing delay. This plan does not authorize new spending, outreach, publication or submission.

## 9. Record evidence as decisions

For every material result, record this row before choosing the next action:

| Observation | Evidence quality and saved source | Hypothesis supported or rejected | Decision changed | Smallest next test or action |
|---|---|---|---|---|
| Fill with the actual measured observation. | Distinguish a new run, a check of saved predictions, outside-source evidence and an untested proposal. | State the exact explanation tested and any unresolved alternative. | Keep, repair, restrict, stop promoting, or confirm the route. | Name the next check that can change the decision. |

Retain independently useful methods with different failure patterns. A better average does not justify ignoring a near-zero subgroup result. Keep chip-quality, counting, biological profiling, translation and control metrics separate.

## 10. Sources and existing evidence

- [Detailed review and screenshot follow-up](14_Results_Review_and_Next_Steps_6_Oct.md): the basis for this plan, including verified corrections and recommendations.
- [Saved audit and decision ledger](Evidence/2026-10-06_results_review_audit.json): recalculated comparisons, group-overlap counts, source hashes and screenshot findings.
- [Original results summary](13_Results_vs_Expected_6_Oct.md): original run references. Apply the corrections above when quoting it.
- [Approved experiment plan](11_Simple_Start_Plan_5_Oct.md): original scope, including A02 light retraining and the B02 modern-model comparison.
- [Experiment schedule and history](10_All_Experiments_Parallel_Schedule.md): full experiment inventory. Historical planning sections are not evidence of completed runs.

The review's uncertainty ranges reuse development predictions after multiple methods were explored. They support choosing what to test next; they do not establish performance on new laboratories or predict winning odds.
