# Review before the write-up, 8 October 2026

## Answer

**The main results are supported by the saved outputs, and the report can be drafted now. Fix the two important issues below before publishing the report and code.** The new RL evaluation and the package-version repeats are complete; do not use the older claim that they are still waiting to run.

This is a review and correction report. It does not replace saved experiment results. No model was trained, no cloud job was started, and no submission was made during this audit. Checks on existing predictions are not new independent experiments.

Reviewed file: [22_Final_Results_Before_and_After.md](../22_Final_Results_Before_and_After.md). Its SHA-256 at the start of the review was `d71fb64eaa7056d36e4d2da4edebac375caf60632a47e624b24080b447c910ec`.

[Machine-readable verification record](2026-10-08_pre_writeup_audit.json): source hashes, recomputed values, diagnostic findings and the decision attached to each important observation. The reviewed results file remained unchanged during this audit.

## Important corrections

| Priority | Finding | Evidence and what it means | Required action |
|---|---|---|---|
| 1: report accuracy | **E08's explanation of the baseline change is not supported.** The row says input clipping explains 0.801 becoming 0.810 and calls it explained | The extractor converts unsigned image pixels to 0–1, then normalizes them. With the actual model's means and scales, values lie between about −2.118 and +2.640. Clipping to −10…+10 changes nothing. The old and new saved features do differ, and the selected baseline regularization changed from 0.001 to 0.0031623, while package versions also changed. These observations do not isolate the cause | Replace the claimed cause with “the baseline changed between runs; the cause has not been isolated.” Lead the report with the consistently rebuilt comparison, about **0.850 vs 0.810**, and keep the earlier row as history. Only claim a cause after a controlled check |
| 1: release/reproduction | **The shipped photo scorer and the install file use different scikit-learn versions** | Loading `model_x01_3x3.joblib` under the declared 1.9.1 emits version warnings: the scaler, classifier and pipeline were saved under **1.6.1**. A cached-feature scoring check still gave AUC **0.849662**, so this does not disprove the ranking result. However, loading models across versions is unsupported by the library | Rebuild and save the small scoring model under the declared environment, using the existing frozen training procedure and development data only, or publish its exact tested environment. Test the saved artifact with the same public installation steps. Do not retune on the final test |

Sources for the first finding: [A02 feature extraction](../../code/scripts/a02_frozen.py), [later extractor](../../code/scripts/a_views.py), [first final result](../../code/runs/20261007-201747_F01_chip_final_eval/results.json), [clean final result](../../code/runs/20261007-205052_F03_chip_final_eval_clean/results.json), and the two saved development feature arrays. The mean absolute feature difference was 0.0017685, maximum 0.2011719, after checking identical image order. The baseline's chosen regularization is stored in each final result's `settings.whole.C`.

Sources for the second finding: [installation versions](../../code/requirements.txt), [saved scorer metadata](../../code/model_x01_3x3.json), [scorer rebuild script](../../code/scripts/train_final_model.py), and the [official scikit-learn model-persistence documentation](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations). The local test loaded the existing model and scored the saved clean final features; it was not a fresh-image or public-repository reproduction test.

## Wording and source details to correct or clarify

| Location | Issue | Suggested wording or action |
|---|---|---|
| Section 6 introduction | “Everything here is a simulator result” incorrectly includes R05 | “R01 to R04 concern simulated droplet control. R05 uses the real chip-photo dataset to study which photos to label.” |
| Methods and model metadata | “Native-resolution tiles” can sound like tiles from the original camera image | State the actual steps: convert to grey, resize the long side to **1036 pixels**, cut the 3×3 tiles from that prepared image, then round dimensions for the model. The whole-photo view uses long side 518. See [inference](../../code/inference.py) and [the extractor](../../code/scripts/a_views.py) |
| Task description | “Bad photos” can be mistaken for labels specifically identifying blur or exposure | The source labels are expert ratings of **good/bad chip-sample quality**. Describe the tool as ranking photos by predicted sample quality; do not invent a specific cause for the bad label. [Dataset description](https://zenodo.org/records/10203721) |
| E11 / remaining rights work | JUMP is called merely “open,” and its data licence is still listed as unconfirmed | The [Cell Painting Gallery registry](https://registry.opendata.aws/cellpainting-gallery/) specifies **CC0 1.0** for the data, with dataset-paper citation requested. C02's actual profile URL is in that gallery's `cpg0016-jump-assembled` collection. Cite that data source and its paper; keep the metadata repository's licence separate |
| Input-gate claims | The gate checks format, not whether all biological content is supported | Say “format checks route the listed unsupported formats to manual review.” A supported width, height and colour mode do not establish that an unfamiliar lab or biological sample is validated |
| Statistical claims about RL | The intervals resample scenarios after averaging the five fixed trained policies | Describe the result as evidence for those policies in this simulator's scenario ranges. Do not present it as proof about arbitrary new training runs, different simulator families or physical chips |

## Results independently checked in this review

| Check | Outcome | Decision |
|---|---|---|
| Final chip predictions, first evaluation | 672 automatic photos, 284 bad, 13 filename groups; whole-photo AUC **0.801165**, X01 **0.849717** | Matches the recorded first run |
| Final chip predictions, clean rebuild | Same scope; whole-photo AUC **0.809768**, X01 **0.849599** | Use this consistent comparison for the main result; its stored paired range is positive |
| Shipped scorer on saved clean final features | AUC **0.849662**; largest prediction difference from the clean evaluation 0.000818; version warnings described above | Headline ranking score holds, but repair the release environment |
| B02 final count setting | Recomputed from per-image rows: 50 distinct images, mean count error **4.308**, pooled strict-outline score averaged across fold models **0.846598** | Matches 4.31 and 0.847 |
| B03 final outline setting | Recomputed from per-image rows: 50 distinct images, mean count error **9.896**, pooled strict-outline score averaged across fold models **0.901026** | Matches 9.90 and 0.901; do not pair this outline score with the count-setting error |
| Repaired R02 to R04 data | **9,700 episode rows**, no duplicate study/variant/training-seed/condition/scenario keys, 50 scenarios in each policy/condition group. Recomputed paired means and ranges from raw episode rows | The repaired conclusions survive independent calculation |
| RL environment repair | Two different fixed-action policies with scenario seed 100000 had identical target schedules. Across 300 steps, reported diameter error always equalled error against the target used for the action | Both earlier evaluation problems are repaired in the new environment |
| F03 all-version repeats | Independently compared both per-image tables and all five settings-grid tables for each of the four runs against the earlier run | All **28 table comparisons matched**. This checks the repeated clean-up steps, not training every model again |

The nucleus baseline B01 has summary results and a run log, but its final per-image rows are not included in `final_per_image.csv`. Its 4.928 count error and 0.802967 strict-outline score were checked in the saved summary, not independently recomputed from per-image records here. Saving those per-image rows would make the baseline easier to audit; this is an evidence-storage gap, not proof that the score is wrong.

## RL answer after checking the repaired results

On the harder simulator scenarios, mean diameter error is **1.333** for RL, **1.946** for the model-assisted fixed-rule controller, and **2.062** for the simpler controller. That is about **31% and 35% lower mean error**, respectively. Independent whole-scenario resampling retained negative difference intervals in both comparisons. The largest change from the saved interval endpoints across all recorded comparisons/metrics was about 0.00334 when using a different resampling seed, which did not overturn the reported R02 mean-error result.

Keep the exceptions: the model-assisted controller has better settled error within the training ranges; an RL episode still spent **25.3%** of its steps outside the operating range; no physical chip was tested. R02, R03 ordinary and R04 all-inputs share the same five trained networks, not three independent sets of evidence. Staged training has worse size error; its slightly lower outside-range average was not a confirmed benefit because the paired interval includes zero. R05 label picking remains a negative result.

## Kaggle and primary-source checks

Read live on **8 October 2026**:

- [Official submission requirements](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview): declare a category; provide the technical report, public demo video and public code; complete the separate registration form. Video guidance is a maximum of five minutes. Public materials must be accessible without login/payment. An interactive website is optional.
- The same page showed an **in-progress draft** and deadline **10 October 2026, 9:29 PM IST** (the exact page timestamp is 21:29:59). A draft is not a submitted entry. Separate registration completion was not verified.
- [Official rules](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/rules): simulated data and AI tools are permitted with truthful disclosure, appropriate usage rights and the stated public-code conditions. There is no requirement to finish our internal 36-ID experiment plan.
- [DINOv2 model card](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m): Apache-2.0. [BBBC039](https://bbbc.broadinstitute.org/BBBC039) and [BBBC038](https://bbbc.broadinstitute.org/BBBC038): CC0. [Cell Painting Gallery](https://registry.opendata.aws/cellpainting-gallery/): CC0 1.0.
- [Zenodo OOC record](https://zenodo.org/records/10203721): the live browser page shows **Creative Commons Attribution 4.0 International**. The earlier paper-versus-record licence difference has not been resolved with the authors by this audit. Keep source attribution and avoid claiming that the conflict is settled.
- [Cellpose repository](https://github.com/MouseLand/cellpose): code is BSD-3-Clause; its README identifies CC-BY-NC training data. Keep code, data and model-usage terms distinct. “Not used because the team took a conservative licence position” is more precise than claiming that Kaggle explicitly bans Cellpose.

C01/RxRx1 and C03/Allen inclusion rights were not newly cleared here. Decide their inclusion using the exact terms for the material to be shown/shared. Public availability alone is not permission for every use.

## Before writing versus before submitting

**Before writing:** use the clean-rebuild comparison as the main photo result, correct E08's unsupported cause, separate real-photo and simulator evidence, and use the repaired RL numbers. No new large training experiment is necessary just to report these measured findings.

**Before submitting:** make the saved scorer and installation versions match; test the exact public instructions; choose the code licence and add credits; settle which supporting material can be shown; finish the report, team/category details and public video; complete the separate registration form; and submit rather than only save the draft.

C04, real camera noise, tiny-nucleus improvements and a timed user study remain possible future work. They become necessary only for the stronger claims they would support. Keep current limits explicit and protect the final test from further model selection.

## Limits of this audit

This audit checked the headline predictions, repaired RL results and code, the four package-version repeats, key method claims and live requirements/selected licences. It did not retrain the full model collection, independently annotate images, prove the public package works from scratch, reconcile Modal billing, clear every third-party term, or independently verify every historical number. The stored path-count check is only an existence check.
