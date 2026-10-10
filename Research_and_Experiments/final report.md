# Final experiment report before the write-up

**Updated and checked: 8 October 2026.**

**We have enough evidence to begin the write-up. The photo-review tool is the main result. Nucleus analysis and RL control are separate supporting studies. The submission package still needs finishing.**

This report covers **all 36 original plan IDs**, **all 13 extra work items**, and the later **F05 scorer rebuild and F06 strict re-test**. Of the 29 research experiment IDs within the original plan, **28 have recorded results**. **C04 has not run** because suitable independent labelled chip data is missing. The other seven original IDs cover checks and final tasks. Extra work items overlap existing experiments; they are not 13 new independent experiments.

The detailed numbers and run paths remain in [the before-and-after results](22_Final_Results_Before_and_After.md). This report gives the current conclusions without carrying forward the old “not started” labels or withdrawn explanations.

## 1. Main conclusions

| Area | Result to use in the write-up | What it means |
|---|---|---|
| **Chip-sample review, main result** | On the same **672 automatically scored final photos**, the clean rebuild gives **0.850** for X01 3×3 and **0.810** for the whole-photo method. Gain **+0.040**, with a 95% range of **+0.015 to +0.067** | Small image tiles help rank photos of poor-quality chip samples. This is a ranking score, not 85% accuracy |
| **Nucleus counting** | On 50 final images, B02's average count error is **4.31 nuclei per image**, against **4.93** for the simple B01 method | A modest counting improvement |
| **Nucleus outlines** | At their outline settings, B03 scores **0.901**, against **0.855** for B01 on the final images | The outline improvement is the stronger nucleus result. These are different settings from the counting comparison |
| **RL control, simulator only** | On harder simulated conditions, mean droplet-size error is **1.333 micrometres** for RL, **1.946** for the model-assisted rule controller and **2.062** for the simpler rule controller | About **31% and 35% less average error**, respectively, after the evaluation was repaired. No physical chip was tested |
| **Working tool** | Rebuilt scorer uses scikit-learn **1.9.1**, matching the install file. A fresh Kaggle installation completed the 17-photo demo: **12 scored, 5 sent to manual review** | The normal install-and-demo path has passed. A separate strict re-test failed to run and is not counted as passed |

**Terms:** AUC is the photo-ranking score; 0.5 is chance and higher is better. Count error is how many nuclei the count is off by; lower is better. F1@0.75 measures how well predicted nucleus outlines match the reference at a strict overlap requirement; higher is better. RL means learning actions through feedback. A micrometre is one millionth of a metre.

The chip labels are biology experts' ratings of **good or bad sample quality**. They are not labels specifically identifying blur, exposure or a particular biological defect.

## 2. All 36 original plan IDs

“Tested” means the stated experiment has a saved result. It does not mean the idea helped, every possible version was tried, or every weakness was solved. Unconfirmed small changes are kept separate from supported improvements.

### Chip photos: A01 to A12 and X01

| ID | What was tested | Current result | Decision / remaining work |
|---|---|---|---|
| A01 | Simple image measurements | **Tested.** Ranking score 0.695, above chance | Keep as a starting comparison |
| A02 | Larger frozen model and light retraining | **Tested, including the later retraining.** Frozen large model 0.843 vs small model 0.768 in development; light retraining 0.819 | Keep the frozen model. Extra training did not help in this comparison |
| A03 | Cell types not used for training | **Tested and corrected.** With the held-out type and its related filename groups excluded: whole photo 0.756, X01 3×3 0.799 | Patches still help. Withdraw the old “no drop on unseen types” claim |
| A04 | Cell content, edges and camera formats | **Tested.** Edges retain much of the signal; equal weighting of formats did not solve the small-photo problem | Describe possible photo-wide shortcuts. Do not claim broad lab independence |
| A05 | Flips, turns and brightness changes | **Tested and repeated.** No confirmed gain for the chosen X01 method | Not used |
| A06 | Different class weights | **Initial test recorded.** 0.847 vs 0.843; paired confirmation not run | No confirmed improvement claim. Further confirmation only if this idea is revived |
| A07 | Add sample details | **Initial test recorded.** 0.845 vs 0.843 | No demonstrated useful gain; not used |
| A08 | Correct confidence scores | **Tested for the chosen model.** The tested corrections did not help | Use scores to order review, not as guaranteed probabilities |
| A09 | Order photos for review | **Tested, including final data.** Final review-area score 0.703 for X01 vs 0.675 whole photo and 0.499 random | Better ordering is supported. Actual minutes saved are unmeasured |
| A10 | Average two photo models / use disagreement | **Tested.** Averaging reduced the development score; the models share mistakes | Do not describe this pair as an independent backup |
| A11 | Choose photos for labelling | **Tested.** Estimated score at 800 labelled photos: uncertainty picking 0.810 vs random 0.794; the gain is noisy | Supporting result only; no proven human labelling-time saving |
| A12 | Heat maps as explanations | **Tested.** Hiding supposedly important regions did not pass the explanation checks | Do not show them as trusted explanations |
| X01 | Whole photo plus small tiles | **Tested and confirmed on the reserved final photos.** Frozen choice: 3×3; clean-rebuild final score 0.850 vs 0.810 | Main photo method. Small and unfamiliar formats go to manual review |

The main method converts a photo to grey, resizes its long side to **1036 pixels**, then cuts a 3×3 grid. Each tile is processed at its prepared size, rounded to the model's required multiple of 14. The whole-photo view uses a long side of 518 pixels. Tiles are not cut directly from the original camera-resolution image.

### Nuclei: B01 to B07

| ID | What was tested | Current result | Decision / remaining work |
|---|---|---|---|
| B01 | Simple counting and outline method | **Tested, including final data.** Final count-setting error 4.93; outline-setting F1@0.75 0.855 | Keep as the simple reference |
| B02 | Learned nucleus counting | **Tested, repeated with a new training seed, final test done.** Final count error 4.31, second seed 4.36 | A modest final counting gain. Development improvement was larger |
| B03 | Learn nucleus edges | **Tested, repeated, wider settings checked.** Final outline score 0.901; its count-setting error 4.90 | Stronger for outlines. Keep counting and outline settings separate |
| B04 | Blur, dim images and added noise | **Tested; noise training also tested.** It sharply improved artificial-noise performance but made clean counting worse | Keep as a labelled alternative. Real camera-noise robustness remains untested |
| B05 | Tiny nuclei and crowded images | **Tested, causes examined, settings swept.** Only about 43% of the smallest tenth are found at the looser overlap level | Tiny nuclei remain a weakness. More recovery through the tested settings costs other quality |
| B06 | Warning from model disagreement | **Tested, including shared failures and final review simulation.** Both models make the same-direction errors on many images | Useful only as a partial warning. Shared mistakes can remain invisible |
| B07 | A different nucleus dataset | **Tested and repeated.** Performance weakens on outside grey images; colour images have strict-outline score 0 | Limit the claim to the tested grey fluorescence setting; colour goes to manual review |

The final 50 nucleus images share plates with the development images. This is not a new-lab validation. The reported final model scores average results across the five fold models; the spread across models is not a confidence interval over independent labs.

### Supporting datasets: C01 to C04

| ID | What was tested | Current result | Decision / remaining work |
|---|---|---|---|
| C01 | Different lab batches in RxRx1 | **Tested.** Accuracy 86% on new batches vs 97% within batches; possible prior exposure and usage limits remain | Research record. Settle exact inclusion rights and claim scope before publishing this supporting study |
| C02 | Re-scale JUMP profiles to remove batch differences | **Tested and corrected.** Biological matching 0.006 to 0.059, but batch identification 0.404 to 0.860, which is worse for batch removal | Mixed result. Do not claim the method removes batch effects |
| C03 | Predict fluorescence before measuring nuclei | **Tested and corrected.** With depth chosen from the plain input only, count error 12.09 for predict-then-measure vs 7.25 for direct measurement | Direct measurement is better in this task. Settle rights for any material included in the submission |
| C04 | Chip photos from another lab | **Blocked, not run.** The recorded search found no suitable independent labelled dataset | Do not claim outside-lab chip validation. Missing data is not proof that no dataset exists |

### RL: R01 to R05

| ID | What was tested | Current result | Decision / remaining work |
|---|---|---|---|
| R01 | Build and check a control simulator | **Equation check done.** The practice model follows published equations; some response settings are assumptions | Useful for simulation. Independent physical dynamics remain unvalidated |
| R02 | Learned control versus two fixed-rule controllers | **Tested and re-scored under the repaired evaluation.** On harder conditions: 1.333 vs 1.946 / 2.062 micrometres mean error | Supported simulator improvement. Keep the exception and worst cases below |
| R03 | Train RL in stages | **Tested and re-scored.** Harder-condition size error 1.401 staged vs 1.333 ordinary | Staged training did not improve the primary error measure in this setup |
| R04 | Different inputs for RL | **Tested and re-scored.** Harder-condition error 1.333 with all inputs vs 1.738 camera-only | Extra inputs help in these tests. Report missing-input and noise tests separately |
| R05 | RL chooses photos to label | **Tested, including 10× training.** Estimated score at 800 photos: 0.767 RL, 0.794 random, 0.810 uncertainty; longer RL training 0.755 | This label picker did not help. It uses real chip-photo data and is separate from droplet simulation |

The repaired evaluation fixes error timing and gives each controller the same scenario. It uses **50 new scenarios and five saved training seeds**, with no retraining. Training seeds are averaged within each scenario before estimating uncertainty from whole scenarios. R02, R03's ordinary controller and R04's all-input controller share **the same five networks**; they are not three independent confirmations.

### Checks and final work: G01 to G03, F01 to F04

| ID | Work | Current status | What remains |
|---|---|---|---|
| G01 | Data, model and version checks | **Technical checks done; publication rights partly open.** Checksums, duplicate checks, model files and versions recorded | Finish credits, code licence and exact rights for any displayed/shared material |
| G02 | Test splits and scoring checks | **Image checks done; RL evaluation repaired and checked** | Do not use the opened final sets to choose new models. Filename groups are clues, not verified chip IDs |
| G03 | Speed, memory and cost | **Measurements recorded.** Modal A02 cloud files confirm an A100. Current October API reading: $17.17 metered, $0 billed | Older $24.26 versus $17.08 readings remain unreconciled. Omit the unverified Mac speed from claims |
| F01 | Confirm choices on final data | **Done for the frozen photo and nucleus choices.** Repaired RL evaluation also complete | New variants need development-only choices and new independent confirmation |
| F02 | Review workload | **Saved-prediction simulations done.** Reviewing 10 of 50 final nucleus images would remove 37% of the error in the averaged B02 count predictions | Actual time savings need a timed user study |
| F03 | Reproduce from a clean setup | **Chip pipeline and one B03 training split rebuilt.** Four later clean-up repeats matched under the declared versions | Do not say every model or supporting study was rebuilt from scratch |
| F04 | Submission package | **Partly complete.** Scorer, local demo, report draft and video script exist; rebuilt scorer passes the normal fresh-install path | Final public-package check, rights/credits, team/category, public video, registration and actual submission |

## 3. All 13 extra work items

These are supporting checks or artifacts, often belonging to the original IDs above.

| ID | Extra work | Current result and limit |
|---|---|---|
| E01 | Input-format checks | Tested formats are routed as intended. Three of 675 final chip photos go to manual review. The check does not establish that an unfamiliar lab or biological sample is supported |
| E02 | Recorded choice rule | Selected X01 3×3 before final evaluation; small photos remain manual review |
| E03 | Train only the output layer | Development 0.845 vs frozen 0.843; no clear gain. Different from the failed last-block retraining |
| E04 | Find where tiny nuclei are lost | Diagnosis completed: merging, thresholds and size filtering account for different losses |
| E05 | Test changes at those stages | Sweeps completed. More tiny nuclei can be recovered, but the tested changes hurt other results |
| E06 | Check shared nucleus failures | Completed. Both models fail in the same direction on 83 of 150 development images, carrying 78% of B02's count error in the first seed |
| E07 | Compare a modern ready-made model | InstanSeg development best-count error 3.97; best-outline F1@0.75 0.913, using different settings. Possible training exposure prevents a clean superiority claim |
| E08 | Explain the changed whole-photo reference | **Old clipping explanation withdrawn. Cause not isolated.** Lead with the consistent clean rebuild, 0.850 vs 0.810; keep 0.850 vs 0.801 as history |
| E09 | Working scorer and demo | Scorer rebuilt under 1.9.1. The 17-photo demo preserves the earlier rank order and routes. It illustrates operation, not perfect accuracy |
| E10 | Exact model files and software versions | Recorded. Rebuilt scorer metadata now records its software versions and training-feature source |
| E11 | Data/model usage rights | Records exist; some publication choices remain. JUMP gallery data is CC0 1.0. Separate data, code and model terms |
| E12 | Evidence path and number checks | Checks completed within their stated scope. A path existing or a number appearing somewhere is not proof that every claim is correct |
| E13 | Documents | Detailed results, decision ledger, instructions, technical-report draft and video script exist. The public submission package is not complete |

## 4. Latest scorer rebuild and installation checks

These later verification jobs are additional work, not changes to the original 36-ID count.

| Job | Current result | What can be claimed |
|---|---|---|
| **F05: scorer rebuild and fresh install** | Kaggle status **COMPLETE**, checked live. All 15 pinned versions match. Installation took **1420.6 seconds, about 23.7 minutes**. Scorer fitting took **17 seconds**; the 17-photo CPU demo took **402.8 seconds** | The normal fresh-install and inference path worked on Kaggle CPU. This timing belongs to that run, not a promise for another machine |
| **F06: strict re-test** | Kaggle status **ERROR**, checked live; downloaded log is an empty list, with no useful failure diagnosis | The extra strict re-test did not complete. Do not present it as passed |

In F05, the command-line switch meant to turn version warnings into errors was ignored. The old-model control caught that problem. The rebuilt model loaded and the normal demo ran without the old version-warning text, but the cloud test was not a successful strict warning-as-error check. A proper strict load was recorded separately on the Mac. See [the rebuild and installation evidence](Evidence/2026-10-08_scorer_rebuild_and_fresh_install.md).

The rebuilt scorer's stored yes/no cut-off is **0.451172**, about **0.45**, instead of the old **0.57**. **`inference.py` does not use that cut-off.** It ranks images by the continuous score. This is not a change to the tool's ranking rule. The scorer still uses **C = 0.001** and **2,397 development photos**; final photos were not used for training or choosing settings.

On saved clean final features, the rebuilt scorer gives **AUC 0.849599**, matching the clean evaluation, with a largest score difference below **0.0000001**. This confirms the saved-feature check; the separate 17-photo fresh-install demo confirms the image-to-output path on those examples. Neither is new independent test data.

The recorded script changes add version/source metadata and make the tile description precise. They do not change the training procedure. The old scorer and metadata are preserved in the evidence backup.

## 5. What the live cloud check confirmed

The cloud check performed for this report was read-only. It did not start compute, change jobs or submit anything.

| Source | Live check | Finding |
|---|---|---|
| Kaggle `ai4s-rl-eval` | Completion status and downloaded outputs | **COMPLETE. All eight compared result/log files exactly match the local copies** |
| Modal `ai4s-data` volume, A02 | Downloaded original result and environment files | **Both match the local copies exactly.** Recorded GPU: NVIDIA A100-SXM4-40GB |
| Kaggle `ai4s-f05-scorer` | Completion status, rebuilt model, metadata, logs and install report | **COMPLETE. All 14 compared files match the saved job evidence.** The local scorer and metadata also match the cloud output |
| Kaggle `ai4s-f03` | Live job status | **COMPLETE.** The earlier clean-rebuild outputs were checked locally; this pass did not download them again |
| Kaggle `ai4s-f06-install-test` | Live status and log | **ERROR; empty log.** Cause remains unknown |
| Modal current account records | App list and October billing summary | No apps listed in the checked profile. **$17.17317520 metered usage; $0 billed**, with credits/allowances applied. Metered usage is not the amount billed |

In total, **24 downloaded cloud files matched their local counterparts**. This establishes where those records came from; it is not a new experiment or a guarantee that every methodological choice is valid. The billing reading does not explain the earlier inconsistent amounts.

Exact hashes and scope: [cloud verification record](Evidence/2026-10-08_cloud_verification.json).

## 6. What to say about RL

**RL produced a useful, supported result in the repaired simulator comparison. It did not help the separate label-picking task.**

- On harder scenarios, RL minus the model-assisted controller is **−0.613 micrometres**, with a 95% range **−0.723 to −0.503**. Against the simpler controller it is **−0.729**, with a range **−0.816 to −0.633**. Lower error is better.
- Each of the five saved training seeds had lower mean error than both references in both tested condition ranges. The intervals concern the sampled scenarios after averaging those fixed policies.
- RL is not best in every measure: after the first half of each target block, the model-assisted controller has lower error within the training ranges.
- On one harder RL episode, **25.3% of steps were outside the simulator's operating range**. This is a different episode from the one with the highest mean size error, **2.826 micrometres**. No safety guarantee or validated physical fallback is claimed.
- Staged training worsened the main size-error result. Its slightly lower outside-range average was not a confirmed gain because its interval includes zero.
- Real-chip control, general performance on other simulators and actual biological benefits remain untested.

Full evidence: [repaired RL evaluation](Evidence/2026-10-08_RL_repaired_evaluation.md).

## 7. Remaining limits and possible future experiments

| Remaining question | Current limit | Test needed for a stronger claim |
|---|---|---|
| Another lab's chip images, C04 | No suitable independent labelled set found | Obtain compatible independent data, verify rights/duplicates and test the frozen method |
| Small/unfamiliar chip formats | Too little representative evidence | More independent examples, especially poor-quality samples, before replacing manual review |
| Real camera noise | Training changes were tested with artificial noise | Compare retained and noise-trained models on real noisy images with reliable labels |
| Tiny nuclei | Most of the smallest objects remain missed | Test a structurally different way to separate small touching objects; measure false detections and overall quality too |
| Shared failures | Two models can make the same mistake | Evaluate a warning based on different information at the same review workload |
| InstanSeg | Possible prior exposure to related benchmark images | A comparison on independently sourced data with training exposure checked |
| Time saved | Review curves are simulated from stored predictions | A timed study of people performing the review task |
| C02 batch correction | Biological matching improved while batch differences worsened | One different correction method under the repaired, matched protocol |
| Physical RL | Simulator assumptions are not validated physical dynamics | Independent response measurements or an authorized physical evaluation |
| Full reproduction | Only the stated pipeline, split and clean-up checks were repeated | Rebuild additional components only before claiming the entire collection was reproduced |

These are not all conditions for starting the write-up. They are conditions for the stronger claims in the last column. Do not use the already opened final sets to choose new variants, and do not restart failed methods merely to obtain a positive result.

## 8. Before submission

The [official Kaggle requirements](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview) were read in this chat on 8 October. The displayed deadline was **10 October 2026, 9:29 PM IST**. The required package is a category declaration, technical report, public code and public demo video, plus the separate registration form. A public interactive website is optional. The [rules](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/rules) permit simulated data and AI tools with truthful disclosure and appropriate rights.

| To do | Current position |
|---|---|
| Finish the write-up using the current evidence | Ready to draft. Use the consistent 0.850 vs 0.810 photo comparison and repaired RL results |
| Check the final public package | F05 passed on Kaggle; F06 did not finish. Test the actual public instructions and use a proper in-Python version-warning check. Public-link access has not been verified |
| Choose code licence and complete credits | Still open. Identify the outside datasets, pretrained models, software and material AI-tool use |
| Settle supporting-study inclusion | C01/C03 rights choices and the OOC paper-versus-record licence difference remain open. Do not infer unrestricted use from public availability |
| Complete team and category details | Confirm actual members, roles and team leader; declare the category at the start |
| Record and publish the demo video | Local demo and script exist; public video remains to be completed. Keep it within five minutes and accessible without login/payment |
| Confirm the separate registration form | Completion is unverified; accepting Kaggle rules is not the same as completing the form |
| Submit the entry | The last page check showed a draft. Saving a draft is not submitting |

Recommended write-up order: the chip-sample review problem, the working tool, the controlled photo comparison, failure cases and manual review, nucleus support, then the separate RL study. Keep the negative results and limits visible.

## 9. Evidence and audit trail

- [Detailed before/after results and all run paths](22_Final_Results_Before_and_After.md)
- [Decision ledger](17_Decision_Ledger.md)
- [Pre-write-up audit, with the earlier issues](Evidence/2026-10-08_pre_writeup_audit.md)
- [Scorer rebuild and fresh-install result, including the failed strict re-test](Evidence/2026-10-08_scorer_rebuild_and_fresh_install.md)
- [Live cloud verification and exact file hashes](Evidence/2026-10-08_cloud_verification.json)
- [RL repaired evaluation](Evidence/2026-10-08_RL_repaired_evaluation.md)
- [Package-version repeat checks](Evidence/2026-10-08_F03_package_version_repeat.md)
- [Earlier numerical table check and its scope](Evidence/2026-10-08_results_table_verification.md)
- [Current technical-report draft](20_Technical_Report_Draft.md)
- [Demo instructions and evidence](../code/demo/NOTES.md)

Primary data/model sources checked during the review: [OOC dataset](https://zenodo.org/records/10203721), [BBBC039](https://bbbc.broadinstitute.org/BBBC039), [BBBC038](https://bbbc.broadinstitute.org/BBBC038), [JUMP/Cell Painting Gallery data licence](https://registry.opendata.aws/cellpainting-gallery/), and [DINOv2 model card](https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m). These source checks do not establish permissions for unrelated material.
