# Demo video script

**Version:** 10 October 2026. **Category:** Model & Algorithm.

**Actual video:** 4 minutes 34 seconds (273.938 seconds). **Original target:** 4:28. **Hard limit:** 5:00. Scene timings below are rounded from the rendered schedule.

The video presents recorded, verified outputs. It does not pretend to be a live recording. The GitHub link is pending at the user's request.

## 0:00 to 0:18 | A review aid for chip samples

**Screen:** figures/kaggle_card_thumbnail.png

**Visible message:** Research use. A person makes the sample-quality decision.

**Narration:**

This is AI4S Chip Photo Review, a research tool for microscope photos of organ-on-a-chip samples. It places likely poor-quality samples earlier in a review list. A person makes the decision.

**Evidence:** Project method and expert good/bad sample-quality labels.

## 0:18 to 0:42 | One whole photo, plus nine local tiles

**Screen:** figures/01_workflow_redesigned.png

**Visible message:** The tiles come from the prepared image, not the original camera resolution.

**Narration:**

The tool first checks the image format. Small or unfamiliar formats go to manual review without an automatic score. Accepted images become gray images with a long side of 1,036 pixels. Frozen DINOv2 features combine a whole-photo view with nine tiles. A small trained model produces the review score.

**Evidence:** code/inference.py and model_x01_3x3.json

## 0:42 to 1:18 | Recorded operation: 17 input photos

**Screen:** csv

**Visible message:** Recorded F05 output, 8 October 2026. CPU inference took 402.8 seconds.

**Narration:**

This is the command that runs the photo tool. The table contains actual saved output from a verified fresh Kaggle installation on October eighth. Twelve photos received scores and ranks. Five photos went to manual review, including small and unfamiliar formats. The output includes the filename, format, route, score, and rank. This CPU demonstration took 402.8 seconds.

**Evidence:** F05 ranking_fresh.csv and job_report.json

## 1:18 to 1:40 | What the review list looks like

**Screen:** figures/04_real_demo.png

**Visible message:** The first four and last four scored photos are shown. This is an illustration.

**Narration:**

The top row shows the first four photos in the review list. The bottom row shows the last four. This small demonstration illustrates operation. It is not an accuracy estimate.

**Evidence:** Images: Movčana et al., Organ-on-a-Chip Image Dataset, Zenodo 10203721.

## 1:40 to 2:11 | Reserved photos: a measured ranking increase

**Screen:** figures/02_chip_final.png

**Visible message:** Same 672 photos. AUC 0.850 vs 0.810. Paired increase 0.040.

**Narration:**

The primary result uses 672 reserved photos scored automatically by both methods. The ranking score, called AUC, was 0.850 with tiles and 0.810 with the whole photo alone. The paired increase was 0.040, with a 95 percent range from 0.015 to 0.067. AUC measures ranking quality. It does not mean 85 percent accuracy. Review curves do not measure human time savings.

**Evidence:** F03 clean final evaluation; filename-group uncertainty ranges.

## 2:11 to 2:35 | Corrections changed the decisions

**Screen:** figures/09_validation_and_formats.png

**Visible message:** Related filename groups are clues, not verified physical chip identifiers.

**Narration:**

Scores decreased when related filename groups were also excluded from training. We withdrew the earlier claim of no loss on unseen cell types. The tile method still ranked better in the corrected comparison. Small photos remained a weakness, so they go to manual review. Extra training and heat-map explanations did not justify stronger claims.

**Evidence:** Corrected A03 and X01 format tests; development data only.

## 2:35 to 2:56 | Nuclei: separate settings for separate goals

**Screen:** figures/05_nucleus_final.png

**Visible message:** 50 reserved nucleus images. Counting and outline results use different settings.

**Narration:**

Separate nucleus studies used fifty reserved images. B02 reduced average count error from 4.93 to 4.31 nuclei per image. At different settings, B03 increased the strict outline score from 0.855 to 0.901. The images share plates with development images, so this is not a new-laboratory test.

**Evidence:** F01 nucleus results; these images share plates with development images.

## 2:56 to 3:18 | Stress tests exposed important limits

**Screen:** figures/06_nucleus_limits.png

**Visible message:** Only about 43% of the smallest tenth of nuclei were found at overlap 0.50.

**Narration:**

Strong artificial noise caused large errors. Training with noise helped under that noise type, but made clean counting worse. Only about forty-three percent of the smallest tenth of nuclei were found at the looser overlap level. Outside color images had strict outline scores below 0.001. The models also shared mistakes.

**Evidence:** B04/B05 development records. B07 color-image scores are below 0.001.

## 3:18 to 3:41 | Separate study: simulated droplet control

**Screen:** figures/08_rl_simulator.png

**Visible message:** Simulator only. Worst learned-control episode: 25.3% of steps outside the operating range.

**Narration:**

We also compared controllers in a repaired droplet simulator. On harder conditions, learned control had mean size error of 1.333 micrometers, versus 1.946 and 2.062 for two rule controllers. One learned-control episode still spent 25.3 percent of steps outside the operating range. No physical chip was operated.

**Evidence:** Repaired R02 evaluation: 50 new scenarios and five saved training seeds.

## 3:41 to 4:03 | Reproduction checks have a stated scope

**Screen:** reproduction

**Visible message:** Rebuilt scorer: scikit-learn 1.9.1. F06 did not pass the additional strict test.

**Narration:**

A fresh Kaggle setup reproduced the photo pipeline and one nucleus training split. It did not rebuild every model or supporting study. The rebuilt scorer completed the ordinary fresh-install demonstration. The additional strict cloud test failed with no useful diagnostic log. That failure remains in the report.

**Evidence:** F03 clean rebuild, F05 fresh install, and F06 error record.

## 4:03 to 4:23 | Measured results and the next validation

**Screen:** limits

**Visible message:** The contribution is a tested review method with clear limits and human control.

**Narration:**

The measured benefit is better review order on the stated dataset. Human time savings, transfer to another chip laboratory, and physical control remain unproven. The next steps are independently labeled chip data and a timed study with researchers. A person stays responsible for sample decisions.

**Evidence:** Decision ledger and current write-up; future tests are proposals, not results.

## 4:23 to 4:34 | Sources, credits, and full experiment record

**Screen:** credits

**Visible message:** Full methods, all experiments, and limitations are in the Kaggle write-up.

**Narration:**

The write-up contains the full experiment record and source credits. The full write-up contains the reproduction instructions and resource links. Thank you for reviewing AI4S Chip Photo Review.

**Evidence:** Synthetic narration. Recorded outputs. No fabricated measurements or microscopy images.

## Production and review plan

1. Use the real recorded CSV and demo sheet for the operation scene.
2. Label saved output as recorded. Do not animate a fake live run.
3. Use the verified result figures and the stated comparison settings.
4. Keep the primary photo method first. Keep the nucleus and simulator studies separate.
5. Include the failure cases, reproduction limits and source credits.
6. Create clear narration and synchronized subtitles.
7. Check the actual MP4 duration, sound, text and image clarity.
8. Attach the MP4 and public video link where Kaggle permits them.
9. Add the public GitHub link when the user creates the repository.

## Resource credits

- Dataset images: Movčana and colleagues, Organ-on-a-Chip Image Dataset, Zenodo 10203721. The record states CC BY 4.0; the paper states CC BY-SA. The report keeps this difference explicit.
- Nucleus data: BBBC039 and BBBC038, Broad Bioimage Benchmark Collection, CC0.
- Image model: DINOv2 ViT-L/14; code and model credits remain in the report.
- Quantitative charts: saved experiment records, rendered with Matplotlib.
- Narration: the installed macOS Samantha synthetic voice. No music or external stock footage.

## Remaining final-submission items

The user will add the GitHub repository link later. Team details, the stated rights decisions, separate registration and the final submission action remain outside this video-production step.
