# Breaking the Frame: Seeing More in Every Chip

This repository contains the code, saved results and research records for one competition entry. The competition is [AI4S Open Innovation: AI for Life Science](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien) on Kaggle. The category is Model & Algorithm.

| Item | Location |
|---|---|
| Technical report | [Kaggle writeup](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/writeups/breaking-the-frame-seeing-more-in-every-chip). The report text is also in `Research_and_Experiments/Writeup/kaggle_writeup.md`. |
| Demo video | [YouTube video](https://youtu.be/HXp1MqyvRRE), 4 minutes 34 seconds. The file is `Research_and_Experiments/Writeup/Video/AI4S_Demo.mp4`. |
| Tool entry point | `code/inference.py` |
| Map from each experiment to its saved records | `Research_and_Experiments/Writeup/source_map.md` |

## 1. Purpose

The tool is a review aid for microscope photos of organ-on-a-chip samples. It puts likely poor-quality samples first in a review list. A person makes the final decision.

The tool does not operate a physical chip. It is not a diagnostic device. It does not predict drug response.

## 2. Method

1. The tool examines the image format.
2. Small or unfamiliar formats go to manual review without a score.
3. The tool changes each accepted image to a gray image.
4. A frozen DINOv2 image model calculates features for the whole photo.
5. The same model calculates features for nine smaller tiles in a 3×3 grid.
6. A small trained scorer calculates one review score for each photo.
7. The tool sorts the photos by score and writes a review list.

A frozen model is an image model whose weights do not change. A feature is a numerical description that the image model produces. A tile is one rectangular part of a prepared image.

## 3. Primary results

The technical report gives each result, failure and limit. This table shows only the primary numbers from that report.

| Study | Result | Scope |
|---|---|---|
| Chip photos, whole photo plus 3×3 tiles | Ranking score (AUC) 0.850 | 672 reserved final photos that the tool scored automatically. |
| Chip photos, whole photo only | Ranking score (AUC) 0.810 | The same 672 photos in the same clean rebuild. |
| Chip photos, paired gain | 0.040, with a 95% range from 0.015 to 0.067 | The same 672 photos. |
| Nucleus counting | Mean count error decreased from 4.93 to 4.31 nuclei | 50 final images from BBBC039. |
| Nucleus outlines | Outline score F1@0.75 increased from 0.855 to 0.901 | 50 final images. Counting and outlines use different settings. |
| Droplet control simulator | Mean size error 1.333 micrometers, compared with 1.946 and 2.062 for two rule controllers | Harder simulated conditions. No physical chip was operated. |

AUC measures ranking quality. An AUC of 0.850 does not mean 85% accuracy. The measured benefit is a better review order on the stated dataset. Human time savings were not measured.

## 4. Repository contents

| Path | Contents |
|---|---|
| `code/inference.py` | The entry point of the photo tool. |
| `code/model_x01_3x3.joblib` and `code/model_x01_3x3.json` | The saved scorer and its metadata. |
| `code/final_config.json` | The fixed choices for the final evaluations. |
| `code/requirements.txt` | The package versions used for the clean rebuild. |
| `code/src/ai4s` | Shared code. |
| `code/scripts` | Experiment scripts. |
| `code/splits` | The fixed data split files with their hashes. |
| `code/runs` | Saved results. `code/runs/registry.csv` is an index with some older incomplete entries. |
| `code/kaggle` | Kaggle job files. `code/kaggle/out` contains job outputs that are not stored in `code/runs`. |
| `code/cloud` | Helper scripts for Modal cloud jobs. |
| `code/logs` | Job logs. |
| `code/demo` | Example outputs from one recorded run. Read `code/demo/NOTES.md`. |
| `Research_and_Experiments` | Research records, evidence files, the report, figures and the video. |
| `Resources` | Planning notes about the competition. |

## 5. Run the photo tool

1. Install Python 3.12 or 3.13.
2. Download this repository.
3. Open a terminal in the `code` directory.
4. Install the declared packages.
5. Run the tool on a folder of photos.

```bash
python -m pip install -r requirements.txt
python inference.py --photos path/to/photos --out ranking.csv --sheet sheet.png
```

The input is image files or folders with `.png`, `.jpg` or `.tif` files. The output file `ranking.csv` contains `photo`, `format`, `route`, `p_bad` and `rank`. Rank 1 is the first photo to review. The optional file `sheet.png` shows example photos with their scores.

Photos in an unsupported format get the route `manual_review` and no score. The 640×480 format also goes to manual review. The value `p_bad` is a ranking score. It is not a calibrated probability.

A CPU can run the tool. A compatible GPU decreases the processing time. The first run downloads the DINOv2 model, thus an internet connection is necessary.

Use the versions in `requirements.txt`. The saved scorer was made with scikit-learn 1.9.1. The scikit-learn library does not support loading this scorer with a different version. The command `python scripts/train_final_model.py` makes the scorer again.

## 6. Reproduce the results

No paid service is necessary for the tested inference path. Paid cloud computing was used during the research.

| Task | Command or file | Recorded scope |
|---|---|---|
| Clean rebuild | Run `code/kaggle/jobs/ai4s-f03/run.py` as a Kaggle GPU notebook. Then run `python scripts/f03_compare.py`. | About 42 minutes on a Kaggle T4. The chip pipeline and one B03 training split. Not all models. |
| Nucleus cleanup repeats | Run each `code/kaggle/jobs/ai4s-f03c-*/run.py` as a Kaggle CPU notebook. Then run `python scripts/f03_pinned_compare.py NEW_RUNS_FOLDER`. | Saved nucleus models were used again. This was not new training. |
| Final evaluations | `python scripts/f01_final.py --part chip_features`, then `--part chip_eval`, then `--part nucleus`. | These use `code/final_config.json`. |
| Scorer rebuild | `python scripts/train_final_model.py` | Recorded in F05. |
| Simulator evaluation | `python scripts/r_eval_repaired.py` | 50 new scenarios and five saved training seeds. |
| Path trace | `python scripts/f04_trace.py` | This examines named path references. It does not examine each number. |

The public Kaggle notebooks below contain the recorded runs.

| Kaggle notebook | Purpose | Recorded status |
|---|---|---|
| [ai4s-f03](https://www.kaggle.com/code/itsasup/ai4s-f03) | F03 clean rebuild in a fresh GPU environment | Completed. |
| [ai4s-f03c-b02s1](https://www.kaggle.com/code/itsasup/ai4s-f03c-b02s1) | F03 cleanup repeat | Completed. |
| [ai4s-f03c-b03s1](https://www.kaggle.com/code/itsasup/ai4s-f03c-b03s1) | F03 cleanup repeat | Completed. |
| [ai4s-f03c-b03x0](https://www.kaggle.com/code/itsasup/ai4s-f03c-b03x0) | F03 cleanup repeat | Completed. |
| [ai4s-f03c-b03x1](https://www.kaggle.com/code/itsasup/ai4s-f03c-b03x1) | F03 cleanup repeat | Completed. |
| [ai4s-f05-scorer](https://www.kaggle.com/code/itsasup/ai4s-f05-scorer) | F05 scorer rebuild and fresh CPU installation | Completed. The warning setting was ignored, thus this was not a strict test. |
| [ai4s-f06-install-test](https://www.kaggle.com/code/itsasup/ai4s-f06-install-test) | F06 stricter installation test | Kaggle reported ERROR. The cause is unknown. This test did not pass. |
| [ai4s-demo](https://www.kaggle.com/code/itsasup/ai4s-demo) | Demonstration run of the photo tool | Completed. |
| [ai4s-rl-eval](https://www.kaggle.com/code/itsasup/ai4s-rl-eval) | Simulator evaluation with `scripts/r_eval_repaired.py` | Completed. |

These notebooks read three public Kaggle datasets: `itsasup/ai4s-code`, `itsasup/ai4s-scorer-pkg` and `itsasup/ai4s-rl`. The datasets contain copies of this code and saved results.

## 7. Data

The raw data is not in this repository. The scripts download the data from the public sources.

| Data | Use in this project | Source and terms |
|---|---|---|
| Organ-on-a-Chip Image Dataset | Primary sample-quality task | [Zenodo 10203721](https://zenodo.org/records/10203721). The record states CC BY 4.0. The paper states CC BY-SA. This difference is open. |
| BBBC039 | Nucleus counting and outlines | [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/BBBC039), CC0. |
| BBBC038 | Outside nucleus test | [BBBC038](https://bbbc.broadinstitute.org/BBBC038), CC0. |
| JUMP Cell Painting profiles | C02 supporting study | [Cell Painting Gallery](https://registry.opendata.aws/cellpainting-gallery/), CC0 1.0 for gallery data. |
| RxRx1 | C01 supporting study | [RxRx1](https://www.rxrx.ai/rxrx1). Research record only. |
| Allen Cell images | C03 supporting study | [Allen Cell](https://www.allencell.org/). Research and non-commercial terms are recorded. Redistribution is not cleared. |

## 8. Files that are not in this repository

| Item | Reason |
|---|---|
| Raw datasets (the local `data` directory) | Size and source terms. Download them from the sources in section 7. |
| The 17 raw chip photos of the demonstration | They are dataset photos from another owner. `code/demo/NOTES.md` identifies them. |
| `Research_and_Experiments/Evidence/OOC_datasheet.xlsx` | It is a file from the dataset of another owner. |
| C01 file `per_site_correct.csv` | It is derived from RxRx1 data. The C01 summary results are included. |
| C03 model files `route*_fold*.pt` and C03 images `channels_*.png` | They are derived from Allen Cell data. The C03 result tables are included. |
| Kaggle upload bundles in `code/kaggle` | They are copies of files in this repository. Some copies have earlier dates. |
| Copies in `code/kaggle/out` | Only files that are not stored in a different location are included. |
| The Python environment and cache files | The file `code/requirements.txt` gives the package versions. |
| Two ZIP copies of the writeup package | They are copies of files in `Research_and_Experiments/Writeup`. |
| The ASD-STE100 standard document | This project does not redistribute it. |

## 9. Models and software

| Item | Use | Terms |
|---|---|---|
| DINOv2 ViT-L/14, through `timm` (`vit_large_patch14_dinov2.lvd142m`) | Image features for chip photos | Apache-2.0 |
| InstanSeg | Comparison method only | Apache-2.0 |
| U-Net design and watershed | Nucleus models and cleanup | Published methods |
| PyTorch, timm, scikit-learn, scikit-image, NumPy, SciPy, pandas, Pillow | Core software | Permissive licenses |
| Gymnasium and Stable-Baselines3 | Simulator control studies | Permissive licenses |

## 10. Limits

- The development groups are filename clues. They are not verified chip identifiers.
- The work uses one chip dataset and one fluorescence dataset.
- The small 640×480 format is not validated.
- Noise results use artificial noise only.
- The nucleus models do not find many of the smallest nuclei.
- Color nucleus images are not supported.
- The review workload numbers come from saved predictions. They are not measured time savings.
- The simulator result is not a physical chip result.
- The opened final sets must not be used to select future methods.
- An independent chip laboratory did not do a test of the tool.

## 11. License and credits

The code in this repository has the Apache License 2.0. Read the `LICENSE` file. Data, models and software from other owners keep their own terms.

Images and feature files derived from the Organ-on-a-Chip Image Dataset require credit to Movčana and colleagues. The license difference in section 7 is open. To be safe, apply CC BY-SA 4.0 terms to those derived files.

The project uses the Organ-on-a-Chip Image Dataset, BBBC data, DINOv2, U-Net, watershed and published simulator equations. The simulator equations come from [Pärnamets and colleagues, 2022](https://doi.org/10.1109/ACCESS.2022.3226623). The dataset paper is [Organ-On-A-Chip Image Dataset for Machine Learning and Tissue Model Evaluation](https://www.mdpi.com/2306-5729/9/2/28).

The saved experiment records supplied the results. The video narration is a synthetic voice.

## 12. Writing note

This file uses the short-sentence and consistent-term principles of [ASD-STE100 Issue 9](https://www.asd-ste100.org/). A sentence-length check of this file passed. A full dictionary check of this file was not done. Thus this file is STE-guided. It is not fully ASD-STE100 compliant.

The files in `Research_and_Experiments`, `Resources` and `code` are working records. Most of them do not use this writing standard. The review of the report is in `Research_and_Experiments/Writeup/ASD_STE100_review.md`.
