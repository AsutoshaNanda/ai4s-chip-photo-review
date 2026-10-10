# Data sources and checks before use

Updated: **2 October 2026**. The competition allows teams to choose suitable data. The live [overview](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview#data) encourages testing on real chips. Listing a source does not mean its files have been downloaded, its use is allowed, or a valid test has been done.

| Project option | Source | What is ready and what to do next |
|---|---|---|
| A: chip quality | [Zenodo OOC dataset](https://zenodo.org/records/10203721) | The spreadsheet and a check of the archive's file list are saved locally. The provider lists an image archive. The actual images, which physical samples belong together, and the recorded disagreement about the license still need checking. |
| B: nucleus counting | [BBBC039](https://bbbc.broadinstitute.org/BBBC039) | Provider page checked again: images, colored masks that mark nuclei, details of the training/test groups, and CC0 terms. Load and check all three file types together. Use the example linked by the provider to read the masks. |
| B07: another nucleus test | [BBBC038](https://bbbc.broadinstitute.org/BBBC038) | An earlier option; BBBC039 confirms that some images appear in both. Check the selected images and whether models trained earlier have seen them before calling this an independent test. |
| C01: measurements of cell appearance or behavior | [RxRx1](https://www.rxrx.ai/rxrx1) | The provider describes lab batches and repeated image locations. We still need to check a specific small file of measurements, its size, how it was processed, and its terms. |
| C02: processing tables of cell measurements | [JUMP profile tutorial](https://broadinstitute.github.io/jump_hub/howto/notebooks/11_retrieve_profiles.html) | File-list and processing instructions checked again. Earlier corrections may have used samples that we later choose for testing; check which samples they used. |
| C03: predict one image from its matching image | [Allen label-free training data](https://downloads.allencell.org/publication-data/label-free-prediction/index.html) | A new, specific source to investigate. Before training, check that matching image signals are real, how samples are grouped, which subset is small enough to use, and whether the intended use is allowed. |
| C04: an independent chip measurement to test | [Skin-on-chip paper](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2026.1787959/full) | An earlier lead, not a checked dataset that is ready to use. Obtain real reference measurements that match the task and files we can use. |
| R01 to R04: control | [Droplet-control research](https://arxiv.org/abs/2609.16369) | A research reference, not an available, tested practice environment. We need records of actions and their outcomes, or checked rules for how the system changes, plus a separate test. |

The [resource readiness guide](19_Resource_Readiness.md) gives the exact remaining checks and links. The [local data check](../Research_and_Experiments/03_Data_and_Model_Audit.md) keeps the counts measured on 26 September and explains their limits.

## Do these checks per source

1. Record the exact file/version, terms, file hash (a value used to identify the file), and intended use. A software license does not establish the rights for a model or dataset.
2. Inspect examples from the training group only, along with their reference answers. Keep labels, sample IDs, and related image pieces matched correctly.
3. Use the actual sample details to decide which samples are independent. An unchecked filename beginning is only a possible clue, not a chip ID.
4. Save the sample lists for training, choosing settings, adjusting confidence estimates, and final testing before selecting models. Any data preparation that learns from examples must use only the examples allowed by the chosen test plan.
5. Check for identical or nearly identical examples and whether an already-trained model has seen them. Keep an early test with unknown prior exposure separate from a test with no such overlap.
6. Record missing information as missing. Do not invent times, flow values, matching fluorescence images (images of glowing signals), or what happens after an action.

Do not call nucleus-test results proof that a method works on chips. Do not call predicted fluorescence a real measured answer, or a computer-made environment a verified biological system.

## Other sources to search, kept from the earlier guide

The organizer lists [CellNet](https://cellnet.hms.harvard.edu/), [BBBC](https://bbbc.broadinstitute.org/), [IDR](https://idr.openmicroscopy.org/), [CytoData](https://www.cytodata.org/), and [JUMP](https://jump-cellpainting.broadinstitute.org/). These are places to look for data, not datasets checked again in this update. The earlier [BBBC021](https://bbbc.broadinstitute.org/BBBC021) lead is still available for a separately defined treatment task. Its exact usage rights and the outcome to measure still need checking.
