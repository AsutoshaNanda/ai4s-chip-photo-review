# Data and model checks

Sample-detail measurements: **26 September 2026**. Plan/source update: **2 October 2026**. The original check files are unchanged, and the checks were not rerun. Source descriptions and measurements made directly from files are shown separately below.

See [the resource guide](../Resources/19_Resource_Readiness.md) for what is ready now. The OOC file list and BBBC039 page were checked again. The OOC API could not be refreshed, so the earlier license disagreement remains unresolved. Allen offers a new C03 lead for matching training-image pairs, but the files and sharing terms still need checking. No new images or trained-model files were tested.

## Real organ-on-a-chip dataset: sample details checked directly

Source: [Zenodo record](https://zenodo.org/records/10203721). Local originals: [spreadsheet](Evidence/OOC_datasheet.xlsx) and [record JSON](Evidence/zenodo_10203721_record.json). Analysis: [audit JSON](Evidence/ooc_metadata_audit.json), [archive inventory](Evidence/ooc_archive_inventory.json).

The spreadsheet's MD5 file-check value is `a3d4875e3da4da83bac45c0ce727cb40`, matching the provider's value. Only the last 1,048,576 bytes of the 6,710,767,405-byte archive were downloaded. The HTTP 206 partial-download response and ZIP64 file list confirm that all 3,237 entries in the list were read. The actual images were not opened.

| Check | Measured finding | Consequence |
|---|---:|---|
| Nonempty image records | 3,072 | Ignore blank formatting rows; the worksheet's 3,119 rows are not the image count |
| Unique IDs | 3,072 | No repeated IDs in the sample details |
| PNG files in archive file list | 3,072 | All sample-detail IDs match archive filenames |
| Published train / val / test groups | 2,130 / 286 / 656 | Save these original training, settings-checking, and final-test assignments before any approved regrouping |
| Good / bad | 1,727 / 1,345 | Folder labels agree with spreadsheet decisions 1 / 2 |
| Missing exact hours after seeding, when cells were placed | 2,244 | Do not assume there are complete sequences over time |
| Missing seeding density, how many cells were placed in a given space | 344 | Keep a record of what is missing; learn any way of filling gaps from training data only |
| Missing flow rate | 859 | Do not claim a flow-response result for the full dataset |
| Filename beginnings, called prefixes | 59 | Their biological meaning is still unverified |
| Prefixes in more than one published data group | 59 | Check for related samples across groups before claiming independence |
| Prefixes in all three data groups | 49 | The published way of balancing groups does not prove that chips were kept separate |
| Repeated CRC32 file-check value plus unpacked-size pairs | 0 | This weak check found no possible duplicates; identical-pixel and near-duplicate checks are still not done |

These measurements support the availability of data and show possible problems with testing. They are not model experiments.

### Counts for each cell type

| Cell type as stored | Good | Bad | Total |
|---|---:|---:|---:|
| A549 | 537 | 238 | 775 |
| CACO | 109 | 237 | 346 |
| HPMEC | 798 | 664 | 1,462 |
| HSAEC | 163 | 81 | 244 |
| HUVEC | 15 | 92 | 107 |
| NHBE | 105 | 33 | 138 |
| Total | 1,727 | 1,345 | 3,072 |

HUVEC has only 15 good examples, making precise estimates for that group difficult. Report how many samples each score uses and its uncertainty range. If a test group contains only one category and a score cannot be calculated, show it as unavailable, not zero or silently left out.

### Limits of the labels and sample details

- Quality labels combine sample appearance with image-taking problems such as blur. Do not turn good/bad into alive/dead or effective/ineffective treatment. [Paper](https://www.mdpi.com/2306-5729/9/2/28)
- The columns are image ID, cell type, seeding density, hours after seeding, day, decision, and flow rate. Seeding means placing cells; density describes how many were placed in a given space. There is no explicit chip, donor, or run column.
- Filename beginnings that look like dates are only possible clues for grouping. Confirm their meaning before treating them as experiment IDs.
- Flow-rate text uses several spellings for what appears to be the same unit. Record how the text is read and converted, and keep the original value. A missing number does not mean zero flow.
- Never give the model the folder path containing the label, the filename, or the spreadsheet's decision as input. Use cell type or cell-growing details only in comparisons clearly marked as using them.

### License conflict

The [paper](https://www.mdpi.com/2306-5729/9/2/28) says the dataset is CC-BY-SA. The [downloaded API record](Evidence/zenodo_10203721_record.json) says `cc-by-4.0`. Do not silently choose the less restrictive statement. Keep the credits and settle the exact terms before choosing how to share the final files. This research does not explain away the disagreement.

## Other dataset candidates

| Dataset | Verified source facts | What it can test | Remaining check |
|---|---|---|---|
| [BBBC039](https://bbbc.broadinstitute.org/BBBC039) | 200 images of the DNA signal, masks marking separate objects, published training/test groups, CC0 | Separating nuclei and counting them accurately | Read colored masks correctly, open examples, and check whether already-trained models have seen them |
| [BBBC038](https://bbbc.broadinstitute.org/BBBC038) | Varied nucleus images, published object markings, CC0 | Different imaging conditions | First remove images shared with BBBC039 and any known model-training overlap |
| [RxRx1](https://www.rxrx.ai/rxrx1) | Images after experimental changes and repeated lab batches | Compare cell appearance and behavior across batches | Exact chosen file, license, origin of the model that made the image measurements, and outcome to measure |
| [JUMP profiles](https://broadinstitute.github.io/jump_hub/howto/notebooks/11_retrieve_profiles.html) | Downloadable tables of cell measurements and different processing options | Match repeated versions of an experiment across batches | Prevent test information entering through data preparation applied to the full dataset; use a limited subset |
| [Skin-on-chip study, 2026](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2026.1787959/full) | A task-specific process that marks regions in brightfield images and checks quality | A possible future test specific to chips | The exact usable download and license were not checked |

BBBC object masks do not prove results for brightfield chip quality, whole-cell outlines, harmful drug effects, or organ function. Chip-quality labels do not test whether nuclei are outlined correctly. Keep these tasks and result tables separate.

## Model options and checks before use

| Candidate | Why test it | Important limit |
|---|---|---|
| Small already-trained category model, for example MobileNetV3 or ResNet18 | Quick comparison for A using an existing model through current [Torchvision](https://docs.pytorch.org/vision/stable/models.html) tools | Record the exact weight enum, the named choice of model files, image-preparation steps, origin, rights, and training exposure; no local time or performance has been measured |
| Brightness cutoff plus watershed, a method for splitting touching objects | An easy-to-inspect, independent comparison for outlining objects | May struggle with uneven brightness or touching objects; measure this |
| Small U-Net image model trained from the beginning | Clear record of the training done here | Is not automatically a new idea or better than current methods |
| StarDist | Predicts nucleus shapes as separate objects | Check its shape assumptions and whether earlier DSB training included test images |
| Cellpose 4.2 family | Current maintainer lists Cellpose-SAM updates and DINO-based options | Maintainer notes CC-BY-NC training data; the code license alone does not clear model/data use; check which terms apply instead of assuming use is banned |
| InstanSeg | A separate method for outlining objects, with efficient prediction and a clear model list | Match the image type and real size per pixel; inspect the license and training data for each exact saved model |
| Segment Anything for Microscopy | Corrections made with user input could reduce labeling work | Needs a real correction task and measured effort; automatic outlining quality is a separate question |

Sources: [StarDist](https://github.com/stardist/stardist), [Cellpose](https://github.com/MouseLand/cellpose), [InstanSeg](https://github.com/instanseg/instanseg), [microscopy SAM paper](https://www.nature.com/articles/s41592-024-02580-4).

Choose at most two AI candidates for the first small trial. Published model rankings do not choose the winner for us.

## Before any model is accepted

Record the software version, model ID, file-check value for the trained model, recorded training sources, required image scale and signals, data terms, and whether the model may already have seen our test images. State when prior exposure is unknown. Do not call a model with unknown exposure a clean independent test.

Use the current official programming interface when building the code. This folder describes experiments, not a tested software setup. It does not recommend copying installation commands from old competition code repositories.
