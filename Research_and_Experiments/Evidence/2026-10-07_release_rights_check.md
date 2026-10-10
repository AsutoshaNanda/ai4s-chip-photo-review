# Release and rights check — 7 Oct 2026 (G01 / F04)

**Later check on 7 October:** the official overview and rules were read again in the browser. Writing supported results can proceed now. Simulated data and simulation projects are permitted, but their limitations and resource use must be disclosed. The account shows accepted Kaggle rules and an in-progress draft. Separate registration-form completion is unverified. The current displayed deadline is **10 October 2026, 9:29 PM IST**. See [the current write-up checklist](../22_Final_Results_Before_and_After.md#section-7-can-we-write-and-submit-the-write-up). This live rules check does not resolve the dataset/licence questions below.

Read live today: the [competition rules](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/rules) (browser), the [Zenodo record 10203721](https://zenodo.org/api/records/10203721) (API), and the Hugging Face model pages for DINOv2. Not legal advice; open items are listed at the end.

## What the rules require (competition-specific rules, sections 2–4, plus Kaggle foundational rules, section 6)

| Rule | What we must do |
|---|---|
| Public datasets, open models and pretrained models may be used if licences and attribution are observed | Keep every licence and credit in the report |
| Major data sources, access methods, licences and processing must be disclosed; material third-party models and software must be named with source and purpose | Add a "Data and models used" table to the report (below) |
| Prior work must be disclosed and the new work identified | State what existed (published datasets, DINOv2, U-Net design, InstanSeg) and what was added here |
| Open-source code used must carry an OSI-approved licence that does not limit commercial use; publicly shared competition code is deemed licensed under such a licence | Our code: choose an OSI licence (user decision, below). No non-commercial code in the pipeline |
| Joint ownership of submitted materials with the organizers (for evaluation, exhibition, promotion); third-party material keeps its own licence | Tell the user; do not put third-party material in the package that cannot be shown publicly |
| Fabricating results is prohibited; test or validation data may not be hand-labelled | Report negative results and limitations honestly. Image F01 is now complete; do not use the opened final set to choose new variants. RL comparison claims need the later scoring repair |

## Data and models in the pipeline

| Item | Licence found | Used for | Decision |
|---|---|---|---|
| Organ-on-Chip photos (Zenodo 10203721) | **Live API record: CC-BY 4.0.** The paper says CC-BY-SA. The conflict is unresolved at the source | Chip-photo experiments (A, X) | Credit the authors. Do not redistribute the dataset. Any example photos shown in the video/report get attribution; to be safe under the stricter reading, release shown derived images under CC-BY-SA |
| BBBC039 | CC0 | Nucleus training/validation (B) | OK |
| BBBC038 (Kaggle DSB 2018 stage 1 train) | CC0 | Outside-data test (B07) | OK; 43 copies of BBBC039 photos removed |
| DINOv2 ViT-L weights (`timm/vit_large_patch14_dinov2.lvd142m`, `facebook/dinov2-large`) | **Apache-2.0** (both model pages, read today) | Chip features (A02, X01) | OK; exact file hashes recorded by `model_versions.py` |
| InstanSeg code and `single_channel_nuclei` weights | Apache-2.0 | Modern-model comparison | OK to use; **training-data exposure**: its README names BBBC038/DSB, so results are labelled "possible prior exposure" |
| StarDist 2D_versatile_fluo | not used | — | Trained on a DSB2018 subset: likely saw our images |
| Cellpose-SAM | not used | — | Training data CC-BY-NC: fails the licence check |
| JUMP-CP ORF profiles + metadata | recorded earlier as BSD-3 (metadata repo); data on the open Cell Painting Gallery | C02 | OK to use; confirm the data licence line before naming it in the report |
| RxRx1 | non-commercial / share-alike (earlier check) | C01 | **Do not include in the submitted package** (C01 stays in the research record only) |
| Allen Cell DIC→lamin B1 stacks | research/non-commercial terms, redistribution limits (2 Oct check) | C03 | **Do not redistribute.** Whether C03 appears in the submission at all is a user decision; if shown, cite as research use, no data in the package |
| Python libraries (PyTorch, timm, scikit-learn, scikit-image, NumPy, pandas, SciPy, Stable-Baselines3, Gymnasium) | permissive (BSD/MIT/Apache) | all | OK; versions recorded in `model_versions.py` output |

## Open items (need the user)

1. **Our own code licence** (OSI-approved, commercial use allowed, e.g. Apache-2.0 or MIT): your choice; the rules require one when code is public.
2. **C03 (Allen) and C01 (RxRx1):** include in the submission, or keep as research-only records?
3. **OOC licence conflict** (CC-BY vs CC-BY-SA): handled conservatively above; a precise answer would need the dataset authors.
4. **Joint ownership clause:** the user should read section 2 of the rules before submitting.
