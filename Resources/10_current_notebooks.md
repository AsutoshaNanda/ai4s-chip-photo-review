# Current competition notebooks

## Notebook list visible on 2 October

The live competition All list showed these ten notebook links on the page we checked. This records that page at that time, not all private or future notebooks. Only the notebook that checks image-acquisition groups had its contents newly reviewed today. None was rerun.

| Visible notebook | Inspection status |
|---|---|
| [Neural MEA Dose Forecast](https://www.kaggle.com/code/loxigicck/neural-mea-dose-forecast) | Seen in the list only; data and results unverified |
| [External Clinical Action Replay](https://www.kaggle.com/code/loxigicck/external-clinical-action-replay) | Seen in the list only; do not assume it supplies valid data for controlling chips |
| [Chip Decision Contract Map](https://www.kaggle.com/code/loxigicck/chip-decision-contract-map) | Seen in the list only |
| [BioFluidNet-OoC: Multimodal Phenotypic Profiling](https://www.kaggle.com/code/avikdas567/biofluidnet-ooc-multimodal-phenotypic-profiling) | Earlier 26 September review below; current source not checked again |
| [Neural organoid phenotype audit across labs](https://www.kaggle.com/code/loxigicck/neural-organoid-phenotype-audit-across-labs) | Seen in the list only; suitability as independent data unverified |
| [NeuroChip Twin v2 demo](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-v2-demo) | Seen in the list only; not treated as the version reviewed earlier |
| [Audit acquisition groups in organ-on-chip images](https://www.kaggle.com/code/loxigicck/audit-acquisition-groups-in-organ-on-chip-images) | Version 3 visible text, code sections, and saved output inspected; not rerun |
| [Biofluidnet V2 Ooc Physics Informed Deep Learning](https://www.kaggle.com/code/limbasu/biofluidnet-v2-ooc-physics-informed-deep-learning) | Earlier review below; current source not checked again |
| [AI4S Organ-on-a-Chip Cellular Phenotyping](https://www.kaggle.com/code/liokingo/ai4s-organ-on-a-chip-cellular-phenotyping) | Earlier review below; current source not checked again |
| [Neurochip Twin AI4S Reproducible Synthetic Demo](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-ai4s-reproducible-synthetic-demo) | Earlier review below; current source not checked again |

### Finding from the new focused check

The notebook about image-acquisition groups checks IDs in the data sheet and groups in the ZIP file list. It clearly says that filename beginnings are clues to how images were collected, not IDs for physical chips. It reports numerical differences between a clean run and its earlier reference. For its broader comparison with matching conditions, it also says the uncertainty range includes zero, meaning no difference is still possible. The web reader could not retrieve the linked full code repository here. The full model experiment was not repeated.

**Decision:** Cite this earlier work. Do not call checking filename beginnings alone our new contribution, or copy a participant's measurements into our log. Before using a component, check its source/version/terms and rerun it. Continue testing a separate benefit in the amount of review work and use valid independent tests.

## 26 September reviews retained below

These are the four notebook versions originally inspected. Their dates, limits, and reported observations describe that earlier state.

**Snapshot: 26 September 2026.** The live Code list showed four notebooks. We inspected source views and saved outputs; none was rerun. Displayed running times belong to those saved runs.

All four notebook pages displayed Apache 2.0 licensing. Check separate data, model, and linked-repository terms before reuse.

## 1. Biofluidnet V2 Ooc Physics Informed Deep Learning

[Lim Ba Su, version 5](https://www.kaggle.com/code/limbasu/biofluidnet-v2-ooc-physics-informed-deep-learning)

- **What it does:** Combines simulated chip flow, tables of cell measurements, and responses to treatment.
- **Data:** Clearly stated to be computer-simulated. The notebook states its assumptions and limits.
- **Testing shown:** Keeps test compounds out of training; separately tests unfamiliar flow settings and ways that treatments act.
- **Useful parts:** Comparison tables, testing with components removed, saved results, and clear limits.
- **Limit:** Its model and simulator use related mathematical rules. Success here does not establish accuracy on real chips.
- **Our action:** Borrow how the tests are organized. An approach using simulation would also need tests with a different simulator or real measurements.
- **Displayed run:** About 28 minutes on T4 GPU resources, not a runtime guarantee.

## 2. AI4S Organ-on-a-Chip Cellular Phenotyping

[Lio Kingo, version 2](https://www.kaggle.com/code/liokingo/ai4s-organ-on-a-chip-cellular-phenotyping)

- **What it does:** Loads saved trained-model files and predictions, then shows category scores and a map of image features.
- **Data named:** BBBC041. The [provider](https://bbbc.broadinstitute.org/BBBC041) identifies this as malaria-infected human blood smears.
- **Useful parts:** Error tables for each category and visual summaries.
- **Limit:** The displayed notebook does not repeat training from the original images. Results on blood smears do not prove that chip monitoring works.
- **Our action:** Reuse the reporting idea. Check how the original images were divided for training and testing, and make fresh predictions before trusting model quality.
- **Displayed run:** About 76 seconds on CPU.

## 3. BioFluidNet-OoC: Multimodal Phenotypic Profiling

[Avik Das, version 2](https://www.kaggle.com/code/avikdas567/biofluidnet-ooc-multimodal-phenotypic-profiling)

- **What it does:** Makes artificial treatment/flow data, compares models, and saves a prediction table.
- **Useful parts:** Simple comparison models, saved result files, and an illustration of the steps.
- **Specific concern:** The source learns how to rescale inputs from all rows before dividing the data. It divides individual rows just once, which does not establish performance on compounds absent from training.
- **Our action:** Learn data-preparation settings only from training data, and keep entire compounds out of training for the test. Do not copy the published scores.
- **Submission reminder:** Its prediction CSV, a table saved as text, is an output example. AI4S requires the full Writeup submission.
- **Displayed run:** About 55 seconds on T4 GPU resources.

## 4. Neurochip Twin AI4S Reproducible Synthetic Demo

[Francisco Angulo de Lafuente, version 3](https://www.kaggle.com/code/franciscoangulo/neurochip-twin-ai4s-reproducible-synthetic-demo)

- **What it does:** Generates artificial image sequences, measures changes, compares prediction methods, and saves tables and a report.
- **Useful parts:** A small CPU demonstration, saved predictions, and a clear separation between evidence from artificial data and evidence about biology.
- **Limit:** The displayed run uses artificial data. The notebook says its separate BBBC038 check tests only how images are divided into objects.
- **Our action:** Follow its careful reporting, without copying its numerical results or conclusions about biology. A model that uses changes over time needs real image sequences before making real-world claims.
- **Displayed run:** About 58 seconds on CPU.

**Rule for choosing what to reuse:** These are participant projects, not confirmed winning solutions. Use a component only when its purpose, license, inputs, and independent test are clear.

