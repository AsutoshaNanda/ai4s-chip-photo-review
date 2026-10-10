# Project decision

Date: 26 September 2026. Status: **project choice depends on checks; model performance has not been measured**.

**Work-plan update, 3 October:** The user requested all experiments, with important work first and independent work happening together. Use [the complete schedule](10_All_Experiments_Parallel_Schedule.md). Evidence decides the project focus; that does not restrict us to exploring one option at a time. The requested start was 2 October; all unchecked setup now carries into 3 October. The current GPU provider/count, budget, account allowance, and available daily hours are unverified. Dates are targets, not assumptions about how long work takes.

## Recommended first small trial: chip-image quality review

Build a tool that helps a researcher review brightfield organ-on-a-chip images, made with light passing through the sample. It would show the image, a predicted sample-quality category, and an ordered list for human review. Save the prediction, model version, and review status in an output file.

The aim is to match the dataset's expert quality labels. This does not measure whether cells are alive and healthy, whether a treatment is harmful, whether an organ works, or whether something is safe for patients. The source's bad category combines cell appearance with problems caused while taking images. [Dataset paper, S04](02_Research_Library.md#s04)

**Proposed category:** Tool & Platform. A category declaration is required; this choice is our recommendation. [Official overview, S01](02_Research_Library.md#s01)

**Possible contribution:** show that a tested rule for ordering reviews catches more bad images for the same amount of review work, and still works when cell type and image-taking conditions change. A category-predicting model, a colored image showing where it focused, or an attractive screen alone would not prove this benefit.

## Compare the project options fairly

| Project option | Evidence in its favor | Main risk | Decision for now |
|---|---|---|---|
| A: Real-chip quality review | Public data directly fits the topic; sample details and image IDs match; papers show the task is already studied | Unresolved license disagreement, uncertain chip groups, quality labels that combine different issues | First trial to see whether it can work; moderate confidence that it is possible, no claim about measured improvement |
| B: Nucleus counting and review | Small BBBC039 dataset, masks marking each object, clear CC0 terms, established comparisons | Many existing methods; models may already have seen test images; only indirectly related to chips | A small trial alongside A and a separate project option; moderate to high confidence that it is possible after loading checks |
| C: Cell Painting treatment analysis | Rich public data about cell appearance and behavior, with tables of image measurements | Differences caused by lab batches, large downloads, uncertain outcome to measure and model rights | Check now; run C01/C02 trials once a clear biological question and test using separate sample groups exist |
| D: Predict fluorescent images from brightfield images | An established research area that fits the competition | Needs correctly matched images; convincing-looking outputs can mislead | Check now; run C03 once suitable pairs and reference measurements for the intended later use are ready |
| E: Digital twin or flow-control simulation | Published control research supports studying this option; a digital twin is a computer model of a real system | Testing with the same simulator can favor the rules built into it; no real measured outcome for prediction is ready here | R01 checks requirements now; R02 to R04 need a checked environment and separate confirmation before any claim about biology |

The sources and smallest tests behind these decisions are in [the research library](02_Research_Library.md) and [experiment guide](04_Experiment_Playbook.md).

## Checks that decide whether to continue

| Check | What counts as passing | What to do if it fails |
|---|---|---|
| Data | Files open correctly; labels mean what we claim; sources and rights are recorded | Fix loading or select the alternative; do not invent labels |
| Independence | Physical sample groups are known, or the claim is clearly limited to this dataset and a recorded test shows the effect of grouping by an unconfirmed clue | Drop claims about new chips, labs, or donors; decide whether the smaller project is still useful |
| Simple comparison | Compare AI with a simple method and any shortcut using sample details on the same groups kept out of training | Do not claim that AI adds useful value; investigate the failure |
| Practical value | An ordered review list or correction process improves a defined task with the same effort | Remove the benefit claim; keep only measured model results |
| New contribution | Show a new test, ability to handle changes, or practical benefit beyond the cited work | Reduce the claim; do not call an existing model a new method |
| Submission | Public code that others can run, a report, and a working video can be completed | Cut optional models and features first |

Checkpoint: review unmet requirements for option A after the first focused day. Option B and independent source checks continue alongside it. Do not start training that depends on an unmet requirement. Keep the evidence and the exact requirement needed to continue. This is a planned checkpoint, not a prediction of how long checks will take.

**Related-work update:** A participant already provides an [acquisition-prefix check](https://www.kaggle.com/code/loxigicck/audit-acquisition-groups-in-organ-on-chip-images), which examines filename beginnings as clues to image groups. Its source and saved output were inspected on 2 October, not rerun. Our proposed addition must therefore show an actual, separate benefit; checking filename beginnings alone is not a supported claim of a new idea. C03 now has [Allen training-data leads](../Resources/19_Resource_Readiness.md), but the image pairs and intended-use rights still need checking.

## What could overturn this recommendation?

- Failure on separately held-out sample groups matters more than a strong result from randomly divided images.
- If a model using sample details alone matches the image model, check whether the image model actually uses biological information. This warns of a possible shortcut; it does not automatically prove the image model is useless.
- If a simpler category model reliably beats the already-trained network, use the simpler model where it serves the intended task.
- If the review list misses failures in a cell type, stop claiming broad reliability until a separate test confirms a warning or review rule that catches them, or the claimed use is restricted.
- A meaningful, repeated benefit from the nucleus-review process can justify choosing B even though A matches the topic more closely.
- An accessible, independent outside chip dataset could strengthen the tests for A. A related paper without downloadable data cannot.

No source establishes which option will win. Make the final decision from actual runs using [the selection rules](05_Validation_and_Selection.md).

## Preserve independent alternatives

Within the chosen task, keep predictions from the simple comparison method and the AI method. They may fail on different examples. A backup is useful only after its performance and any rule for switching to it have been checked independently.

Options A and B solve different tasks. Their scores cannot be averaged, and B cannot directly replace a failing chip-quality model. Keep B as another project option, not as an untested member of a combined model.
