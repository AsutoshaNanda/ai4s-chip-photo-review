# Coverage audit: did we skip any experiment or step? (Tue 6 Oct, ~00:20 IST)

Asked by the user: "verify did you skip any exps or anything".

## 1. Whole experiments: none skipped

A script counted every experiment ID in each planning file.

| File | Experiment IDs found | Note |
|---|---|---|
| Old schedule (file 10, older section) | 36 / 36 | — |
| Approved plan (file 11) | 36 / 36 | — |
| Live status table (file 10, top) | 36 / 36 | — |
| Playbook (file 04) | 30 / 36 | Expected: the 6 extras (X01, R01–R05) were defined later in file 10, not in the playbook |
| Old E0–E6 list | all 7 mapped in the plan | — |

## 2. Steps inside experiments: what was missing or only implied, now added

The playbook's steps were compared with the plan line by line. Found:

| ID | Step in the playbook / schedule | Was it in the plan? | Action |
|---|---|---|---|
| **G03** | Real-data pilot: 16 real training examples, one short training pass, **save → reload → same predictions**, data-preparation time measured separately from prediction time | **No.** Tonight's speed test used random practice data only. **This was a real gap.** | Added: real-data pilot for each pipeline (chip photos, nuclei, RL) on Tue, on each machine we use |
| A01 | Tables of correct/wrong per class; share found per cell type; running time | Implied, not in the code yet | Added before A01 runs. The cut-off is chosen on practice data inside each fold (rule recorded), never fixed in advance |
| A04 | Small crop-position shifts; check whether cropping removes information the label depends on | Not spelled out | Added |
| X01 | Combine with sample details only after A07 shows they help | Not spelled out | Added |
| B04 | One justified training change if the stress tests show a weakness | Not spelled out | Added |
| C03 | Baseline that makes the same measurement **directly from the plain photo**, without the predicted glowing image | Not spelled out | Added |
| R04 | Scaling learned on training runs only; no future information in inputs; combine with R03 only after each effect is measured alone | Partly (inputs, missing/noisy tests were listed) | Added |
| G01 | Licence of each **model file** (pretrained weights), not only datasets | Partly (only for B02) | Added for A02, B02, C03 |
| G01 | Resolve the chip-data licence conflict (paper: CC-BY-SA; website: CC-BY 4.0) | Recorded, not resolved | Resolving needs contacting the dataset authors. **Needs your OK** (no outreach without it). Meanwhile we don't re-share the photos. |
| F04 | Report how many comparisons were tried (more tries = more chance of a lucky score) | Not spelled out | Added |
| F04 | Recheck the competition discussion page before submitting; 200–300-word summary | Not spelled out | Added |
| — | Open questions for organisers (team-formation deadline, bonus wording) | Not in plan | **Needs your OK** to contact organisers; otherwise they stay recorded as open |

## 3. Changes to the old plan (flagged, all for a reason)

- Fixed cut-offs and fixed sizes are replaced by full curves and ranges (user rule, D46).
- One big modern model is added to A02 (user's standing rule).
- B01: settings are chosen inside 5 rotating plate-grouped folds of the 150 practice images. The playbook said "validation only" (50 images); rotating folds are a bigger, fairer practice test. The 50 test images stay locked. The first B01 run (tune on 100 training, report on 50 validation) is kept as a recorded run.
- Chip final test: whole filename-groups held out, sized like the authors' own test share (21.4%). The published split is kept for reference only. Decided from data, recorded in `code/splits/README_ooc.md`.

## 4. Status of all 36 at audit time

| Status | IDs |
|---|---|
| Started, measured parts | G01 (nuclei done; chip photos downloading, 81%), G02 (scoring tests, nucleus splits), G03 (Kaggle T4 speed; L4 not available), R01 (search started) |
| Running | B01 |
| Planned on their scheduled day (not skipped) | A01–A12, X01, B02–B07, C01–C04, R02–R05, F01–F04 |

Nothing is marked skipped, rejected or blocked. C04 and R02–R04 can end "blocked" only with written evidence. For R02–R04, the fallback is the labelled practice simulator, so they will still run.

## 5. New measured fact from tonight

The Kaggle L4 option is **not available to this account**. Two requests, one via the settings file and one via the push option, both ran on 2× T4 (`code/runs/g03_kaggle_l4*/`). Kaggle lane = free T4 ×2 only.
