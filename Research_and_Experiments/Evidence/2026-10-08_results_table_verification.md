# Verification of 22_Final_Results_Before_and_After.md, 8 October 2026

## What was checked

| Check | How | Result |
|---|---|---|
| Structure | All 36 IDs present in Section 2 (plus two new rows) and Section 4; Section 1 keeps the 6 October layout (33 rows, F01–F04 as one row); 13 extra rows in Section 3; links and anchors; column count of every table row | No problems |
| Stored paths | `code/scripts/f04_trace.py` | 192 paths found, 0 missing |
| Numbers exist in the named files | `code/scripts/f04_numbers.py`: 533 decimal numbers in the Result and Had-to-beat cells of Section 2 and 122 in Section 3; whole numbers 41 and 44 | Each number not found was traced by hand (below) |
| Independent recomputation | R02–R04 recomputed from the raw `episodes.csv` (every mean, per-seed value, out-of-range share, settled error, worst case, paired difference and share of scenarios; confidence limits re-drawn with another resampling seed agree within 0.002); C02 means, ratio and difference from the five fold files; A11 and R05 values at 800 photos from the stored curves; E05 sweep table from `stage_sweep.csv`; E06 from the overlap and warning-miss curves; F03 nucleus comparison, F03 step times, G03 timings, fold wall times | All agree except the items under Corrections |

## Corrections made

1. **R02 worst cases (file 22 row R02 and Section 6, and the repaired-evaluation record).** The highest-error episode and the most-time-outside-range episode are different episodes, but they were written as one ("2.826 µm, 25.3% outside"). Stored values: highest-error PPO episode 2.166 µm (0.0% outside) in the training ranges and 2.826 µm (1.3% outside) in the harder ranges; most time outside in any PPO episode 11.0% and 25.3%. Rule controllers: highest error 2.346 / 3.213 µm (PI) and 3.122 / 4.419 µm (FF+PI); most time outside 35.7% / 38.0% and 60.7% / 49.0%. The conclusions do not change.
2. **F03 nucleus fold 0 retrained:** outline-setting count error 11.49 → **11.48** (stored 11.4848; 11.485 was rounded twice).
3. **G03 (file 22; one line in file 16):** the A100 figure of 167 photos/s is from the real A02 feature run (`feature_timing.json`, 166.76); the speed test gives 145.5. The Mac figure of 5.9 photos/s has no stored measurement file (it comes from the 6 October table) and is now marked so. Light-retraining time on the T4 was measured at 1.9, 2.1, 2.3, 2.6 and 2.4 h per fold (mean 2.3 h); the text said "about 2.4 h". The Modal bill figures are provider figures and are not stored in a run folder.
4. **Stored-at lists** completed where a row quotes numbers from another run: A01 (format-only review area, 0.5436), X01 (repaired-rule numbers), B02 (first run, B01 reference), G03 (A02 timing runs), E09 (`model_x01_3x3.json`).
5. E05 now says the marker-spacing rows are the count setting.

## Numbers that were not in the named files and why

* Means or ratios made from stored fold values: C02 (0.860, 0.059, 0.404, 0.006, 0.084, 0.947, 0.049, ratio 9.8, difference 0.456).
* Values read off stored curves at 800 photos: A11 and R05 (0.810, 0.794, 0.767, 0.755).
* Numbers quoted from another row's run: B02 6.73 / 0.787 (B01), B02 modern, B03, B05, B07 reference values, X01 repaired-rule numbers, F02 review areas.
* Version strings picked up as numbers (F03, E10, E11); percentages written without a percent sign in E05.

## Limits of this verification

* A number found in a named file shows it exists there, not that it sits in the right place; at three decimals a coincidence is possible. That is why the R-series, C02, A11/R05, E05, E06 and F03 were recomputed rather than only searched.
* Not every number in Sections 1 and 4 to 7 was recomputed. Section 1 is the historical 6 October table and was left unchanged.
* The Mac speed (5.9 photos/s) is unverified and the Modal billing figures are unreconciled.
* Values stored with four decimals are rounded half up in the tables (for example C02 fold 4: 0.1065 → 0.107).
