# Freeze decision (7 Oct 2026)

Mechanical rule fixed before the light-retraining results were known: see the docstring of `code/scripts/freeze_choice.py`.

## Chip photos

- fixed_3x3 minus whole: +0.0430, 95% range [+0.0083, +0.0827] -> MOVE to fixed_3x3
- fixed_5x5 minus fixed_3x3: +0.0129, 95% range [-0.0113, +0.0364] -> stay with fixed_3x3
- retrain_last_block minus fixed_3x3: -0.0715, 95% range [-0.1158, -0.0374] -> stay with fixed_3x3

**Frozen chip-photo candidate: `fixed_3x3`.** Small 640×480 photos: manual review (no validated rule).

Context (repaired A03 rule, same photos):

| candidate | pooled_AUC_A03_rule | weakest_cell_type | weakest_type_AUC |
|---|---|---|---|
| whole | 0.7562 | A549 | 0.768 |
| fixed_3x3 | 0.7992 | A549 | 0.805 |
| fixed_5x5 | 0.8121 | A549 | 0.781 |
| retrain_last_block | 0.7277 | HPMEC | 0.736 |

## Nucleus

B02 and B03 both stay as separate candidates, each with its best-count and best-outline settings (v2 search), seeds 0 and 1; the noise-trained variant (seed 0) is an extra labelled row. No selection among them on the final test.

## Not decided here

Whether C01/C03 appear in the submission, and the code licence: user decisions (see `Evidence/2026-10-07_release_rights_check.md`).