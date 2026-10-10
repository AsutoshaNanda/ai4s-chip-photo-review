# Write-up checks, 9 October 2026

This file records checks performed while preparing the Kaggle draft. It does not record new training or new test data.

| Observation | Evidence quality | Supported or rejected explanation | Decision | Smallest next action |
|---|---|---|---|---|
| The clean rebuild has whole-photo review area 0.6795984239, not 0.675. X01 is 0.7025957830. | Direct saved JSON and prediction curves from `20261007-205052_F03_chip_final_eval_clean`. | The 0.675 number belongs to the earlier final evaluation. It must not describe the clean rebuild. | Use 0.680 vs 0.703 in the draft and figures. Keep the earlier number as history only. | Recompute curve areas from the saved CSV. |
| The clean automatic-only final HUVEC group has 51 photos: 39 bad and 12 good. | Direct saved JSON and prediction rows. | Reject the older draft statement that this group has three good photos. | Use the actual counts. Explain that a perfect score on this small group does not prove broad reliability. | Count labels by cell type from the prediction CSV. |
| The current final report includes the repaired RL evaluation and scorer rebuild. | Saved results, evaluation notes and source code. | Older statements that R02-R04 repairs or the scorer rebuild remain pending are stale. | Use the repaired simulator numbers and F05 success. Preserve F06 failure. | Compare plotted numbers with raw episode rows and the saved installation report. |
| The Kaggle account contains an existing empty draft titled AI4S Writeup. | Live browser observation. | A draft is present; no new draft is needed. | Fill the existing draft and save it without submitting. | Verify the rendered report, hosted images and saved draft status. |

## Language

Use ASD-STE100 Issue 9 as the writing guide. Use short sentences, active wording, defined technical terms and consistent names. Check sentence length as an editing aid. A full dictionary and grammar compliance review is not claimed.

Official reference: https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf

## Source priority

Use saved result files for plotted numbers. Use `final report.md` for current decisions. Use Section 2 and Section 3 of `22_Final_Results_Before_and_After.md` for historical comparisons and run locations. The repaired RL result supersedes earlier simulator scores.

## Figure and draft checks

- Recomputed final AUC and discrete review-area scores from saved predictions and curves. Values match the clean result JSON. The review-area score is the mean catch rate for review counts 1 through N, not a trapezoidal integral.
- Recomputed B02 count error and B03 outline-setting count error from their 50-image records. Values match.
- Recomputed R02 harder-condition means and worst single episodes from the episode CSV. Values match.
- Checked 51 unique ledger rows: all 36 original IDs, 13 extra items, F05 and F06.
- Checked the 224-word summary and eight figure references. No em dashes occur in the report. Prose sentence-length checks found no sentences above 25 words under the editing check. This is not full STE certification.
- Uploading the extra evidence ZIP was rejected by automatic approval review because the internal reports were outside the requested figure upload. The ZIP remains local. Only the requested report text and figures are being added to Kaggle.

## Final saved-page outcome

The Kaggle draft was saved with the full text, all 51 ledger rows, 19 rendered tables and eight hosted figures. Each figure reports a positive loaded image size. The editor text exactly matched the generated hosted Markdown before saving. The Submit button remained disabled. No final submission occurred. See `draft_validation.json` and `kaggle_render_check.json`.

## More figures and STE review, 9 October 2026

Six additional figures were generated from saved records or copied unchanged from saved scientific output. The document now has 14 figures, with sequential captions and all 51 original ledger rows preserved. No model training or new test data was used.

The partial STE review corrected 32 occurrences of the selected British spellings, selected word-use errors, combined action steps, and several passive sentences. The conservative text check has a maximum of 21 words in a checked sentence and 15 in a numbered step. No checked paragraph exceeds six sentences. These checks do not establish full dictionary or grammar compliance. See `ASD_STE100_review.md` and `ste_style_check.json`.

The new figure sources exposed two precision issues. B07 color strict scores are below 0.001, not all exactly zero; the manual-review decision is unchanged. C03 direct count error is stored as 7.255 and is now shown exactly. Its reference masks come from fluorescence measurements, not manual annotations. See `more_figures_verification.json`.

## Final expanded draft save, 10 October 2026

The expanded report was saved on Kaggle with exact editor-to-local text equality. All 14 displayed figures loaded with positive natural image sizes. The page has 20 rendered tables and keeps Submit disabled. The revised text preserves all 51 experiment/check rows. The local source and hosted Markdown match after image-URL substitution. Full ASD-STE100 compliance remains unverified; see the scoped language review.
