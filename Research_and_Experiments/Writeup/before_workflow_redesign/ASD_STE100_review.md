# ASD-STE100 review, 9 October 2026

**Verdict: the revised report is guided by STE. Full compliance is not verified.**

ASD-STE100 controls vocabulary, word meanings and grammar as well as sentence structure. Short sentences alone do not establish compliance. The current [official FAQ](https://www.asd-ste100.org/STE_faq.html) explains this distinction and the use of technical terms. The [official Issue 9 standard](https://www.asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf) is the primary reference.

## Checks and corrections

| Area | Action in this revision | Status |
|---|---|---|
| American spelling | Changed gray, color, labeling, license, center, program and micrometer spellings in editable text. | Checked for the listed words. |
| Sentence length | Tested descriptions against 25 words and numbered instructions against 20. | See exact counts in `ste_style_check.json`. |
| Paragraph length | Tested for more than six sentences per paragraph or caption. | Passed the stated editing check. |
| Instructions | Split the scorer-and-sort step and other sequential actions. Moved the download condition before its command. | Corrected inspected cases. |
| Word use | Replaced selected uses of main, both, perform, figurative fall and check as a verb. | Selected corrections only. |
| Terms | Added definitions for model, feature, scorer, tile, nucleus and computing terms. | Project glossary added; not blanket dictionary approval. |
| Full vocabulary and grammar | Every word still needs a contextual check of its meaning and grammatical role. | Not completed. |

These checks do not measure a percentage of compliance. The automated test uses conservative whitespace counts and excludes code, URLs, headings and figure-number labels. A human must still review sentence meaning, noun groups, technical-term eligibility, verb forms and every dictionary entry in context. Existing scientific output images retain their original labels as quoted outputs.

Direct PDF retrieval returned HTTP 403. Indexed excerpts from the official PDF and the current official FAQ were available. The review therefore does not claim a complete review of all dictionary entries or every rule.

## Examples corrected

- The original step combined application of the scorer with sorting the photos. The revision gives each action its own step.
- The original text used British spellings. The revision uses American spellings in editable text and corrected generated labels.
- The original text used a form of “fall” for a lower score. The revision uses “decrease.”
- The original report called itself uncertified. The revision states the actual review limit: full compliance remains unverified.

## Evidence precision corrected while making the new graphs

| Observation | Evidence quality | Meaning | Decision and next action |
|---|---|---|---|
| The B07 color F1@0.75 values are not all exactly zero. Across the eight saved model/setting/seed summaries, they range from 0 to about 0.000295. | Direct saved JSON for both B07 seeds. | The earlier zero wording overstates the rounded values. Performance is still extremely poor. | Report below 0.001. Retain manual review. No new model selection follows from this correction. |
| C03 direct count error is stored as 7.255. | Direct corrected input-only result JSON. | Earlier text used the rounded 7.25 value. | Use 7.255 for precision. The predicted route remains worse at 12.09. |

`more_figures_verification.json` records the new graph values and source hashes. This writing revision did not train a model, tune a model or create new test data.
