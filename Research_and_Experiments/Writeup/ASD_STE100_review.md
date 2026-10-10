# ASD-STE100 review, updated 10 October 2026

**Verdict: the report is STE-guided, but it is not fully ASD-STE100 compliant.** The full standard was obtained and selected dictionary entries were checked in this pass. This conclusion replaces the earlier statement that the source PDF could not be read.

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
| Full vocabulary and grammar | The full source and selected entries were inspected. A word's approved grammatical role is essential. | Remaining deviations are present; no full pass. |

These checks do not measure a percentage of compliance. The automated test uses conservative whitespace counts and excludes code, URLs, headings and figure-number labels. A human must still review sentence meaning, noun groups, technical-term eligibility, verb forms and every dictionary entry in context. Existing scientific output images retain their original labels as quoted outputs.

The official 434-page Issue 9 PDF was downloaded successfully on 10 October 2026. Its title, issuer and issue date were checked. Rules 1.1, 1.2, 5.1-5.4, 6.3 and 6.6 were read from the full source. Selected dictionary entries were also checked. The entire report has not passed a complete contextual word and grammar audit.

The descriptive sentence limit is in Rule 6.3. Rule 6.1 concerns the order in which information is given.

## Important findings from the full standard

- Rule 1.2 approves `test` as a noun, not as a general verb. The earlier substitution of a verb form of `check` with a verb form of `test` did not establish compliance. Clear verb uses in the prose were rewritten with `examine`, `compare`, or noun constructions such as `do a test`.
- Selected uses of `achieve`, `retain`, `provide` and `whether` were changed without changing the reported results.
- General `final` as an adjective is not approved in the dictionary. The phrase `final decision` remains in the illustrations. This is one concrete reason not to describe the full artifact as strictly compliant.
- Technical terms can qualify under the standard's technical-noun and technical-verb rules. They are not automatically approved merely because they sound scientific.

Source locations in the downloaded PDF: rules at PDF pages 46, 87-90, 95 and 101; selected entries at PDF pages 154, 198, 250, 344, 360, 362, 422 and 428. These are file page numbers, not the standard's printed section-page labels. The source PDF is kept outside the upload package and is not redistributed with the report.

## Examples corrected

- The original step combined application of the scorer with sorting the photos. The revision gives each action its own step.
- The original text used British spellings. The revision uses American spellings in editable text and corrected generated labels.
- The original text used a form of “fall” for a lower score. The revision uses “decrease.”
- The original report called itself uncertified. The revision states the actual result: remaining deviations prevent a claim of full compliance.

## Evidence precision corrected while making the new graphs

| Observation | Evidence quality | Meaning | Decision and next action |
|---|---|---|---|
| The B07 color F1@0.75 values are not all exactly zero. Across the eight saved model/setting/seed summaries, they range from 0 to about 0.000295. | Direct saved JSON for both B07 seeds. | The earlier zero wording overstates the rounded values. Performance is still extremely poor. | Report below 0.001. Retain manual review. No new model selection follows from this correction. |
| C03 direct count error is stored as 7.255. | Direct corrected input-only result JSON. | Earlier text used the rounded 7.25 value. | Use 7.255 for precision. The predicted route remains worse at 12.09. |

`more_figures_verification.json` records the new graph values and source hashes. This writing revision did not train a model, tune a model or create new test data.
