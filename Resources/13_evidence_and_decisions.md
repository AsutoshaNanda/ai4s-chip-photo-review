# Evidence and decisions

**2 October update on where to record work:** The observations below keep their 26 September date. Put new evidence and decisions in [the central decision record](../Research_and_Experiments/06_Evidence_and_Decisions.md), alongside [2 October source checks](../Research_and_Experiments/Evidence/2026-10-02_source_checks.md). Do not treat the earlier schedule as completed work.

Research observations recorded on 26 September 2026. These are not model-test results.

| Observation | Evidence quality | What it supports or rejects | Decision and smallest next action |
|---|---|---|---|
| A video, code, and report are required | High: [official overview](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview) | Rules out a plan to submit only a file of predictions | Set aside time for all three required parts |
| Real chip data is optional | High: [host reply](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/discussion/737840) | Supports starting with public cell images | Inspect a small public dataset |
| BBBC039 has reference outlines and CC0 terms | High: [provider](https://bbbc.broadinstitute.org/BBBC039) | Supports an image-analysis test small enough to manage | Open 20 training images with their matching masks |
| Some BBBC039 images also appear in BBBC038 | High: [provider](https://bbbc.broadinstitute.org/BBBC039) | Rules out assuming the two datasets are independent | Check overlap before testing on the other dataset |
| Several notebooks use computer-simulated data | High for what the source contains; not proof of biological accuracy: [notebook review](10_current_notebooks.md) | Rules out treating their scores as real-chip evidence | Reuse only parts with a clear justification; test independently |
| BioFluidNet learns how to rescale inputs from all rows before dividing the data | High: source inspected, code not rerun | Reduces trust in the stated test | Learn data-preparation settings separately within each training group |
| Cellular Phenotyping uses saved predictions from BBBC041 | High for code; [dataset source](https://bbbc.broadinstitute.org/BBBC041) confirms blood smears | Does not support claims about monitoring chips | Require the work to be repeated from raw data before using it |
| Current Cellpose page says training data is for non-commercial use | High: [maintainer source](https://github.com/MouseLand/cellpose) | Rules out assuming unrestricted reuse | Check the exact model usage rights or use our own small model |

## Rule for future results

For every important result, record what happened, how strong the evidence is, which idea it supports or rejects, what decision changes, and the next test. Update this immediately.

- If a stronger test contradicts the plan, pause that option and investigate.
- Confirm improvements with a fresh run or independent data.
- Keep a separately tested alternative if it survives different failures.
- Reject a method with very serious failures unless a tested backup catches them.
- Never hide a failing group inside an overall average.

## Final selection, still pending

Chosen method: **not selected**.

Before submitting, link the runs that decided the choice, worst cases, usage-rights check, and check that the work can be repeated. Explain why the chosen method follows the strongest evidence. If it does not, resolve the conflict before continuing.

