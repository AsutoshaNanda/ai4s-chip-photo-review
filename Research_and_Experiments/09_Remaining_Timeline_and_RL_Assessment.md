# Remaining timeline from 3 October

Updated: **3 October 2026, Asia/Kolkata**. This continues the requested 2 October start and carries all unfinished setup into today. All **30 original plans plus six feature/RL entries** remain in [the complete register](10_All_Experiments_Parallel_Schedule.md), including the E0 to E6 mapping.

## Verified starting point

- You requested a 2 October start. The plan refresh resumed on 3 October. No previous target is marked complete without a recorded result.
- The inspected workspace contains the earlier source/metadata audit and planning files, but no recorded model runs. Work outside this workspace was not checked.
- The competition page, checked live on 2 October, gives **10 October 2026, 21:29:59 IST** as the cutoff. Its required package is a Kaggle Writeup with a category, public code, report, and a public demonstration video. [Official overview](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview)
- GPU provider/count, remaining quota, budget, daily work hours, independent external data, and operator availability are unverified. No values are assumed.

**Dates below are proposed targets, not predicted runtimes or evidence of feasibility.** Every dependent experiment waits for its data and evaluation requirements. No experiment is deleted to make the calendar look complete; unresolved work remains visibly uncompleted. Whether all full experiments fit can only be determined after the first pilots.

## Daily parallel plan

| Date | First-priority work | Work alongside it | Required record |
|---|---|---|---|
| **3 Oct, today** | Complete unfinished G01/G02/G03 checks and registration/compute review first. Then measure A01/B01 and start A02/B02 when ready. | All B07/C01 to C04/R01/R05 source checks; prepare variants, the run register, minimal demo/report, and F02 task definition. | Verified prerequisites, baseline outputs when available, actual timing/memory, and explicit pending work. No earlier completion or available hours are assumed. |
| **4 Oct** | Complete eligible A02/B02 pilots and begin A03/A04/B05 plus early confirmation. | A05/A06/A07/A11/A12/B03/B04/X01; C01/C02/C03 pilots and R02 when their own checks pass. A08/A09/B06 can start from saved eligible predictions. | One-change comparisons against fixed references; exact blockers for anything not ready. |
| **5 Oct** | Continue all eligible comparisons, weakest-group diagnosis, A08/A09/B06, and F01 repeats. | A10, eligible B07/C04 checks, C pilots, and R03/R04/R05; demo construction and reproduction rehearsal. | Overall and per-group evidence, cost, and keep/reject decisions. External data role chosen before use. |
| **6 Oct** | Finish comparisons that can still receive proper confirmation; select and freeze supported candidates. | Final feature combinations only after isolated effects are known; F02 workflow checks, F03 rehearsal, and report/video drafts. | Every one of the 36 entries has a result or a precise uncompleted status. Data checks are not reported as model results. |
| **7 Oct** | F01 locked final evaluation and eligible timed F02 checks. | Finish the selected demo and populate the report from saved evidence. | Final results, uncertainty, worst failures, and a decision ledger. Do not tune on final-test outcomes. |
| **8 Oct** | F03 clean reproduction and F04 claim/access checks. | Finish the video and technical report. | Reproduced numbers and a complete submission package. Remove any unsupported claim. |
| **9 Oct** | Check registration, eligibility, links, and claims, then submit. | Verify submission status and preserve the exact submitted version. | Confirmed submitted Writeup. This is our internal target. |
| **10 Oct** | Buffer for verified fixes. | Recheck the live deadline/status before **21:29:59 IST**. | Accessible, complete entry before cutoff. |
| **10 to 20 Oct** | Expert review period. Preserve the submitted version and prepare presentation answers. | Further work remains separately labeled; its eligibility is not assumed. | Clear explanation of usefulness, comparisons, and limits. |
| **20 to 30 Oct** | If selected, use the organizer's assigned final-presentation slot. | Demonstration and questions. | Present the measured contribution and known failures. |
| **Before November** | Check the official result. | Follow organizer instructions. | No invented placement or winning probability. |

The last three date ranges come from the [official timeline](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview#timeline), last checked on 2 October. Experiment priorities and daily targets are our proposed work plan.

## Suggested improvements to the earlier plan

1. **Use the complete resource map before each run.** [Resource readiness](../Resources/19_Resource_Readiness.md) states what was checked and what still needs proof, including a new paired-image lead for C03.
2. **Reuse fixed computation carefully.** Cache compatible image features and saved predictions; run scoring and review analysis while another model trains. Changed encoders or preprocessing require new features. Time savings are not yet measured.
3. **Test the real contribution early.** A09/B06 measure review value while model comparisons continue. A published or participant method is prior work, not evidence that our version improves anything.
4. **Strengthen originality checks.** A newly inspected participant notebook already audits OOC acquisition prefixes. Cite it and test a distinct benefit; do not describe prefix auditing alone as a new contribution. [Related audit notebook](https://www.kaggle.com/code/loxigicck/audit-acquisition-groups-in-organ-on-chip-images)
5. **Build delivery alongside research.** Start the runnable entry point and report outline now, leaving results blank. This protects a complete entry without dropping experiments from the register.
6. **Choose compute from actual measurements.** Compare total run cost, memory, storage/transfer, and interruptions. [Compute guide](../Resources/08_tools_and_free_compute.md). No provider has been selected or declared cheapest.

These are proposals to improve execution and evidence quality. They are not measured model gains. A strong result from any route can change priorities; a failing required group reopens selection immediately.

## Does RL or multi-stage PPO fit?

**RL, reinforcement learning, learns which actions to take from their consequences. PPO is one method for training that behavior.** Standard PPO gathers experience by interacting with an environment. [Original PPO paper](https://arxiv.org/abs/1707.06347)

| Idea | Fit for the present project | Smallest useful test or prerequisite |
|---|---|---|
| Whole-image features | Good baseline candidate. These are measurements such as focus, brightness, contrast, and texture. | Compare them with class-frequency and sample-information controls on the same development groups. |
| Whole image plus small regions | Scheduled as X01 after A04; useful to test whether resizing loses local defects. | Compare whole-image-only with combined information. Keep every crop from one source image in the same data partition and score the source image. A bad image does not imply every crop is bad. |
| Images plus culture information | Conditional. Cell type, culture day, density, or flow rate may help, but can also encode acquisition shortcuts. | Compare image-only, information-only, and combined models. Fit missing-value handling on training data only and test absent fields. Never use target labels, label-containing paths, or IDs as model inputs. |
| Staged supervised training | Suitable candidate: reuse existing image features, then fine-tune a small part only if justified, then test the review rule. | Keep each stage only if its benefit survives the same validation protocol. This is not PPO. |
| RL for choosing the next image to label | R05 screens feasibility now and has a conditional pilot window; no demonstrated advantage here. | First run A11 random and simple active-selection controls at equal label count. Existing labels can simulate availability, not actual human annotation time. |
| Multi-stage PPO for pump, pressure, or fluid control | R01 to R04 are included, starting with environment checks. The current image dataset does not support this control task by itself. | Obtain a validated simulator or usable action-and-outcome recordings and independent evaluation before training. Compare ordinary PPO, staged training, and extra global inputs under matched budgets. |

The saved OOC audit reports 3,072 image records, no explicit chip/run identifier, and 2,244 missing exact time values. It does not establish complete trajectories connecting an action to a later outcome. A flow-rate column alone cannot establish what would happen if a controller changed that rate. This makes control-data validation a prerequisite for PPO, not a reason to exclude the RL feasibility check or dismiss RL across life science.

### Research references, inspected on 28 September

- **Gyimah et al., 2023:** [Deep reinforcement learning-based digital twin for droplet microfluidics control](https://doi.org/10.1063/5.0159981) uses PPO in a fluid simulator, with pressure and droplet-size observations and inlet-pressure actions. This supports a fluid-control use case, not static image classification. The primary abstract was accessible in indexed text from the [author's university-hosted thesis](https://digikogu.taltech.ee/et/Download/1b5ac438-b9b1-437e-a12c-f54205a6bb9c); the publisher page could not be retrieved and the full thesis exceeded the reader's size limit. No reported result was independently reproduced here.
- **Anand and Kothare, 14 September 2026:** [Autonomous Droplet Navigation via Model-Based Reinforcement Learning](https://arxiv.org/abs/2609.16369) reports learning from physical interaction data to guide droplets through different paths. It is a recent preprint and uses model-based RL, not evidence that PPO is optimal. The primary abstract was inspected; code and experiments were not audited. Its relevance is the need for real action consequences and transfer tests.

If a credible control environment becomes available, a possible research sequence is: compare with a simple feedback controller; train one PPO policy in ordinary conditions; expand training to measured noise and varied dynamics; then test on unseen settings and independent real measurements. Inputs could include current pressure, measured flow, image measurements, target, and recent actions. The reward would measure target tracking and penalties for instability or constraint violations. These are proposed design choices, not a reproduced multi-stage recipe.

Additional inputs are justified only when they improve independent results. A simulation judged solely by the assumptions used to build it cannot establish real biological value. Testing more stages or more features does not repair a missing validation endpoint.

**Before this deadline:** run R01 beside the other data checks. Launch R02 to R04 if a working environment with verified dynamics, a meaningful reference controller, a measurable target, and confirmation capacity is available. Otherwise record the exact blocker. R05 uses a separate label-selection task. New controlled evidence can change priorities immediately.

## Evidence rules for every branch

- Carry unfinished Route A and Route B prerequisites into 3 October together. A blocked route does not stop an independent eligible route. Nucleus results remain a separate task and must not be presented as chip validation.
- Keep the first model pilots small per route, then run every scheduled independent comparison when eligible. Remove the earlier project-wide limit of two improvements; allocate further work from actual evidence and measured cost.
- Protect the proposed 7 to 9 October final-testing, reproduction, and submission targets. Available work hours and compute remain unknown until checked.
- If the proposed review order cannot beat predicted bad-image score, use the simpler order and remove the unsupported benefit claim.
- If two controlled attempts fail, stop cosmetic changes and revisit the task or failure explanation. Do not keep searching until one favorable score appears.
- If capacity is insufficient, reduce pilot size, reuse valid shared computations, or add explicitly budgeted capacity. Record any work still blocked. Reduce repeat tuning and visual decoration before sacrificing confirmation or required deliverables.

At the end of each work session, update [the run log](08_Run_Log.md): observation, evidence quality, supported or rejected explanation, changed decision, and smallest next test. Before submission, link the decisive evidence and explicitly resolve any conflict between the selected method and stronger contrary results.
