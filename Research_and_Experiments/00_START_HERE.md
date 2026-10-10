# AI4S research and experiment guides

Updated: **3 October 2026, Asia/Kolkata**. The earlier measurements of sample details keep their 26 September date.

**Starting today:** All 30 original plans and six feature/RL entries are still in [the complete schedule for work that can run together](10_All_Experiments_Parallel_Schedule.md). Features are image measurements; RL means reinforcement learning, or learning from rewards. [The daily timeline](09_Remaining_Timeline_and_RL_Assessment.md) brings unfinished setup into 3 October. No earlier task is assumed to be complete. The live deadline was checked on 2 October; no model results were added.

**Give real organ-on-a-chip quality review first priority. Start nucleus counting and independent checks alongside it.** Use measured results to decide what the final project adds. Other project options can start their own small trials once their data, usage rights, test setup, and computing resources have passed the required checks.

The earlier library keeps 31 research/data/method references, with more dated sources added. The experiment guide keeps all 30 original plans; the schedule adds six feature/RL entries. [Which resources are ready](../Resources/19_Resource_Readiness.md) lists the exact sources and checks still needed. This is not a review of every relevant paper.

**Evidence status:** It has been checked that all 36 entries are included and that the current deadline is recorded. Priorities and dates are proposed choices. Completion time, model improvements, practical benefit, and final placing are unknown until data and computer-resource tests have been done.

## Read in this order

| File | Purpose |
|---|---|
| [01 Project decision](01_Project_Decision.md) | Which project to test first, why, and when to switch |
| [02 Research library](02_Research_Library.md) | Original sources, useful findings, limits, and related experiments |
| [03 Data and model checks](03_Data_and_Model_Audit.md) | Checked data facts, measured sample-detail counts, and unresolved issues |
| [04 Experiment guide](04_Experiment_Playbook.md) | Fair comparisons, confidence, cost limits, and rules for keeping or rejecting a method |
| [05 Testing and choosing](05_Validation_and_Selection.md) | Training/test groups, measurements, uncertainty, worst cases, and rules for the final choice |
| [06 Evidence and decisions](06_Evidence_and_Decisions.md) | What the evidence changed and the next required action |
| [07 Schedule and submission](07_Schedule_and_Submission.md) | What takes priority and what must be submitted for judging |
| [08 Run log](08_Run_Log.md) | Where actual measurements must be recorded |
| [09 Daily timeline and RL](09_Remaining_Timeline_and_RL_Assessment.md) | Work that can happen together each day, and whether extra image measurements or learning from rewards fit |
| [10 All experiments](10_All_Experiments_Parallel_Schedule.md) | All 30 plans, where E0 to E6 fit, six extra feature/RL checks, and how to share computing resources |

## Most valuable findings

1. A public real-chip dataset has **3,072 image records**. Its spreadsheet entries were independently matched with its archive filenames here. [Data check](03_Data_and_Model_Audit.md)
2. There are no clear IDs for physical chips/runs. All 59 filename beginnings appear in more than one published training/test group. This is a reason to check for related samples, **not proof that test information has leaked into training**. [Saved measurements](Evidence/ooc_metadata_audit.json)
3. The dataset paper lists CC-BY-SA, but its repository API lists CC BY 4.0. The exact rules for sharing the data are still unclear. [Sources and decision](06_Evidence_and_Decisions.md)
4. Current methods for outlining objects in images mean using a basic U-Net model alone is weak evidence of a new idea. A useful addition could be a measured reduction in review work or missed failures. That benefit must be tested. [Research](02_Research_Library.md)

## First actions

- Complete **G01 and G02**: settle what data we can use and how, and define valid training/test groups before training.
- Run **A01 and A02** if the real-chip option passes its checks. Compare simple image measurements, possible shortcuts based on sample details, and one small already-trained model that assigns image categories.
- Run **A04 and A09**: check what the category model uses and whether its order for reviewing images catches bad samples better than simpler choices.
- Start **B01 and B02** on BBBC039 alongside this work once their own checks pass. Results on fluorescent images of nuclei do not prove that a method works on chips.
- Start **C01 to C04 and R01/R05 checks of what is needed first** immediately; start method comparisons that pass those checks in their scheduled periods.
- Save time for **F01 to F04**: repeat results, measure practical benefit, repeat work from a clean setup, and check the final submission.

The [full experiment schedule](10_All_Experiments_Parallel_Schedule.md) gives every plan a priority, date, requirements, and expected output. Run independent work together; clearly show unmet requirements and resource limits. The [submission guide](07_Schedule_and_Submission.md) keeps time available for final confirmation and submission.

## Recorded work and this update

- On 26 September, moved the 19 original guides unchanged into [Resources](../Resources/00_READ_FIRST.md). The resources are now updated at the user's request; their previous versions are in [the pre-update snapshot](Evidence/plans_before_2026-10-02.zip).
- Read the live competition overview and rules in the browser.
- Reviewed the linked papers, provider pages, and current software-maintainer guides, with access limits stated in the library.
- Downloaded the small public OOC spreadsheet and dataset-record details, and read the archive's file list with a 1 MiB request for part of the file.
- Checked IDs, counts, labels, missing fields, and shared filename beginnings.

**This planning update checked no new dataset images, trained no model, and submitted no entry.** The project file list and central run log contain no recorded model runs. Work outside this project folder was not checked. No paper's or participant's score is presented as our result.

The file-check values from the earlier move are in [the organization record](organization_manifest.json); they do not describe today's edited resources. New source observations are in [the 2 October evidence note](Evidence/2026-10-02_source_checks.md). The [evidence folder](Evidence/ooc_metadata_audit.json) holds the small downloaded source files and measured check results.
