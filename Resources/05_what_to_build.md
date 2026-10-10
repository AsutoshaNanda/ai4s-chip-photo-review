# What to build and compare

Updated: **2 October 2026**. These are project ideas to test, not proven improvements. Run independent experiments at the same time, then choose the final entry using measured results.

## First option to try: review chip-image quality

Input: a real brightfield chip image, meaning an image made with light passing through the sample, with optional sample details that would be available when the tool is used.

Output: a predicted quality category, which images to review first, model/version details, and a downloadable results table. Show probability estimates or explanations only if checks support them.

Possible improvement: catch more poor-quality images for the same amount of review work than a simple image rule or sorting by the predicted poor-quality score. Test this in A09. No benefit has been measured yet. The quality category does not prove that a drug is safe or that cells are alive and healthy.

## Parallel candidate: nucleus counting and review

Input: a fluorescence image, which shows glowing signals, with a suitable reference mask marking the correct objects for testing.

Output: nucleus outlines, count, object measurements, tested error warnings, and a file to save the results. Compare a simple method with checked AI methods in B01/B02; test the benefit of review and corrections in B06/F02.

This is a separate project option. Its scores cannot show that the chip-quality model works, and must not be averaged with chip-quality scores.

## All other project options are still included

C01/C02 cover tables of cell appearance and behavior, C03 predicts fluorescence from matching images, C04 tests on independent real-chip data, X01 compares whole images with local details, and R01 to R05 cover reinforcement learning (RL), where a system learns from rewards, for control or choosing samples to label. Each has a date and clear requirements in [the complete experiment list](../Research_and_Experiments/10_All_Experiments_Parallel_Schedule.md).

Carry all unfinished data/source checks into 3 October. Aim to run comparisons that meet their requirements on 4 to 6 October, choose candidates on 6 October, do final testing on 7 October, repeat the work from a clean setup on 8 October, and submit on 9 October. These are planning targets, not assumptions about how long runs will take.

The proposed category is still Tool & Platform for a review tool. Check that it fits what the final project actually adds. Credit reused models and related work. The new [participant audit notebook](https://www.kaggle.com/code/loxigicck/audit-acquisition-groups-in-organ-on-chip-images) means checking filename beginnings alone must not be called a new idea.
