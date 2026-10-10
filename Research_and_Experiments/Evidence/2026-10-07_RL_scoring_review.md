# RL scoring review, 7 October 2026

## Decision

The saved R02 results show a promising simulator result. The exact improvement and the claim that controllers faced matching test episodes are now **provisional**. Re-score saved controllers with the two issues below corrected before making a confirmed or statistically significant RL improvement claim. R03 and R04 use the same environment, so their reported comparisons also need this check. R05 is a separate photo-label selection task and is not affected by these environment issues.

No model was trained, no saved result was overwritten, and no experiment code was changed during this review. The check used the existing local Python environment and the current source files.

## What was checked

Sources: [environment](../../code/src/ai4s/rl/droplet_env.py), [controller evaluation](../../code/scripts/r_control.py), [fresh-episode confirmation](../../code/scripts/r02_confirm.py), and [saved R02 episode results](../../code/runs/20261007-022746_R02_confirm_fresh_conditions/episodes.csv).

The local probe used episode seed 50000 and the recorded training-range settings: pump delay 0.5 to 5 seconds, diameter mismatch 0.03, rate mismatch 0.20, diameter noise 2 micrometres, and Poisson rate noise. These are simulator settings, not newly measured physical conditions.

### 1. The reported error uses the next target at a target change

In `DropletEnv.step`, the reward is calculated against the current target. The code then calls `_new_targets()` every 100 steps before filling the reported diameter and rate errors. `run_episode` averages those reported errors.

With zero pump actions and seed 50000:

| Check | Step 99 | Step 100 |
|---|---|---|
| Diameter target used for the action | 65.83115718007839 | 65.83115718007839 |
| Diameter target after the step | 65.83115718007839 | 62.264779807467704 |
| Actual simulated diameter | 54.52272707568002 | 54.52272707568002 |
| Reported diameter error | 11.308430104398376 | 7.7420527317876875 |
| Error against the target used for the action | 11.308430104398376 | 11.308430104398376 |

At step 100, the reported rate error was 0.6944360756933117, while the error against the target used for the action was 1.2469207256792947. Target changes also occur at steps 200 and 300; step 300 ends the episode.

**Evidence quality:** direct source inspection plus a local execution of the existing environment. This proves inconsistent target timing, not the size or direction of its effect on the saved PPO advantage.

**Decision changed:** do not describe the stored R02 to R04 error measurements as final. The smallest next action is to calculate the error before changing the target and evaluate the saved policies again. Training rewards already use the current target; a new training run is not automatically necessary for this scoring correction.

### 2. The same episode seed does not keep later targets the same

The environment uses one random-number stream for both measurement noise and future targets. Poisson sampling depends on the controller's flow rate and can consume the stream differently. A shared seed therefore does not guarantee a shared target sequence.

This was demonstrated with two fixed action policies under the same seed and settings:

| Actions at every step | Initial target: diameter / rate | Target after step 100: diameter / rate |
|---|---|---|
| `[0, 0]` | 65.83115718007839 / 928.5492507984749 | 62.264779807467704 / 1231.3102786007391 |
| `[0.2, 0.2]` | 65.83115718007839 / 928.5492507984749 | 73.11358071457698 / 467.51281828580585 |

This is a diagnostic probe, not a new PPO-versus-controller experiment. Later target traces were not stored with the original per-episode results, so those CSV files alone cannot establish identical target sequences for the actual controllers.

**Evidence quality:** direct source inspection plus a local counterexample to the shared-target assumption.

**Decision changed:** the stored paired win percentages do not establish that PPO won on identical tasks. Freeze or pre-generate each episode's target schedule, keep target generation separate from measurement noise, and compare all saved controllers on those same schedules and parameter settings. Preserve all five training seeds and both condition ranges; do not select the best seed after seeing the result.

## Smallest next comparison

1. Preserve the old results and label the corrected evaluation as a new protocol version.
2. Make error timing consistent with the action's target. Record a fixed target schedule independently of each controller's actions.
3. Evaluate all five saved R02 PPO policies and both tuned rule-based controllers on the same new schedules in both condition ranges, without retuning on those schedules.
4. Report each training seed, each condition, average error, rate error, out-of-range share, and worst episode. Compute uncertainty using whole independent scenarios; do not treat time steps or five repeats of one scenario as independent samples.
5. Apply the same corrected evaluation to R03 and R04 before retaining their comparison claims. Keep real-chip performance explicitly untested.

Until this is done, the write-up may describe the RL work as an exploratory simulator study with provisional results and this known evaluation issue. It must not claim that RL is statistically proven better, always safer, or validated on physical chips.


## Separate R05 label-picking check

R05 does not use the droplet environment. The existing 800-photo figures were reproduced by linearly interpolating each stored fold/repeat curve at 800 photos, then averaging 25 curves (five folds and five repeats). All 25 curves for every method had points on both sides of 800, so no extrapolation was needed. These are curve estimates, not direct tests at exactly 800 labels.

| Training rounds for the RL picker | RL | Random | Uncertainty picking |
|---|---|---|---|
| 150 | 0.766607 | 0.793999 | 0.809645 |
| 1500 | 0.755052 | 0.793999 | 0.809645 |

**Evidence quality:** reanalysis of existing `budget_curves.csv` files, no new model training. This supports keeping this label picker rejected; it does not reject RL for other tasks. Exact source paths, file hashes and calculations are recorded in [the verification record](2026-10-07_consolidation_verification.json).
