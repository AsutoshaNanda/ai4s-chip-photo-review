# RL repaired evaluation (protocol v2), 8 October 2026

Follows [the RL scoring review of 7 October](2026-10-07_RL_scoring_review.md). **No model was trained or tuned.** The saved R02, R03 and R04 policies and the two tuned rule-based controllers were scored again in a repaired evaluation environment. The original results and the original code are unchanged; the repaired evaluation is a new protocol version.

## What was repaired (both problems from the review were first re-checked in the code)

1. **Error timing.** In the original `DropletEnv.step` the target was replaced (steps 100 and 200) before the reported errors were filled in, so those steps were scored against the next target. The repaired environment reports the error against the target in force when the action was applied. The training reward was never affected.
2. **Matching scenarios.** One random stream served both measurement noise and later targets, and the Poisson draw uses a different number of random numbers at different flow rates, so one seed gave different target sequences to different controllers. In the repaired environment every scenario seed is split into three independent streams: plant parameters and starting pumps; the whole target schedule (made at reset); and measurement noise (drawn through the inverse cumulative distribution with exactly two random numbers per measurement). Every controller therefore faces the same targets, plant and noise numbers.

Direct check (`src/ai4s/rl/droplet_env_v2.py`, seed 50000, two different fixed-action policies): original environment, target after step 100 = 62.265 vs 73.114 (different); repaired environment, identical three-block target schedule for both; error reported at step 100 = |target used − actual| (17.205 against target 59.509; the next target is 72.708).

## Protocol

* 47 saved policies/controllers: both rule-based controllers (tuned gains from the R02 run) and 45 saved PPO policies (R02 `PPO_all`, R03 `ordinary` / `staged` / `nominal_only`, R04 five input sets; five training seeds each). The best seed is never chosen.
* **50 new scenarios** (seeds 100000–100049; never used for training, tuning, validation or any earlier test). Conditions: the training ranges and the harder ranges not used for training. R04 also under its stress tests (inputs missing 10% / 30%, measurement noise ×2 / ×4, training ranges).
* Uncertainty: each scenario is one sample. Training seeds are averaged inside a scenario; differences are paired by scenario; 95% ranges resample whole scenarios (10,000 draws, seed 20261008). Time steps and training seeds are not treated as independent samples.
* Metrics: mean diameter error (µm), mean diameter error in the second half of each target block ("settled"), relative rate error, share of steps outside the measured operating range, worst episode.

**Finding on the way:** R02's `PPO_all`, R03's `ordinary` and R04's `inputs_all` are the **same five trained networks** (weights and input scaling identical for every seed; training with a fixed seed is deterministic). The three studies share one policy set; it was scored once per study for transparency and gives identical numbers.

## Results (mean over 50 scenarios; training seeds averaged within each scenario)

### R02: learned controller vs rule-based controllers

| | Train ranges | Harder ranges |
|---|---|---|
| Mean diameter error, PPO (5 seeds) | **1.008** | **1.333** |
| PI | 1.382 | 2.062 |
| FF+PI | 1.175 | 1.946 |
| PPO minus PI | **−0.374 [−0.431, −0.312]**, better in 94% of scenarios | **−0.729 [−0.816, −0.633]**, 98% |
| PPO minus FF+PI | **−0.167 [−0.257, −0.080]**, 70% | **−0.613 [−0.723, −0.503]**, 96% |
| Relative reduction vs PI / FF+PI | 27% / 14% | 35% / 31% |
| Rate error (PPO / PI / FF+PI) | 0.070 / 0.154 / 0.147 | 0.103 / 0.272 / 0.244 |
| Out-of-range share (PPO / PI / FF+PI) | 0.004 / 0.066 / 0.105 | 0.008 / 0.084 / 0.130 |
| Settled diameter error (PPO / PI / FF+PI) | 0.620 / 0.912 / 0.516 | 0.643 / 1.303 / 1.062 |
| PPO minus FF+PI, settled error | **+0.104 [+0.041, +0.167]** (PPO worse) | −0.419 [−0.551, −0.295] |
| Highest-error single episode, diameter error in µm (PPO / PI / FF+PI) | 2.166 / 2.346 / 3.122 | 2.826 / 3.213 / 4.419 |
| Most time outside the operating range in any single episode, a different episode (PPO / PI / FF+PI) | 0.110 / 0.357 / 0.607 | 0.253 / 0.380 / 0.490 |

Each of the five PPO seeds beats both rule-based controllers on mean diameter error in both ranges. Train ranges: 1.011, 1.048, 0.980, 1.033, 0.970; harder ranges: 1.315, 1.369, 1.344, 1.335, 1.303.

For comparison, the earlier (flawed) evaluation gave 1.484 vs 2.096 (PI) and 1.950 (FF+PI) in the harder ranges, reductions of 29% and 24%.

### R03: staged training

| | Train ranges | Harder ranges |
|---|---|---|
| Mean diameter error: ordinary (the R02 policies) / staged / nominal-only | **1.008** / 1.084 / 1.170 | **1.333** / 1.401 / 1.568 |
| Staged minus ordinary | **+0.076 [+0.059, +0.093]** (staged better in 6% of scenarios) | **+0.068 [+0.048, +0.089]** (16%) |
| Nominal-only minus ordinary | +0.162 [+0.123, +0.197] | +0.235 [+0.184, +0.280] |
| Out-of-range share (ordinary / staged / nominal-only) | 0.004 / 0.004 / 0.013 | 0.008 / 0.006 / 0.019 |

### R04: extra inputs

| Mean diameter error | Train ranges | Harder ranges | Out-of-range share (train / harder) |
|---|---|---|---|
| all inputs | **1.008** | **1.333** | 0.004 / 0.008 |
| history only | 1.007 | 1.431 | 0.031 / 0.043 |
| minimal | 1.302 | 1.729 | 0.055 / 0.051 |
| pump set-points | 1.401 | 1.682 | 0.003 / 0.003 |
| camera only | **1.402** | **1.738** | 0.005 / 0.008 |

Paired differences (train / harder): all minus camera **−0.394 [−0.436, −0.352] / −0.405 [−0.457, −0.355]**; all minus minimal −0.293 [−0.344, −0.239] / −0.396 [−0.448, −0.336]; history minus camera −0.395 [−0.450, −0.342] / −0.307 [−0.370, −0.249]; pumps minus camera −0.001 [−0.041, +0.039] / −0.056 [−0.090, −0.020]; minimal minus camera −0.100 [−0.158, −0.049] / −0.010 [−0.060, +0.037].

Stress tests (training ranges, mean diameter error):

| Inputs | Normal | 10% inputs missing | 30% missing | Noise ×2 | Noise ×4 |
|---|---|---|---|---|---|
| all | 1.008 | 1.074 | 1.228 | 1.268 | 1.816 |
| history only | 1.007 | 1.101 | 1.296 | 1.355 | 1.963 |
| minimal | 1.302 | 1.314 | 1.386 | 1.717 | 2.302 |
| pumps | 1.401 | 1.421 | 1.526 | 1.724 | 2.225 |
| camera only | 1.402 | 1.417 | 1.450 | 1.737 | 2.242 |

## What this supports and what it does not

* **Supported (simulator only):** with identical scenarios and consistent error timing, the learned controller has lower mean diameter and rate error and spends much less time outside the measured operating range than both rule-based controllers in both ranges; the 95% ranges for mean diameter error exclude zero in all four comparisons; every training seed agrees. Staged and nominal-only training are worse than ordinary training. Giving the controller all inputs (or its recent history) clearly beats camera-only input.
* **Exceptions:** after the first half of each target block (settled error) the model-assisted rule controller is better in the training ranges (PPO +0.104 µm). History-only and minimal inputs have lower error than camera-only but spend 3–6% of steps outside the operating range.
* **Not supported:** physical-chip performance, safety guarantees (in the harder ranges one PPO episode still spent 25% of its steps outside the range, although the rule-based worst cases were worse; the highest-error PPO episode there spent 1.3% outside), or generality beyond this simulator family. 50 scenarios and five training seeds; the scenarios come from the same simulator that produced the training episodes. No physical fallback was tested.

## Files

Run folder `code/runs/20261007-234732_R02_repaired_evaluation_v2/` (`results.json`, `paired_differences.csv`, `summary_by_policy.csv`, `episodes.csv`, `log.txt`). Code: `code/src/ai4s/rl/droplet_env_v2.py`, `code/scripts/r_eval_repaired.py`. Original results remain in `code/runs/20261006-011158_R02_pi_vs_ppo/`, `…012739_R03_staged_vs_ordinary_ppo/`, `…013244_R04_input_sets/` and `…20261007-022746_R02_confirm_fresh_conditions/`.
