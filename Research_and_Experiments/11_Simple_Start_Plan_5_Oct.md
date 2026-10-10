# Plan: start all 36 AI4S experiments (from Mon 5 Oct, night)

## Context — why this plan exists

- Your folder `Research_and_Experiments` lists **36 experiments**: 30 original ones plus 6 extras (X01 and the reward-learning tests R01–R05). **None has been run yet.** The run log (`08_Run_Log.md`) is empty.
- The old calendar expected a start on 2–3 Oct. It is now **Mon 5 Oct, ~11:30 PM India time**. The deadline is **Sat 10 Oct, 9:29:59 PM India time** (last checked on the live page 2 Oct — we recheck before submitting). That leaves **about 4 days 22 hours**.
- You asked for: every experiment started (**no skipping, reward-learning included**), plain language, the cheapest GPU options, a budget of **about $30**, and **no assumed numbers** (no fixed pass marks, no hard-coded limits, no guessed run times).
- Already done (you confirmed): the competition's separate registration form, and a Kaggle account with phone verified.

## Bottom line

1. **All 36 experiments get started**, each on a set day. Each ends with a real result or the exact reason it couldn't finish. Nothing is deleted or quietly dropped.
2. **Money:** free first (Kaggle's free GPUs + Modal's free $30/month, if your sign-up shows it). Then a cheap rented RTX 3090 on Vast.ai — but **only after you see measured costs**. Hard cap: your ~$30.
3. **First real step (Tue morning): a speed test on every machine** (your Mac, Kaggle, Modal, Vast). Speeds and costs are measured, not guessed. You approve any spending after seeing that table.
4. **Submit target: Fri 9 Oct night.** Sat 10 Oct is the safety buffer.
5. **Computer power level: Low (your choice).** That means free machines plus RTX 3090-class rentals (~$0.19/hour on Vast). For comparison: "Mid" would be RTX 4090 / A100 cards ($0.47–0.60/hour on Vast), and "High" would be H100 / H200 cards ($2.21–5.00/hour on Vast). These are hourly prices. Total hours are unknown until Tuesday's speed test.

## Ground rules (from your instructions)

- **No assumed numbers.** Each comparison shows the **full picture**, not one pass mark: the whole curve across every amount of reviewing, or every overlap level, with uncertainty ranges. If one cut-off is ever needed to make a decision, it is either picked from practice data **before** the final test and written down, or you pick it.
- **Changes from the old plan (flagged):**
  - The old files suggest fixed cut-offs: "+2 points", "+5 points at 20% review", "10% fewer counting errors", "a big error = 5 nuclei or 10%", "overlap 0.5", "2 GPU-hours per trial", "stop after 2 failed tries". Following your rule, **none of these is used as a fixed rule anymore**. Those values only appear as points on the full curves.
  - Fixed sizes such as "25% / 50% / 100% of the data" or "check 10% / 20% / 30% of photos" also become full ranges. The exact steps are set from the real number of photo groups.
  - I'm also adding **one big modern AI model at a large image size, early** (your standing rule), next to the small model the old plan named. I'm not assuming it wins; we measure.
  - All dates have moved about 2.5 days later.
- **No guessed run times.** Tuesday's speed tests set the time and money budgets.
- **GPU by default for model training, as you asked.** Is your Mac slow? Not necessarily; it depends on the job. We measure it on Tuesday and use it only for jobs where it keeps up (it's free). Pure bookkeeping work (scoring saved results, small tables) runs on the Mac, because a GPU has nothing to speed up there.
- **Tasks are never mixed.** Chip-photo scores, nucleus-counting scores and reward-learning scores are reported separately and never averaged together.
- **Final-test images are locked away before any training** and opened only once, on Thursday.
- **Ideas are judged on rotating practice tests across all the practice data (cross-validation), not on one small slice.** If a small test and a bigger one disagree, we trust the bigger one, and I'll tell you loudly.
- **Copy every model file and log to your Mac** before stopping or deleting any cloud machine.
- **Nothing is paid for, published or submitted without your OK.**

## Words used (plain meanings)

- **GPU**: a graphics chip that makes AI training much faster.
- **Model**: the AI program that learns from examples.
- **Training**: letting the model learn from labelled examples.
- **Baseline**: the simple method the AI must beat.
- **Practice data / final test**: practice data is for trying things out. The final test is opened once, at the end, for an honest score.
- **Random start (seed)**: re-running with a different random start, to see whether a result is luck.
- **RL (reward learning)**: an AI that learns by trial and error from a score.
- **PPO**: one common recipe for RL.

## What we are building (one paragraph)

**Main idea (A):** a tool that sorts organ-on-a-chip microscope photos (3,072 expert-labelled photos, 6 cell types, labelled "good" or "bad") so a researcher checks the likely-bad ones first.
**Backup idea (B):** a nucleus-counting helper (BBBC039: 200 small images, free licence).
**Extras:** four more datasets (C), a whole-photo vs zoomed-patch test (X01), and five reward-learning tests (R).
**Category to declare:** decided Thursday from the results; the proposed one is "Tool & Platform".

## The machines ("lanes") that run side by side

| Lane | Machine | Cost | Main use |
|---|---|---|---|
| 1 | Your Mac (M4 Pro, 24 GB memory, Apple GPU) | Free | Data checks, scoring saved results, report. Also any job the speed test shows is fast enough here. |
| 2 | Kaggle (2× T4 GPUs, already set up) | Free, about 30 GPU-hours/week | Chip-photo model training; two jobs at once, one per GPU |
| 3 | Modal (if the free $30 shows up) | Free $30/month; up to 10 GPUs at once | Many short runs side by side: variants, repeats, RL |
| 4 | Vast.ai rented RTX 3090 | Paid, only after your OK | Big downloads, heavy jobs (C03), extra capacity |

Each cloud machine downloads the data **straight from the source**. That's faster than uploading from your home internet, and it saves your Mac's disk.

## Day-by-day (India time)

| When | Main work (me) | Alongside | You |
|---|---|---|---|
| **Mon night → Tue morning** | Make a clean Python setup with fixed versions. (Your Mac has a *preview* build of PyTorch, which is risky when others need to re-run our work.) Write the data-loading and scoring code. Start downloads. | Keep the Mac awake | Sign up for Modal; create a Vast.ai account (no money yet); free up disk space |
| **Tue 6 Oct** | G01–G03 checks → **speed and cost table** → you approve spending. A01, B01 simple methods; first A02/B02 runs. Quick checks for B07, C01–C04, R01, R05. | Report and demo skeletons | Approve spending; tell me Kaggle GPU hours left |
| **Wed 7 Oct** | All A, B and X runs side by side. A08–A10 and B06 on saved results. C01, C02, C03 runs; R02 and R05. | Report drafting | Short check-ins |
| **Thu 8 Oct** | R03, R04. C04 test if data was found. Repeats of promising results (F01). **Afternoon: freeze the choices. Evening: one locked final test.** F02. | Demo + video script | Review results; approve the frozen choice |
| **Fri 9 Oct** | F03 fresh rebuild on a new free Kaggle notebook. Report + video. F04 checks. | — | Record the video (your voice, my script); approve publishing; press **Submit** |
| **Sat 10 Oct** | Buffer: fixes only. Recheck the deadline live. Confirm "submitted" before 9:29:59 PM. | — | — |

**Honest note:** all 36 will be started. Whether every one reaches a fully confirmed result by Thursday depends on measured run times. **On Tuesday evening, after the speed tests, I'll tell you which ones are at risk**, and we'll decide together: more GPUs side by side within budget, or smaller runs. Results are never invented to fill the calendar.

## All 36 experiments: what each asks, how it starts, where, when

"GPU" below means Kaggle, Modal or Vast, whichever the speed and cost test picks.

### G: shared checks (these come first)

| ID | Plain question | How we start it | Where | When |
|---|---|---|---|---|
| G01 | Is each dataset what it claims to be, and are we allowed to use it? | Download each one, check the file fingerprints and open samples. Record the licence, including the chip data's conflict: the paper says CC-BY-SA, the website says CC-BY 4.0. | Mac + cloud | Tonight → Tue AM; again for each new dataset |
| G02 | Are the final-test images truly kept apart? | Lock the final-test images first. Which ones is decided from the group check and written down. Build splits three ways: the published split, grouped by filename start (an *unconfirmed* clue), and one-cell-type-left-out. Check for exact and near-duplicate photos. Test the scoring code on fake perfect, all-wrong and empty answers. | Mac | Tue AM |
| G03 | How fast is each job on each machine, and what does it cost? | Run the same small job on the Mac (CPU and Apple GPU), Kaggle T4, Modal and a Vast 3090. Record seconds, memory, disk and $. | All | Tue AM–noon |

### A: chip-photo quality (main idea)

| ID | Plain question | How we start it | Where | When |
|---|---|---|---|---|
| A01 | Can simple measurements (sharpness, brightness, contrast, texture), or the sample details alone, already spot bad photos? | Three simple methods on identical splits | Mac | Tue |
| A02 | Does a pretrained image AI beat A01? | One small model and one big modern model, each at a smaller and a larger image size (sizes picked from the real photo size). First "frozen": the AI only reads features and a simple classifier sits on top. Then light retraining. | GPU | Tue–Wed |
| A03 | Does it still work on a cell type it never saw? | Leave one cell type out at a time; report every type, worst one included | GPU | Wed |
| A04 | Is it looking at the cells, or at edges and background? | Whole photo vs centre crop vs edges-only | GPU | Wed |
| A05 | Do safe flips, turns or brightness changes during training help? | Only changes that keep the label true. No added blur, because blur can be the reason a photo is "bad". | GPU | Wed |
| A06 | Does giving the rarer class more weight help? | Weighted vs unweighted; every cell type checked (one type has only 15 good photos) | GPU | Wed |
| A07 | Do sample details (cell type, day, flow) add anything to the photo? | Photo-only vs details-only vs both; also with details missing | Mac (uses saved AI features) | Wed |
| A08 | Can we trust the AI's "% confident" numbers? | Measure, then correct them using separate photos | Mac | Wed–Thu |
| A09 | Does our review order catch more bad photos for the same checking effort? (the main new idea) | Full curve from 0% to 100% of photos checked: random vs sharpness rule vs AI score vs AI score + "unsure" score | Mac | Wed–Thu |
| A10 | Do two methods make different mistakes, and does combining them or a "second opinion" warning help? | Compared on saved predictions | Mac | Wed–Thu |
| A11 | How many labelled photos are really needed? | Growing slices of the training groups; random vs smart picking at equal label counts; picks repeated | GPU | Wed |
| A12 | Do heat-map explanations show what the AI really uses? | Compare with a scrambled AI; hide regions and watch the prediction change | GPU | Wed–Thu |
| X01 | Does adding zoomed-in patches to the whole photo help? | Whole vs whole + patches. Patches never automatically get the photo's label; scoring is per photo. | GPU | Wed |

### B: nucleus counting (backup idea; BBBC039, ~81 MB, free CC0 licence)

| ID | Plain question | How we start it | Where | When |
|---|---|---|---|---|
| B01 | How well does a simple brightness cut-off plus a "splitter" for touching nuclei count them? | Settings chosen on practice images only | Mac | Tue |
| B02 | Do AI models count better? | A small U-Net trained here (an AI that marks each pixel) vs one ready-made modern model: StarDist, Cellpose-SAM or InstanSeg, whichever passes the licence check and the "has it already seen our test images?" check. Unknowns are reported separately. | GPU | Tue–Wed |
| B03 | Does teaching it nucleus edges or distances separate touching nuclei better? | One change vs B02 | GPU | Wed |
| B04 | Does it break on darker, noisier or blurrier images? | Several strength levels each; every level reported | GPU | Wed |
| B05 | How does it do on crowded images and tiny nuclei? | Groups defined from the training data; a gallery of the worst cases | Mac | Wed |
| B06 | Can a warning flag bad counts for a person to fix? | Full curve for every review amount: methods disagreeing vs a blur/brightness warning vs random | Mac | Wed–Thu |
| B07 | Does it work on a different nucleus dataset? | BBBC038, after removing images shared with BBBC039 and checking what the models have already seen | GPU | Wed–Thu |

The old E0–E6 list is fully covered: E0 → G01, G02, B01; E1 → B01; E2 → B02; E3 → B03; E4 → B04; E5 → B05, B06; E6 → F01, F03; the extra outside-data test → B07.

### C: extra datasets

| ID | Plain question | How we start it | Where | When |
|---|---|---|---|---|
| C01 | Can ready-made cell measurements tell treatments apart across different lab batches? | RxRx1's 51 MB ready-made AI-features file. Its licence is non-commercial and share-alike, so it's checked against the competition terms first. | Picked by speed test | Check Tue, run Wed |
| C02 | Does re-scaling JUMP cell-profile tables remove lab-batch noise but keep the biology? | Small recorded subset; raw tables vs one re-scaling method learned on training batches only | Mac (table work) | Check Tue, run Wed |
| C03 | Can AI predict the glowing (fluorescent) image from the plain photo, in a useful way? | Allen paired images, a small subset on a rented GPU (your Mac's disk is too small). Judged by a real measurement made from the output, not by looks. Rights check first: Allen limits re-sharing. | Rented GPU | Check Tue, run Wed–Thu |
| C04 | Does our chip tool work on chip photos from another lab? | Search for downloadable outside chip photos with matching labels (the skin-on-chip paper lead plus public archives). If found: test the unchanged model. If none exist: recorded as "blocked: no data" with the search log (data can't be invented). | GPU if data | Search Tue, test Wed–Thu |

### R: reward learning (none skipped)

| ID | Plain question | How we start it | Where | When |
|---|---|---|---|---|
| R01 | Is there a real, checked chip-flow control set-up to learn on? | Search published simulators and recordings, e.g. from the droplet-control papers in your notes. Check the licence and whether it was compared against real chips. **Output: the environment for R02–R04.** A real one if found; otherwise a simple practice flow simulator built from basic physics, clearly labelled **"practice only, not proof about real chips"**. | Mac (search and reading) | Tue |
| R02 | Does PPO (trial-and-error AI) beat a standard rule-based controller? | Same environment for both. Compare how closely each follows the target, safety-limit breaks, worst conditions and training cost. | GPU (timed against the Mac first) | Wed |
| R03 | Does training in stages (easy first, then noisy and varied) beat plain PPO with the same total training? | Tested on conditions never seen in training | GPU | Thu |
| R04 | Do extra inputs (pressure, flow, image measures, history) help compared with minimal inputs? | Same training budget; also tested with missing or noisy inputs | GPU | Thu |
| R05 | Can an RL agent choose which photos to label better than random or simple smart picking? | Uses the chip training pool only, with equal label budgets; final-test labels stay hidden | GPU | Wed–Thu |

### F: confirm and submit

| ID | Plain question | How we start it | Where | When |
|---|---|---|---|---|
| F01 | Do the wins hold up? | Repeat promising results with different random starts (as many as measured time and budget allow), then one locked final test | GPU | Thu |
| F02 | Does it really save checking work? | Workload numbers from saved predictions. A timed human test only with a suitable volunteer and your OK; nobody is contacted without it. | Mac | Thu |
| F03 | Can anyone re-run it from scratch, for free? | Rebuild on a fresh free Kaggle notebook and compare the numbers | Kaggle | Fri |
| F04 | Does every claim match saved evidence, and is the entry complete? | Report, code, video and links checked while signed out. **You** press Submit. | Mac + you | Fri (Sat buffer) |

## GPU options: prices checked live today (Mon 5 Oct 2026, ~11 PM India time)

Marketplace prices change hourly. Always check the actual offer before renting.

### Free

| Option | What you get | Catch |
|---|---|---|
| Your Mac | Free | Only **30 GB disk free**. Speed measured Tuesday. |
| **Kaggle** (you're set up) | 2× T4 (16 GB each) or 1× P100; about 30 GPU-hours/week ("sometimes higher"); 12-hour sessions; 20 GB of saved output | Queues at busy times; hours left are shown in your Kaggle settings |
| **Modal** | Its own pricing and sign-up pages say **$30 free every month** (Starter plan); up to **10 GPUs at once**; billed per second | Neither page says whether a card is needed; you'll see at sign-up. CPU and memory are charged on top of the GPU price (example below). |
| Lightning AI | 5 free credits at sign-up, +25 if you add a card | GPU rates not shown on its pricing page |
| Google Colab | Free T4 when available; paid option $11.79 per 100 "compute units" (seen 3 Oct) | No guaranteed hours |

### Rent: USD per GPU per hour (GPU only)

For Vast.ai: **cheapest listing / typical (middle) listing.** The cheapest can be an "interruptible" machine that may be taken back mid-run, so we save progress often.

| GPU (memory) | Vast.ai | Runpod | TensorDock* | Others |
|---|---|---|---|---|
| RTX 3090 (24 GB) | 0.12 / 0.19 | 0.50 | 0.20 | Salad 0.09 (can be interrupted) / 0.17 |
| **2× RTX 3090** | ≈0.24 / ≈0.38 (2 × single) | 1.00 (2 × single) | ≈0.40 (2 × single) | — |
| RTX 4090 (24 GB) | 0.29 / 0.47 | 0.74 | 0.35 | Salad 0.16 / 0.33 |
| RTX 5090 (32 GB) | 0.21 / 0.60 | 0.99 | — | Salad 0.25 / 0.50 |
| RTX A5000 (24 GB) | 0.19 / 0.23 | **0.27** | — | — |
| RTX A6000 (48 GB) | 0.36 / 0.40 | — | — | Thunder 0.35; Lambda 1.09 |
| A40 (48 GB) | 0.28 / 0.40 | 0.49 | — | E2E 1.44 |
| T4 (16 GB) | 0.15 / 0.15 | — | — | Modal 0.59; Kaggle free |
| L4 (24 GB) | 0.25 / 0.32 | 0.49 | — | Jarvislabs 0.44; E2E 0.57; Modal 0.80 |
| L40S (48 GB) | 0.47 / 0.73 | 1.09 | — | E2E 1.20; Modal 1.95 |
| A100 40 GB | — | — | — | Jarvislabs 0.89; E2E 1.98; Lambda 1.99; Modal 2.10 |
| A100 80 GB | PCIe 0.39 / 0.60; SXM 0.31 / 0.92 | 1.59 | 1.50–1.80 | Thunder 1.09; Jarvislabs 1.49; E2E 2.10; Modal 2.50 |
| H100 80 GB | PCIe 1.25 / 2.27; SXM 1.73 / 2.21 | 2.89–3.49 | 2.25 | E2E 2.69; Thunder 3.20; Lambda 3.29–4.29; Jarvislabs 3.49; Modal 3.95 |
| H200 (141 GB) | 2.63 / 5.00 | 4.59 | — | E2E 3.99; Modal 4.54; Jarvislabs 4.59 |

\*TensorDock: "typical" prices from its own page (via search today); CPU, memory and disk are charged on top. Salad runs only packaged "containers", which means more set-up work; its cheapest tier can be interrupted. "2×" rows are worked out from single-card prices, not real two-card listings. "H4" isn't a GPU name, so I listed H100 and T4, the closest. Tell me if you meant another.

### What $30 buys (GPU-only arithmetic, before storage and transfer). Not a forecast of how much we need.

| Choice | Hours for $30 |
|---|---|
| Vast RTX 3090 at typical $0.19 | ~157 h |
| Vast 2× RTX 3090 at ~$0.38 | ~79 h (two GPUs every hour) |
| Runpod RTX A5000 at $0.27 | ~111 h |
| Vast RTX 4090 at $0.47 | ~63 h |
| Vast A100 80 GB at $0.60 | 50 h |
| Vast H100 SXM at $2.21 | ~13.5 h |
| Modal T4 with 4 CPU cores + 16 GB memory: $0.59 + 4×$0.047 + 16×$0.008 ≈ $0.91/h | ~33 h, from the *free* $30 |

### Storage and data-transfer costs (the hidden costs)

- **Vast.ai:** each machine's owner sets disk and internet-transfer prices, shown on the offer before you rent. Disk is still charged after the machine is stopped, so we copy results home and then delete it.
- **Runpod:** disk $0.10/GB/month while running; $0.20/GB/month for disk kept while stopped; shared storage $0.07/GB/month. Transfer fees are not mentioned on its pricing page.
- **Modal:** storage free up to 1 TiB/month; data sent out costs $0.04/GB (our results are small).
- **Thunder:** no transfer fees; 100 GB of disk per GPU included. **Jarvislabs:** disk $0.10/GB/month, billed per minute. **Lambda:** says no transfer fees, but its GPUs cost more.
- **TensorDock:** CPU, memory and disk are charged on top of the GPU price.
- **E2E Networks (India):** accepts UPI and net banking. A fallback if an Indian card is declined on foreign sites, but it costs more.

### My pick: cheapest options that fit your rules

1. **Free first:** Kaggle (2× T4), plus Modal's $30 (if your sign-up shows it), plus your Mac where it measures fast enough.
2. **Paid:** a **Vast.ai single RTX 3090** (cheapest per hour seen) for long jobs and C03. For two jobs side by side on one copy of the data: a **2× RTX 3090** machine.
3. **Backup if Vast machines are unreliable:** **Runpod RTX A5000** ($0.27/h, one fixed price).
4. **A100 80 GB** (Vast, from $0.39/h) only if a job is *measured* to need more than 24 GB of GPU memory.
5. **H100/H200** only if the speed test shows a job can't finish in time otherwise. On Vast they cost about 10× or more per hour than a 3090.

About 2 GPUs: splitting *one* training run across 2 GPUs needs extra code and doesn't always speed up small models, so we'd measure before relying on it. Two GPUs are mainly useful for running two experiments at once.

### Money safety

- **Prepaid credit only** on Vast and Runpod: add small amounts, so you can't overspend.
- **No spending until you've seen Tuesday's measured cost table.**
- I report money spent at every check-in and warn you well before $30.
- Models and logs are copied to your Mac before any machine is stopped or deleted; then rented disks are deleted.

## Your to-do list (only you can do these)

1. **Tonight or Tue morning:** sign up at modal.com. Check that the $30 free credit shows and whether it asks for a card, and tell me what you saw. I'll then install Modal's helper tool, and you run one command (`modal setup`), which opens a browser login.
2. **Tonight or Tue morning:** create a Vast.ai account (and/or Runpod). **Don't add money yet**; wait for Tuesday's cost table.
3. When adding money, enter the access key yourself. I'll give exact steps; I won't type keys.
4. **Free up disk space:** 30 GB is free now, and the chip photos need about 13.4 GB while unpacking. More room means fewer problems.
5. **Keep the Mac plugged in and awake overnight.** I'll use the Mac's built-in keep-awake command.
6. Check Kaggle Settings → GPU hours left, and tell me.
7. Approve: each spend, each public post (Kaggle notebook / GitHub), the Friday video (your voice, my script), and the final **Submit** click.

## Files I'll create or update after you approve

- **New code folder `AI4S/code/`** containing:
  - data download and checks;
  - splits;
  - scoring, tested on fake answers first;
  - simple methods and image-AI training (frozen and retrained);
  - U-Net;
  - review-order curves and confidence correction;
  - RL (environment + rule-based controller + PPO + staged training + label-picking agent);
  - extra-dataset scripts;
  - launch scripts for Kaggle, Modal and Vast.

  Every run gets its own folder, `code/runs/<run-id>/`, with settings, log, predictions, scores, and measured time and cost.
- **Copy this plan into the project** as `Research_and_Experiments/11_Simple_Start_Plan_5_Oct.md`.
- **Update, never deleting entries:**
  - `10_All_Experiments_Parallel_Schedule.md`: new dates, live status of all 36;
  - `08_Run_Log.md`: every run, before and after;
  - `06_Evidence_and_Decisions.md`: your new rules (no fixed cut-offs, no skipping RL, $30 cap);
  - `Resources/08_tools_and_free_compute.md`: today's prices.
- **Save your rules to my memory** (no assumed numbers or fixed cut-offs, no skipping experiments, ~$30 cap for AI4S), so future sessions follow them too.

## How we'll check it's working

- **Before any model:** the scoring code passes fake perfect, all-wrong and empty tests. The final-test images are locked, with their file list and fingerprint saved.
- **Every run:** registered in the run log before it starts. Afterwards: measured time, memory, cost, and saved predictions.
- **Daily:** a status table for all 36: planned / running / measured / confirmed / rejected / blocked, with the exact reason. Plus money spent vs $30.
- **Thursday:** one locked final test.
- **Friday:**
  - a fresh Kaggle rebuild reproduces the saved numbers;
  - every number in the report is traced to a saved file;
  - links are checked while signed out;
  - the Writeup shows "submitted", not "draft".

## Known risks (honest)

- **Free GPU queues** → Modal and Vast as backup.
- **Mac disk** → big data stays in the cloud; tidy up as we go.
- **Licences:**
  - The chip data's licence conflicts (CC-BY-SA vs CC-BY 4.0), so we won't re-share the photos; our code downloads them from the source.
  - RxRx1 is non-commercial and share-alike, Allen limits re-sharing, and Cellpose's training data is non-commercial. Each is checked before its results are used.
- **C04** may have no usable outside chip data → recorded as blocked, with the search evidence.
- **No real, checked chip-flow simulator** may exist → R02–R04 run on the clearly labelled practice simulator.
- **Time:** if the speed tests show something can't finish, I'll tell you Tuesday evening with options. Nothing is quietly dropped.

## Added after the coverage audit (Tue 6 Oct, ~00:20 IST)

See [12_Coverage_Audit_6_Oct.md](12_Coverage_Audit_6_Oct.md). No experiment was missing. These steps from the playbook were not spelled out above and are now part of the plan:

- **G03:** a real-data pilot for each pipeline (16 real training examples, one short training pass, save → reload → same predictions, data-preparation time measured separately), in addition to the random-data speed test.
- **A01:** tables of correct/wrong per class and the share found per cell type. Any cut-off is chosen on practice data inside each fold, with the rule recorded.
- **A04:** small crop-position shifts, plus a check of whether cropping removes what the label depends on.
- **X01:** combine with sample details only after A07 shows they help.
- **B04:** one justified training change if the stress tests show a weakness.
- **C03:** a baseline that makes the same measurement directly from the plain photo.
- **R04:** scaling learned on training runs only; no future information; combine with R03 only after each effect is measured alone.
- **G01:** the licence of every pretrained model file.
- **F04:** report how many comparisons were tried; recheck the competition discussions before submitting; 200–300-word summary.
- **Needs your OK (no outreach without it):** contacting the chip-dataset authors about the licence conflict; contacting organisers about the open rule questions.
- **Measured:** Kaggle's L4 option is not available to this account (two tries both ran on 2× T4). The Kaggle lane is 2× T4.
