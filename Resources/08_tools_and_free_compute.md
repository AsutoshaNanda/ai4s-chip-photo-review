# Computers, costs, and tools for building the project

Updated: **3 October 2026, IST**. GPU means a processor used to speed up some AI work. The user requested free options and options around a **$20 total budget**. This is a shopping limit, not permission to rent or spend. The resources actually available in the account and the speed of our work are still unchecked.

## Shortlist for approximately $20 or less

Prices below were read on 3 October. They are displayed offers or public guidance, not resources we have booked. The hour calculations leave out storage, data transfers, taxes, and setup time.

| Website | Checked offer | Use for this plan |
|---|---|---|
| [Kaggle Notebooks](https://www.kaggle.com/docs/efficient-gpu-usage) | Public guidance says a weekly GPU allowance of 30 hours or sometimes more; the actual hours left vary. | First free option for small trial runs of image models. Check the account's actual allowance. |
| [Google Colab](https://research.google.com/colaboratory/faq.html) | Free GPU access varies, with no fixed guaranteed weekly hours. | Extra small trial runs when a GPU is available. |
| [Colab Pay As You Go](https://colab.research.google.com/signup) | The page displayed **$11.79 for 100 compute units**, without a subscription. Pro displayed $11.79/month for 100 units. | A paid notebook option within the requested budget. Compute units are credits, not hours; how quickly they are used depends on the assigned machine. |
| [Lightning AI](https://lightning.ai/pricing) | The live pricing FAQ says **5 free credits on registration, plus 25 for adding a card**. | A promotional offer if the account qualifies. Check the actual credited balance; do not treat advertised hours as guaranteed. |
| [Vast.ai](https://vast.ai/pricing) | Live public cards displayed **RTX 3090 24 GB from $0.11/hour, median $0.19/hour**; RTX 4090 24 GB from $0.29/hour, median $0.45/hour. Median means the middle listed price. | A possible low-cost rental. If an actual offer costs $0.19/hour, $15 buys about 79 hours before extras. The middle public price is not itself an offer we can reserve. |
| [Runpod Pods](https://www.runpod.io/pricing) | The pricing page lists **RTX A5000 24 GB at $0.27/hour**, RTX 3090 at $0.50/hour, and RTX 4090 at $0.74/hour. | Another rental option. $15 at $0.27/hour is about 56 hours before extras. Check availability and the full price quote. |

**Suggested budget split:** Try free options first. If a rental is needed, tentatively set aside $15 for computer use and $5 for other charges. Then check the full actual price against the $20 limit. This does not prove that all 36 experiments fit the budget. Save model progress files, called checkpoints, and test that they can be loaded before using a machine the provider may pause or take back.

Lightning's older official guide mentions credits each month, while the live pricing FAQ describes credits given at registration. Use the actual account offer; do not promise 80 free hours each month. Its displayed $20/month Pro price is charged for a full year, so it is not a $20 one-month purchase. [Current pricing](https://lightning.ai/pricing), [older guide](https://lightning.ai/docs/overview/ai-studio/)

[AWS now states that SageMaker Studio Lab is closed to new customers](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html), so it is not a new-signup option in this shortlist.

**We still need to measure whether these options fit:** These services provide online notebooks or GPU computers on which we can try our image models. GPU memory, time taken, availability, and whether the required software works together must be checked in an actual trial. No provider or GPU card has been tested for this project.

## Computer options

| Option | Verified source information | Check before choosing |
|---|---|---|
| [Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage) | The public guide describes a changing weekly GPU allowance and controls for managing sessions. Read in the browser on 2 October. | Actual hours left, available hardware, session limits, storage, and whether saved work reloads in this account. |
| [Google Colab](https://research.google.com/colaboratory/faq.html) | Access, hardware, and limits vary. Free sessions may end; they are not a set of workers for one job spread across several computers. | What this session actually provides, whether the job fits, and whether saved files survive a restart. |
| [Vast.ai](https://github.com/vast-ai/docs/blob/main/guides/pricing.mdx) | Hosts set rates for computer use, storage, and data transfers. A stopped rented machine can still have storage charges. | Actual offer, memory, host reliability, disk space, transfer rates, total cost, and how to save files and remove rented resources. |
| [Runpod](https://github.com/runpod/docs/blob/main/accounts-billing/billing.mdx) | Computer use and storage are billed separately; kept storage can still cost money after the computer stops. | Current offer, required credit, how long storage will be kept, and the full run cost. Back up needed outputs before removing resources. |
| Existing computer | No cloud rental is needed to inspect local files. | Actual hardware, free disk space and working memory (RAM), and measured speed. Do not assume CUDA, the NVIDIA GPU software, or any particular processor. |

These are options, not a ranking of the cheapest providers. None establishes a guaranteed free A100, free rental credit, unlimited time, or fixed account allowance. No account was created and no money was spent.

## Computer budget status on 5 October

**Correction to the running times suggested in chat:** The suggested 240 T4 / 120 L4 / 24 H100 / 48 RTX 4090 / 72 single-RTX-3090 / 48 dual-RTX-3090 hours were rough amounts to reserve. They were not calculated from working runs. Do not use them as precise forecasts or guaranteed completion times for all 36 plans. Some input sizes, stopping rules, and costs of environments for reinforcement learning (RL, learning from rewards) are still undefined. Giving a narrow range of possible times would be inventing precision.

### Same-day public price reference

The live [Vast.ai pricing cards](https://vast.ai/pricing) displayed these median rates, the middle listed prices, on 5 October. They are market references, not reserved offers. Multiplying suggested hours by a price gives a possible cost if those hours are used. It does not prove that the work takes that long.

| GPU setup | Earlier hours set aside without run measurements | Displayed median hourly rate | Possible cost for GPU use only |
|---|---:|---:|---:|
| T4 | 240 h | $0.15 | $36.00 |
| L4 | 120 h | $0.33 | $39.60 |
| H100 SXM | 24 h | $2.30 | $55.20 |
| RTX 4090 | 48 h | $0.46 | $22.08 |
| RTX 3090 | 72 h | $0.19 | $13.68 |
| 2 x RTX 3090 | 48 elapsed h | $0.38 combined, assuming two offers at the displayed single-card median | $18.24 |

These costs leave out storage, data transfers, taxes, setup, restarts, and any minimum charges. The two-card price is calculated, not a checked listing for a two-card machine. Running two independent experiments together may reduce the time you wait, but does not automatically halve the GPU-hours billed. Do not commit a full rental budget based on these unmeasured times. First time typical A/B workflows. Separately measure C03 and R02 to R05 once their actual tasks exist.

### Additional GPU candidates requested by the user

The following public card prices were read directly from Vast.ai on 5 October. The hours are clearly marked as budget examples. They were not worked out from a completed AI4S run, confidence in them is low, and they must not be presented as times in which all experiments will finish.

| GPU | Listed memory | Example hours set aside | Displayed median rate | Cost of those hours for GPU use only |
|---|---:|---:|---:|---:|
| RTX A5000 | 24 GB | 96 h | $0.23/h | $22.08 |
| RTX 5090 | 32 GB | 36 h | $0.60/h | $21.60 |
| A40 | 48 GB | 84 h | $0.79/h | $66.36 |
| RTX A6000 | 48 GB | 84 h | $0.40/h | $33.60 |
| L40S | 48 GB | 48 h | $0.80/h | $38.40 |
| A100 PCIe | 80 GB | 48 h | $0.66/h | $31.68 |
| A100 SXM4 | 80 GB | 36 h | $0.80/h | $28.80 |

The displayed A100 SXM4 minimum was $0.31/h, much lower than its $0.80 median. This makes a cheaper A100 offer worth checking, but the minimum does not prove that a suitable one-card machine is available. Check the actual card memory, minimum GPU count, rental type, and full charges. More memory alone does not prove faster training; the hours above for the different A100 types are not measured performance results.

The checked [run log](../Research_and_Experiments/08_Run_Log.md) still contains no measured training time. Total GPU hours for all 36 plans are therefore **unknown**, and no quoted estimate should be called measured. Some plans involve data checks, CPU calculations, repeating work, or presenting it. Some train several versions. C03 and R02 to R05 need datasets or practice environments that are still unverified. The number of plans is not the number of GPU training jobs.

From the saved [OOC archive file list](../Research_and_Experiments/Evidence/ooc_archive_inventory.json), the archive is 6,710,767,405 bytes and its unpacked files total 6,709,839,467 bytes. Keeping both would need about **13.42 GB of disk space before model files, outputs, and temporary files**. These numbers describe file sizes. They do not show a completed download or measured training storage needs.

At [Runpod's currently listed A5000 rate of $0.27 per GPU-hour](https://www.runpod.io/pricing), a **proposed $15 budget for computer use** buys about **55.6 GPU-hours before storage, taxes, and other costs**. This does not prove that 55.6 hours will finish the plan. When using the public [Kaggle GPU guide](https://www.kaggle.com/docs/efficient-gpu-usage), count only the allowance actually shown in the user's account. The public weekly figure is not an account balance.

To get an estimate backed by measurements, time each different workflow on the computer it will use: loading data, a typical training pass, making predictions, and saving/reloading. Record the number of training examples and the batch size, meaning how many examples are processed together. Then list every planned version, independent repeat, and final confirmation run. Add their estimated GPU times, actual setup/data-transfer time, and an allowance for restarts based on measurements. Compare this with the available budget. Mark each forecast as an estimate until the full run finishes. C03 and R02 to R05 need separate small trials after their required checks pass.

## Make the first small trial useful

G03 should record the hardware name, GPU memory, computer RAM, free disk space, data-preparation time, training/prediction time, whether saving and reloading works, and saved-file sizes. Start with a small valid set of training samples. Increase it enough to measure steady performance, and report which samples were used. A tiny loading check alone cannot reliably predict the time for full training.

For an actual offer, record its date/time and calculate the proposed cost from the measured work: computer time multiplied by the quoted rate, plus storage, data transfers, setup/restarts, and any required charges. Mark a time projected from a smaller run as an estimate. Leave unknown entries blank. A low hourly price alone does not prove a lower cost for the finished run.

## Running work at the same time

- With one available GPU, train models in priority order while CPU scoring, data checks, and writing happen alongside them.
- With several confirmed independent sessions, give A/B work separate sessions and start C/R work that meets its requirements when resources are free.
- Measure memory use and processing speed before running jobs together on one GPU.
- Use independent notebooks in ways the service allows. Do not assume free services allow training spread across computers, long unattended jobs, or ways around usage limits.
- Save settings, predictions, masks, and model progress files that can resume training outside temporary sessions; check that they reload.
- Release a computer session when its work is done, following the provider's current storage rules. Save the needed files first.

## Guides for building the project

| Work | Reference | Use and limit |
|---|---|---|
| Small model for assigning image categories | [Torchvision weights](https://docs.pytorch.org/vision/stable/models.html) | Choose exact versions of trained model files and their matching image-preparation steps. The documented `weights` interface replaces the old `pretrained` flag; check the installed version. |
| Simple nucleus comparison method | [Watershed example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html) | Existing official example kept; code not run here. |
| BBBC039 masks | [Provider-linked decoder](https://gist.github.com/jccaicedo/15e811722fca51e3ae90e8b43057f075) | Correctly read the masks that mark separate touching nuclei before scoring; check that the code works with our setup instead of copying it blindly. |
| Other current nucleus methods | [StarDist](https://github.com/stardist/stardist), [Cellpose](https://github.com/MouseLand/cellpose), [InstanSeg](https://github.com/instanseg/instanseg) | Check exact model files, usage rights, overlap with training images, and the real size each pixel represents. This list does not establish a winner. |
| Prediction using matching image pairs | [Allen author code](https://github.com/AllenCellModeling/pytorch_fnet) | A specific C03 reference. Its software setup and results have not been reproduced. |
| Learning with notebooks | [ZeroCostDL4Mic](https://henriqueslab.org/pages/zerocostdl4mic), [PyTorch basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html), [CellProfiler](https://cellprofiler.org/examples) | Earlier learning references kept; check the particular notebook and terms when using it. |
| Research on implementing PPO, a reinforcement-learning method | [Stable-Baselines3 documentation](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) | The fetched master page identifies a version still under development. Do not install it just because this link opens. Choose a supported release only after R01 establishes a valid environment. |

After the trial, record the exact software packages and model files that worked. Keep that setup for final testing. Do not call an untested list of software requirements a working setup.
