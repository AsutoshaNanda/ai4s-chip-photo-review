# Similar competitions and ideas to test

These are **findings from other projects**. Their scores do not predict an AI4S result.

## 1. Data Science Bowl 2018: outline cell nuclei

The [dataset provider](https://bbbc.broadinstitute.org/BBBC038) describes images from varied biological and imaging conditions. The [organizer's analysis](https://github.com/carpenterlab/2019_caicedo_dsb) links these top three solutions:

| Place / author | What their public source shows | Small idea to test here |
|---|---|---|
| [1st: ods.ai topcoders](https://github.com/selimsef/dsb2018_topcoders) | Multiple model folders, neural-network and decision-tree models, training in two stages | Save predictions from separate methods so their errors can be compared |
| [2nd: Jacobkie](https://github.com/jacobkie/2018DSB) | A changed U-Net image model with eight position outputs; image patterns learned in earlier training; predictions at four image sizes | Test separation that uses object edges before trying a full system that works at several image sizes |
| [3rd: Deep Retina](https://github.com/Lopezurrutia/DSB_2018) | Mask R-CNN, a model for finding and outlining objects, with image patterns learned earlier, resizing, flips, and rotations | Test flips/rotations one change at a time |

These small test ideas are our adaptations. They do not exactly repeat the winning systems.

**Pitfall:** First-place instructions use CUDA 9 and cuDNN 7. These are historical setups, not current installation guidance. Do not run old scripts blindly.

## 2. Recursion / CellSignal 2019: distinguish biology from lab conditions

The [dataset owner's explanation](https://www.rxrx.ai/rxrx1) describes predicting genetic treatments using separate lab experiment batches.

**Apply here:** When sample details allow it, test on lab runs that were not used for training. For treatment tasks, report each cell type and batch separately.

**Reference we may reuse:** [Official starter code](https://github.com/recursionpharma/rxrx1-utils), especially the code for showing images. Its older training setup needs checking.

## 3. Sartorius 2021: outline individual neuronal cells

[Official competition](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/overview)

Phase-contrast images, made with a method that makes cells easier to see, need separate outlines for irregularly shaped cells. Testing matches each predicted object to a reference object.

**Apply here:** Check objects that were wrongly joined or missed, not just whether the pixels marked as objects look right. Learn from the method; outlining nuclei and outlining whole cells are different tasks.

## What to reuse first

Start with the [current watershed example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html) and [UNet4Nuclei reference](https://github.com/carpenterlab/unet4nuclei/). These are code examples to inspect. They have not been checked to run in our setup.

Do not bring submission formats or tricks for climbing the score table from past competitions into this hackathon.

