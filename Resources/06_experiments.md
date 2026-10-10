# Experiments to run

**Status: these are planned comparisons; no model runs are recorded in the checked project folder as of 2 October.** The full E0 to E6 list below is kept and matched to [the complete 36-entry schedule](../Research_and_Experiments/10_All_Experiments_Parallel_Schedule.md). Independent comparisons can run at the same time once their required checks pass.

## Set up a fair test first

- Use BBBC039's published data groups. Training examples teach the model; validation examples help choose settings; test examples are saved for the final check.
- Keep every piece cut from an image in the same data group as that image.
- Check for repeated images and any images already used to teach an already-trained model.
- Save image IDs, settings, software versions, runtime, and predictions.
- Do not invent missing labels.

## Run in order

| ID | Question and comparison | Record | Decision |
|---|---|---|---|
| E0 | Do sample images match their reference outlines? | 20 training examples, loading errors, size checks | Fix mismatches before proceeding |
| E1 | How well does a brightness cutoff plus watershed, a method for separating touching objects, work? | Count error, object F1, time taken | Keep as the simple comparison |
| E2 | Does a small U-Net, an AI model that labels image pixels, plus object separation beat E1? | Same validation images and measurements | Keep only a useful improvement that can be repeated |
| E3 | Does teaching object edges reduce nuclei wrongly joined together? | E2 versus E2 with edge training | Keep if errors in crowded images fall without large losses elsewhere |
| E4 | Do flips and rotations help the model cope with changed images? | E2 versus one changed training setup | Keep only if validation supports it |
| E5 | Can a warning identify bad counts? | Different methods giving different answers versus a blur/brightness warning | Keep only if it catches more large errors for the same amount of review work |
| E6 | Does the chosen method still work in a fresh run and the final test? | Reloaded model; untouched test; a second number controlling training randomness, called a seed, if affordable | Report the result, including failures |

The order is a suggestion. Stop making small surface changes if the evidence supports a different method.

## Measurements in plain language

- **Count error:** Average absolute difference from the reference number of nuclei. Lower is better.
- **Object F1:** A score that balances missed nuclei and false detections. Higher is better. Match each predicted object with at most one reference object, using a required amount of overlap chosen before testing.
- **IoU:** The area shared by the predicted and reference masks divided by their total combined area, counting the shared area once. Proposed overlap needed for a match: 0.5. This is our testing choice, not the AI4S judging formula.
- **Runtime:** Seconds per image, with computer type recorded.

Show each method separately. Also show images with few/many nuclei, small/large nuclei, and the worst examples. Decide where these groups start and end using training/validation data before the final test.

## Warning experiment details

Use validation data to first define a “large count error” and how many images can be reviewed. Compare methods that flag the same percentage of images. Report errors across all images, errors among images not flagged, and how many bad cases were missed. Never hide hard images by leaving them out.

## Extra test, only if time remains

Test on another real dataset after checking repeated images, usage rights, and overlap with model-training data. Report it separately. Adding artificial blur or changing brightness can test how the method handles difficult conditions, but cannot replace a real second dataset.

Record every result in [the experiment log](14_experiment_log.md). If the final test fails, reduce what you claim the method can do; do not use that test data to choose new settings.

