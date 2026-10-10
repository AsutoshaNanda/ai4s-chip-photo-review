from pathlib import Path
import json
import re

HERE = Path(__file__).resolve().parent
source = HERE / 'before_more_figures/kaggle_writeup.md'
text = source.read_text()
changes = []

def replace(old, new):
    global text
    count = text.count(old)
    if not count:
        raise ValueError(old)
    text = text.replace(old, new)
    changes.append({'before': old, 'after': new, 'count': count})

replace('It uses a frozen DINOv2 image model and a small trained scorer.', 'It uses an existing DINOv2 image model without more training. A small trained model calculates the review score.')
replace('7. Apply the saved scorer and sort photos by descending score.', '7. Apply the saved scorer.\n8. Sort the photos by descending score.')
replace('1. Obtain the project package and the saved scorer from the public link when it is available.', '1. When the public link is available, download the project package and the saved scorer.')
replace('We checked file integrity, image decoding and duplicate overlap.', 'We did checks of file integrity, image decoding and duplicate overlap.')
replace('Later clean rebuilds checked reproduction with the same choice.', 'Later clean rebuilds tested reproduction with the same choice.')
replace('No physical chip was tested.', 'We did not test a physical chip.')
replace('The scorer was rebuilt with 2,397 development photos under scikit-learn 1.9.1.', 'We rebuilt the scorer with 2,397 development photos under scikit-learn 1.9.1.')
replace('The rule was written before the light-retraining results were known.', 'We wrote the rule before we knew the light-retraining results.')
replace('Both groups need more independent examples before a strong reliability claim.', 'More independent examples are necessary for the two groups before a strong reliability claim.')
replace('The first run also needs access to the model download.', 'The model download must be accessible for the first run.')
replace('Public access and the final package still need checking.', 'Public access and the final package still require a check.')
replace('The favourable average does not remove important exceptions:', 'The lower average error does not remove these exceptions:')
replace('At strong added noise, the retained outline models fall to strict scores', 'At strong added noise, the retained outline models decrease to strict scores')
replace('its strict score falls to 0.806', 'its strict score decreases to 0.806')
replace('R01 checks the published steady-state equations used by the simulator.', 'R01 tests the published steady-state equations that the simulator uses.')
replace('Check whether image features transfer across batches.', 'Test whether image features transfer across batches.')
replace('Check whether edges and camera format supply shortcuts.', 'Test whether edges and camera format supply shortcuts.')
replace('Check reproduction in a fresh environment.', 'Test reproduction in a fresh environment.')
replace('Check shared failures behind disagreement warnings.', 'Test shared failures behind disagreement warnings.')
replace('Check source and publication terms.', 'Do a check of source and publication terms.')
replace('Check paths and reported numbers.', 'Do checks of paths and reported numbers.')
replace('The current report covers the recorded work.', 'The current report describes the recorded work.')
replace('These results use different settings.', 'The count and outline results use different settings.')
replace('A match needs at least 0.75 overlap.', 'A match requires at least 0.75 overlap.')
replace('On 108 colour images, the strict outline score is zero for every tested setting.', 'On 108 colour images, all tested strict outline scores are below 0.001. Several values round to 0.000 without being exactly zero.')
replace('Colour nucleus strict score reaches zero; models share errors', 'Colour nucleus strict scores are below 0.001; models share errors')
replace('Colour strict-outline score is zero.', 'Colour strict-outline scores are below 0.001.')
text = re.sub(r'(?<![\d.])7\.25(?!\d)', '7.255', text)
replace('This draft is not certified as fully compliant with its dictionary and grammar rules.', 'We revised the text after a partial language review on 9 October 2026. Full compliance with the dictionary and grammar rules remains unverified.')
replace('- Add the public code link and verify access without login or payment.', '- Add the public code link.\n- Make sure that the public code is accessible without login or payment.')
replace('- Choose the code licence and complete resource credits.', '- Choose the code licence.\n- Add missing resource credits.')
replace('- Settle chip-image licence wording and final C01/C03 publication scope.', '- Resolve the chip-image licence wording.\n- Decide the final C01/C03 publication scope.')
replace("- Check the exact public installation instructions and account for F06's failed strict test.", '- Test the exact public installation instructions.\n- Resolve the failed F06 test before you claim that the strict installation path passes.')

word_map = {'grey': 'gray', 'colour': 'color', 'colours': 'colors', 'labelled': 'labeled', 'labelling': 'labeling', 'licence': 'license', 'licences': 'licenses', 'programme': 'program', 'centre': 'center', 'micrometre': 'micrometer', 'micrometres': 'micrometers', 'metre': 'meter', 'favourable': 'favorable', 'main': 'primary', 'both': 'the two', 'however': 'but', 'further': 'more', 'people': 'persons'}
protected = r'(```[\s\S]*?```|`[^`\n]+`|!?\[[^\]]*\]\([^)]+\))'
parts = re.split(protected, text)
for i in range(0, len(parts), 2):
    for old, new in word_map.items():
        def change_case(match):
            return new.capitalize() if match.group(0)[0].isupper() else new
        parts[i] = re.sub(r'\b' + old + r'\b', change_case, parts[i], flags=re.I)
text = ''.join(parts)

replace('| Training seed | A number that controls random choices during training. Different seeds test repeatability. | No preferred value. |', '| Training seed | A number that controls random choices during training. Different seeds test repeatability. | No preferred value. |\n\nThe project uses these technical terms consistently:\n\n| Project term | Meaning in this report |\n|---|---|\n| Frozen model | An image model whose weights do not change during the experiment. |\n| Feature | A numerical description that an image model produces. |\n| Scorer | The small trained model that converts image features to a review score. |\n| Tile | One rectangular part of a prepared image. |\n| Nucleus | The part of a cell that contains most of its DNA. The plural is nuclei. |\n| Foreground | Image regions that the model identifies as nuclei. |\n| Reference mask | An image that marks the reference object regions. |\n| Model weight | A stored number that determines how a model calculates an output. |\n| Plate | A laboratory container with multiple sample wells. |\n| RL | Reinforcement learning: training that uses feedback from actions. |\n| CPU / GPU | A general computer processor / a processor suited to many parallel calculations. |\n| RGB | An image with red, green and blue color channels. |\n| CSV | A text file that stores a table as comma-separated values. |\n| DIC | Differential interference contrast, a microscope imaging method. |')

replace('### 4. Primary photo results', '![Development scores before and after group exclusion, and scores for the three camera formats.](figures/09_validation_and_formats.png)\n\n*Figure 9. The corrected test reduces the apparent scores. The tile method remains better in that test. Small photos remain a separate failure. The two panels use development data. They do not show new final tests.*\n\n### 4. Primary photo results')
replace('Disagreement between B02 and B03 gives only a partial warning.', '![Actual B03 outline-setting examples with reference and predicted boundaries.](figures/12_b03_outline_examples.png)\n\n*Figure 12. Unchanged B03 development output at the outline settings. The source gallery selects the six largest count errors, not the largest outline errors. Green marks reference outlines. Magenta marks predictions. This selection illustrates limitations and is not a random sample. Images and masks: BBBC039, CC0.*\n\nDisagreement between B02 and B03 gives only a partial warning.')
replace('No timed review study tested that assumption.', 'No timed review study tested that assumption.\n\n![Final nucleus review curves and development error shares from shared model failures.](figures/10_review_and_shared_failures.png)\n\n*Figure 10. The left panel uses the 50 final images. It assumes that a person corrects each reviewed count. The right panel uses 150 development images. Shared errors limit a warning based on disagreement. Perfect knowledge is a reference, not an available review method.*')
replace('InstanSeg performs better in the recorded development comparison.', '![Nucleus count and outline performance on development, outside gray, and outside color images.](figures/11_outside_nucleus_test.png)\n\n*Figure 11. The left panel uses count settings. The right panel uses outline settings. The outside color scores are below 0.001. The datasets are separate tests. Differences between their scores do not isolate the cause of failure.*\n\nInstanSeg gives better results in the recorded development comparison.')
replace('The corrected C02 comparison uses the same ten training batches for every candidate strength.', '![Biological matching and batch identification in C02, plus direct and predicted-image measurement in C03.](figures/13_supporting_route_comparisons.png)\n\n*Figure 13. C02 improves biological matching but increases batch identification. C03 gives higher count error after fluorescence prediction. The graph shows the saved direct error as 7.255. The panels describe separate tasks and do not form a combined score.*\n\nThe corrected C02 comparison uses the same ten training batches for every candidate strength.')
replace('R05 asks a different question using real chip-photo data:', '![Ordinary and staged control training, with a heat map of input and noise stress tests.](figures/14_rl_training_and_inputs.png)\n\n*Figure 14. The left panel uses harder simulator ranges. The right panel uses training ranges. Numbers show mean diameter error in micrometers. Each value averages five saved training seeds within 50 scenarios. These are related simulator comparisons, not independent physical tests.*\n\nR05 asks a different question using real chip-photo data:')

order = re.findall(r'!\[[^\]]*\]\(figures/(\d+)[^)]*\)', text)
mapping = {int(old): i + 1 for i, old in enumerate(order)}
text = re.sub(r'\bFigure (1[0-4]|[1-9])\b', lambda m: f'Figure {mapping[int(m.group(1))]}', text)
for old, new in [('grey', 'gray'), ('colour', 'color'), ('micrometres', 'micrometers')]:
    text = re.sub(r'(!\[[^\]]*)\b' + old + r'\b', lambda m: m.group(0).replace(old, new), text)

(HERE / 'kaggle_writeup.md').write_text(text)
(HERE / 'language_changes.json').write_text(json.dumps({'replacements': changes, 'word_map': word_map, 'figure_number_mapping': mapping, 'numeric_corrections': ['C03 direct count error: exact saved value 7.255', 'B07 color strict scores: below 0.001, not all exactly zero']}, indent=2))
print(json.dumps({'figures': len(order), 'words': len(text.split()), 'figure_number_mapping': mapping}, indent=2))
