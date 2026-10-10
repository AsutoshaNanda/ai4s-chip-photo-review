# Research library

Checked: **26 September 2026**. These are original papers, pages from authors and software maintainers, data providers, and official rules/guides. Each entry connects evidence to an experiment or decision.

**Update completed 3 October:** Original entries keep their source dates. The focused 2 October checks and new entries below do not mean every older paper was reread. See [the resource guide](../Resources/19_Resource_Readiness.md) and [the dated checks](Evidence/2026-10-02_source_checks.md) for what is ready and what could not be accessed.

**Source confidence** means how sure we are that the source supports the finding described. It does not mean its method will improve our project. Published results were not repeated here. Where only paper summaries or extracts shown in search results were readable, that limit is stated.

## Competition requirements

### S01

**[AI4S official overview](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/overview)**. Live browser, high confidence.

Expert judging covers importance/impact 30%, technical approach/innovation 30%, results/testing 20%, ability to repeat/build the work 10%, and presentation 10%. A public video, code, and report go with the Kaggle Writeup. Testing on real chips is encouraged. **Action:** test a useful process and leave time to finish the submission. **Plans:** G01, F02 to F04. These percentages cannot be used to predict an award.

### S02

**[AI4S official rules](https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien/rules)**. Live browser, high confidence in the displayed wording.

The rules allow outside models/data under their own terms, require credit and disclosure of earlier work, and ban invented results. Other owners' rights still apply. **Action:** record licenses separately for code, trained-model files, and data; identify what we actually added. **Plans:** G01, F04. These guides do not settle the legal meaning of disputed terms.

## Real-chip quality and practical use

### S03

**[Movčana and collaborators: Organ-on-a-Chip Image Dataset, 2023](https://zenodo.org/records/10203721)**. Provider record plus downloaded sample details; high confidence about access and the file list.

Real brightfield chip images, made with light passing through samples, have quality labels and cell-type details. The image archive is about 6.7 GB; the spreadsheet is about 120 KB. **Action:** check the small file of sample details first. **Plans:** G01, G02, A01 to A12. Our [direct check](03_Data_and_Model_Audit.md) adds evidence beyond the provider description. Independence between chips/runs has not been established.

### S04

**[Movčana et al., Organ-On-A-Chip Image Dataset for Machine Learning and Tissue Model Evaluation, Data, 2024](https://www.mdpi.com/2306-5729/9/2/28)**. Original text available through publisher search results reviewed; high confidence in the described design, moderate confidence that it applies to our setup.

The paper describes expert quality judgments, bad images containing image-taking defects, and an example using an already-trained MobileNetV3 model on pieces cut from image centers. It lists the dataset as CC-BY-SA, which conflicts with the repository API. **Action:** use a small already-trained model for comparison, test image cropping, and settle usage rights. **Plans:** G01, A02, A04. Existing work limits any claim that simply sorting chip images into two quality categories is new.

### S05

**[Supervised-Learning-Driven Interrogation of Organ-on-a-Chip Quality from Microscopy Images, Chem & Bio Engineering, 2025](https://pubs.acs.org/doi/full/10.1021/cbe.5c00087)**. Extracts in publisher search results reviewed; opening the full page directly returned error 403. Confidence moderate; methods were not fully checked.

The study reports chip-quality prediction for A549 and HSAEC. In an expanded task, its test results were weaker than its training results. **Action:** require results for each cell type and measure performance on new examples instead of borrowing its score. **Plans:** A01, A03, F01. This is earlier related work, not an independent test of our proposed model or proof of an independent dataset.

### S06

**[IQT Labs: An AI-based Application Using Organ-on-a-chip Data](https://www.iqt.org/library/an-ai-based-application-using-organ-on-a-chip-data)**. Description from the project team; moderate evidence, not a fair comparison test for our task.

The authors discuss differences in lighting/background, uncertain chip identities, and using model confidence to decide which quality cases need review. **Action:** test whether the model takes shortcuts from image surroundings and whether review becomes more efficient. **Plans:** A04, A09. Their reported success does not prove that our labels, images, or review order will behave similarly.

### S07

**[Yao et al., AI-assisted quantitative analysis for evaluating melanin distribution in 3D pigmented epidermis-on-a-chip models, 7 April 2026](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2026.1787959/full)**. Parts of the original paper reviewed; moderate evidence that it applies here.

The process connects marking regions in brightfield images with tissue-quality measurements that can be understood. **Action:** define a measurement for the specific task instead of claiming biological value from a general image score. **Plans:** C04, F02. The exact downloadable dataset and reuse terms were not checked. Do not turn its reported agreement rate into our target or expected result.

## Outlining and labeling nuclei

### S08

**[BBBC039: Nuclei of U2OS cells in a chemical screen](https://bbbc.broadinstitute.org/BBBC039)**. Provider page read; high confidence.

It provides 200 images of the DNA signal, colored markings for separate objects, published training/test groups, and CC0 terms. Some images also appear in BBBC038. **Action:** use it for a small nucleus test, reading separate-object masks correctly and checking image overlap. **Plans:** B01 to B07. It has no reference outcome for chip quality or harmful treatment effects.

### S09

**[BBBC038: Kaggle 2018 Data Science Bowl](https://bbbc.broadinstitute.org/BBBC038)**. Provider page read; high confidence.

Varied imaging conditions and nucleus markings make this a possible second dataset for testing object outlines. **Action:** choose a usable subset only after checking shared images and overlap with earlier model training. **Plan:** B07. Variety does not prove independence from BBBC039 or from a general-purpose model's training data.

### S10

**[Caicedo et al., Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl, Nature Methods, 2019](https://www.nature.com/articles/s41592-019-0612-7)**. Parts of the original article reviewed; high confidence in its reasons for the work, but only indirect evidence for choosing a method now.

This work studies object outlining across different experiments and explains why performance on new conditions matters. **Action:** test individual objects and unfamiliar conditions, not only pixels marked as objects. **Plans:** B02, B05, B07. Older winning model designs and software setups are not current recommendations.

### S11

**[Schmidt et al., Cell Detection with Star-convex Polygons, 2018](https://arxiv.org/abs/1806.03535)**. Original paper summary read; high confidence in the method description.

Star-shaped object predictions help separate touching nuclei. **Action:** compare a model that predicts separate objects with pixel labeling followed by watershed, a method for splitting touching regions. **Plans:** B02, B03, B05. Shapes the model cannot represent and different labeling styles may cause problems; any improvement here is unmeasured.

### S12

**[StarDist maintainer repository](https://github.com/stardist/stardist)**. Current code repository reviewed; high confidence that code is available.

Provides maintained code and links to model guides. **Action:** check the training data and rights of the chosen already-trained model before use; record its exact release. **Plans:** G01, B02. A code license does not replace a full check of shared training/test images.

### S13

**[Pachitariu and Stringer, Cellpose 2.0: how to train your own model, Nature Methods, 2022](https://www.nature.com/articles/s41592-022-01663-4)**. Parts describing methods/results in the original paper reviewed; moderate confidence that they apply to our work.

The paper studies repeated correction and further model training with limited labeling effort. **Action:** compare correction effort against a fixed budget of randomly chosen images, keeping the validation data untouched for checking settings. **Plans:** B06, F02. The small numbers of labeled examples that worked in the paper are not guaranteed to work for our images.

### S14

**[Pachitariu, Rariden and Stringer, Cellpose-SAM: Superhuman generalization for cellular segmentation, 2025](https://www.janelia.org/node/69901)**. Paper summary on the authors' institution website read; moderate confidence that it applies here.

The model adapts SAM within Cellpose and tests how it handles several image changes. **Action:** include a current general-purpose model for comparison when rights and earlier training exposure allow it. **Plans:** B02, B04. Its title and published test claims do not prove it is better on our independent test.

### S15

**[Current Cellpose repository](https://github.com/MouseLand/cellpose)**. Maintainer page read; high confidence for current notices.

The June 2026 update lists new DINO-based models and an updated SAM model. It also notes CC-BY-NC training data. **Action:** use current guides, record the exact model, and settle which terms apply. **Plans:** G01, B02. Do not assume old Cellpose programming interfaces still work, or that the code license alone automatically permits or bans the intended use.

### S16

**[Goldsborough et al., InstanSeg: an embedding-based instance segmentation algorithm optimized for accurate, efficient and portable cell segmentation, 2024](https://arxiv.org/abs/2408.15954)**. Original paper summary read; moderate confidence that it applies to our work.

It offers another way to outline separate objects and studies speed as well as accuracy. **Action:** measure actual elapsed time, memory, and object errors on the same images. **Plans:** B02, G03. Published speed improvements depend on the computer and test used.

### S17

**[InstanSeg maintainer repository](https://github.com/instanseg/instanseg)**. Current code repository read; high confidence in the described programming interfaces.

The repository supports fluorescence and brightfield images and records model versions. The image type and real size represented by a pixel matter. **Action:** inspect the exact model entry and training sources, then try a small image sample. **Plans:** G01, G03, B02. An example in a model list does not prove the usage rights of a particular model file.

### S18

**[Segment Anything for Microscopy, Nature Methods, 2025](https://www.nature.com/articles/s41592-024-02580-4)**. Original paper summary and correction notice read; moderate confidence that it applies to our work.

SAM adapted for microscope images supports automatic outlining and outlining with user input through napari, an image-viewing tool. **Action:** test a correction process only if the target is labeling effort. **Plans:** B06, F02. Read the updated article and author correction before trying to repeat it exactly; this is not a tested model of harmful treatment effects.

## Confidence, failure detection, and independent testing

### S19

**[Zenk et al., Comparative Benchmarking of Failure Detection Methods in Medical Image Segmentation: Unveiling the Role of Confidence Aggregation, 2024](https://arxiv.org/abs/2406.03323)**. Original paper summary read; indirect evidence because it studies outlining regions in 3D medical images.

The study recommends measuring error alongside the share of cases kept without review, and finds agreement between pairs of methods a useful simple comparison. **Action:** compare disagreement-based warnings with blur, random review, and models that predict errors directly. **Plans:** A09, A10, B06. Agreement between whole-image masks can hide count errors; test disagreements about objects and counts separately.

### S20

**[Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://arxiv.org/abs/1706.04599)**. Original paper summary read; an established general method that has not been measured on our data.

Temperature scaling adjusts model confidence values to make them more meaningful. **Action:** learn this adjustment on separate calibration data and check Brier score and log loss, two measures of probability error. **Plan:** A08. For a two-category model's raw scores, a positive temperature keeps the same ranking and 0.5 decision boundary. Do not promise a gain in AUC, a ranking score, or in review order at a fixed workload from this adjustment.

### S21

**[Ovadia et al., Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift, NeurIPS 2019](https://arxiv.org/abs/1906.02530)**. Original paper summary read; high confidence that changed data needs testing, but indirect evidence for this dataset.

Confidence methods can become less reliable when the test data changes. **Action:** test probability adjustments and review rules again for each cell type and image-collection group. **Plans:** A03, A08, A09, B07. Good confidence estimates on familiar data do not guarantee reliability on new chips.

### S22

**[Lakshminarayanan, Pritzel and Blundell, Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, 2017](https://arxiv.org/abs/1612.01474)**. Original paper page and summary reviewed; moderate confidence that it applies here.

Separately trained models can help estimate uncertainty. **Action:** first check whether two candidate models make different errors, then compare their combined predictions under an equal budget. **Plan:** A10. Two similar networks are not automatically independent protection against failure; they can agree and both be wrong.

## Cell appearance and behavior, and other outcomes to measure

### S23

**[Recursion: RxRx1](https://www.rxrx.ai/rxrx1)**. Dataset-owner explanation read; high confidence in dataset purpose.

Repeated lab batches make differences caused by the equipment or process a major challenge. **Action:** compare a category model using image measurements, or a method that finds similar examples, across experiments kept out of training. **Plan:** C01. Labels for genetic changes are not labels for harmful effects in patients. Claims about specific genes need care because the provider describes effects on unintended targets.

### S24

**[JUMP Hub: Retrieve JUMP profiles](https://broadinstitute.github.io/jump_hub/howto/notebooks/11_retrieve_profiles.html)**. Official tutorial read; high confidence for documented access options.

Ready-made tables of cell measurements allow small, limited experiments without downloading all the images. Some processing options already correct differences between lab batches. **Action:** record how the table was made and check whether its preparation used samples reserved for testing. **Plan:** C02. Do not call a table corrected using the whole dataset a change learned only from training data.

### S25

**[Treis et al., Same Encoder, Different Winner: A Paired-View Framework for Cell Painting Encoder Evaluation, 11 September 2026](https://arxiv.org/abs/2609.12761)**. Summary of a recent paper released before formal review read; early evidence, not yet confirmed.

The authors report that rankings of models that extract image features depend on the test plan and image surroundings. **Action:** keep matching biological examples across lab batches separate from matching within one study; compare image surroundings under controlled conditions. **Plans:** A04 uses this as a related example, C02 uses it directly. The summary says files will be released; their availability was not checked. Do not make the project depend on them.

### S26

**[Christiansen et al., In silico labeling: Predicting fluorescent labels in unlabeled images, Cell, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6309178/)**. Parts of the original article reviewed; an established research area that can work here only if the requirements are met.

Matching image pairs can teach a model to predict fluorescent structures from unstained images. **Action:** require real, aligned pairs and test the structures used for later measurements, not just whether the images look similar. **Plan:** C03. The checked chip dataset does not establish that matching fluorescence images exist.

## Current guides for writing the code

### S27

**[Torchvision: Models and pretrained weights](https://docs.pytorch.org/vision/stable/models.html)**. Current official guide read; high confidence in the programming guidance, but not tested here.

Use exact trained-model files and matching image-preparation steps instead of old examples with unclear default settings. **Plans:** A02, G03. After a small check that the setup works, keep the working torch/torchvision versions fixed. A current guide does not prove that installed versions work together.

### S28

**[scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)**. Current official documentation read; high confidence.

Related samples can be kept in the same group when dividing data. **Action:** keep physical sample groups together and learn data-preparation changes separately inside each training fold, the training part of a repeated test. **Plans:** G02, A03, C01, C02. Group-splitting software cannot fix an incorrect or unverified definition of a group.

### S29

**[scikit-learn: Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)**. Current official documentation read; high confidence.

Adjusting probability estimates needs data separate from model training and suitable scoring. **Action:** use separate groups for this adjustment, or record a group-aware cross-fitting method, which rotates which groups are used to learn and check the adjustment. **Plan:** A08. For small groups, report uncertainty instead of claiming precise reliability.

### S30

**[scikit-learn: Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)**. Current official documentation read; high confidence.

Test information can leak in through rescaling data, filling missing values, selecting useful measurements, or repeatedly changing choices based on test results. **Action:** keep one fixed set of data-use rules and data-preparation steps. **Plans:** G02, A07, C02, F01. A high score after such leakage is not a reason to keep the method.

### S31

**[scikit-image: Watershed segmentation](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html)**. Current official example read; high confidence for a simple comparison method.

Starting markers can separate connected object regions. **Action:** establish a simple brightness-cutoff/watershed comparison and choose settings only on validation data. **Plans:** B01, B03. A small illustrative example is not a measured result on microscope images.

## Coverage and remaining limits

This search covered research directly about chip quality, current nucleus-outlining methods, human correction, adjusting confidence estimates, finding failures, lab-batch differences, tables of cell measurements, and predicting fluorescence. It gave priority to accessible data, fair tests, current guides, and finishing within roughly two weeks.

The original 26 September notebook reviews describe that earlier state. The [notebook list](../Resources/10_current_notebooks.md) was updated on 2 October, including a focused review of a notebook that checks image-collection groups. Saved outputs were not rerun. Participant notebooks, votes, paper titles, and public test rankings do not prove that an approach will win AI4S.

This does not claim to cover every relevant paper or guarantee experiment success. New direct evidence must update [the decision record](06_Evidence_and_Decisions.md).

## Resource additions checked on 2 October

### S32

**[Allen label-free training-data index](https://downloads.allencell.org/publication-data/label-free-prediction/index.html)**. Provider links checked; image files untested. **Action:** check a real matching-image subset for C03, including alignment between image signals and sample grouping. This gives a specific lead, not a completed test on chips.

### S33

**[Allen Cell terms](https://www.allencell.org/terms-of-use.html)**. Original terms read. **Action:** check the intended sharing of the C03 report/video/code against the conditions that apply. This does not establish final permission or settle the separate disagreement about the OOC dataset.

### S34

**[AllenCellModeling/pytorch_fnet](https://github.com/AllenCellModeling/pytorch_fnet)**. The provider links the authors' code repository, which was checked through the web reader. **Action:** examine a small C03 example and its exact software setup before reuse. No installation, code run, or repeat measurement of performance was completed.

### S35

**[Audit acquisition groups in organ-on-chip images](https://www.kaggle.com/code/loxigicck/audit-acquisition-groups-in-organ-on-chip-images)**. Participant notebook, version 3: visible text, code sections, and saved check output inspected. This is work reported by someone else, not our measurement. **Action:** cite it as related work for G02/A03/A09; test a separate practical benefit. It reports numerical differences when repeating results and limits what can be concluded from filename beginnings. The linked full repository was unavailable through the web reader and its broader model experiments were not rerun.

### S36

**Computer-provider guides:** [Kaggle](https://www.kaggle.com/docs/efficient-gpu-usage), [Colab](https://research.google.com/colaboratory/faq.html), [Vast.ai](https://github.com/vast-ai/docs/blob/main/guides/pricing.mdx), and [Runpod](https://github.com/runpod/docs/blob/main/accounts-billing/billing.mdx). Public terms/guidance checked; no account quote or actual available resources verified. **Action:** use G03 measurements and actual offers to decide resource needs and cost. Do not assume guaranteed free GPUs or a cheapest provider.
