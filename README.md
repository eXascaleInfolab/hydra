# Master's thesis: Prostate Cancer Classification: A Transfer Learning Approach to Integrate Information From Diverse Body Parts


Julien Clément, Johan Jobin, Giuseppe Cuccu and Akansha Bhardwaj

## Structure
```bash
├───challenges
│   └───PROSTATEx
│       └───challenge_with_stacked_images
│           └───dataset_processing
├───datasets
│   ├───dataset_processing
│   │   ├───Kaggle_brain
│   │   │   ├───non_normalized
│   │   │   │   └───non_stacked_images
│   │   │   └───normalized
│   │   │       └───non_stacked_images
│   │   ├───Lung_CT_Challenge
│   │   │   ├───non_normalized
│   │   │   │   └───non_stacked_images
│   │   │   └───normalized
│   │   │       └───non_stacked_images
│   │   └───PROSTATEx
│   │       ├───non_normalized
│   │       │   ├───non_stacked_images
│   │       │   └───stacked_images
│   │       └───normalized
│   │           ├───non_stacked_images
│   │           └───stacked_images
│   ├───medical_format_to_png
│   ├───PROSTATEx_stacked_images_to_png
│   └───visualization
├───models
│   ├───paper_reproduction
│   │   ├───roulette
│   │   ├───test_model
│   │   └───train_model
│   └───transfer_learning
├───results
│   ├───paper_reproduction
│   │   ├───datasets_used
│   │   ├───entire_dataset_as_training_set
│   │   │   ├───20200216_paper_novalidation_notest_lr1e-7_dropout02_batchsize32
│   │   │   │   ├───DS1
│   │   │   │   └───stdout_logs
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_15_075
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_16_075
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_17_076
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_18_076
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_19_076
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_20_076
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_21_076
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_22_076
│   │   │   ├───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_23_075
│   │   │   └───20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_30_075
│   │   ├───images
│   │   ├───roulette
│   │   │   └───20191220_model_roulette_AUC
│   │   └───training_and_validation
│   │       ├───20200216_our_test_set_with_best_model_AUC_0.75
│   │       │   ├───stdout_logs
│   │       │   └───Test
│   │       ├───20200216_paper1_lr1e-7_batch32_100epoch_02dropout_reducelrauc001
│   │       │   ├───DS1
│   │       │   └───stdout_logs
│   │       └───20200217_CHALLENGE_test_set_with_best_model_AUC_071
│   └───transfer_learning
│       ├───datasets_used
│       ├───entire_pipeline
│       │   ├───DS1_Prostate
│       │   │   └───Full
│       │   │       └───20200228_DS1Full_P_1e-8_0.4_2000
│       │   │           ├───DS1
│       │   │           └───stdout_logs
│       │   ├───DS2_Brain
│       │   │   ├───Frozen
│       │   │   │   └───20200229_DS2Frozen_B_1e-7_0.3_noreducelr_2000
│       │   │   │       ├───DS2
│       │   │   │       └───stdout_logs
│       │   │   └───Full
│       │   │       └───20200301_DS2Full_B_1e-8_0.3_2000
│       │   │           ├───DS2
│       │   │           └───stdout_logs
│       │   ├───DS3_Lung
│       │   │   ├───Frozen
│       │   │   │   └───20200302_DS3Frozen_L_1e-5_0.3_2000
│       │   │   │       ├───DS3
│       │   │   │       └───stdout_logs
│       │   │   └───Full
│       │   │       └───20200303_DS3Full_L_1e-8_0.3_2000
│       │   │           ├───DS3
│       │   │           └───stdout_logs
│       │   └───DS4_Prostate
│       │       ├───Frozen
│       │       │   └───20200304_DS4Frozen_P_1e-5_0.3_2000
│       │       │       ├───DS4
│       │       │       └───stdout_logs
│       │       └───Full
│       │           └───20200305_DS4Full_P_1e-9_0.0_2000
│       │               ├───DS4
│       │               └───stdout_logs
│       ├───images
│       └───roulette
│           └───20200208_roulette_PROSTATE_one_channel_2ndsplit_auc
├───tests
│   ├───20191109_conversion_grayscale
│   ├───20191126_compare_numpy_pngs
│   ├───20191204_MNIST
│   ├───20191205_pytorch_functions_check
│   ├───20191206_imbalanced_sampler
│   ├───20191213_crop_align_images
│   ├───20191215_check_t2_dwi_adc_stacking
│   ├───20191219_gradient_flow_checking
│   ├───20191220_check_for_nan
│   ├───20200115_red_dots_checking
│   ├───20200121_global_performance_transfer_learning
│   │   └───.ipynb_checkpoints
│   └───20200124_xargs
└───utils
```
## Folders description

### challenges
Contains scripts to process the official PROSTATEx challenge data and to generate predictions for the challenge.
.
### datasets
Contains processing scripts for the PROSTATEx, the Lung CT Challenge and Kaggle Brain datasets.

### models
Contains scripts for the paper reproduction and our new approach based on transfer learning using multiple body parts.

### results
Contains the results of our experiments (paper reproduction and transfer learning).

### test
Contains test scripts to check the proper functioning of the code.

### utils
Contains useful scripts that can easily be imported into other projects.

## Abstract

Automating the detection of cancer participates to an an early detection and treatment, which increases the chances of recovery. Recent algorithms in artificial intelligence relying on deep learning have shown promising results in this field. Indeed,
the usefulness of convolutional neural networks (CNNs) for segmentation or classification tasks is no longer to be proven. However, the performance of these models is often limited by the amount of available data to train the algorithm.

This thesis first presents a state-of-the-art convolutional neural network for prostate lesion classification. All steps from the data processing to the smallest details regarding the neural network training are explained, ensuring a complete repro-
ducibility of the experiment. This model was evaluated on the official SPIE-AAPM-NCI Prostate MR Classification Challenge dataset and achieved an AUC of 0.76. This result constitutes a solid baseline and confirms the correct functioning of the
implementation.

On top of this implementation, a new transfer learning approach using lesions of multiple body parts (brain and lung) was built. This method shows that integrating information from diverse datasets improves automated prostate cancer diagnosis.
Indeed, it appears that lesions of these different types of cancer share low-level features that can be used to increase the generalization ability and performance of the prostate lesion classifier. This technique provides a concrete solution to the lack of available data for prostate classification and suggests that many other types of cancers can take advantage of it.

## Research paper reproduction
Based on Yang Song et al. “Computer-aided diagnosis of prostate cancer using a deep
convolutional neural network from multiparametric MRI: PCa Classification
Using CNN From mp-MRI”. en. In: Journal of Magnetic Resonance Imaging 48.6
(Dec. 2018), pp. 1570–1577. ISSN: 10531807.

### Process overview
![Process overview](/results/paper_reproduction/images/paper_reproduction_process.png)

### Data processing
![Data processing](/results/paper_reproduction/images/data_processing.png)

### Data splitting
![Data splitting](/results/paper_reproduction/images/paper_reproduction_split.png)

### Visual checking
![Visual checking](/results/paper_reproduction/images/alignment.png)

### Model architecture
![Model architecture](/results/paper_reproduction/images/model_paper_manual.png)

### Training metrics with the data split into a training (80%), a validation (10%) and a test set( 10%)
![Training metrics](/results/paper_reproduction/images/paper_reproduction_results.png)

### Results of the model on our test set
<img src="/results/paper_reproduction/images/auc_our_test_set.png" width="50%">

### Training metrics with the whole data available (100%)
![Training metrics with the whole dataset](/results/paper_reproduction/images/paper_reproduction_results_challenge_full_dataset.png)

### PROSTATEx challenge result with the model saved at epoch 21:
<img src="/results/paper_reproduction/images/paper_reproduction_results_challenge2.PNG" width="30%">

## Freeze/Unfreeze transfer learning using multiple body parts (Prostate, Brain, Lung)

### Main idea: split the model into a features extractor and a decision maker
![Features extractor and decision maker](/results/transfer_learning/images/tl_model_split.png)

### Process overview
![Features extractor and decision maker](/results/transfer_learning/images/tl_overview.png)

### PROSTATEx processing
![PROSTATEx keeping only DWI](/results/transfer_learning/images/PROSTATEx_processing.PNG)

### Lung CT Challenge processing
![Lung CT Challenge processing](/results/transfer_learning/images/lungCTChallenge_processing.PNG)

### Kaggle Brain processing
![Lung CT Challenge processing](/results/transfer_learning/images/kaggle_brain_processing.PNG)

### Training results
![DS1-full](/results/transfer_learning/images/tl_DS1_full.png)
![DS2-frozen](/results/transfer_learning/images/tl_DS2_frozen.png)
![DS2-full](/results/transfer_learning/images/tl_DS2_full.png)
![DS3-frozen](/results/transfer_learning/images/tl_DS3_frozen.png)
![DS3-full](/results/transfer_learning/images/tl_DS3_full.png)
![DS4-frozen](/results/transfer_learning/images/tl_DS4_frozen.png)
![DS4-full](/results/transfer_learning/images/tl_DS4_full.png)

### Exact metrics
![Transfer learning](/results/transfer_learning/images/tl_all_pipeline.PNG)

### Use the decision maker trained on the first dataset (PROSTATEx) on all models to quantify the usefulness of brain and lung features for prostate classification
![Global performance validation auc](/results/transfer_learning/images/tl_global_validation_auc.png)
![Global performance validation accuracy](/results/transfer_learning/images/tl_global_validation_accuracy.png)

## Conclusion
This work presented the process leading to the development of a deep learning system to classify potentially cancerous lesions, as well as strategies to overcome field related issues such as the lack of data. The starting point was the reproduction of Song et al.’s experiment. Reaching good performance in this part despite being able to reproduce every single trick
showed that our processing and training methods were working well. This resulted in a solid baseline that was exploited in order to take part in the "SPIE-AAPM-NCI Prostate MR Classification Challenge”, also called PROSTATEx challenge. Various
hyperparameters and ways of processing the data were tested in order to reach an AUC of 0.76 on this challenge. This score would have placed the model at the 15th position out of 71 submissions at the time of the challenge, which confirms the
robustness of the latter.

Then, the work focused on overcoming one of the main issues in deep learning: the lack of data. To achieve this, transfer learning, as well as more common techniques such as data augmentation, were applied. Our transfer learning implementa-
tion alternated between frozen and unfrozen steps and made use of brain and lung datasets to increase the model performance on a prostate dataset. During frozen steps, the first part of the model (the feature extractor) does not update its weights
at all, whereas the second part (the decision maker) does. Experiments showed that our method allowed to increase the AUC on our test set by approximately 18%, from 0.68 before transfer learning to 0.80 after transfer learning.

Throughout this thesis, various reusable tools were developed: visualization of medical imaging files, conversion of medical imaging files to PNG, a PyTorch sampler using undersampling, easy-to-use processing scripts for multiple datasets
(PROSTATEx, Kaggle Brain, Lung CT Challenge), processing verification tools (red dot images), training verification tools (gradient flow graphs, metrics plots using Tensorboard), an all-in-one training and testing file which can be adapted to new
models and datasets, an end-to-end transfer learning pipeline. All these elements can be used as a baseline for future workds or as additions to existing projects.
