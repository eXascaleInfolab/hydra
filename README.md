# Hydra: Cancer Detection Leveraging Multiple Heads and Heterogeneous Datasets

Authors: Dr. Giuseppe Cuccu, Johan Jobin, Julien Clément, Akansha Bhardwaj, Dr. Carolin Reischauer, Prof. Dr. med. Hariett Thöny, Prof. Dr. Philippe Cudré-Mauroux

## Abstract
We propose an approach combining layer freezing and fine-tuning steps alternatively to train a neural network over multiple and diverse datasets in the context of cancer detection from medical images. Our method explicitly splits the network into two distinct but complementary components: the feature extractor and the decision maker. While the former remains constant throughout training, a different decision maker is used on each new dataset. This enables end-to-end training of the feature extractor on heterogeneous datasets (here MRIs and CT scans) and organs (here prostate, lung and brain). The feature extractor learns features across all images, with two major benefits: (i) extended training data pool, and (ii) enforced generalization across different data. We show the effectiveness of our method by detecting cancerous masses in the SPIE-AAPM-NCI Prostate MR Classification data. Our training process integrates the SPIE-AAPM-NCI Lung CT Classification dataset as well as the Kaggle Brain MRI dataset, each paired with a separate decision maker, improving the AUC of the base network architecture on the Prostate MR dataset by 0.12 (18\% relative increase) versus training on the prostate dataset alone. We also compare against standard end-to-end Transfer Learning over the same datasets for reference, which only improves the results by 0.04 (6\% relative increase).

## Structure
```
├── datasets
│   ├── PROSTATEx_stacked_images_to_png
│   ├── dataset_processing
│   │   ├── Kaggle_brain
│   │   │   ├── non_normalized
│   │   │   │   └── non_stacked_images
│   │   │   └── normalized
│   │   │       └── non_stacked_images
│   │   ├── Lung_CT_Challenge
│   │   │   ├── non_normalized
│   │   │   │   └── non_stacked_images
│   │   │   └── normalized
│   │   │       └── non_stacked_images
│   │   └── PROSTATEx
│   │       ├── non_normalized
│   │       │   ├── non_stacked_images
│   │       │   └── stacked_images
│   │       └── normalized
│   │           ├── non_stacked_images
│   │           └── stacked_images
│   ├── medical_format_to_png
│   └── visualization
├── models
│   └── transfer_learning
├── results
│   ├── hydra
│   │   ├── datasets_used
│   │   ├── entire_pipeline
│   │   │   ├── DS1_Prostate
│   │   │   │   └── Full
│   │   │   │       └── 20200228_DS1Full_P_1e-8_0.4_2000
│   │   │   │           └── DS1
│   │   │   ├── DS2_Brain
│   │   │   │   ├── Frozen
│   │   │   │   │   └── 20200229_DS2Frozen_B_1e-7_0.3_noreducelr_2000
│   │   │   │   │       └── DS2
│   │   │   │   └── Full
│   │   │   │       └── 20200301_DS2Full_B_1e-8_0.3_2000
│   │   │   │           └── DS2
│   │   │   ├── DS3_Lung
│   │   │   │   ├── Frozen
│   │   │   │   │   └── 20200302_DS3Frozen_L_1e-5_0.3_2000
│   │   │   │   │       └── DS3
│   │   │   │   └── Full
│   │   │   │       └── 20200303_DS3Full_L_1e-8_0.3_2000
│   │   │   │           └── DS3
│   │   │   └── DS4_Prostate
│   │   │       ├── Frozen
│   │   │       │   └── 20200304_DS4Frozen_P_1e-5_0.3_2000
│   │   │       │       └── DS4
│   │   │       └── Full
│   │   │           └── 20200305_DS4Full_P_1e-9_0.0_2000
│   │   │               └── DS4
│   │   ├── images
│   │   ├── roulette
│   │   │   └── 20200208_roulette_PROSTATE_one_channel_2ndsplit_auc
│   │   └── test_results
│   │       ├── Test
│   │       └── stdout_logs
│   └── standard_TL
│       ├── datasets_used
│       ├── entire_pipeline
│       │   ├── DS1_Prostate
│       │   │   └── Full
│       │   │       └── 20200501_P_1e-8_0.4_2000
│       │   │           ├── DS1
│       │   │           └── stdout_logs
│       │   ├── DS2_Brain
│       │   │   └── Full
│       │   │       └── 20200510_B_1e-8_0.3_2000_3_ok
│       │   │           ├── DS2
│       │   │           └── stdout_logs
│       │   ├── DS3_Lung
│       │   │   └── Full
│       │   │       └── 20200512_L_1e-8_0.3_2000_ok
│       │   │           ├── DS3
│       │   │           └── stdout_logs
│       │   └── DS4_Prostate
│       │       └── Full
│       │           └── 20200513_P_1e-8_0.0_2000_ok
│       │               ├── DS4
│       │               └── stdout_logs
│       ├── roulette
│       │   └── 20200208_roulette_PROSTATE_one_channel_2ndsplit_auc
│       └── test_results
│           ├── Test
│           └── stdout_logs
└── utils
```
## Folders description

### datasets
Contains processing scripts for the PROSTATEx, the Lung CT Challenge and Kaggle Brain datasets.

### models
Contains scripts to run the actual model training.

### results
Contains the results of our experiments.

### utils
Contains useful scripts.
