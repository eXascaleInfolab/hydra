###### Transfer learning ######

# Data used
Prostate: sftp://hfr_cs@diufrd141.unifr.ch/HOME/hfr_cs/Data/Prostate/ProstateX/l/20200124_prostateDWIOnlyAugmentedNormalizedSize65x60BalancedNewCropping
Brain: sftp://hfr_cs@diufrd141.unifr.ch/HOME/hfr_cs/Data/Brain/Kaggle/brainAugmented200NormalizedNewCroppingV2
Lung: sftp://hfr_cs@diufrd141.unifr.ch/HOME/hfr_cs/Data/Lung/LungCTChallenge/lungAugmented240NewCroppingV2

# Best hyperparameters for each step (learning rate, dropout)
DS1_Prostate/Full: 1e-8, 0.4
DS2_Brain/Frozen: 1e-7, 0.3
DS2_Brain/Full: 1e-8, 0.3
DS3_Lung/Frozen: 1e-5, 0.3
DS3_Lung/Full: 1e-8, 0.3
DS4_Prostate/Frozen: 1e-5, 0.3
DS4_Prostate/Full: 1e-9, 0.0


# Command specificities 

(--freeze, --attach_DM, --last_layer)
DS1_Prostate/Full: False, False, don't specify any
DS2_Brain/Frozen: True, False, last_layer.pckl corresponding to the best model at 1st step
DS2_Brain/Full: False, False, last_layer.pckl corresponding to the best model at 1st step
DS3_Lung/Frozen: True, False, last_layer.pckl corresponding to the best model at 1st step
DS3_Lung/Full: False, False, last_layer.pckl corresponding to the best model at 1st step
DS4_Prostate/Frozen: True, True, last_layer.pckl corresponding to the best model at 1st step
DS4_Prostate/Full: False, False, last_layer.pckl corresponding to the best model at 1st step


# Command
python3 paper_reproduction_TL.py 
--trainingset #see data used# 
--validationset #see data used# 
--reference_training #target dataset, see data used# 
--reference_validation "#target dataset, see data used#" 
--batchsize 128 
--nbepochs 2000 
--lr #See Best hyperparameters for each step# 
--cudadevice 'cuda:5' --modeltoload # Best model from the last step or model from the roulette #  
--dropout #See Best hyperparameters for each step# 
--inputchannel 1 
--freeze #See Command specificities# 
--attach_DM #See Command specificities# 
--tensorboard_output_name #choices=['DS1/Full', 'DS2/Frozen', 'DS2/Full', 'DS3/Frozen', 'DS3/Full', 'DS4/Frozen', 'DS4/Full']# 
--outputdirectory #path to output directory#



###### Testing the model on the test set ######

# Data used
Prostate: sftp://hfr_cs@diufrd141.unifr.ch/HOME/hfr_cs/Data/Prostate/ProstateX/l/20200124_prostateDWIOnlyAugmentedNormalizedSize65x60BalancedNewCropping


# Command
python3 paper_test_model_enhanced_prediction 
--test-set /HOME/hfr_cs/Data/Prostate/ProstateX/l/20200124_prostateDWIOnlyAugmentedNormalizedSize65x60BalancedNewCropping 
--cuda-device 'cuda:5' 
--modeltoload ...../20200222_manual_transfer_learning/......./trainedBaselineInter_best.pth 
--outputdirectory #path to output directory#


# Results

Before TL
            SUMMARY OF THE CLASSIFIER ON TEST SET
            -------------------
            Accuracy: 0.6842105263157895
            Precision:0.3333333333333333
            Recall:   0.5
            F1 score: 0.4
            Specificity: 0.7333333333333333
            AUC: 0.6833333333333333
            --------------------
            Running time: 6.747607946395874


After TL
            SUMMARY OF THE CLASSIFIER ON TEST SET
            -------------------
            Accuracy: 0.6842105263157895
            Precision:0.3333333333333333
            Recall:   0.5
            F1 score: 0.4
            Specificity: 0.7333333333333333
            AUC: 0.8
            --------------------
            Running time: 6.759934663772583

