Data used: 
sftp://jobin_student@diufpc04/mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/l/20191224_prostateStackedAugmentedNormalizedSize65x60Balanced

###### Training the neural network using the training set and the validation set ###### 

Command:
python paper_reproduction.py 
--trainingset /mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/l/20191224_prostateStackedAugmentedNormalizedSize65x60Balanced/train/ 
--validationset /mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/l/20191224_prostateStackedAugmentedNormalizedSize65x60Balanced/val/ 
--batchsize 32 
--nbepochs 100
--outputdirectory ./20200216_paper1_lr1e-7_batch32_100epoch_02dropout_reducelrauc001
--lr 1e-7 
--cudadevice 'cuda:2' 
--dropout 0.2 
--modeltoload ~/Results/0.73/20191220_model_roulette_f1score/baseline.pth
--inputchannel 3

Results: Best model at epoch 21 with an accuracy of 0.7525 and an AUC of 0.765 on the validation set


###### Testing the best model on the test set ###### 
Command:

python paper_test_model_enhanced_prediction.py 
--testset /mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/l/20191224_prostateStackedAugmentedNormalizedSize65x60Balanced/test/ 
--modeltoload ~/Results/paper_reproduction/20200213_0.76/Training_and_validation/20200216_paper1_lr1e-7_batch32_100epoch_02dropout_reducelrauc001/trainedBaselineInter.pth
--inputchannel 3
--outputdirectory ./20200216_our_test_set_with_best_model

Results: 0.75 in AUC

### Testing the best model on the challenge test set ###

Data used: sftp://jobin_student@diufpc04/mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/challenge/Challenge_numpy_arrays_stacked_normalized

Command:

python3 PROSTATEx_challenge_numpy.py 
--modelToLoad ~/Results/paper_reproduction/20200213_0.76/Training_and_validation/20200216_paper1_lr1e-7_batch32_100epoch_02dropout_reducelrauc001/trainedBaselineInter.pth
--input-directory /mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/challenge/Challenge_numpy_arrays_stacked_normalized/ 
--output-directory ~/Results/paper_reproduction/20200213_0.76/Training_and_validation/20200217_CHALLENGE_test_set_with_best_model

Results: 0.71 in AUC
