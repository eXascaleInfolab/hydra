Data used: 
sftp://jobin_student@diufpc04/mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/l/20191224_prostateStackedAugmentedNormalizedSize65x60Balanced

###### Training the neural network using all availabe data as training set ###### 

Command:
python paper_reproduction.py 
--trainingset /mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/l/20200207_prostateStackedAugmentedNormalizedSize65x60Balanced_noval_notest 
--batchsize 32 
--nbepochs 100
--outputdirectory .20200216_paper1_challengetraining_novalidation_notest_lr1e-7_dropout02_batchsize32
--lr 1e-7 
--cudadevice 'cuda:2' 
--dropout 0.2 
--modeltoload ~/Results/0.73/20191220_model_roulette_f1score/baseline.pth
--inputchannel 3


###### Testing different the model at different epochs on the challenge test set ######

Data used: 
sftp://jobin_student@diufpc04/mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/challenge/Challenge_numpy_arrays_stacked_normalized

Command:

python3 ~/Code/2019_Hospital-Fribourg/challenges/PROSTATEx/challenge_with_stacked_images/PROSTATEx_challenge_numpy.py 

--modeltoload ~/Code/2019_Hospital-Fribourg/models/paper_reproduction/train_model/20200216_paper1_challengetraining_novalidation_notest_lr1e-7_dropout02_batchsize32/trainedBaselineInter_epochXXX.pth 
--inputdirectory /mnt/hdd/hospitalFRdataGroup/Data/Prostate/ProstateX/challenge/Challenge_numpy_arrays_stacked_normalized/ 
--outputdirectory ~/Code/2019_Hospital-Fribourg/challenges/PROSTATEx/challenge_with_stacked_images/20200217_CHALLENGE_training_lr1e-7_dropout0.2_batchsize32_decay_epoch_XXX

Replace the XXX by the ith epoch.

Results:

Epoch: 15 -> 0.75
Epoch: 16 -> 0.75
Epoch: 17 -> 0.76
Epoch: 18 -> 0.76
Epoch: 19 -> 0.76
Epoch: 20 -> 0.76
Epoch: 21 -> 0.76
Epoch: 22 -> 0.76
Epoch: 23 -> 0.75
Epoch: 30 -> 0.75



