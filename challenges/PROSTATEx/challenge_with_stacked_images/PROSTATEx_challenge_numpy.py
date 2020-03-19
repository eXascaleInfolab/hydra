#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

# Importation of modules
import torch
import csv
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from PIL import Image
import time
import argparse
import math
from pathlib import Path
import os

###### MODEL ######
class Flatten(nn.Module):
    """Flattens the output of the previous layer"""
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ReproductedModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(ReproductedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Dropout(p=dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Dropout(p=dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Dropout(p=dropout),
            nn.MaxPool2d(2),
        )

        self.features.add_module('flatten', Flatten())

        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ELU(),

            nn.Dropout(p=dropout),

            nn.Linear(256, 64),
            nn.ELU(),

            nn.Dropout(p=dropout),

            nn.Linear(64, 16),
            nn.ELU(),
        )

        self.last_layer = nn.Sequential(
            nn.Linear(16, num_classes),
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x

###### END MODEL ######

### CUSTOM DATASET ####

class get_prostateX_test_set(datasets.DatasetFolder):
    """ProstateX challenge test set"""

    def __getitem__(self, index):
        # this is what DatasetFolder normally returns, i.e (data, label)
        original_tuple = super().__getitem__(index)

        # the image file name without extension
        path = Path(self.samples[index][0]).stem

        # remove the useless label and replace it by the path of the image
        tuple_with_path = (original_tuple[0], path)
        return tuple_with_path

### END CUSTOM DATASET #####

def new_loader(path):
    """
    Description: load the given .npy file and convert it into a tensor
    Params:
        - path: path of the file
    Returns
        - the corresponding tensor
    """
    # Load the data from the numpy array
    np_array = np.load(path)

    # Convert to tensor and send to available device, i.e GPU/CPU
    tensor = torch.from_numpy(np_array).to('cuda' if torch.cuda.is_available() else 'cpu')

    return tensor


def load_prostateX_data(test_data_path):
    """
    Description: load the given data
    Params:
        - test_data_path: path containing the test set
    Returns
        - returns a PyTorch dataloader for the test set
    """
    test_data = get_prostateX_test_set(root=test_data_path, loader=new_loader, extensions=("npy"))
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=1,)

    return (test_data_loader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script makes a prediction for each finding of each patient using the given model in argument. Then, it exports the results in a .csv file with regard to the format used in the PROSTATEx challenge')

    parser.add_argument('--modeltoload',
                        help='model (.pth file) to load',
                        required=True,
                        type=str)

    parser.add_argument('--inputdirectory',
                        help='root of the input directory that contains all test pictures',
                        required=True,
                        type=str)

    parser.add_argument('--outputdirectory',
                        help='Output directory ',
                        required=False,
                        type=str,
                        default='./')

    args = parser.parse_args()

    # Create the output directory if it does not exist
    if not os.path.exists(args.outputdirectory):
        os.makedirs(args.outputdirectory)

    # Run on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Loading the model
    model = ReproductedModel()
    model.load_state_dict(torch.load(args.modeltoload,  map_location=torch.device(device)))
    model.to(device)
    model.eval()
    print("> Model {} loaded".format(args.modeltoload))

    # Loading all data
    dataloader = load_prostateX_data(args.inputdirectory)
    print("> Data loaded")

    # dict {ProstateX-0204-1:[modelOutput1, modelOutput2, modelOutputn...]}
    allResults = {}

    # Counter of the number of images in the dictionary
    nb_images = 0

    print("> Computing NN outputs...")
    # Image and filename are tuple of only 1 element
    for image, filename  in dataloader:

        image = image.to(device)

        # Create the key for the dictionary containing all predictions 'ProstateX-0204-1' : ProstateX-patientID-FindingID
        # ProstateX
        prostate_x = filename[0].split("-")[1]

        # patient id
        patient_id = filename[0].split("-")[2].split('_')[0]

        # finding id
        finding_id = filename[0].split("-")[3].split('_')[0]

        # Creation of the key for the dictionary
        patients_array_prox_fID = str(prostate_x + "-" + patient_id + "-" + finding_id)

        # Initialize the dictionary and add the output of the model for the current image
        allResults.setdefault(patients_array_prox_fID, []).append(model(image).data.cpu().numpy()[0][1])

        # Increment the counter of images added in the dictionary
        nb_images += 1

    print("> All computations done")

    # Sanitiy check
    nb_image_really_stored = 0
    for key in allResults:
        nb_image_really_stored += len(allResults[key])

    if (nb_image_really_stored != nb_images):
        print("The number of images really stored into the dictionary does not match the total number of images")
        print(nb_images)
        print(nb_image_really_stored)
        exit(0)

    # Compute the mean value for each patient for each finding
    for key in allResults:
        allResults[key] = sum(allResults[key])/len(allResults[key])

    print("> Exporting the results to CSV file")

    with open(os.path.join(args.outputdirectory,'PROSTATEx_challenge_results_JohanAndJulien.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', )
        writer.writerow(['ProxID','fid','ClinSig'])

        for key, value in allResults.items():

            # Split 'ProstateX-0204-1' into:
            # ProstateX-0204
            # 1
            splits= key.split('-')
            patientID = str(splits[0]+'-'+splits[1])
            fid = str(splits[2])

            # Write the row with respect to this format: ProstateX-0204,1,0.775326731
            writer.writerow([patientID, fid, str(value)])

    print(">> Done")
