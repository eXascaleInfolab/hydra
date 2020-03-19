#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import importlib
from torchvision import datasets, transforms
import argparse
import torch
from PIL import Image
import torch.utils.data
import numpy as np

#### BEGIN CLASS UNDERBALANCEDSAMPLING #####
class BalancedRandomUndersampling(torch.utils.data.Sampler):

    def __init__(self, dataset, seed=None):
        self.dataset = dataset
        self.seed = seed

    def __iter__(self):

        if(self.seed!=None):
            np.random.seed(self.seed)

        # Store [class: [indexInDataset1, indexInDataset2, ...]]
        indicesForClass = {}
        for index, (_, label) in enumerate(self.dataset.imgs):
            indicesForClass.setdefault(label, []).append(index)

        # Find the class which has the minimum number of images
        smallestKey, smallestLen = min([(i,len(j)) for i,j in indicesForClass.items()], key = lambda t: t[1])

        # For each class, keep only "smallestlen" elements (i.e undersampling)
        undersamplingWithoutReplacementIndicesList = {}
        for label, listOfIndices in indicesForClass.items():
            # If the image does not belong to the smallest class, choose "smallestlen" elements from it
            if(label != smallestKey):
                undersamplingWithoutReplacementIndicesList[label] = np.random.permutation(listOfIndices)[:smallestLen].tolist()
            # If the image belong to the smallest class, keep all images
            else:
                undersamplingWithoutReplacementIndicesList[label] = listOfIndices

        # Create a list of balanced indices of each class --> example [class0, class1, class2, class0, class1, class2, etc]
        balancedIndicesMixed = []
        for commonIndex in range(smallestLen):
            for key in undersamplingWithoutReplacementIndicesList:
                balancedIndicesMixed.append(undersamplingWithoutReplacementIndicesList[key][commonIndex])

        return (i for i in balancedIndicesMixed)

    def __len__(self):
        return len(self.dataset.imgs)
#### END CLASS UNDERBALANCEDSAMPLING #####

###### UTIL ######
def new_loader(path):
    """
    Description:
        - Load an image, convert it to a 8 bits grayscale image
    Params:
        - path: takes an image path
    Returns:
        - A grayscaled image
    """
    img = Image.open(path)

    return img.convert('L')

def loadData(train_data_path, validation_data_path, imageSize, batch_size, seed=None):
    """
    Description:
        - Load the training and validation sets. Folders have to be organized in this way:
            train/label1/image.npy
            test/label2/image2.npy
        - Organize the data in batches of size batch_size
    Params:
        - train_data_path: training set path
        - validation_data_path: validation set path
        - imageSize: size to which images have to be resized
        - batch_size: number of elements per batch
        - seed: randomness initialization
    Returns:
        - A tuple containing (training_loader, validation_loader)
    """
    # Before storing an image in the dataset, resize it, convert it to grayscale and cast it into pytorch tensor
    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((imageSize, imageSize), interpolation=Image.BICUBIC),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        ])

    ##### Training set ####
    train_data = datasets.ImageFolder(root=train_data_path, loader=new_loader, transform=TRANSFORM_IMG)
    sampler1 = BalancedRandomUndersampling(train_data, seed=seed)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=sampler1)

    ##### Validation set ####
    if validation_data_path != None:
        validation_data = datasets.ImageFolder(root=validation_data_path, transform=TRANSFORM_IMG)
        sampler2 = BalancedRandomUndersampling(train_data, seed=seed)
        validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, sampler=sampler2)
    else:
        validation_data_loader = None

    return (train_data_loader, validation_data_loader)

### END UTIL #####

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Display the different batches with their respective labels and count the number of image per class to check that BalancedRandomUndersampling creates balanced batches')

    parser.add_argument('--trainingSet1',
                        help='dataset 1 training folder path',
                        required=True,
                        type=str,
                        default='./')

    args = parser.parse_args()

    # Visually check that the batches are 010101010101...
    training_data_loader, validation_data_loader = loadData(args.trainingSet1, None, 128, 32)
    for images, labels in training_data_loader:
        print(labels)
        counter_of_0 = labels.tolist().count(0)
        counter_of1 = labels.tolist().count(1)
        print(f"There are {counter_of_0} 0\'s and {counter_of1} 1\'s")
        print("-------------------------------------------------------------------")
