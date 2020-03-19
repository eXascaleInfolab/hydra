#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import torch
import torch.utils.data
import numpy as np


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
            # if the image does not belong to the smallest class, choose "smallestlen" elements from it
            if(label != smallestKey):
                undersamplingWithoutReplacementIndicesList[label] = np.random.permutation(listOfIndices)[:smallestLen].tolist()
            # if the image belong to the smallest class, keep all images
            else:
                undersamplingWithoutReplacementIndicesList[label] = listOfIndices

        # create a list of balanced indices of each class --> example [class0, class1, class2, class0, class1, class2, etc]
        balancedIndicesMixed = []
        for commonIndex in range(smallestLen):
            for key in undersamplingWithoutReplacementIndicesList:
                balancedIndicesMixed.append(undersamplingWithoutReplacementIndicesList[key][commonIndex])

        return (i for i in balancedIndicesMixed)



    def __len__(self):
        return len(self.dataset.imgs)
