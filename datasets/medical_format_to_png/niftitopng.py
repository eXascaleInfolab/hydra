#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import sys

def exploreAndConvert(niftiNpArray, npArrayMax, npArrayMin, sliceNumber, outputName, outputFolder, axisName, timeStamp=None, flip=False):
    """
    Description: Converts a NIfTI slice to png
    Params:
        - niftiNpArray: 2d slice (NumPy array)
        - npArrayMax: maximal pixel value
        - npArrayMin: minimal pixel value
        - slice:  index of the given slice
        - outputName: name of the output file
        - timeStamp: timestamp of the given slice
    Returns:
        - no return value
    """

    # normalize pixel values
    niftiNpArray = (niftiNpArray - npArrayMin) * 255.0 / (npArrayMax - npArrayMin)

    # convert to 8 bits (required to export as PNG)
    niftiNpArray = (niftiNpArray).astype('uint8')

    if flip:
        niftiNpArray = 255 - niftiNpArray

    # create a grayscaled image
    png = Image.fromarray(niftiNpArray).convert('L')

    # save as PNG
    if (timeStamp == None):
        png.save("{}{}_{}_slice{}.png".format(outputFolder, outputName, axisName, sliceNumber))
    else:
        png.save("{}{}_{}_time{}_slice{}.png".format(outputFolder, outputName,axisName, timeStamp, sliceNumber))


def niftiToPng(niftiPath, outputName, outputFolder='./', flip=False):
    """
    Description: Converts 3D or 4D .nii images to png
    Params:
        - niftiPath: path to the .nii file
        - outputName: name used in the output file
    Returns:
        - no return value
    """

    # Load the .nii file
    niftiObject = nib.load(niftiPath)

    # get_fdata() casts to 64 bits
    niftiNpArray = niftiObject.get_fdata()

    # 3D
    if (len(niftiObject.shape) == 3):

        # Get the dimensions
        (x, y, z) = niftiObject.shape

        npArrayMin = np.min(niftiNpArray)
        npArrayMax = np.max(niftiNpArray)

        # On X-Axis
        for i in range (0, x):
            exploreAndConvert(niftiNpArray[i,:,:], npArrayMax, npArrayMin, i, outputName, outputFolder, 'x', flip=flip)

        # On Y-Axis
        for i in range (0, y):
            exploreAndConvert(niftiNpArray[:,i,:], npArrayMax, npArrayMin, i, outputName, outputFolder, 'y', flip=flip)

        # On Z-Axis
        for i in range (0, z):
            exploreAndConvert(niftiNpArray[:,:,i], npArrayMax, npArrayMin, i, outputName, outputFolder, 'z', flip=flip)

    # 4D
    elif (len(niftiObject.shape) == 4):
        # x = nb of Rows, y = nb of Columns, z = nb of Slices, t = time frame
        (x, y, z, t) = niftiObject.shape

        for ythTimeStamp in range(0, t):

            # Get maximal and mini values
            npArrayMin = np.min(niftiNpArray[:,:,:,ythTimeStamp])
            npArrayMax = np.max(niftiNpArray[:,:,:,ythTimeStamp])

            # On X-Axis
            for ithSlice in range (0, x):
                exploreAndConvert(niftiNpArray[ithSlice, :, :, ythTimeStamp], npArrayMax, npArrayMin, ithSlice, outputName, outputFolder, 'x', ythTimeStamp, flip=flip)

            # On Y-Axis
            for ithSlice in range (0, y):
                exploreAndConvert(niftiNpArray[:, ithSlice,:, ythTimeStamp], npArrayMax, npArrayMin, ithSlice, outputName, outputFolder, 'y', ythTimeStamp, flip=flip)

            # On Z-Axis
            for ithSlice in range (0, z):
                exploreAndConvert(niftiNpArray[:, :, ithSlice, ythTimeStamp], npArrayMax, npArrayMin, ithSlice, outputName, outputFolder, 'z', ythTimeStamp, flip=flip)
