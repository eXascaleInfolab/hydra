#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import numpy as np
import SimpleITK as sitk
from PIL import Image
import csv
import matplotlib.pyplot as plt
import sys

def load_itk(filename):
    """
    Description: Load a .mhd file, get the data of the corresponding .raw file and shuffle the dimensions to get axis in the order z,y,x
    Params:
        - filename: path to the .mhd file
    Returns:
        - actual slice, origin and spacing
    Source:
        - https://stackoverflow.com/questions/37290631/reading-mhd-raw-format-in-python
    """
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing


def convertb8bitsAndGeneratePNG(currentSlice, minRaw, maxRaw, outputName, index, axisName, outputFolder, flip=False):
    """
    Description: converts an image to 8 bits and save as PNG
    Params:
        - currentSlice: 2D slice
        - minRaw: min value in the slice (used to normalize)
        - maxRaw: max value in the slice (used to normalize)
        - outputName: file output name
        - index: slice number
        - axisName: axis name (RAW files are 3-dimensional and we export a 2D slice)
        - outputFolder: output folder
        - flip: True to flip colors, False otherwise
    Returns:
        - no return value
    """

    # Normalize pixel values
    currentSlice = (currentSlice - minRaw) * 255.0 / (maxRaw - minRaw)

    # Convert to 8 bits (required to export as PNG)
    currentSlice = (currentSlice).astype('uint8')

    if flip:
        currentSlice = 255 - currentSlice

    # Create the Pillow image
    img = Image.fromarray(currentSlice)

    # Export image
    img.save("{}{}_{}_{}.png".format(outputFolder, outputName, axisName, index))


def rawToPng(mhdPath, outputName, outputFolder="./", flip=False):
    """
    Description: Convert a .raw file into a 8-bit grayscaled .png
    Params:
        mhdPath -Required: path to the .mhd file that needs to be converted
    Returns:
        - no return value
    """

    # Load scan
    ctScan, _,  _ = load_itk(mhdPath)

    # Get min and max values (to normalize)
    npArrayMax = np.max(ctScan)
    npArrayMin = np.min(ctScan)

    # Get the dimensions
    (x, y, z) = ctScan.shape

    # On X-Axis
    for i in range (0, x):
        convertb8bitsAndGeneratePNG(ctScan[i, :, :], npArrayMin, npArrayMax, outputName, i, 'x', outputFolder, flip=flip)

    # On Y-Axis
    for i in range (0, y):
        convertb8bitsAndGeneratePNG(ctScan[:, i, :], npArrayMin, npArrayMax, outputName, i, 'y', outputFolder, flip=flip)

    # On Z-Axis
    for i in range (0, z):
        convertb8bitsAndGeneratePNG(ctScan[:, :, i], npArrayMin, npArrayMax, outputName, i, 'z', outputFolder, flip=flip)
