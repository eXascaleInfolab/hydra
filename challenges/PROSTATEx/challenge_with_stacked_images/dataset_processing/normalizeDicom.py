#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import numpy as np
from PIL import Image
import pydicom

def get_LUT_value(numpyArray, windowWidth, windowCenter, rescaleSlope, rescaleIntercept):
    """
    Description:
        - Apply the RGB Look-Up Table for the given data and window/level value.
    Params:
        - numpyArray:  NumPy array containing the value of each pixel, 16 bits per pixel
        - windowWidth: dataset.WindowWidth (cannot appear without WindowCenter)
        - windowCenter: dataset.WindowCenter (cannot appear without WindowWidth)
        - rescaleSlope: specify the linear transformation from pixels in their stored on disk representation to their
          in memory representation
        - rescaleIntercept - Required: specify the linear transformation from pixels in their stored on disk representation to their
          in memory representation
    Returns:
        - the pixel values normalized
    """

    # Hounsfield unit:
    # Hounsfield Units (HU) are used in CT images it is a measure of radio-density, calibrated to distilled water and free air.
    # HUs can be calculated from the pixel data using the Slope and Intercept value from the Dicom image
    # and applying it to a target pixel.
    # HU = m * P + b
    # Where: m is the Dicom attribute (0028,1053) “Rescale slope”
    #        b is the Dicom attribute (0028,1052) “Rescale intercept”
    #        P is the value of that particular pixel in the pixels array.
    # https://www.medicalconnections.co.uk/kb/Hounsfield-Units/

    if rescaleSlope != None and rescaleIntercept != None:
        numpyArray = numpyArray * rescaleSlope + rescaleIntercept

    if isinstance(windowCenter, pydicom.multival.MultiValue):
        windowCenter = windowCenter[0]

    if isinstance(windowWidth, pydicom.multival.MultiValue):
        windowWidth = windowWidth[0]

    # np.piecewise iterates through an array, checks the given conditions on each element, returns the corresponding value of the last array
    # Example:
    ### x = np.array([1,2,3,4])
    ### np.piecewise(x, [x < 3, x >= 3], [0,1])
    ### --> array([0, 0, 1, 1])
    # http://dicom.nema.org/medical/dicom/2014a/output/pdf/part03.pdf page 1057
    numpyArray = np.piecewise(numpyArray,
    [numpyArray<= windowCenter-0.5-(windowWidth-1)/2, numpyArray>windowCenter - 0.5 + (windowWidth-1) /2],
    [0, 255, lambda x: ((x - (windowCenter - 0.5)) / (windowWidth-1) + 0.5) * (255 - 0) + 0 ])

    # Conversion 16 bits to 8 bits array: [0:MAXARRAY] -> [0:255]
    ratio = np.max(numpyArray) / 255 ;
    return (numpyArray/ ratio).astype('uint8')


def get_normalized_array(dataset, flip=False):
    """
    Description:
        - Get normalized NumPy array from DICOM file
    Params:
        - dataset: FileDataset (Pydicom) corresponding to one specific slice
        - flip: if true, the color are inversed (black becomes white and vice versa)
    Returns:
        - return the normalized numpy array of the given DICOM file
    """
    # Dataset without pixels
    if ('PixelData' not in dataset):
        raise TypeError("DICOM dataset does not have pixel data")

    # Dataset without windowWidth/WindowCenter -> unable to compute the linear transformation
    if ('WindowWidth' not in dataset) or ('WindowCenter' not in dataset):
        # Number of bits allocated for each pixel (each sample/channel should have the same number of bits allocated)
        # either 1 or a multiple of 8
        bits = dataset.BitsAllocated

        # Number of "channels" (RGB = 3, greyscale = 1 etc.) / number of separates planes in the image
        # either 1 or 3 planes
        samples = dataset.SamplesPerPixel

        # Get raw pixel values of the DICOM
        ds = dataset.pixel_array

        # 1 bit per pixel, 1 plane
        if bits == 1 and samples == 1:
            if flip:
                ds = 1 - ds

        # 1 plane = greyscale
        elif bits == 8 and samples == 1:
            if flip:
                ds = 255 - ds

        # 3 planes = RBG
        elif bits == 8 and samples == 3:
            if flip:
                ds = 255 - ds

        elif bits == 16:
            ds = (ds.astype(np.float)-ds.min())*255.0 / (ds.max()-ds.min())

            if flip:
                ds = 255 - ds

            ds = ds.astype(np.uint8)

        else:
            raise TypeError("Don't know PIL mode for %d BitsAllocated and %d SamplesPerPixel" % (bits, samples))

        # return the normalized pixel values
        return ds

    # WindowWidth and WindowCenter are available in dataset
    else:
        if ('RescaleSlope' not in dataset) or ('RescaleIntercept' not in dataset):
            image_array = get_LUT_value(dataset.pixel_array, dataset.WindowWidth, dataset.WindowCenter, None, None)
        else:
            image_array = get_LUT_value(dataset.pixel_array, dataset.WindowWidth, dataset.WindowCenter,
            dataset.RescaleSlope,dataset.RescaleIntercept)

        if flip:
            image_array = 255 - image

        # Return the normalized pixel values
        return image_array
