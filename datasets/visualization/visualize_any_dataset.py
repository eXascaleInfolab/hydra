#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################


import matplotlib.pyplot as plt
import pydicom
import pandas as pd
from PIL import Image
import numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("rawtopng", "../png_conversion/rawtopng.py")
rtp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtp)

import normalizeDicom as n
import math
import nibabel as nib
import os
from glob import glob
import argparse


def visualizeDicom(filepath, flip=False):
    """
    Description: Visualizes a dcm file
    Params:
        - filepath: path to the DICOM file
    Returns:
        - no return value
    Source:
        - https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html
    """
    # Read DICOM file
    dataset = pydicom.dcmread(filepath)

    # Normal mode:
    print()
    print("Filename.........:", filepath)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name...:", display_name)
    print("Patient id.......:", dataset.PatientID)
    print("Modality.........:", dataset.Modality)
    print("Study Date.......:", dataset.StudyDate)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

    # Use .get() if not sure the item exists, and want a default value if missing
    print("Slice location...:", dataset.get('SliceLocation', "(missing)"))

    # Normalize and convert DICOM to image
    img = n.get_PIL_image(dataset, flip)

    # Display image
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()


def visualizeDicoms(folderpath, flip=False):
    """
    Description: visualizes all dcm files within a directory (and subdirectories)
    Params:
        - folderpath: path to the folder containing DICOM files
    Returns:
        - no return value
    Source:
        - https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_read_dicom.html
    """
    # Initialize list of slices
    slices = []

    # Get all .dcm files in folderpath and subdirectories
    # https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python
    files = [y for x in os.walk(folderpath) for y in glob(os.path.join(x[0], '*.dcm'))]

    # Load all dcm files
    datasets = [pydicom.dcmread(filepath) for filepath in files]

    # Sort dicom since the order of the files != the right order
    datasets.sort(key=lambda x: x.InstanceNumber)

    # Iterate through all dicoms and create the visualizable version
    for dataset in datasets:
        img = n.get_PIL_image(dataset, flip)
        numpyArray = np.array(img)
        slices.append(numpyArray)

    # Visualize slices
    visualizeHelper(slices, 'Z')


def visualizeRaw(filepath, axis='all', flip=False):
    """
    Description: Visualizes a raw file
    Params:
        - filepath: path to the .mhd file
        - axis: axis to go through. Either 'all', x or y or z, a combination of x, y, z.
    Returns:
        - no return value
    """
    # Load scan
    rawNpArray, _,  _ = rtp.load_itk(filepath)

    # Split the path based on slashes
    splitPath = filepath.split('/')

    # "splitPath[len(splitPath) - 1][:-4]"
    ### splitPath[len(splitPath) - 1] --> keeps the file name from the file path
    ### [:-4] --> removes the file extension from the file name (i.e. '.mhd')
    figureTitle = 'Raw file {}'.format(splitPath[len(splitPath) - 1][:-4])

    # Get the dimensions
    (x, y, z) = rawNpArray.shape

    # Initialize lists of slices for each axis
    allXs = []
    allYs = []
    allZs = []

    # Get min and max values (to normalize)
    npArrayMax = np.max(rawNpArray)
    npArrayMin = np.min(rawNpArray)

    # On X-Axis
    if axis == 'all' or 'X' in axis or 'x' in axis:
        for i in range (0, x):
            currentSliceX = convertTo8bits(rawNpArray[i, :, :], npArrayMin, npArrayMax)
            # Inverse black and white
            if flip:
                currentSliceX = (255 - currentSliceX)

            allXs.append(currentSliceX)

        visualizeHelper(allXs, 'X')

    # On Y-Axis
    if axis == 'all' or 'Y' in axis or 'y' in axis:
        for i in range (0, y):
            currentSliceY = convertTo8bits(rawNpArray[:, i, :], npArrayMin, npArrayMax)
            # Inverse black and white
            if flip:
                currentSliceY = (255 - currentSliceY)
            allYs.append(currentSliceY)

        visualizeHelper(allYs, 'Y')

    # On Z-Axis
    if axis == 'all' or 'Z' in axis or 'z' in axis:
        for i in range (0, z):
            currentSliceZ = convertTo8bits(rawNpArray[:, :, i], npArrayMin, npArrayMax)
            # Inverse black and white
            if flip:
                currentSliceZ = (255 - currentSliceZ)
            allZs.append(currentSliceZ)

        visualizeHelper(allZs, 'Z')


def visualizeNifti(niftiPath, axis= 'all', flip=False):
    """
    Description: Visualizes a Nifti file (3D or 4D)
    Params:
        - niftiPath: path to the .nii file
        - axis: axis to go through. Either 'all', x or y or z, a combination of x, y, z.
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

        # Initialize lists of slices for each axis
        allXs = []
        allYs = []
        allZs = []

        # get min and max values (to normalize)
        npArrayMax = np.max(niftiNpArray)
        npArrayMin = np.min(niftiNpArray)

        # On X-Axis
        if axis == 'all' or 'X' in axis or 'x' in axis:
            for i in range (0, x):
                currentSliceX =  convertTo8bits(niftiNpArray[i, :, :], npArrayMin, npArrayMax)
                # Inverse black and white
                if flip:
                    currentSliceX = (255 - currentSliceX)
                allXs.append(currentSliceX)

            visualizeHelper(allXs, 'X')

        # On Y-Axis
        if axis == 'all' or 'Y' in axis or 'y' in axis:
            for i in range (0, y):
                currentSliceY = convertTo8bits(niftiNpArray[:, i, :], npArrayMin, npArrayMax)
                # Inverse black and white
                if flip:
                    currentSliceY = (255 - currentSliceY)
                allYs.append(currentSliceY)

            visualizeHelper(allYs, 'Y')

        # On Z-Axis
        if axis == 'all' or 'Z' in axis or 'z' in axis:
            for i in range (0, z):
                currentSliceZ = convertTo8bits(niftiNpArray[:, :, i], npArrayMin, npArrayMax)
                # Inverse black and white
                if flip:
                    currentSliceZ = (255 - currentSliceZ)
                allZs.append(currentSliceZ)

            visualizeHelper(allZs, 'Z')

    # 4D
    elif (len(niftiObject.shape) == 4):

        # Get the dimensions (x, y, z, timestamp)
        (x, y, z, t) = niftiObject.shape

        # Initialize lists of slices for each axis
        allXs = []
        allYs = []
        allZs = []

        # On X-Axis
        if axis == 'all' or 'X' in axis or 'x' in axis:
            for timeStamp in range(0, t):
                # get min and max values for each timestamp (to normalize)
                npArrayMin = np.min(niftiNpArray[:, :, :, timeStamp])
                npArrayMax = np.max(niftiNpArray[:, :, :, timeStamp])

                for i in range (0, x):
                    currentSliceX = convertTo8bits(niftiNpArray[i, :, :, timeStamp], npArrayMin, npArrayMax)
                    # Inverse black and white
                    if flip:
                        currentSliceX = (255 - currentSliceX)
                    allXs.append(currentSliceX)

            visualizeHelper(allXs, 'X', t)

        # On Y-Axis
        if axis == 'all' or 'Y' in axis or 'y' in axis:

            for timeStamp in range(0, t):
                # get min and max values for each timestamp (to normalize)
                npArrayMin = np.min(niftiNpArray[:, :, :, timeStamp])
                npArrayMax = np.max(niftiNpArray[:, :, :, timeStamp])

                for i in range (0, y):
                    currentSliceY = convertTo8bits(niftiNpArray[:, i, :, timeStamp], npArrayMin, npArrayMax)
                    # Inverse black and white
                    if flip:
                        currentSliceY = (255 - currentSliceY)
                    allYs.append(currentSliceY)

            visualizeHelper(allYs, 'Y', t)

        # On Z-Axis
        if axis == 'all' or 'Z' in axis or 'z' in axis:
            # On Z-Axis
            for timeStamp in range(0, t):
                # get min and max values for each timestamp (to normalize)
                npArrayMin = np.min(niftiNpArray[:, :, :, timeStamp])
                npArrayMax = np.max(niftiNpArray[:, :, :, timeStamp])

                for i in range (0, z):
                    currentSliceZ = convertTo8bits(niftiNpArray[:, :, i, timeStamp], npArrayMin, npArrayMax)
                    # Inverse black and white
                    if flip:
                        currentSliceZ = (255 - currentSliceZ)
                    allZs.append(currentSliceZ)

            visualizeHelper(allZs, 'Z', t)


class IndexTracker(object):
    """
    Allows to scroll through the slices of 3d images in the matplotlib window
    Adapted from: https://matplotlib.org/devdocs/gallery/event_handling/image_slices_viewer.html
    """
    def __init__(self, ax, X, axisName):
        self.ax = ax
        ax.set_title('Going through {}-axis'.format(axisName))

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :],cmap='gray', vmin=0, vmax=255)

        self.update()

    def onscroll(self, event):
        # scroll upwards -> next slice
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices

        # scroll downwards -> previous slice
        else:
            self.ind = (self.ind - 1) % self.slices

        self.update()

    def update(self):
        # update the displayed slice
        self.im.set_data(self.X[self.ind, :, :])

        # update the slice number in the label
        self.ax.set_ylabel('slice %s' % self.ind)

        self.im.axes.figure.canvas.draw()


class IndexTracker4D(object):
    """
    Allows to scroll through the slices and the timestamps of 4d images in the matplotlib window
    Adapted from: https://matplotlib.org/devdocs/gallery/event_handling/image_slices_viewer.html
    """
    def __init__(self, ax, X, axisName, timeStamp):
        self.ax = ax
        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices//(2 * timeStamp)
        self.indTS = 0
        self.axisName = axisName
        self.timeStamp = timeStamp

        self.im = ax.imshow(self.X[self.ind, :, :],cmap='gray', vmin=0, vmax=255)

        self.update()

    def onscroll(self, event):
        # scroll upwards -> next slice
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        # scroll downwards -> previous slice
        else:
            self.ind = (self.ind - 1) % self.slices

        self.update()

    def on_key(self, event):
        # press left arrow -> previous timestamp
        if event.key == 'left':
            self.indTS = (self.indTS - 1) % self.timeStamp
            self.ind = int((self.ind - self.slices / self.timeStamp) % self.slices)
        # press right arrow -> next timestamp
        else:
            self.indTS = (self.indTS + 1) % self.timeStamp
            self.ind = int((self.ind + self.slices / self.timeStamp) % self.slices)

        self.update()

    def update(self):
        # update the displayed slice
        self.im.set_data(self.X[self.ind, :, :])

        # update the slice number in the label
        self.ax.set_ylabel('slice %s' % self.ind)

        # update the timestamp in the main title
        self.ax.set_title('Going through {}-axis, time stamp {}'.format(self.axisName, math.floor(self.ind / (self.slices / self.timeStamp))))

        self.im.axes.figure.canvas.draw()


def visualizeHelper(slices, axisName, t=None):
    """
    Description: Used to visualize slices and enable scrolling.
    Params:
        - slices: slices to display (list of NumPy arrays)
        - axisName: axis to iterate through (x,y,z)
        - t = number of timestamps. Must be used for 4D data only.
    """

    # Create a NumPy array from the list of slices
    a = np.array(slices)

    # Create the matplotlib figure
    fig, ax = plt.subplots(1,1)

    # If 3 dimensions (no timestamp)
    if (t == None):
        tracker = IndexTracker(ax, a, axisName)
    # If 4 dimensions (timestamp)
    else:
        tracker = IndexTracker4D(ax, a, axisName, t)
        # detect key press events
        fig.canvas.mpl_connect('key_press_event', tracker.on_key)

    # Detect scrolling events
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


def convertTo8bits(currentSlice, npArrayMin, npArrayMax):
    """
    Description: Converts the given slice into a 8-bit array
    Params:
        - currentSlice: 2D slice
        - npArrayMin: the minimum voxel value of the entire volume
        - npArrayMax: the maximum voxel value of the entire volume
    Returns:
        - Converted image
    """

    # Normalize pixel values
    currentSlice = (currentSlice - npArrayMin) * 255.0 / (npArrayMax - npArrayMin)

    # Convert to 8 bits (required to export as PNG)
    currentSlice = (currentSlice).astype('uint8')

    return currentSlice


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='')

    parser.add_argument('--format',
                        choices=['dicom', 'dcm','dicoms','dcms', 'nifti', 'nii', 'niigz', 'raw', 'mhd'],
                        help='file format to visualize',
                        required=True,
                        type=str,
                        default=None)

    parser.add_argument('--path',
                        help='input file path or input directory',
                        required=True,
                        type=str,
                        default='./')

    parser.add_argument('--flip',
                        help='flip the colors of the images',
                        action='store_true')


    parser.add_argument('--axes',
                        choices=['x','y','z','xy','yx','zy','yz','xz','zx','xyz','xzy','yxz','yzx','zxy','zyx','all'],
                        help='axes',
                        required=False,
                        type=str,
                        default='all')

    args = parser.parse_args()

    # Dicom (single file)
    if args.format == 'dcm' or args.format == 'dicom':
        if args.path[-4:] == '.dcm' and os.path.isfile(args.path):
            if args.flip:
                visualizeDicom(args.path, True)
            else:
                visualizeDicom(args.path)
        else:
            parser.error('path must point to a DICOM file')

    # Dicoms (multiple files)
    if args.format == 'dcms' or args.format == 'dicoms':
        if os.path.isdir(args.path):
            if args.flip:
                visualizeDicoms(args.path, True)
            else:
                visualizeDicoms(args.path)
        else:
            parser.error('path must point to a folder')

    # NIfTI
    elif args.format == 'nii' or args.format == "nifti" or args.format =="niigz":
        if (args.path[-4:] == '.nii' or args.path[-7:] == '.nii.gz') and os.path.isfile(args.path):
            if args.axes is not None:
                if args.flip:
                    visualizeNifti(args.path, args.axes, True)
                else:
                    visualizeNifti(args.path, args.axes)
            else:
                parser.error('--axes is necessary when chosen format is nii')
        else:
            parser.error('path must point to a NIFTI file')

    # RAW
    elif args.format == 'mhd' or args.format == 'raw':
        if args.path[-4:] == '.mhd' and os.path.isfile(args.path):
            if args.axes is not None:
                if args.flip:
                    visualizeRaw(args.path, args.axes, True)
                else:
                    visualizeRaw(args.path, args.axes)
            else:
                parser.error('--axes is necessary when chosen format is raw')
        else:
            parser.error('path must point to an mhd file')
