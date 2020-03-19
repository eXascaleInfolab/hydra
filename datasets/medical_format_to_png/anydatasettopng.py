#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import os
from glob import glob
import dicomtopng
import rawtopng
import niftitopng
import platform
import argparse

def anyDatasetToPng(folderPath, flip=False):
    """
    Description: Converts any files (.dcm, .nii.gz/.nii, .mhd) to PNG files.
    Params:
        - folder_path: path to the dataset directory
        - flip: True to flip the colors, False otherwise
    Returns:
        - no return value
    """

    print("Collecting file names ...")

    # Get all .dcm files in folderpath and subdirectories and ignore hidden files
    # https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python
    dicomFiles = [y for x in os.walk(folderPath) for y in glob(os.path.join(x[0], '[!._]*.dcm'))]
    niftiFiles = [y for x in os.walk(folderPath) for y in glob(os.path.join(x[0], '[!._]*.nii'))]
    niftiFiles += ([y for x in os.walk(folderPath) for y in glob(os.path.join(x[0], '[!._]*.nii.gz'))])
    rawFiles = [y for x in os.walk(folderPath) for y in glob(os.path.join(x[0], '[!._]*.mhd'))]

    print(">> Filenames collected")

    print("Beginning of dicom files processing")

    # Convert all DICOM files to png
    for i in dicomFiles:
        outputFolder, outputName = getOutputFolderAndName(i)
        print(">> " + i)

        # Convert to png and save
        dicomtopng.dicomToPng(i, outputName, outputFolder, True, flip=flip)

    print(">> End of dicom files processing")

    print("Beginning of NIfTI files processing ")

    # Convert all NIFTI files to png
    for i in niftiFiles:
        outputFolder, outputName = getOutputFolderAndName(i)
        print(">> "+i)

        # Convert to png and save
        niftitopng.niftiToPng(i, outputName, outputFolder, flip=flip)

    print(">> End of nifti files processing")

    print("Beginning of raw files processing")

    # Convert all RAW files to png
    for i in rawFiles:
        outputFolder, outputName = getOutputFolderAndName(i)
        print(">> "+i)

        # Convert to png and save
        rawtopng.rawToPng(i, outputName, outputFolder, flip=flip)

    print(">> Ending of raw files processing")


def getOutputFolderAndName(filePath):
    """
    Description: Get output folder name and file name from file path
    Params:
        - filePath: path to the dataset directory
    Returns:
        - no return value
    """

    # If OS is Windows split by \
    if platform.system() == 'Windows':
        splitPath = filePath.split('\\')

    # If OS is not Windows split by /
    else:
        splitPath = filePath.split('/')

    # Path without the file name
    outputFolder = filePath[0: len(filePath)-len(splitPath[-1])]
    outputName = splitPath[-1]

    return (outputFolder, outputName)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='')

    parser.add_argument('--folder',
                        help='folder path',
                        required=True,
                        type=str,
                        default='./')

    parser.add_argument('--flip',
                        help='flip the colors of the images',
                        action='store_true')

    args = parser.parse_args()

    if os.path.isdir(args.folder):
        if args.flip:
            anyDatasetToPng(args.folder, True)
        else:
            anyDatasetToPng(args.folder)
