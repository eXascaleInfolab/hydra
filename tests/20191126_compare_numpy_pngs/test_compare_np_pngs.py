#########################################
#                                       #
#  Julien clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import argparse
import numpy as np
from PIL import Image
import os
import pprint

def get_all_files_in_folders_and_subfolders_without_extension(root_dir):
    """
    Description: get all the files in a folder and sub-folders.
    Params:
        - root_dir: all files in this directory and it's sub-folders will be returned by this method.
    Returns:
        - dict containing name files associated with their path
    """
    file_names = {}
    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            file_names[name[:-4]]= os.path.join(path, name)
    return file_names


def compare(numpy_folder, png_folder):
    """
    Description: this function compare the pixel values of all NumPy arrays from the numpy_folder to the pixels values of all correspondings PNG images from the png_folder
    Params:
        - numpy_folder: folder containing all NumPy arrays
        - png_folder: folder containing all correspondant PNG images
    Returns:
    """
    # Get a dict of all numpy arrays containing names and corresponding full path
    allNumpyArray = get_all_files_in_folders_and_subfolders_without_extension(numpy_folder)

    #  Get a dict of all pngs containing names and corresponding full path
    allPngs = get_all_files_in_folders_and_subfolders_without_extension(png_folder)

    results = {}
    counter_identical = 0
    counter_different = 0
    for numpyArray in allNumpyArray.keys():
        # Load the numpy array
        npArray = np.load(allNumpyArray[numpyArray])

        # Load the corresponding png
        im = Image.open(allPngs[numpyArray])
        npArray_png = np.array(im)

        # If the sum of all pixel values is the same in the NumPy array and in the PNG image
        if np.sum(npArray) == np.sum(npArray_png):
            results[numpyArray] = "identical"
            counter_identical += 1
        else:
            results[numpyArray] = "different"
            counter_different += 1

    # Display all results
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(results)

    # Summarize the results
    print()
    print(f"""-----------SUMMARY-------------
        - Number of identical images: {counter_identical}
        - Number of different images: {counter_different}
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Check that the pixels value of a numpy array are the same in the corresponding png')

    parser.add_argument('--numpy-folder',
                        help='Path to the file containing the numpy array',
                        required=True,
                        type=str
                        )

    parser.add_argument('--png-folder',
                        help='path to the corresponding .png',
                        required=True,
                        type=str
                        )

    args = parser.parse_args()

    compare(args.numpy_folder, args.png_folder)
