#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

def main(root_folder, output_folder):
    """
    Description: this function display all stacked NumPy array (T2-DWI-ADC) contained in a root_folder and generate plot to check the alignment in the output_folder
    Params:
        - root_folder: folder that contains all stacked NumPy array
        - output_folder: destination folder
    Returns:
        - No return value
    """
    # Check if the output_folder exists. If not, create it.
    output_folder = Path(output_folder)
    if (not output_folder.exists()):
        output_folder.mkdir(parents=True)

    # Create the path to the root folder containing all np stacked arrays
    root_folder = Path(root_folder)

	# Get the list of all files in directory tree at given path
    listOfFiles = []
    for (dirpath, dirnames, filenames) in os.walk(root_folder):
        listOfFiles += [Path(os.path.join(dirpath, file)) for file in filenames]

    # Display each stacked image: T2 on the left, DWI in the middle and ADC on the right
    for stacked_image in listOfFiles:
        stacked_np_arrays = np.load(stacked_image)
        fig=plt.figure(figsize=(15, 6),facecolor='w', edgecolor='k')
        for index, mri_type in enumerate(stacked_np_arrays):
            # matplotlib index starts at 1
            index +=1

            # Row, columns, position
            fig.add_subplot(2,3, index)

            # T2
            if(index == 1):
                plt.title("T2")
            # DWI
            if(index == 2):
                plt.title("DWI")

                # Display information about the different images
                first_part = str(stacked_image.stem).split("_")[0] + str(stacked_image.stem).split("_")[1]
                t2 = str(stacked_image.stem).split("_")[2]
                dwi = str(stacked_image.stem).split("_")[3]
                adc = str(stacked_image.stem).split("_")[4]
                extra = str(stacked_image.stem).split("_")[5]
                extra2 = str(stacked_image.stem).split("_")[6]
                plt.text(0.25, 130, f"[Patient]: {first_part}\n[T2]: {t2}\n[DWI]: {dwi}\n[ADC]: {adc}\n[Visit]: {extra}\n[Augmentation]: {extra2}", fontsize=13)

            # ADC
            if(index == 3):
                plt.title("ADC")

            plt.imshow(mri_type, cmap='gray', vmin=0, vmax=255)

        plt.savefig(os.path.join(output_folder, f"{stacked_image.stem}.png"))
        plt.close()
        print(f"{stacked_image.stem}.png saved")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script creates plots containing the 3 stacked mri types on an numpy array to check alignment')

    parser.add_argument('--dataset-folder',
                        help='Path to the root of the dataset.',
                        required=True,
                        type=str
                        )

    parser.add_argument('--output-folder',
                        help="",
                        required=True,
                        type=str)

args = parser.parse_args()
main(args.dataset_folder, args.output_folder)
