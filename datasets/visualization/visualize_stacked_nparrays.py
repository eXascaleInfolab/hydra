from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Visualize the stacked NumPy arrays of the given folder')

    parser.add_argument('--inputfolder',
                        help='path to the input directory',
                        required=True,
                        type=str)

    args = parser.parse_args()

    # create a path object
    path_input_folder = Path(args.inputfolder)

    # look for all NumPy arrays in input-folder and subfolders
    for np_file in path_input_folder.rglob('*.npy'):

        # load array
        array = np.load(path_input_folder / np_file)

        # display image
        for img in array:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
            plt.title(np_file.stem)
            plt.show()
