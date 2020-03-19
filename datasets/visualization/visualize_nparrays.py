from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    # arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Create a dataset containing PNG images from a directory containing NumPy arrays')

    parser.add_argument('--inputfolder',
                        help='path to the input directory',
                        required=True,
                        type=str)

    parser.add_argument('--mode',
                        help='rgb or gray',
                        required=False,
                        type=str,
                        default='gray',
                        choices=['rgb', 'gray'])

    parser.add_argument('--stacked',
                        help='True if images are stacked images',
                        required=False,
                        type=str,
                        default='False',
                        choices=['True', 'False'])

    args = parser.parse_args()

    # create a path object
    path_input_folder = Path(args.inputfolder)

    # look for all NumPy arrays in input-folder and subfolders
    for np_file in path_input_folder.rglob('*.npy'):

        # load array
        array = np.load(path_input_folder / np_file)

        if args.stacked == 'True':
            # display stacked image
            for img in array:
                plt.imshow(img, cmap='gray', vmin=0, vmax=255)
                plt.title(np_file.stem)
                plt.show()
        else:
            # display image
            if args.mode == 'gray':
                plt.imshow(array, cmap='gray', vmin=0, vmax=255)
                plt.title(np_file.stem)
                plt.show()
            else:
                plt.imshow(array)
                plt.title(np_file.stem)
                plt.show()
