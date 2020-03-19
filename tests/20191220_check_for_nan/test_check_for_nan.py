#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import numpy as np
import os
import argparse
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script checks that no NaN value is in a NumPy array of the given directory')

    parser.add_argument('--input-directory',
                        help='directory to check',
                        required=True,
                        type=str)

    args = parser.parse_args()

    # Get all directories and their files
    for root, dirs, files in os.walk(f'{args.input_directory}'):
        for f in files:
            # If it is a NumPy array
            if f.endswith('.npy'):
                # Load the NumPy array
                np_array = np.load(os.path.join(root, f))

                # If the NumPy array contains a NaN
                if np.isnan(np_array).any():
                    print('> NAN')
                    import IPython; IPython.embed(); exit()
                else:
                    print('.', end='', flush=True)

    print('> No NaN value found ¯\\_(ツ)_/¯')
