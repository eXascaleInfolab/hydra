#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import pydicom
import numpy as np
import os
from PIL import Image
import normalizeDicom
import pandas as pd
import argparse
from tqdm import tqdm
import sys
import math
import glob
# cross-platform path
from pathlib import Path


def main(dataset_folder, lesions_csv, output_folder):
    """
    Description: this script takes as input the Kaggle Brain dataset path, the .csv groundtruth and creates numpy arrays in the given output folder
    Params:
        dataset_folder   - Required  : folder containing the dataset
        lesions_csv      - Required  : csv file containing the nodule position, etc.
        output_folder    - Required  : output folder
    Returns:
        - No return value
    """
    # Create a True/False directory in the output folder
    os.makedirs(f"{output_folder}/{True}", exist_ok=True)
    os.makedirs(f"{output_folder}/{False}", exist_ok=True)

    # Read csv file
    csv_lesions = pd.read_csv(lesions_csv, sep=',', usecols=['PatientID','Nodule Center Position','fid','Diagnosis'] )

    # Get the total number of findings = number of rows in the csv file
    nfindings = len(csv_lesions)

    # Number of slices exported
    nexported = 0

    # Iterate over the rows of the csv file
    for nrow, slice_data in csv_lesions.iterrows():

        # Print progress
        if (nrow+1) % 10 == 0: print(f"\n  ROW {nrow+1}/{nfindings}")

        # Gather data
        patient_id = slice_data['PatientID'].strip() # CT-Training-lc001 for training, LUNGx-CT001 for testing
        pos_x, pos_y = slice_data['Nodule Center Position'].strip().split(' ')
        finding_id = slice_data['fid']
        diagnosis = slice_data['Diagnosis'] # True or False

        # Load file
        image = Image.open(os.path.join(dataset_folder, f'{1 if diagnosis else 0}/{patient_id}.jpg'))
        image = image.convert('L')

        # Create output name, add nrow at the end to avoid duplication
        dest_fname = f"{patient_id}_fid-{finding_id}_pos-{pos_x}-{pos_y}"
        dest = Path(f"{output_folder}/{diagnosis}/{dest_fname}.npy")

        # Save the numpy array
        np.save(dest, image)

        # Print progress
        print('v', end='', flush=True)

        # Increment the number of exported files
        nexported += 1



    # Sanity checks
    print()
    print(f"Findings in the CSV (nfindings): {nfindings}")
    print(f"Expected: nexported == nfindings")
    print(f"{nexported} (expected {nfindings})")
    if nexported != nfindings:
        print('  ERROR! Size mismatch!')
    print()

    ntrue = len(os.listdir(f"{output_folder}/{True}"))
    nfalse = len(os.listdir(f"{output_folder}/{False}"))
    print(f"Expected: ntrue + nfalse == nexported")
    print(f"{ntrue} + {nfalse} = {ntrue + nfalse} (expected {nexported})")
    if ntrue + nfalse != nexported:
        print('  ERROR! File numbers do not match!')
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='')

    parser.add_argument('--datasetfolder',
                        help='path to root of the dataset folder.',
                        required=True,
                        type=str
                        )

    parser.add_argument('--lesionscsv',
                        help='Path to the .csv file containing the training labels',
                        required=True,
                        type=str
                        )

    parser.add_argument('--outputfolder',
                        help='path to output folder.',
                        required=True,
                        type=str
                        )

    args = parser.parse_args()

    main(args.datasetfolder, args.lesionscsv, args.outputfolder)
