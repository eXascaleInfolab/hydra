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


def main(dataset_folder, train_csv, test_csv, output_folder):
    """
    Description: this script takes as input the Lung CT Challenge dataset path, the .csv groundtruth for the training and test sets, and creates numpy arrays in the given output folder
    Params:
        dataset_folder   - Required  : folder containing the data set
        train_csv        - Required  : csv file containing the training set information (labels, nodule position, etc.)
        test_csv         - Required  : csv file containing the test set information (labels, nodule position, etc.)
        output_folder    - Required  : output folders
    Returns:
        - No return value
    """
    # Create a True/False directory in the output folder
    os.makedirs(f"{output_folder}/{True}", exist_ok=True)
    os.makedirs(f"{output_folder}/{False}", exist_ok=True)

    # Read both csv files
    csv_training = pd.read_csv(train_csv, sep=',', usecols=['Scan number','Nodule Number','Nodule Center Position','Nodule Center Image','Diagnosis'] )
    csv_test = pd.read_csv(test_csv, sep=',', usecols=['Scan number','Nodule Number','Nodule Center Position','Nodule Center Image','Diagnosis'])

    # Concatenate the two csv files on common columns: on top of each other
    csv_concatenated = pd.concat([csv_training, csv_test], ignore_index=True, sort=False)

    # Get the total number of findings = number of rows of the resulting csv file
    nfindings = len(csv_concatenated)

    # Size check
    if (nfindings != len(csv_training) + len(csv_test)):
        print("Concat failed, the number of rows of csv_training + csv_test should be eqal to the number of findings")
        exit(0)

    # Number of slices exported
    nexported = 0

    # Number of slices with unusable labels
    nsuspicious_malignant = 0

    # Number of slices not found
    nnot_found_slices = 0

    # Iterate over the rows of the concatenated csv file
    for nrow, slice_data in csv_concatenated.iterrows():

        # Print progress
        if (nrow+1) % 10 == 0: print(f"\n  ROW {nrow+1}/{nfindings}")

        # Gather data
        patient_id = slice_data['Scan number'].strip() # CT-Training-lc001 for training, LUNGx-CT001 for testing
        slice_number = int(slice_data['Nodule Center Image']) # corresponds to the InstanceNumber
        pos_x, pos_y = slice_data['Nodule Center Position'].strip().split(' ')
        finding_id = float(slice_data['Nodule Number'])
        diagnosis = slice_data['Diagnosis'].strip() # ['benign', 'malignant', 'Benign nodule', 'Suspicious malignant nodule', 'Primary lung cancer']

        # Convert labels into False/True labels
        if diagnosis in ['malignant', 'Primary lung cancer']:
            diagnosis = True
        elif diagnosis in ['benign', 'Benign nodule',]:
            diagnosis = False
        elif diagnosis in ['Suspicious malignant nodule']:
            nsuspicious_malignant += 1
            continue
        else:
            print('unknown label, exiting...')
            exit(0)

        # Path of the corresponding patient_id
        patient_path = Path(f"{dataset_folder}/{patient_id}")

        # Check if the patient_path contains more that one folder
        if (len([x for x in os.listdir(patient_path) if not x.startswith('.')]) > 1):
            print("ATTENTION, this patient has multiple visits")
            exit(0)

        # For a patient ID
        for patient_visit in os.listdir(patient_path):
            if(patient_visit.startswith(".")):
                continue

            patient_visit_path = Path(f"{patient_path}/{patient_visit}")

            # Check if there are multiple mri types for one visit
            if (len([x for x in os.listdir(patient_visit_path) if not x.startswith('.')]) > 1):
                print("ATTENTION, for one patient and one visit, multiple mri-types found ")
                exit(0)

            # For one visit, iterate over the mri types
            for mri_type in os.listdir(patient_visit_path):
                if(mri_type.startswith(".")):
                    continue

                mri_type_path = Path(f"{patient_visit_path}/{mri_type}")

                # Boolean that indicates if the slice was found
                sliceFound = False

                # Iterate over the DICOM files of an mri type
                for fname in glob.glob(f"{mri_type_path}/*.dcm"):
                    # Load the file
                    dcm = pydicom.read_file(fname)

                    # Get the instance number of the .dcm
                    dcm_slice_number = int(dcm.InstanceNumber)

                    # If the instance number corresponds to the slice we are looking for -> process and save
                    if dcm_slice_number == slice_number:
                        # Use for counting errors
                        sliceFound = True

                        # Get the normalized numpy array
                        slice_ary = normalizeDicom.get_normalized_array(dcm)

                        # Create output name, add nrow at the end to avoid duplication
                        dest_fname = f"{patient_id}_fid-{finding_id}_pos-{pos_x}-{pos_y}_slice-{slice_number}"
                        dest = Path(f"{output_folder}/{diagnosis}/{dest_fname}")

                        # Save the numpy array with the dest_fname
                        np.save(dest, slice_ary)

                        # Print progress
                        print('v', end='', flush=True)

                        # Increment the number of exported files
                        nexported += 1

                        break
                    else:
                        print('.', end='')

                # if the slice was not found in the repository
                if not sliceFound:
                    nnot_found_slices += 1


    # Sanity checks
    print()
    print(f"Findings in the concatenated CSV (nfindings): {nfindings}")
    print(f"Expected: nexported + nsuspicious_malignant + nnot_found_slices == nfindings")
    print(f"{nexported} + {nsuspicious_malignant} + {nnot_found_slices} = {nexported + nsuspicious_malignant + nnot_found_slices} (expected {nfindings})")
    if nexported + nsuspicious_malignant + nnot_found_slices != nfindings:
        print('  ERROR! Array sizes do not match!')
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


    parser.add_argument('--traincsv',
                        help='Path to the .csv file containing the training labels',
                        required=True,
                        type=str
                        )

    parser.add_argument('--testcsv',
                        help='Path to the .csv file containing the test labels',
                        required=True,
                        type=str
                        )

    parser.add_argument('--outputfolder',
                        help='path to output folder.',
                        required=True,
                        type=str
                        )

    args = parser.parse_args()

    main(args.datasetfolder, args.traincsv, args.testcsv, args.outputfolder)
