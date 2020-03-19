#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import os, shutil, glob
import numpy as np
import pandas as pd
import pydicom
import argparse
import normalizeDicom
# cross-platform path
from pathlib import Path

def main(datasetFolder, findingsCSV, slicesCSV, outputFolder):
        """
        Description: this script takes as input the PROSTATEx test set path, the .csv groundtruth for findings and the .csv groundtruth for slices and generate numpy arrays into the given output folder
        Params:
            - datasetFolder: path containing the PROSTATEx test set
            - findingsCSV: path containing the .csv groundtruth for findings
            - slicesCSV: path containing the .csv groundtruth for slices
            - outputFolder: destination folder
        Returns:
            - No return value
        """
    # Create a directory unknown to store the numpy arrays
    os.makedirs(f"{outputFolder}/unknown", exist_ok=True)

    # Read and join slices and findings
    slices = pd.read_csv(slicesCSV)
    findings = pd.read_csv(findingsCSV)
    meta = findings.merge(slices, on=['ProxID','fid','pos'])

    # Store the number of findings, i.e. the number of lines of the ProstateX-Findings-Train.csv file or the number of line of the merge
    nfindings = len(meta)

    # Check that the number of lines of the ProstateX-Findings-Train.csv file corresponds to the number of line of the merge
    if nfindings != max((len(slices), len(findings))):
        raise "Join failed!"

    # List that contains tuples of (slice_data, img_k) for all errors
    errors = []

    # When there are mutliple visits for one patients, more than one slice should be generated for the corresponding finding
    additional_slices_multiple_visits = 0

    # Number of slices exported
    nexported = 0

    # For each metadata entry (slice_data works as a dictionary)
    for nrow, slice_data in meta.iterrows():
        # Print progress
        if (nrow+1) % 100 == 0: print(f"\n  ROW {nrow+1}/{nfindings}")

        # Gather data
        patient_id = slice_data['ProxID'] # ProstateX-0000
        mri_type_n = slice_data['DCMSerNum'] # the right mri folder starts with this number
        finding_id = int(slice_data['fid']) # lesion id
        img_i, img_j, img_k = slice_data['ijk'].split(" ") # lesion position in MRI volume
        voxel_i, voxel_j, voxel_k = slice_data['VoxelSpacing'].replace('"','').split(",")

        # Fix for 0-indexed: InstanceNumber from DICOM files starts at index 1, in csv file it starts at 0
        slice_n = int(img_k) + 1

        # Catch negative slice numbers ([slice -1] + [+1 above] == 0)
        if slice_n < 1:
            print('_X11_', end='', flush=True) # make error visible
            # errors.append((slice_data, img_k))
            errors.append((nrow, 'Negative img_k'))
            continue

        # Search for files
        patient_path = Path(f"{datasetFolder}/{patient_id}")

        # Check if the patient_path contains at least 1 visit folder
        if (len(os.listdir(patient_path)) == 0):
            print('_X22_', end='', flush=True)
            errors.append((nrow, f'VISIT FOLDERS NOT FOUND for patient {patient_id}'))

        # Iterate over patient visit_folders
        for visit_index, patient_visit in enumerate(os.listdir(patient_path)):
            # Take multiple visits into account in the total number of files to generate
            if visit_index != 0:
                additional_slices_multiple_visits += 1

            patient_visit_path = Path(f"{patient_path}/{patient_visit}")
            patient_visit_path_string = str(patient_visit_path)

            # Check if the MRI_type folder exists
            # Sometimes, it doesn't exist at all !
            if (len(glob.glob(f"{patient_visit_path_string}/{mri_type_n}-*")) == 0):
                print('_X33_', end='', flush=True)
                errors.append((nrow, f'MRI_TYPE NOT FOUND: {mri_type_n}'))

            # Get into right MRI type
            # NOTE: the dash `-` is fundamental for multi-digit MRI numbers!
            for mri_folder_index, mri_type_path in enumerate(glob.glob(f"{patient_visit_path_string}/{mri_type_n}-*")):
                if mri_folder_index != 0:
                    additional_slices_multiple_visits += 1

                # Go through dicom files
                mri_type = Path(mri_type_path).stem
                available_slices = []

                if 't2' not in mri_type and 'ADC' not in mri_type and 'BVAL' not in mri_type:
                    print('_X_', end='', flush=True)
                    errors.append((nrow, f'NOT T2 OR ADC OR BVAL: {mri_type}'))
                    continue

                if 't2' in mri_type and 'tra' not in mri_type:
                    print('_X_', end='', flush=True)
                    errors.append((nrow, f'NOT TRANSVERSE: {mri_type}'))
                    continue

                # Boolean that indicates if the slice was found
                sliceFound = False

                for fname in glob.glob(f"{mri_type_path}/*.dcm"):
                    # Load the file and check the slice number
                    dcm = pydicom.read_file(fname)

                    # Get the instance number of the .dcm
                    dcm_slice_number = int(dcm.InstanceNumber)

                    # If the instance number corresponds to the slice we are looking for
                    if dcm_slice_number == slice_n:
                        sliceFound  = True

                        # slice_ary = get_image_array(dcm)
                        slice_ary = normalizeDicom.get_normalized_array(dcm)

                        # Build destination
                        # Create output name, add nrow at the end to avoid duplication
                        dest_fname = f"{patient_id}_fid-{finding_id}_mri-{mri_type}_pos-{img_i}-{img_j}-{img_k}_voxel-{voxel_i}-{voxel_j}-{voxel_k}_visit-{visit_index}_slice-{slice_n}_nexported-{nexported}.npy"
                        dest = Path(f"{outputFolder}/unknown/{dest_fname}")

                        # Save the numpy array with the dest_fname
                        np.save(dest, slice_ary)

                        # Make it visible
                        print('v', end='', flush=True)
                        nexported += 1

                        # Slice found and processed: break out of for loop
                        break
                    else:
                        # File loaded is wrong slice: move on
                        available_slices.append(dcm_slice_number)
                        print('.', end='')#, flush=True)


                # End for: when going through all .dcm files is finished
                # print('_ok_', end='', flush=True)

                if not sliceFound:
                    print('_X44_', end='', flush=True)
                    errors.append((nrow, visit_index, f'Slice not found among DCM files nrow {nrow} visit folder index {visit_index}'))

    # New line
    print()

    # Final check
    print(errors)

    # Get the number of errors
    nerrors = len(errors)
    print(f"Findings in CSV (nfindings): {nfindings}")
    print(f"Additional slices -> caused by multiple visits and by multiple mri_type folders having the same number at the beginning (additional_slices_multiple_visits): {additional_slices_multiple_visits}")
    print()

    print(f"Expected: nexported + nerrors == nfindings + additional_slices_multiple_visits")
    print(f"{nexported} + {nerrors} = {nexported + nerrors} (expected {nfindings + additional_slices_multiple_visits})")
    if nexported + nerrors != nfindings + additional_slices_multiple_visits:
        print('  ERROR! Array sizes do not match!')
    print()

    all_unknown = len([elem for elem in os.listdir(f'{outputFolder}/unknown') if not elem.startswith('.')])
    print(f"Expected: all_unknown == nexported")
    print(f"all_unknown = {all_unknown} (expected {nexported})")
    if all_unknown != nexported:
        print('  ERROR! File numbers do not match!')
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script takes as input the PROSTATEx test set, the .csv groundtruth for findings and the .csv groundtruth for slices and generate numpy arrays')

    parser.add_argument('--datasetfolder',
                        help='Path to the root of the dataset.',
                        required=True,
                        type=str
                        )

    parser.add_argument('--findings',
                        help='Path to the .csv file containing information about the findings for the test set',
                        required=True,
                        type=str
                        )

    parser.add_argument('--slices',
                        help='Path to the .csv file containing information about the slices for the test set',
                        required=True,
                        type=str
                        )

    parser.add_argument('--outputfolder',
                        help='Destination folder',
                        required=True,
                        type=str
                        )

    args = parser.parse_args()
    main(args.datasetfolder, args.findings, args.slices, args.outputfolder)
