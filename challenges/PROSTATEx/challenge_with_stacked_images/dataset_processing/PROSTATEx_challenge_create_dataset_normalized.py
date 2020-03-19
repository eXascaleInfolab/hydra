#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import argparse
from pathlib import Path
import os
from PIL import Image
import math
import numpy as np
import random
import pandas as pd
import sys

def check_args(dataset_folder, output_folder):
    """
    Description: this function checks that the given dataset_folder is a directory, contains an "unknown" directory and check if the output exists otherwise it creates it
    Params:
        - dataset_folder: path of the directory containing the 'unknown' directory
        - output_folder: path of the destination folder
    Returns:
        - No return value
    """
    # Check if dataset_folder is a directory
    path_datasetfolder = Path(dataset_folder)

    if (not os.path.isdir(dataset_folder)):
        print('>> The argument --dataset-folder has to be a directory.')
        exit()

    # Check if dataset_folder contains unknown directory
    for (root, dirs, _) in os.walk(path_datasetfolder):
        if root == path_datasetfolder.stem:
            if ('unknown' not in dirs):
                print('>> The argument --dataset_folder must contain unknown directory')
                exit()

    # Check if the output_folder exists. If not, create it.
    path_outputfolder = Path(output_folder)

    if (not path_outputfolder.exists()):
        path_outputfolder.mkdir(parents=True)

def align_stack(folder_path, t2_file_name, dwi_file_name, adc_file_name, mean_std_dict):
    """
    Description: this function takes 3 mri types of the same patients, same lesion, crop the same amount of tissue in the 3 mri types and resize to the desired size before stacking them into a list
    Params:
        - folder_path: path of the folder containing the 3 mri types
        - t2_file_name: name of the t2 mri
        - dwi_file_name: name of the dwi mri
        - adc_file_name: name of the adc mri
        - mean_std_dict: mean_std_dict ={'mean_t2': ..., 'mean_dwi': ..., 'mean_adc': ..., 'std_t2': ...., 'std_dwi': ..., 'std_adc': ...}  mean and standard deviation for each sequence for each patient
    Returns:
        - the stacked and aligned image from the 3 mri types
    """
    # load the three arrays
    t2_np_array = np.load(folder_path / t2_file_name)
    dwi_np_array = np.load(folder_path / dwi_file_name)
    adc_np_array = np.load(folder_path / adc_file_name)

    # Normalize arrays
    t2_np_array = (t2_np_array - mean_std_dict['mean_t2']) / mean_std_dict['std_t2']
    dwi_np_array = (dwi_np_array - mean_std_dict['mean_dwi']) / mean_std_dict['std_dwi']
    adc_np_array = (adc_np_array - mean_std_dict['mean_adc']) / mean_std_dict['std_adc']

    # Convert arrays to PIL image
    t2_img = Image.fromarray(t2_np_array)
    dwi_img = Image.fromarray(dwi_np_array)
    adc_img = Image.fromarray(adc_np_array)

    # Get position of the lesion
    t2_pos = [int(elem) for elem in t2_file_name.stem.split('_')[3].split('-')[1:3]]
    dwi_pos = [int(elem) for elem in dwi_file_name.stem.split('_')[3].split('-')[1:3]]
    adc_pos = [int(elem) for elem in adc_file_name.stem.split('_')[3].split('-')[1:3]]

    # Use "VoxelSpacing" from the CSV file to crop the same amount of tissue on each image
    t2_voxel_spacing = np.array([float(elem) for elem in t2_file_name.stem.split('_')[4].split('-')[1:3]]) # (0.5,0.5,3) -> only working on 1 slice, so we can omit the third dimension
    dwi_voxel_spacing = np.array([float(elem) for elem in dwi_file_name.stem.split('_')[4].split('-')[1:3]])
    adc_voxel_spacing = np.array([float(elem) for elem in adc_file_name.stem.split('_')[4].split('-')[1:3]])

    # Crop a large patch. The patch size from the biggest image is fixed. Others must be computed
    t2_patch_size = np.array([100,100])

    # Dwi resolution < t2 resolution. Hence, we need less dwi pixels to get the same amount of tissue as on the t2
    dwi_patch_size = t2_patch_size // (dwi_voxel_spacing / t2_voxel_spacing) # [16,16]
    adc_patch_size = t2_patch_size // (adc_voxel_spacing / t2_voxel_spacing)

    # Crop the images
    t2_cropped = t2_img.crop((t2_pos[0] - t2_patch_size[0], t2_pos[1] - t2_patch_size[1], t2_pos[0] + t2_patch_size[0], t2_pos[1] + t2_patch_size[1]))
    dwi_cropped = dwi_img.crop((dwi_pos[0] - dwi_patch_size[0], dwi_pos[1] - dwi_patch_size[1], dwi_pos[0] + dwi_patch_size[0], dwi_pos[1] + dwi_patch_size[1]))
    adc_cropped = adc_img.crop((adc_pos[0] - adc_patch_size[0], adc_pos[1] - adc_patch_size[1], adc_pos[0] + adc_patch_size[0], adc_pos[1] + adc_patch_size[1]))

    # Resize images
    dwi_cropped_resized = dwi_cropped.resize(t2_cropped.size, Image.BICUBIC)
    adc_cropped_resized = adc_cropped.resize(t2_cropped.size, Image.BICUBIC)

    return [t2_cropped, dwi_cropped_resized, adc_cropped_resized]


def create_pngs(dataset_folder, output_folder):
    # Create Path objects from args
    path_datasetfolder = Path(dataset_folder)
    path_outputfolder = Path(output_folder)

    # Create Path objects for the True and False directories, where the NumPy arrays are going to be loaded from
    unknown_nparrays_path = path_datasetfolder / 'unknown'

    # Create lambda which flattens a list. https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    flatten = lambda l: [item for sublist in l for item in sublist]

    # [true|false]_nparrays: list of lists of strings containing file names in the [true|false] directories. List of lists => to be flattened
    unknown_nparrays = [files for (_, _, files) in os.walk(unknown_nparrays_path)]

    # Flatten boths lists to create a list of strings, remove hidden files, add filenames as Path objects (useful later)
    unknown_nparrays = [Path(file) for file in flatten(unknown_nparrays) if not file.startswith('.')]

    # Regroup files per patient
    unknown_nparrays_dict = {}

    # True
    for file_name in unknown_nparrays:
        # Get patientID from file_name
        patientID = file_name.stem.split('_')[0]

        # Add file_name to the corresponding patient's list of files
        unknown_nparrays_dict.setdefault(patientID, []).append(file_name)

    # Total number of expected combinations to be output
    nb_combinations = 0

    # Number of stacked images saved
    nb_save = 0

    # Create Path objects for each output directory
    path_outputfolder_unknown = path_outputfolder / 'unknown'
    if (not path_outputfolder_unknown.exists()):
        path_outputfolder_unknown.mkdir(parents=True)

    # Iterate over the patients, convert NumPy arrays, export them
    for index, (patientID, file_names) in enumerate(unknown_nparrays_dict.items()):
        global_array_t2 = np.array([])
        global_array_dwi = np.array([])
        global_array_adc = np.array([])

        # {'visit': {'fid': {'t2': [filenames], 'dwi': [filenames], 'adc': [filenames]}}}
        visit_to_fid_to_sequence_type_to_filename = {}

        # Iterate over patient's files to classify files by sequence type
        for file_name in file_names:
            # Load numpy array
            image_nparray = np.load(unknown_nparrays_path / file_name)

            # Get finding id and visit id
            fid = file_name.stem.split('_')[1].split('-')[1]
            visit = file_name.stem.split('_')[5].split('-')[1]
            visit_to_fid_to_sequence_type_to_filename.setdefault(visit, {})
            visit_to_fid_to_sequence_type_to_filename[visit].setdefault(fid, {})
            visit_to_fid_to_sequence_type_to_filename[visit][fid].setdefault('t2', [])
            visit_to_fid_to_sequence_type_to_filename[visit][fid].setdefault('adc', [])
            visit_to_fid_to_sequence_type_to_filename[visit][fid].setdefault('dwi', [])

            if 't2' in file_name.stem:
                global_array_t2 = np.concatenate((global_array_t2, image_nparray), axis=None)
                visit_to_fid_to_sequence_type_to_filename[visit][fid]['t2'].append(file_name)
            elif 'ADC' in file_name.stem:
                global_array_adc = np.concatenate((global_array_adc, image_nparray), axis=None)
                visit_to_fid_to_sequence_type_to_filename[visit][fid]['adc'].append(file_name)
            else:
                global_array_dwi = np.concatenate((global_array_dwi, image_nparray), axis=None)
                visit_to_fid_to_sequence_type_to_filename[visit][fid]['dwi'].append(file_name)

        # Check if a sequence is missing for this patient
        for visit, fid_to_sequence_type_to_filename in visit_to_fid_to_sequence_type_to_filename.items():
            for fid in fid_to_sequence_type_to_filename:
                nb_t2 = len(visit_to_fid_to_sequence_type_to_filename[visit][fid]['t2'])
                nb_adc = len(visit_to_fid_to_sequence_type_to_filename[visit][fid]['adc'])
                nb_dwi = len(visit_to_fid_to_sequence_type_to_filename[visit][fid]['dwi'])

                nb_combinations += nb_t2 * nb_adc * nb_dwi
                print(f'Combinations: {nb_combinations}')

        # Compute mean and standard deviation for each sequence
        mean_std_dict = {'mean_t2': np.mean(global_array_t2),
                         'std_t2': np.std(global_array_t2),
                         'mean_dwi': np.mean(global_array_dwi),
                         'std_dwi': np.std(global_array_dwi),
                         'mean_adc': np.mean(global_array_adc),
                         'std_adc': np.std(global_array_adc)}

        # Iterate over each t2-dwi-adc combination
        for visit, fid_to_sequence_type_to_filename in visit_to_fid_to_sequence_type_to_filename.items():
            for fid, sequences_to_filenames in fid_to_sequence_type_to_filename.items():
                for t2_file_name in sequences_to_filenames['t2']:
                    for dwi_file_name in sequences_to_filenames['dwi']:
                        for adc_file_name in sequences_to_filenames['adc']:
                            # Align and stack 200x200 patches
                            stacked = align_stack(unknown_nparrays_path, t2_file_name, dwi_file_name, adc_file_name, mean_std_dict)

                            # Find the center of the stack images
                            width, height = stacked[0].size
                            x_middle = math.floor(width/2)
                            y_middle = math.floor(height/2)

                            # Crop a region around the center
                            final_stacked_image = [img.crop((x_middle - 32, y_middle - 32, x_middle + 33, y_middle + 33)) for img in stacked]

                            # Create the stacked numpy array: one channel for each mri type
                            final_stacked_image = np.array([ np.array( img ) for img in final_stacked_image ])

                            # Increment the counter of stacked images saved
                            nb_save += 1

                            # Generate the output name
                            output_name = path_outputfolder_unknown / f"t2-{'_'.join(t2_file_name.stem.split('_')[0:3])}_dwi-{'_'.join(dwi_file_name.stem.split('_')[2:3])}_adc-{'_'.join(adc_file_name.stem.split('_')[2:3])}_visit{visit}.npy"

                            # Check that this output_name does not already exist
                            if os.path.exists(output_name):
                                print('DUPLICATE')
                                import IPython; IPython.embed()

                            # Save the stacked image
                            np.save(output_name, final_stacked_image)

    # Aanity checks => had to add the condition because of .DS_Store files
    n_unknown = len([elem for elem in os.listdir(path_outputfolder_unknown) if not elem.startswith('.')])
    print()
    print(f"Number of elements in output_folder/unknown/: {n_unknown}")
    print()
    print(f"Expected: nb_combinations  == n_unknwon")
    print(f" {nb_combinations}  = {nb_combinations} (expected {n_unknown})")
    if nb_combinations !=  n_unknown:
        print('  ERROR! Number of elements does not match')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Create a dataset containing PNG images from a directory containing NumPy arrays')

    parser.add_argument('--datasetfolder',
                        help='path to the dataset directory (has to contain True and False folders)',
                        required=True,
                        type=str)

    parser.add_argument('--outputfolder',
                        help='output directory',
                        required=True,
                        type=str)

    args = parser.parse_args()

    random.seed(42)

    # Check if the arguments are valid
    check_args(args.datasetfolder, args.outputfolder)

    # Load the nparrays in dataset_folder, generate PNG files, output them to output_folder
    create_pngs(args.datasetfolder, args.outputfolder)
