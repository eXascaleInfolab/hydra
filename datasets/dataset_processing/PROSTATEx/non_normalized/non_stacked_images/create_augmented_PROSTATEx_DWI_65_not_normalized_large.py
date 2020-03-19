import argparse
from pathlib import Path
import os
from PIL import Image
import math
import numpy as np
import random
import pandas as pd
import sys

def check_args(dataset_folder, output_folder, split, split_test, nb_augmented_images):
    """
    Description: This function checks that the command line arguments are valid arguments.
    Params:
        - dataset_folder: folder containing the dataset
        - output_folder: output folder
        - split: Percentage of the data used as training data
        - spilt_test: Percentage of the data used as test data
        - nb_augmented_images: dictionary containing the number of augmented images to generate
    Returns:
        - No return value
    """

    # Check if dataset_folder is a directory
    path_datasetfolder = Path(dataset_folder)

    if (not os.path.isdir(dataset_folder)):
        print('>> The argument --dataset-folder has to be a directory.')
        exit()

    # Check if dataset_folder contains True and False directories
    for (root, dirs, _) in os.walk(path_datasetfolder):
        if root == path_datasetfolder.stem:
            if ('True' not in dirs or 'False' not in dirs):
                print('>> The argument --dataset_folder must contain True and False directories.')
                exit()

    # Check if the output_folder exists. If not, create it.
    path_outputfolder = Path(output_folder)

    if (not path_outputfolder.exists()):
        path_outputfolder.mkdir(parents=True)

    # Check if the output_folder contains [train|val][0|1] folders. If not, create them.
    path_outputfolder_train_zero = path_outputfolder / 'train' / '0'
    path_outputfolder_train_one = path_outputfolder / 'train' / '1'

    path_outputfolder_val_zero = path_outputfolder / 'val' / '0'
    path_outputfolder_val_one = path_outputfolder / 'val' / '1'

    path_outputfolder_test_zero = path_outputfolder / 'test' / '0'
    path_outputfolder_test_one = path_outputfolder / 'test' / '1'

    if (not path_outputfolder_train_zero.exists()):
        path_outputfolder_train_zero.mkdir(parents=True)

    if (not path_outputfolder_train_one.exists()):
        path_outputfolder_train_one.mkdir(parents=True)

    if (not path_outputfolder_val_zero.exists()):
        path_outputfolder_val_zero.mkdir(parents=True)

    if (not path_outputfolder_val_one.exists()):
        path_outputfolder_val_one.mkdir(parents=True)

    if (not path_outputfolder_test_zero.exists()):
        path_outputfolder_test_zero.mkdir(parents=True)

    if (not path_outputfolder_test_one.exists()):
        path_outputfolder_test_one.mkdir(parents=True)

    # Check if the split value is in the right range
    if (split < 0.0 or split > 1.0):
        print('>> The argument --split has to be a float value between 0.0 (included) and 1.0 (included).')
        exit()

    # Check if the split value is in the right range
    if (split_test < 0.0 or split_test > 1.0):
        print('>> The argument --split-test has to be a float value between 0.0 (included) and 1.0 (included).')
        exit()

    if (split + split_test > 1.0):
        print('>> The result of split + split-test has to be a float value smaller than 1.0.')
        exit()

    # Check if the number of augmented images is the right range
    if (nb_augmented_images['nbaugmentedimages_training'] < 0 and nb_augmented_images['nbaugmentedimages_training'] > 60 ):
        print('>> The argument --nbaugmentedimages_training has to be an int value larger than 0 but smaller than 60')
        exit()

    if (nb_augmented_images['nbaugmentedimages_validation'] < 0 and nb_augmented_images['nbaugmentedimages_validation'] > 60 ):
        print('>> The argument --nbaugmentedimages_validation has to be an int value larger than 0 but smaller than 60')
        exit()

    if (nb_augmented_images['nbaugmentedimages_test'] < 0):
        print('>> The argument --nbaugmentedimages_test has to be an int value larger than 0.')
        exit()


def crop_resize_large_patch(folder_path, dwi_file_name, t2_file_name, mean_std_dict):
    """
    Description: Aligns and stack images from the 3 sequences (T2, DWI, ADC).
    Params:
        - folder_path: path to the folder containing the 3 images
        - t2_file_name: name of the T2 file to align with the DWI and ADC
        - dwi_file_name: name of the DWI file to align with the T2 and ADC
        - mean_std_dict: mean and standard deviation per sequence for this patient
    Returns:
        - Pillow image
    """

    # Load the array
    dwi_np_array = np.load(folder_path / dwi_file_name)

    # Convert arrays to PIL image
    dwi_img = Image.fromarray(dwi_np_array)

    # Get position of the lesion
    dwi_pos = [int(elem) for elem in dwi_file_name.stem.split('_')[3].split('-')[1:3]]

    # Using "VoxelSpacing" from the CSV file to crop the same amount of tissue on each image
    t2_voxel_spacing = np.array([float(elem) for elem in t2_file_name.stem.split('_')[4].split('-')[1:3]]) # (0.5,0.5,3) -> only working on 1 slice, so we can omit the third dimension
    dwi_voxel_spacing = np.array([float(elem) for elem in dwi_file_name.stem.split('_')[4].split('-')[1:3]])

    # Crop a large patch. The patch size from the biggest image is fixed. Others must be computed
    patch_size = np.array([200,200])

    # dwi resolution < t2 resolution. Hence, we need less dwi pixels to get the same amount of tissue as on the t2
    dwi_patch_size = patch_size // (dwi_voxel_spacing / t2_voxel_spacing)

    # Crop the image
    dwi_cropped = dwi_img.crop((dwi_pos[0] - dwi_patch_size[0], dwi_pos[1] - dwi_patch_size[1], dwi_pos[0] + dwi_patch_size[0], dwi_pos[1] + dwi_patch_size[1]))

    # resize image
    dwi_cropped_resized = dwi_cropped.resize((400,400), Image.BICUBIC)

    return dwi_cropped_resized


def augment(image, non_picked_list):
    """
    Description: Performs the augmentation and the cropping.
    Params:
        - image: Pillow image
        - non_picked_list: list containing the different augmentation possibilities as tuples (to avoid duplicates)
    Returns:
        - Pillow image
    """

    # Randomly select the augmentation possibility
    index = random.randint(0, len(non_picked_list)-1)

    # Pick a degree, a flipping value and a shifting value
    degree, prob_flipping, shifting = non_picked_list[index]

    # Remove from the list of possibilites (in order to avoid duplication)
    del(non_picked_list[index])

    # Rotate the image
    temp_image = image.rotate(degree, resample=Image.BICUBIC)
    print(f'Rotation: {degree} degrees, shifting {shifting}')

    # Crop the image
    width, height = temp_image.size
    x_middle = math.floor(width/2)
    y_middle = math.floor(height/2)
    temp_image = temp_image.crop((x_middle - 65 + shifting, y_middle - 65, x_middle + 65 + shifting, y_middle + 65))

    # Random horizontal flipping
    if prob_flipping > 0.5:
        print(f'Flipping')
        is_flipped = True
        temp_image = temp_image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        print(f'No flipping')
        is_flipped = False

    # Resize image to 65x65
    temp_image = temp_image.resize((65,65), Image.BICUBIC)

    return temp_image


def augment_images(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images, input_folder, output_folder, mean_std_dict):
    """
    Description: Augments the different DWI images for a patient.
    Params:
        - visit_to_fid_to_sequence_type_to_filename: dictionary {'visit': {'fid': {'t2': [filenames], 'dwi': [filenames]}}}
        - nb_augmented_images: number of augmented images
        - input_folder: folder containing the T2, DWI and ADC images
        - output_folder: output_folder
        - mean_std_dict: mean and standard deviation per sequence for this patient
    Returns:
        - number of saved images (int)
    """

    # Count the number of saved images
    nb_saved = 0

    for visit, fid_to_sequence_type_to_filename in visit_to_fid_to_sequence_type_to_filename.items():
        for fid, sequences_to_filenames in fid_to_sequence_type_to_filename.items():
            for dwi_file_name in sequences_to_filenames['dwi']:

                t2_file_name = sequences_to_filenames['t2'][0]

                # Align images and get DWI
                cropped_image = crop_resize_large_patch(input_folder, dwi_file_name, t2_file_name, mean_std_dict)

                # All possibilities
                rotation_flipping_shifting_not_picked = [(i,j,k)  for i in range(-20, 21) for j in [0,1] for k in [-1,0,1]]

                # Data augmentation
                for augmented_index in range(nb_augmented_images):
                    # Augment the image
                    augmented_image = augment(cropped_image, rotation_flipping_shifting_not_picked)
                    nb_saved += 1
                    print(nb_saved)

                    # Export the image
                    output_name = output_folder / f"dwi-{'_'.join(dwi_file_name.stem.split('_')[2:3])}_fid-{fid}_visit{visit}_augmented-{augmented_index}.png"
                    if os.path.exists(output_name):
                        print('DUPLICATE')
                        import IPython; IPython.embed(); exit()

                    augmented_image.save(output_name)

    return nb_saved


def count_combination(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images):
    """
    Description: Counts the number of combinations for the T2, DWI and ADC images of a specific patient.
    Params:
        - visit_to_fid_to_sequence_type_to_filename: dictionary {'visit': {'fid': {'t2': [filenames], 'dwi': [filenames], 'adc': [filenames]}}}
        - nb_augmented_images: number of augmented images
    Returns:
        - number of combinations for a specific patient (int)
    """

    # Count the number of saved images
    nb_saved = 0

    for visit, fid_to_sequence_type_to_filename in visit_to_fid_to_sequence_type_to_filename.items():
        for fid, sequences_to_filenames in fid_to_sequence_type_to_filename.items():
            for dwi_file_name in sequences_to_filenames['dwi']:
                for augmented_index in range(nb_augmented_images):
                    nb_saved += 1
                    print(nb_saved)

    return nb_saved


def augment_a_class(belonging_class, class_nparrays_dict, split, split_test, nb_augmented_images, input_folder, path_outputfolder, dict_balancing):
    """
    Description: Performs augmentation and folder organization for a specific class.
    Params:
        - belonging_class: label
        - class_nparrays_dict:
        - split: percentage of the dataset used as training set
        - split_test: percentage of the dataset used as test set
        - nb_augmented_images: dictionary containing the number of augmented images to generate
        - input_folder: dataset input folder (for this class)
        - path_outputfolder: path to the output folder
        - mean_std_dict: dictionary containing the mean and the std dev for the dataset
    Returns:
        - number of images saved (int)
    Note:
        - dict_balancing stores by how much each class has to be augmented in order to have a balanced dataset.
          The full process is performed twice. The 1st time to compute dict_balancing. The 2nd time to actually perform the augmentation.
    """

    print(nb_augmented_images)

    # Total number of expected combinations to be output
    nb_combinations = 0

    # Number of stacked images saved
    nb_saved = 0

    # Dictionary that will contain the number of training, val and test samples saved
    dict_for_balancing = {'train': 0, 'val': 0, 'test': 0}

    if dict_for_balancing != None:
        temp_counter = {'train': 0, 'val': 0, 'test': 0}

    # Create Path objects for each output directory
    path_outputfolder_train = path_outputfolder / 'train' / belonging_class
    path_outputfolder_val = path_outputfolder / 'val' / belonging_class
    path_outputfolder_test = path_outputfolder / 'test' / belonging_class

    # Get the number of patients belonging to the class
    number_of_patients_class = len(class_nparrays_dict.keys())

    # Get the number of  patients that will be used for the training set
    number_of_training_patients_class = math.floor(split * number_of_patients_class)

    # Get the number of patients that will be used for the test set
    number_of_test_patients_class = math.floor(split_test * number_of_patients_class)

    # Get the number of patients that will be used for the validation set
    number_of_val_patients_class = number_of_patients_class - number_of_training_patients_class - number_of_test_patients_class

    # Iterate over the patients, convert NumPy arrays, export them
    for index, (patientID, file_names) in enumerate(class_nparrays_dict.items()):

        global_array_dwi = np.array([])

        # {'visit': {'fid': {'t2': [filenames], 'dwi': [filenames]}}}
        visit_to_fid_to_sequence_type_to_filename = {}

        # Iterate over patient's files to classify files by sequence type
        for file_name in file_names:
            # Load numpy array
            image_nparray = np.load(input_folder / file_name)

            # Get finding id and visit id
            fid = file_name.stem.split('_')[1].split('-')[1]
            visit = file_name.stem.split('_')[5].split('-')[1]

            # Create the structure of the dictionary
            visit_to_fid_to_sequence_type_to_filename.setdefault(visit, {})
            visit_to_fid_to_sequence_type_to_filename[visit].setdefault(fid, {})
            visit_to_fid_to_sequence_type_to_filename[visit][fid].setdefault('dwi', [])
            visit_to_fid_to_sequence_type_to_filename[visit][fid].setdefault('t2', [])

            # Classify the file
            if 't2' in file_name.stem:
                visit_to_fid_to_sequence_type_to_filename[visit][fid]['t2'].append(file_name)
            elif 'ADC' in file_name.stem:
                pass
            else:
                global_array_dwi = np.concatenate((global_array_dwi, image_nparray), axis=None)
                visit_to_fid_to_sequence_type_to_filename[visit][fid]['dwi'].append(file_name)

        # Compute mean and standard deviation for each sequence
        mean_std_dict = {'mean_dwi': np.mean(global_array_dwi),
                         'std_dwi': np.std(global_array_dwi)}


        # Train images
        if (index < number_of_training_patients_class):
            if dict_balancing != None:
                 temp_counter['train'] += count_combination(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images['nbaugmentedimages_training'])
            else:
                nb_saved_train = augment_images(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images['nbaugmentedimages_training'], input_folder, path_outputfolder_train, mean_std_dict)
                dict_for_balancing['train'] += nb_saved_train
                nb_saved += nb_saved_train

        # Val or test images
        else:
            # Val images
            if (index - number_of_training_patients_class < number_of_val_patients_class):
                if dict_balancing != None:
                     temp_counter['val'] += count_combination(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images['nbaugmentedimages_validation'])
                else:
                    nb_saved_val = augment_images(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images['nbaugmentedimages_validation'], input_folder, path_outputfolder_val, mean_std_dict)
                    dict_for_balancing['val'] += nb_saved_val
                    nb_saved += nb_saved_val

            # Test images
            else:
                if dict_balancing != None:
                     temp_counter['test'] += count_combination(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images['nbaugmentedimages_test'])
                else:
                    nb_saved_test = augment_images(visit_to_fid_to_sequence_type_to_filename, nb_augmented_images['nbaugmentedimages_test'], input_folder, path_outputfolder_test, mean_std_dict)
                    dict_for_balancing['test'] += nb_saved_test
                    nb_saved += nb_saved_test

    if dict_balancing != None:
        nb_augmented_images['nbaugmentedimages_training'] = math.floor(nb_augmented_images['nbaugmentedimages_training'] * (dict_balancing['train']/temp_counter['train']))
        if temp_counter['val'] != 0:
            nb_augmented_images['nbaugmentedimages_validation'] = math.floor(nb_augmented_images['nbaugmentedimages_validation'] * (dict_balancing['val']/temp_counter['val']))
        else:
            nb_augmented_images['nbaugmentedimages_validation'] = 0

        if temp_counter['test'] !=0:
            nb_augmented_images['nbaugmentedimages_test'] = math.floor(nb_augmented_images['nbaugmentedimages_test'] * (dict_balancing['test']/temp_counter['test']))
        else:
            nb_augmented_images['nbaugmentedimages_test'] = 0
        (nb_saved_new, nb_combinations_new, dict_for_balancing_new) = augment_a_class(belonging_class, class_nparrays_dict, split, split_test, nb_augmented_images, input_folder, path_outputfolder, None)

        return (nb_saved_new, nb_combinations_new, dict_for_balancing_new)

    return (nb_saved, nb_combinations, dict_for_balancing)


def create_images(dataset_folder, output_folder, split, split_test, nb_augmented_images):
    """
    Description: Main function coordinating the augmentation process from end to end.
    Params:
        - dataset_folder: root folder of the dataset
        - output_folder: root of the output folder
        - split: percentage of the dataset used as training set
        - split_test: percentage of the dataset used as test set
        - nb_augmented_images: dictionary containing the number of augmented images to generate
    Returns:
        - no return value
    """

    # Create Path objects from args
    path_datasetfolder = Path(dataset_folder)
    path_outputfolder = Path(output_folder)

    # Create Path objects for the True and False directories, where the NumPy arrays are going to be loaded from
    true_nparrays_path = path_datasetfolder / 'True'
    false_nparrays_path = path_datasetfolder / 'False'

    # Create lambda which flattens a list. https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
    flatten = lambda l: [item for sublist in l for item in sublist]

    # [true|false]_nparrays: list of lists of strings containing file names in the [true|false] directories. List of lists => to be flattened
    true_nparrays = [files for (_, _, files) in os.walk(true_nparrays_path)]
    false_nparrays = [files for (_, _, files) in os.walk(false_nparrays_path)]

    # Flatten boths lists to create a list of strings, remove hidden files, add filenames as Path objects (useful later)
    true_nparrays = [Path(file) for file in flatten(true_nparrays) if not file.startswith('.')]
    false_nparrays = [Path(file) for file in flatten(false_nparrays) if not file.startswith('.')]

    # Regroup files per patient
    true_nparrays_dict = {}
    false_nparrays_dict = {}

    # True
    for file_name in true_nparrays:
        # Get patientID from file_name
        patientID = file_name.stem.split('_')[0]

        # Add file_name to the corresponding patient's list of files
        true_nparrays_dict.setdefault(patientID, []).append(file_name)

    # False
    for file_name in false_nparrays:
        # Get patientID from file_name
        patientID = file_name.stem.split('_')[0]

        # Add file_name to the corresponding patient's list of files
        false_nparrays_dict.setdefault(patientID, []).append(file_name)

    # Augment all images belonging to class False
    (nb_saved_false, nb_combinations_false, dict_for_balancing_false) = augment_a_class('0', false_nparrays_dict, split, split_test, nb_augmented_images, false_nparrays_path, path_outputfolder, None)

    # Augment all images belonging to class True
    (nb_saved_true, nb_combinations_true, dict_for_balancing_true) = augment_a_class('1', true_nparrays_dict, split, split_test, nb_augmented_images, true_nparrays_path, path_outputfolder, dict_for_balancing_false)

    # Total of saved images and combinations
    nb_saved = nb_saved_true + nb_saved_false
    nb_combinations = nb_combinations_true + nb_combinations_false

    path_outputfolder_train_zero = path_outputfolder / 'train' / '0'
    path_outputfolder_train_one = path_outputfolder / 'train' / '1'

    path_outputfolder_val_zero = path_outputfolder / 'val' / '0'
    path_outputfolder_val_one = path_outputfolder / 'val' / '1'

    path_outputfolder_test_zero = path_outputfolder / 'test' / '0'
    path_outputfolder_test_one = path_outputfolder / 'test' / '1'

    # Sanity checks => had to add the condition because of .DS_Store files
    ntrue = len([elem for elem in os.listdir(true_nparrays_path) if not elem.startswith('.')])
    nfalse = len([elem for elem in os.listdir(false_nparrays_path) if not elem.startswith('.')])

    ntrain_zero = len([elem for elem in os.listdir( path_outputfolder / 'train' / '0') if not elem.startswith('.')])
    ntrain_one = len([elem for elem in os.listdir(path_outputfolder / 'train' / '1') if not elem.startswith('.')])

    nval_zero = len([elem for elem in os.listdir(path_outputfolder / 'val' / '0') if not elem.startswith('.')])
    nval_one = len([elem for elem in os.listdir(path_outputfolder / 'val' / '1') if not elem.startswith('.')])

    ntest_zero = len([elem for elem in os.listdir(path_outputfolder / 'test' / '0') if not elem.startswith('.')])
    ntest_one = len([elem for elem in os.listdir(path_outputfolder / 'test' / '1') if not elem.startswith('.')])

    print()
    print(f"Number of elements in dataset_folder/False directory (nfalse): {nfalse}")
    print(f"Number of elements in dataset_folder/True directory (ntrue): {ntrue}")

    print(f"Number of elements in output_folder/train/0 directory (ntrain_zero): {ntrain_zero}")
    print(f"Number of elements in output_folder/train/1 directory (ntrain_one): {ntrain_one}")
    print(f"Number of elements in output_folder/val/0 directory (nval_zero): {nval_zero}")
    print(f"Number of elements in output_folder/val/1 directory (nval_one): {nval_one}")
    print(f"Number of elements in output_folder/test/0 directory (ntest_zero): {ntest_zero}")
    print(f"Number of elements in output_folder/test/1 directory (ntest_one): {ntest_one}")
    print()
    print(f"Argument nb_augmented_images: {nb_augmented_images}")
    print()
    print(f"Expected: nb_saved == ntrain_zero + ntrain_one + nval_zero + nval_one + ntest_zero + ntest_one")
    print(f" {nb_saved} = {nb_saved} (expected {ntrain_zero + ntrain_one + nval_zero + nval_one + ntest_zero + ntest_one})")
    if nb_saved != ntrain_zero + ntrain_one + nval_zero + nval_one + ntest_zero + ntest_one:
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

    parser.add_argument('--split',
                        help='Ratio of the split for training set.',
                        type=float,
                        required=True)

    parser.add_argument('--splittest',
                        help='Ratio of the split for the test set. In this case, the data is split into training, validation and test sets.',
                        type=float,
                        required=False,
                        default=0.0)

    parser.add_argument('--nbaugmentedimagestraining',
                        help='Number of augmented slice for each slice.',
                        type=int,
                        required=True)

    parser.add_argument('--nbaugmentedimagesvalidation',
                        help='Number of augmented slice for each slice.',
                        type=int,
                        required=True)

    args = parser.parse_args()

    random.seed(42)

    nbaugmentedimages = {'nbaugmentedimages_training': args.nbaugmentedimagestraining , 'nbaugmentedimages_validation': args.nbaugmentedimagesvalidation , 'nbaugmentedimages_test': 11 }

    # Check if the arguments are valid
    check_args(args.datasetfolder, args.outputfolder, args.split, args.splittest, nbaugmentedimages)

    # Load the nparrays in dataset_folder, generate files, output them to output_folder
    create_images(args.datasetfolder, args.outputfolder, args.split, args.splittest, nbaugmentedimages)
