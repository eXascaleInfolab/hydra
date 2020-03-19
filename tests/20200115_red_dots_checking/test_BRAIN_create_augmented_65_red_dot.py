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
from PIL import ImageDraw
import math
import numpy as np
import random
import pandas as pd

def check_args(dataset_folder, output_folder, split, split_test, nb_augmented_images):
    """
    Description: this function checks that the given arguments respect certain conditions
    Params:
        - dataset_folder: path of the folder containing the dataset
        - output_folder:  name of the output folder
        - split: indicates the proportion of data to use as training/validation-test. For instance 0.8 means 80 percents as training set 20 percetns as validation-test
        - split_test: indicates the proportion of data to use as validation/test set. For instance 0.1 means 10 percents as validation set and 10 percent as test set
        - nb_augmented_images: number of times that each image will be augmented
    Returns:
        - No return value
    """
    path_datasetfolder = Path(dataset_folder)

    # Check if dataset_folder is a directory
    if (not os.path.isdir(dataset_folder)):
        print('>> The argument --dataset-folder has to be a directory.')
        exit()

    # Check if dataset_folder contains True and False directories
    for (root, dirs, _) in os.walk(path_datasetfolder):
        if root == path_datasetfolder.stem:
            if ('True' not in dirs or 'False' not in dirs):
                print('>> The argument --dataset_folder must contain True and False directories.')
                exit()

    path_outputfolder = Path(output_folder)

    # Check if the output_folder exists. If not, create it.
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

    # Pick a degree and flipping value
    degree, prob_flipping, shifting = non_picked_list[index]

    # Remove from the list of possibilites (in order to avoid duplication)
    del(non_picked_list[index])

    # Random rotation from -20 to 20°
    temp_image = image.rotate(degree, resample=Image.BICUBIC)

    print(f'Rotation: {degree} degrees, shifting {shifting}')

    # Crop the images
    width, height = temp_image.size
    x_middle = math.floor(width/2)
    y_middle = math.floor(height/2)
    temp_image = temp_image.crop((x_middle - 32 + shifting, y_middle - 32, x_middle + 33 + shifting, y_middle + 33))

    # Random horizontal flipping
    if prob_flipping > 0.5:
        print(f'Flipping')
        is_flipped = True
        temp_image = temp_image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        print(f'No flipping')
        is_flipped = False

    return temp_image


def crop_large_patch(folder_path, file_name, mean_std_dict, output_folder):
    """
    Description: Crops a large patch centered on the lesion. This is done in order to make the rotation easier (lesion still in the center).
    Params:
        - folder_path: folder containing the image (NumPy array)
        - file_name: name of the NumPy array file
        - mean_std_dict: {'mean': mean_for_the_sequence_of_each_patient, 'std': std_for_the_sequence_of_each_patient }
    Returns:
        - Pillow image
    """
    # load the three arrays
    np_array = np.load(folder_path / file_name)

    # Convert arrays to PIL image
    img = Image.fromarray(np_array)

    # Get position of the lesion
    pos = [int(elem) for elem in file_name.stem.split('_')[2].split('-')[1:3]]

    # TEST
    # Convert to RGB (allows us to set the red dot)
    img = img.convert('RGB')

    # Create a drawer object
    draw = ImageDraw.Draw(img)

    # Draw a red ellipse
    draw.ellipse((pos[0]-3, pos[1]-3, pos[0]+3, pos[1]+3), fill=(255,0,0,255))

    # Save the image with the red dot
    img.save(output_folder / f'{file_name.stem}.png')

    # Crop a large patch. The patch size from the biggest image is fixed
    patch_size = np.array([100,100])

    # Crop the images
    img_cropped = img.crop((pos[0] - patch_size[0], pos[1] - patch_size[1], pos[0] + patch_size[0], pos[1] + patch_size[1]))

    return img_cropped


def augment_images(file_names, nb_augmented_images, input_folder, output_folder, mean_std_dict):
    """
    Description: Crops a large patch centered on the lesion. This is done in order to make the rotation easier (lesion still in the center).
    Params:
        - folder_path: folder containing the image (NumPy array)
        - file_name: name of the NumPy array file
        - mean_std_dict: dictionary containing the mean and std dev for the dataset
    Returns:
        - number of images saved (int)
    """
    # Count the number of saved images
    nb_saved = 0

    for file_name in file_names:
        # Align and stack 200x200 patches
        large_patch = crop_large_patch(input_folder, file_name, mean_std_dict, output_folder)

        # All possibilities
        rotation_flipping_shifting_not_picked = [(i,j,k)  for i in range(-20, 21) for j in [0,1] for k in [-1,0,1]]

        # Data augmentation
        for augmented_index in range(nb_augmented_images):
            augmented_image = augment(large_patch, rotation_flipping_shifting_not_picked)
            nb_saved += 1
            print(nb_saved)
            output_name = output_folder / f"{file_name}_augmented-{augmented_index}.png"
            if os.path.exists(output_name):
                print('DUPLICATE')
                import IPython; IPython.embed()
            augmented_image.save(output_name)

    return nb_saved


def augment_a_class(belonging_class, class_nparrays_dict, split, split_test, nb_augmented_images, input_folder, path_outputfolder, mean_std_dict):
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
    """
    print(nb_augmented_images)

    # Total number of expected combinations to be output
    nb_combinations = 0

    # Number of stacked images saved
    nb_saved = 0

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
        # Train images
        if (index < number_of_training_patients_class):
            nb_saved_train = augment_images(file_names, nb_augmented_images['nbaugmentedimages_training'], input_folder, path_outputfolder_train, mean_std_dict)
            nb_saved += nb_saved_train

        # Val or test images
        else:
            # Val images
            if (index - number_of_training_patients_class < number_of_val_patients_class):
                nb_saved_val = augment_images(file_names, nb_augmented_images['nbaugmentedimages_validation'], input_folder, path_outputfolder_val, mean_std_dict)
                nb_saved += nb_saved_val
            # Test images
            else:
                nb_saved_test = augment_images(file_names, nb_augmented_images['nbaugmentedimages_test'], input_folder, path_outputfolder_test, mean_std_dict)
                nb_saved += nb_saved_test

    return nb_saved


def create_pngs(dataset_folder, output_folder, split, split_test, nb_augmented_images):
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

    # Used to compute mean and std dev
    global_array = np.array([])

    # True
    for file_name in true_nparrays:
        # Get patientID from file_name
        patientID = file_name.stem.split('_')[0]

        # Add file_name to the corresponding patient's list of files
        true_nparrays_dict.setdefault(patientID, []).append(file_name)

        # Mean-std: load numpy array
        image_nparray = np.load(true_nparrays_path / file_name)
        global_array = np.concatenate((global_array, image_nparray), axis=None)

    # False
    for file_name in false_nparrays:
        # Get patientID from file_name
        patientID = file_name.stem.split('_')[0]

        # Add file_name to the corresponding patient's list of files
        false_nparrays_dict.setdefault(patientID, []).append(file_name)

        # Mean-std: load numpy array
        image_nparray = np.load(false_nparrays_path / file_name)
        global_array = np.concatenate((global_array, image_nparray), axis=None)

    # Compute mean and standard deviation for each sequence
    mean_std_dict = {'mean': np.mean(global_array),
                     'std': np.std(global_array)}

    # Augment all images belonging to class False
    nb_saved_false= augment_a_class(belonging_class='0', class_nparrays_dict=false_nparrays_dict, split=split, split_test=split_test, nb_augmented_images=nb_augmented_images, input_folder=false_nparrays_path, path_outputfolder=path_outputfolder, mean_std_dict=mean_std_dict)

    # Augment all images belonging to class True
    nb_saved_true= augment_a_class(belonging_class='1', class_nparrays_dict=true_nparrays_dict, split=split, split_test=split_test, nb_augmented_images=nb_augmented_images, input_folder=true_nparrays_path, path_outputfolder=path_outputfolder, mean_std_dict=mean_std_dict)

    # Total of saved images and combinations
    nb_saved = nb_saved_true + nb_saved_false

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Create a dataset containing PNG images with red dots from a directory containing NumPy arrays')

    parser.add_argument('--dataset-folder',
                        help='path to the dataset directory (has to contain True and False folders)',
                        required=True,
                        type=str)

    parser.add_argument('--output-folder',
                        help='output directory',
                        required=True,
                        type=str)

    parser.add_argument('--split',
                        help='Ratio of the split for training set.',
                        type=float,
                        required=True)

    parser.add_argument('--split-test',
                        help='Ratio of the split for the test set. In this case, the data is split into training, validation and test sets.',
                        type=float,
                        required=False,
                        default=0.0)

    parser.add_argument('--nbaugmentedimages_training',
                        help='Number of augmented slice for each slice.',
                        type=int,
                        required=True)

    parser.add_argument('--nbaugmentedimages_validation',
                        help='Number of augmented slice for each slice.',
                        type=int,
                        required=True)

    args = parser.parse_args()

    nbaugmentedimages = {'nbaugmentedimages_training': args.nbaugmentedimages_training , 'nbaugmentedimages_validation': args.nbaugmentedimages_validation , 'nbaugmentedimages_test': 11 }

    random.seed(42)

    # Check if the arguments are valid
    check_args(args.dataset_folder, args.output_folder, args.split, args.split_test, nbaugmentedimages)

    # Load the nparrays in dataset_folder, generate PNG files, output them to output_folder
    create_pngs(args.dataset_folder, args.output_folder, args.split, args.split_test, nbaugmentedimages)
