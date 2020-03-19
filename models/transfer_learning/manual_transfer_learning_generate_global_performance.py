#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

# Importation of modules
import matplotlib.pyplot as plt
import argparse
import pickle
from glob import glob
from pathlib import Path
import numpy as np
import math

###### UTIL ######
def show_ref_performance(global_performance, output_directory, training_or_validation):
    """
    Description:
        - Plots the performance of the current model with the last layer of DS1/full model.
          This shows if the features learned in other datasets than DS1 improve the DS1 classification model
    Params:
        - global_performance:  dict {loss: {dataset1_frozen: [], dataset1_full: [], dataset2_frozen: [], ...}, ...}
        - output_directory: path to the directory which the images are going to be saved into
        - training_or_validation: string, either 'training' or 'validation', depending on the images to generate
    Returns:
        - No return value
    """
    # For each metric, create a new figure
    for metric, datasets in global_performance.items():
        fig = plt.figure(figsize=(12,8))
        length = 0
        legends = []

        # For all datasets of one metric
        for dataset, values in datasets.items():

            # For the first dataset
            if length == 0:
                temp = length + len(values)
            # For the other datasets, continue at the same epoch as the previous dataset
            else:
                length = length - 1
                temp = length + len(values)
            plt.plot(list(range(length, temp)), values)
            length=temp

            # Store all legends
            legends.append(dataset+"\n")

            # Set title name
            plt.title('Model performance on the target dataset, with the Decision Maker from DS1')

            # Set x-axis name
            plt.xlabel('Number of epochs')

            # Set y-axis name (i.e the name of the metric)
            plt.ylabel(metric)

        # End of all dataset for one metric
        # Put a tick every "length * 0.10" epochs
        plt.xticks(np.arange(0, length, step=math.ceil(length*0.10)))
        plt.legend(legends, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=len(datasets.keys()))
        plt.tight_layout()
        plt.savefig(output_directory / f'{training_or_validation}_{metric}.png')


def load_dicts(input_directory, training_or_validation):
    """
    Description:
        - Loads every serialized file containing global performance dictionaries
    Params:
        - input_directory: path to the root directory which the dictionaries are going to be loaded from
        - training_or_validation: string, either 'training' or 'validation', depending on the images to generate
    Returns:
        - A merged global perfomance dictionary with all the transfer learning steps
    """
    
    global_performance_dicts = []
    global_performance_dict = {}

    # Load training global_performance dictionaries
    paths = list(Path(input_directory).rglob(f'global_performance_dict_{training_or_validation}.pckl'))

    # Re-order paths
    ordered_paths = []
    order = ['DS1Full', 'DS2Frozen', 'DS2Full', 'DS3Frozen', 'DS3Full', 'DS4Frozen', 'DS4Full']

    for dataset_name in order:
        for path in paths:
            if dataset_name in str(path):
                ordered_paths.append(path)

    # Load each dictionary
    for path_to_dict in ordered_paths:
        f = open(path_to_dict, 'rb')
        dict = pickle.load(f)
        global_performance_dicts.append(dict)

    # Merge dictionaries
    # global_performance_dicts = [{metric: {dataset_name: [values, ...]}}, ...]
    for dict in global_performance_dicts:
        for metric, dataset_values in dict.items():
            global_performance_dict.setdefault(metric, {})

            for dataset, values in dataset_values.items():
                global_performance_dict[metric][dataset] = values

    return global_performance_dict

###### END UTIL ######


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='')

    parser.add_argument('--input-directory',
                        help='root of the output directory used by Tensorboard and to save the models',
                        required=True,
                        type=str)

    parser.add_argument('--output-directory',
                        help='root of the output directory used by Tensorboard and to save the models',
                        required=True,
                        type=str)

    args = parser.parse_args()

    global_performance_dict_training = load_dicts(args.input_directory, 'training')
    global_performance_dict_validation = load_dicts(args.input_directory, 'validation')

    show_ref_performance(global_performance_dict_training, Path(args.output_directory), 'training')
    show_ref_performance(global_performance_dict_validation, Path(args.output_directory), 'validation')
