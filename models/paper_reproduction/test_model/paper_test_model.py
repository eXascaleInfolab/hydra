#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

# Importation of modules
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from brutelogger import BruteLogger

###### UTIL ######
def new_loader(path):
    """
    Description:
        - Load an image, convert it to a tensor with one single channel and send the latter to cuda/cpu
    Params:
        - path: takes an image path
    Returns:
        - A tensor with one channel instead of 3 (default loader automatically converts to RGB)
    """
    # This variable is global
    global INPUT_NB_CHANNEL

    # Load the data into a numpy array
    np_array = np.load(path)

    # Count the number of channel of the data
    if(len(np_array.shape)==2):
        number_channel = 1
    elif(len(np_array.shape)==3):
        number_channel = 3
    else:
        print(f"This number of input channel is not allowed")
        exit(0)

    # Check that the given input dimension match with the real number of channel of the data
    if (INPUT_NB_CHANNEL != number_channel):
        print(f"The given number of channel ({INPUT_NB_CHANNEL}) does not match with the number of channel of the input ({number_channel})")
        exit(0)

    # If the non-stacked images (1 channel) are given to the neural network
    if(number_channel == 1):
        # Add an extra dimension to the tensor
        wrap = np.array([np_array])
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(wrap).to('cuda' if torch.cuda.is_available() else 'cpu')

    # If the stacked images (3 channels) are given to the neural network
    elif(number_channel == 3):
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(np_array).to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        print(f"{len(np_array.shape)}-dimensional can't be fed to the neural network")

    return tensor

def load_data(test_data_path):
    """
    Description:
        - Load the training and validation sets. Folders have to be organized in this way:
            train/label1/image.npy
            test/label2/image2.npy
        - Organize the data in batches of size batch_size
    Params:
        - test_data_path: training set path
    Returns:
        - A PyTorch data loader containing the test set
    """
    # Create a PyTorch DatasetFolder object  containing all numpy arrays with their ground-truth
    test_data = datasets.DatasetFolder(root=test_data_path, extensions=("npy"), loader=new_loader)

    # Create a test data loader to organize the DatasetFolder in batches of given size
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1)

    return test_data_loader


def confusion(prediction, truth):
    """
    Description:
        - Compute the confusion matrix with the given predictions and truth

    Params:
        - prediction: list/array of model outputs
        - truth: ground-truth for each prediction

    Returns:
        A dict {'TP': true_positives, 'FP': false_positives, 'TN': true_negatives, 'FN': false_negatives} containing the confusion matrix:
        - 1 and 1 (True Positive)
        - 1 and 0 (False Positive)
        - 0 and 0 (True Negative)
        - 0 and 1 (False Negative)
    """
    prediction = prediction.tolist()
    truth = truth.tolist()

    # sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None): Computes confusion matrix to evaluate the accuracy of a classification
    # numpy.ravel(): Returns a contiguous flattened array.
    # Doc: "In the binary case, we can extract true positives, etc as follows:
    # >>> tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    # >>> (tn, fp, fn, tp)
    # (0, 2, 1, 1)"
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(truth, prediction, labels=[0,1]).ravel()

    return {'TP': true_positives, 'FP': false_positives, 'TN': true_negatives, 'FN': false_negatives}



def get_metrics(loss_function, confusion_matrix_dict, predictions_per_batch):
    """
    Description:
        - Compute loss, accuracy, precision, recall, f1score, specificity, auc using the predictions and the confusion matrix
    Params:
        - loss_function: function used to compute the loss
        - confusion_matrix_dict: {'TP': [TP_batch_0, TP_batch_1, ...], 'FP': [], 'TN': [], 'FN': []}
        - predictions_per_batch: dictionary containing all predictions with ground-truth for the loss computation
    Returns:
        - a dictionary containing each metric and their value. Usually the average over an epoch.
    """
    true_positives = sum(confusion_matrix_dict['TP'])
    false_positives = sum(confusion_matrix_dict['FP'])
    true_negatives = sum(confusion_matrix_dict['TN'])
    false_negatives = sum(confusion_matrix_dict['FN'])

    if (true_positives + false_positives + true_negatives + false_negatives != 0):
        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
    else:
        accuracy = 0

    if (true_positives + false_positives != 0):
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if (true_positives + false_negatives != 0.0):
        recall = true_positives / (true_positives + false_negatives) # = True positive rate
    else:
        recall = 0.0

    if (precision + recall != 0):
        f1score = 2 * ((precision*recall) / (precision + recall))
    else:
        f1score = 0.0

    if (true_negatives + false_positives != 0):
        specificity = true_negatives / (true_negatives + false_positives)
    else:
        specificity = 0.0

    auc = roc_auc_score(predictions_per_batch['labels'], predictions_per_batch['predictions_positive_class'])

    loss = loss_function(predictions_per_batch['predicted_class'], predictions_per_batch['labels_tensor']).item()

    return {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1score': f1score, 'specificity': specificity, 'auc': auc}

def plot_tensorboard(plot_name, nb_epochs, dict_metrics, writer):
    """
    Description:
        - Plots the different metrics using Tensorboard
    Params:
        - plot_name: name of the tensorboard plot
        - nb_epochs: number of epochs
        - dict_metrics:  {'loss': [losses], 'accuracy': [accuracies], 'precision': [precision], 'recall': [recalls], 'f1score': [f1scores], 'specificity': [specificities], 'auc': [aucs]}
        - writer: tensorboard writer object
    Returns:
        - No return value
    """
    for epoch in range(nb_epochs):
        for key, value in dict_metrics.items():
            writer.add_scalar(f'{plot_name}/{key}', value[epoch], epoch)

def report_tensorboard(plot_name, text, writer):
    """ Makes a written report using Tensorboard"""
    writer.add_text(plot_name, text)
###### END UTIL ######

###### MODEL ######
class Flatten(nn.Module):
    """Flattens the output of the previous layer"""
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ReproductedModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0, input_dimension=3, last_layer_size='small'):
        super(ReproductedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=input_dimension, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Dropout(p=dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Dropout(p=dropout),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Dropout(p=dropout),
            nn.MaxPool2d(2),
        )

        self.features.add_module('flatten', Flatten())

        if last_layer_size == 'small':
            self.classifier = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ELU(),

                nn.Dropout(p=dropout),

                nn.Linear(256, 64),
                nn.ELU(),

                nn.Dropout(p=dropout),

                nn.Linear(64, 16),
                nn.ELU(),
            )

            self.last_layer = nn.Sequential(
                nn.Linear(16, num_classes),
                nn.Softmax(1),
            )
        elif(last_layer_size == 'medium'):
            self.classifier = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ELU(),

                nn.Dropout(p=dropout)
            )

            self.last_layer = nn.Sequential(
                nn.Linear(256, 64),
                nn.ELU(),

                nn.Dropout(p=dropout),

                nn.Linear(64, 16),
                nn.ELU(),

                nn.Linear(16, num_classes),
                nn.Softmax(1)
            )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x
###### END MODEL ######

###### TESTING FUNCTIONS ######

def test_model(model, test_data_loader, device, loss_function, writer=None, plot_name='', output_directory="./default"):
    """
    Description:
        - Test a model on the given data and stores the results on tensorboard
    Params:
        - model: a trained model
        - test_data_loader: unseen test data (inputs, labels)
        - device: device used to run the model, i.e 'cuda' or 'cpu'
        - loss_function: loss function
        - writer: Tensorboard SummaryWriter
        - plot_name: Tensorboard report name (if == '' => nothing is displayed)
    Returns:
        - metrics values {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1score': f1score, 'specificity': specificity, 'auc': auc}
    """
    model.to(device)

    # model.eval() disables specific layers such as dropout and batchnorm.
    model.eval()

    with torch.no_grad():

        # List of predictions to compute AUC (float)
        predictions_per_batch = {'labels': [], 'predictions_positive_class': [], 'predicted_class': torch.tensor([]).to(device), 'labels_tensor': torch.tensor([]).to(device)}

        confusion_matrix_dict = {}

        # For each batch
        start = time.time()
        for inputs, labels in test_data_loader:

            # Send data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Feed the model and get output
            output = model(inputs)

            # Compute loss
            loss = loss_function(output, labels)

            # Doc Torch.max():
            ## "Returns a namedtuple (values, indices) where values is the maximum value of each row of the input tensor
            ## in the given dimension dim. And indices is the index location of each maximum value found (argmax)."
            _, predicted = torch.max(output.data, 1)


            # output.data = [prob_class_0, prob_class_1] => keep only prob_class_1 to compute AUC
            # torch.index_select for "output.data", on dimension 1 (=> indexing column by column => index 1 = element in the current row, column 1), torch.tensor([1]) = indices'
            all_positives = torch.index_select(output.data, 1, torch.tensor([1]).to(device))

            predictions_per_batch['labels'] = predictions_per_batch['labels'] + labels.cpu().numpy().tolist()
            predictions_per_batch['predictions_positive_class'] = predictions_per_batch['predictions_positive_class'] + all_positives.cpu().numpy().tolist()
            predictions_per_batch['predicted_class'] = torch.cat((predictions_per_batch['predicted_class'], output), 0)
            predictions_per_batch['labels_tensor'] = torch.cat((predictions_per_batch['labels_tensor'].long(), labels), 0)

            # {'TP': int, 'FP': int, 'TN': int, 'FN': int}
            conf_matrix_per_batch = confusion(predicted, labels)

            # Tranform dictionary of {str: int} into {str: list(int)} => required for get_metrics()
            # conf_matrix_per_batch = {key: [value] for (key, value) in conf_matrix_per_batch.items()}
            for key, value in conf_matrix_per_batch.items():
                confusion_matrix_dict.setdefault(key, []).append(value)

        end = time.time()

        metrics_test =  get_metrics(loss_function, confusion_matrix_dict, predictions_per_batch)


        if plot_name != '':
            summary_text = f"""
            SUMMARY OF THE CLASSIFIER ON TEST SET
            -------------------
            AUC: {metrics_test['auc']}
            --------------------
            Running time: {end-start}

            """

            print(summary_text)

            # Filename to write
            f = f'{output_directory}/summary.txt'

            # Open the file with writing permission
            myfile = open(f, 'w')

            # Write a line to the file
            myfile.write(summary_text)

        return metrics_test

###### END TRAINING AND TESTING FUNCTIONS ######

##### MAIN TRAINING FUNCTIONS


##### END MAIN TRAINING FUNCTIONS
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script tests the performance of the given model on the given dataset (no enhanced prediction)')

    parser.add_argument('--testset',
                        help='test set folder path',
                        required=True,
                        type=str)

    parser.add_argument('--lossfunction',
                        help='loss function. [L1Loss, MSELoss, CrossEntropyLoss]',
                        required=False,
                        type=str,
                        choices=['L1Loss', 'MSELoss', 'CrossEntropyLoss'],
                        default='CrossEntropyLoss')

    parser.add_argument('--cudadevice',
                        help='cuda device',
                        required=False,
                        type=str,
                        default='cuda')

    parser.add_argument('--modeltoload',
                        help='pth file to load as initial model',
                        required=True,
                        type=str)

    parser.add_argument('--inputchannel',
                help='number of channel of the input',
                required=True,
                type=int,
                choices=[1,3])

    parser.add_argument('--outputdirectory',
                        help='root of the output directory used by Tensorboard and to save the models',
                        required=True,
                        type=str,
                        default='./runs')

    parser.add_argument('--lastlayer',
                    help="choose the last layer size among small and medium",
                    required=False,
                    type=str,
                    choices=['small', 'medium'],
                    default='small')

    args = parser.parse_args()

    # Number of channels of the input data
    INPUT_NB_CHANNEL = int(args.inputchannel)

    # Choose which last layer to use
    SIZE_LAST_LAYER = args.lastlayer

    BruteLogger.save_stdout_to_file(path=f"{args.outputdirectory}/stdout_logs")

    # Loss functions
    loss_functions_dict = {
    'L1Loss': nn.L1Loss(),
    'MSELoss': nn.MSELoss(),
    'CrossEntropyLoss': nn.CrossEntropyLoss()
    }

    # Load the model to test
    model = ReproductedModel(input_dimension=INPUT_NB_CHANNEL, last_layer_size=SIZE_LAST_LAYER)

    # check that the model to load has the same number dimension input as the one given in args
    input_dimension_model_to_load = torch.load(args.modeltoload)['features.0.weight'].size()[1]
    if(INPUT_NB_CHANNEL != input_dimension_model_to_load ):
        print(f"The given number of channel ({INPUT_NB_CHANNEL}) does not match the number of input channel of the model to load ({input_dimension_model_to_load})")
        exit(0)
    else:
        model.load_state_dict(torch.load(args.modeltoload))

    print("> Model created")

    test_data_loader = load_data(test_data_path=args.testset)

    print("> Data loaded")

    test_model(model, test_data_loader, args.cudadevice, loss_functions_dict[args.lossfunction], writer=None, plot_name='Summary', output_directory=args.outputdirectory)
