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
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
import argparse
from tqdm import tqdm
from pathlib import Path

###### UTIL ######
def new_loader(path):
    """
    Description: load the given .npy file and convert it into a tensor
    Params:
        - path: path of the file containing the stacked images
    Returns
        - the corresponding tensor
    """
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



def load_data(test_data_path, batch_size):
    """
    Description:
        - Load the training and validation sets. Folders have to be organized in this way:
            train/label1/image.npy
            test/label2/image2.npy
        - Organize the data in batches of size batch_size
    Params:
        - test_data_path: training set path
        - batch_size: number of elements per batch
    Returns:
        - A PyTorch data loader containing the test set
    """
    # Create a PyTorch DatasetFolder object  containing all numpy arrays with their ground-truth
    test_data = datasets.DatasetFolder(root=test_data_path, extensions=("npy"), loader=new_loader)

    # Create a test data loader to organize the DatasetFolder in batches of given size
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    return test_data_loader

def confusion(prediction, truth):
    """
    Description:
        - Computes the confusion matrix with the given predictions and truth

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
###### END UTIL ######

###### MODEL ######
class Flatten(nn.Module):
    """Flattens the output of the previous layer"""
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ReproductedModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3, input_dimension=3, last_layer_size='small'):
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


###### TRAINING AND TESTING FUNCTIONS ######
def test_model(model, test_data_loader, device, loss_function):
    """
    Description:
        - Test a model on the given data
    Params:
        - model: a trained model
        - test_data_loader: unseen test data (inputs, labels)
        - device: device used to run the model, i.e 'cuda' or 'cpu'
        - loss_function: loss function
    Returns:
        - metrics values {'loss': loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1score': f1score, 'specificity': specificity, 'auc': auc}
    """

    model.to(device)

    # model.eval() disables specific layers such as dropout and batchnorm.
    model.eval()

    with torch.no_grad():

        # list of predictions to compute AUC (float)
        predictions_per_batch = {'labels': [], 'predictions_positive_class': [], 'predicted_class': torch.tensor([]).to(device), 'labels_tensor': torch.tensor([]).to(device)}

        # dict {'TP': [TP_batch0, TP_batch1, TP_batch2, ...], 'FP': [FP_batch0, FP_batch1, ...], 'TN':[...], 'FN': [...]}
        confusion_matrix_dict = {}

        # For each batch
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

        metrics_test =  get_metrics(loss_function, confusion_matrix_dict, predictions_per_batch)

        return metrics_test
###### END TRAINING AND TESTING FUNCTIONS ######

##### MAIN TRAINING FUNCTIONS
def main_generate_test(models_dict, paths_dict, fixed_parameters_dict):
    """
    Description: generate a given number of random model initializations and pick the best according to the metric to optimize
    Params:
        - models_dict: dict {'model_to_output': ...} contains the output path
        - paths_dict: dict {'test_set': ...} contains the path used as test set
        - fixed_parameters_dict: {'loss_function': ...,
                                'cuda_device': ....,
                                'nb_models': ...,
                                'optimized_metric': ...}
                                 contains the loss function, the cuda-device name, the number of models to generate and the metric to optimize
    Returns:
        - No return value
    """
    # These variables are global
    global INPUT_NB_CHANNEL
    global SIZE_LAST_LAYER

    # Get available device
    device = fixed_parameters_dict['cuda_device'] if torch.cuda.is_available() else 'cpu'

    # Load all data as training data
    test_data_loader = load_data(paths_dict['test_set'], 32)

    print("> Data loaded")

    best_model = {}

    # Generate nb_models models
    for index in tqdm(range(fixed_parameters_dict['nb_models'])):
        model = ReproductedModel(input_dimension=INPUT_NB_CHANNEL, last_layer_size=SIZE_LAST_LAYER)
        metrics_test = test_model(model, test_data_loader, device, fixed_parameters_dict['loss_function'])

        # For the first model
        if index == 0:
            best_model['dict'] = model.state_dict()
            best_model['best_metrics'] = metrics_test
        else:
            # If the current random initialization is better than the previous best initialization
            if metrics_test[fixed_parameters_dict['optimized_metric']] >= best_model['best_metrics'][fixed_parameters_dict['optimized_metric']]:
                best_model['dict'] = model.state_dict()
                best_model['best_metrics'] = metrics_test

    # Save the model
    torch.save(best_model['dict'], models_dict['model_to_output'])
    print(f"Best model saved as {models_dict['model_to_output']}")
    print(f"Best model metrics: {best_model['best_metrics']}")
##### END MAIN TRAINING FUNCTIONS

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Generate a given number of random model initializations and pick the best according to the metric to optimize')

    parser.add_argument('--testset',
                        help='validation set 1 folder path',
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

    parser.add_argument('--nbmodels',
                        help='number of different models/initializations to test',
                        required=False,
                        type=int,
                        default=100)

    parser.add_argument('--inputchannel',
                        help='number of channel of the input',
                        required=True,
                        type=int,
                        choices=[1,3])

    parser.add_argument('--optimizedmetric',
                        help="model with the best --optimized_metric will be saved, choices=['auc', 'accuracy', 'precision', 'recall', 'f1score', 'specificity']. Default = accuracy",
                        required=False,
                        type=str,
                        choices=['auc', 'accuracy', 'precision', 'recall', 'f1score', 'specificity'],
                        default='accuracy')

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

    # loss functions
    loss_functions_dict = {
    'L1Loss': nn.L1Loss(),
    'MSELoss': nn.MSELoss(),
    'CrossEntropyLoss': nn.CrossEntropyLoss()
    }

    # Global variables : number of channel and last layer choice
    INPUT_NB_CHANNEL = int(args.inputchannel)
    SIZE_LAST_LAYER = args.lastlayer

    path_output_directory = Path(args.outputdirectory)

    if (not path_output_directory.exists()):
        path_output_directory.mkdir(parents=True)

    # Generate all initializations and pick the best one according to the metric to optimize
    main_generate_test(
    models_dict={'model_to_output': str(path_output_directory / 'baseline.pth')},
    paths_dict={'test_set': args.testset},
    fixed_parameters_dict={'loss_function': loss_functions_dict[args.lossfunction],
                            'cuda_device': args.cudadevice,
                            'nb_models': args.nbmodels,
                            'optimized_metric': args.optimizedmetric})
