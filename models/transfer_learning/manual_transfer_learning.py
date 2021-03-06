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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle

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
        tensor = torch.from_numpy(wrap).to(args.cudadevice if torch.cuda.is_available() else 'cpu')

    # If the stacked images (3 channels) are given to the neural network
    elif(number_channel == 3):
        # Send the tensor to the GPU/CPU depending on what device is available
        tensor = torch.from_numpy(np_array).to(args.cudadevice if torch.cuda.is_available() else 'cpu')
    else:
        print(f"{len(np_array.shape)}-dimensional can't be fed to the neural network")

    return tensor


def load_data(train_data_path, validation_data_path, batch_size):
    """
    Description:
        - Load the training and validation sets. Folders have to be organized in this way:
            train/label1/image.npy
            test/label2/image2.npy
        - Organize the data in batches of size batch_size
    Params:
        - train_data_path: training set path
        - validation_data_path: validation set path
        - batch_size: number of elements per batch
    Returns:
        - A tuple containing (training_loader, validation_loader)
    """
    # Before storing an image in the dataset, resize it, convert it to grayscale and cast it into pytorch tensor
    # interpolation=Image.BICUBIC: BICUBIC works well when upscaling and downscaling images
    # TRANSFORM_IMG = transforms.Compose([
    #     #transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
    #     #transforms.Grayscale(num_output_channels=1),
    #     transforms.ToTensor(),
    #     ])

    ##### Training set ####
    train_data = datasets.DatasetFolder(root=train_data_path, extensions=("npy"), loader=new_loader)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    ##### Validation set ####
    if validation_data_path != None:
        validation_data = datasets.DatasetFolder(root=validation_data_path, extensions=("npy"), loader=new_loader)
        validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    else:
        validation_data_loader = None

    return (train_data_loader, validation_data_loader)

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

def plot_val_and_train(training_metric_per_epoch, val_metric_per_epoch, metric_name, writer, best_epoch, best_lr, tensorboard_output_name):
    """
    Description:
        - Plots training and validation values for each metric
    Params:
        - training_metric_per_epoch: list containing the metric value for the training set at each epoch
        - val_metric_per_epoch: list containing the metric value for the validation set at each epoch
        - metric_name: metric name (i.e 'loss', 'accuracy', 'precision', 'recall', 'f1score', 'specificity', 'auc' )
        - writer: tensorboard writer object
        - number_epoch: number of epoch, y-axis
        - lr: learning rate use
        - tensorboard_output_name: tensorboard figure name
    Returns:
        - No return value
    """
    fig = plt.figure()
    plt.plot(training_metric_per_epoch)

    if val_metric_per_epoch != None:
        plt.plot(val_metric_per_epoch)

    # plot training loss before and after backpropagation
    plt.legend([f'Training {metric_name}', f'Validation {metric_name}'], loc='upper left')

    plt.title(f'{metric_name}, Best HP: {best_epoch} epochs, {best_lr} lr')
    plt.xlabel('Number of epochs')
    plt.ylabel(metric_name)

    # Add the previous plot on tensorboard
    writer.add_figure(f"{tensorboard_output_name}/trainingAndValidation{metric_name}", fig)

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
    def __init__(self, num_classes=2, dropout=0.3, input_dimension=3):
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
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x


###### END MODEL ######

###### TRAINING AND TESTING FUNCTIONS ######
def train_model(model, train_data_loader, learning_rate, dropout, nb_epoch, loss_function, optimizer, device, writer, tensorboard_output_name, output_directory, optimized_metric, test_reference, last_layer_path, validation_data_loader=None, save_intermediate='', freeze=False):
    """
        Description:
            - train the given model without cross-validation with respect to the given hyperparameters and display results on tensorboard
        Params:
            - model: the PyTorch model to be trained
            - train_data_loader: PyTorch data loader containing the training set
            - learning_rate: learning rate for the optimizer
            - nb_epoch: number of epochs used for the training
            - loss_function: loss function used
            - optimizer: PyTorch optimizer object containing the parameters to optimize
            - device: device used to train the model, i.e gpu/cpu
            - writer: tensorboard writer object
            - tensorboard_output_name: tensorboard output name
            - optimized_metric: choose the best intermediate model to keep with respect to this metric: auc, accuracy, precision, recall, f1-score, specificity
            - validation_data_loader: PyTorch data loader containing the validation set
            - save_intermediate: save the intermediate best model
        Returns:
            - training metrics are returned
    """
    global INPUT_NB_CHANNEL

    # Send model to GPU if available
    model.to(device)

    # Metrics per epoch
    metrics_per_epoch_training = {}

    if validation_data_loader != None:
        metrics_per_epoch_val = {}
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=5 )
        # Best model
        best_model = {optimized_metric: 0, 'dict': None, 'epoch': -1, 'accuracy_condition': 0}

    losses_before_backpropagation_per_epoch = []

    for epoch in range(nb_epoch):
        # {'TP': [TP_batch_0, TP_batch_1, ...], 'TN': [], 'FP': [], 'FN': []}
        confusion_matrix_dict = {}

        # List of predictions to compute AUC (float)
        predictions_per_batch = {'labels': [], 'predictions_positive_class': []}

        # model.train() tells your model that you are training the model.
        # So effectively layers like dropout, batchnorm etc. which behave
        # differently on the train and test procedures.
        model.train()

        if freeze:
            model.features.eval()

        # For each batch
        for index, (inputs, labels) in enumerate(train_data_loader):

            # Send data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Feed the model and get output
            output = model(inputs)

            # Compute loss
            loss = loss_function(output, labels)

            # Backprop
            # Computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            loss.backward()

            # Optimizer.step is performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule.
            optimizer.step()

            # Every time a variable is back propagated, the gradient will be accumulated instead of being replaced.
            # (This makes it easier for rnn, because each module will be back propogated through several times.)
            # zero_grad() sets all gradients to zero
            optimizer.zero_grad()

        # End of an epoch:
        metrics_epoch_training = test_model(model=model, test_data_loader=train_data_loader, device=device, loss_function=loss_function)

        # Decrease the learning rate when the training AUC decreases -> optimize the training performance
        if validation_data_loader != None:
            scheduler.step(metrics_epoch_training['auc'])

        # Add all metrics for this epoch to the corresponding list
        for key, value in metrics_epoch_training.items():
            metrics_per_epoch_training.setdefault(key, []).append(value)

        # If we want to see the performance of the current model (i.e at each epoch) on the validation set
        if(validation_data_loader != None):
            metrics_epoch_val = test_model(model=model, test_data_loader=validation_data_loader, device=device, loss_function=loss_function)

            # Add all validation metrics for this epoch to the corresponding list
            for key, value in metrics_epoch_val.items():
                metrics_per_epoch_val.setdefault(key, []).append(value)

        print(f"> Epoch {epoch}")
        print(f">>Train: {metrics_epoch_training}")
        if (validation_data_loader != None):
            print(f">>Val: {metrics_epoch_val}")

        # Test the current feature extractor + a reference decision maker on the reference dataset
        if test_reference:
            # First step of the pipeline: testing the current model, nothing to load
            if last_layer_path == 'no_path':
                new_model = ReproductedModel(dropout=0.3, input_dimension=INPUT_NB_CHANNEL) # get a new instance
                new_model.load_state_dict(model.state_dict()) # copy weights and stuff
            # Otherwise, load the decision maker, attach it and test this model
            else:
                new_model = ReproductedModel(dropout=0.3, input_dimension=INPUT_NB_CHANNEL) # get a new instance
                new_model.load_state_dict(model.state_dict()) # copy weights and stuff
                f = open(last_layer_path, 'rb')
                last_l = pickle.load(f)
                f.close()
                new_model.last_layer = last_l

            # Compute the metrics of the current model with the last layer of the model after DS1 full
            metrics_epoch_ref_validation = test_model(model=new_model, test_data_loader=dataset_of_reference_val, device=device, loss_function=loss_function)
            metrics_epoch_ref_training = test_model(model=new_model, test_data_loader=dataset_of_reference_training, device=device, loss_function=loss_function)

            # Get the dataset number and frozen/full string
            tensorboard_output_name_split = tensorboard_output_name.split('/')
            dataset_number = tensorboard_output_name_split[0][2]
            full_or_frozen = 'Frozen' if freeze else 'Full'

            # Add the metrics in the global performance dictionary key = metric, value= value of the metric:
            # global_performance_dict_validation = {loss: {dataset1_full: [], dataset2_frozen: [], ...}, ...}
            for key, value in metrics_epoch_ref_validation.items():
                global_performance_dict_validation.setdefault(key, {})
                global_performance_dict_validation[key].setdefault(f'DS{dataset_number}_{full_or_frozen}', []).append(value)

            for key, value in metrics_epoch_ref_training.items():
                global_performance_dict_training.setdefault(key, {})
                global_performance_dict_training[key].setdefault(f'DS{dataset_number}_{full_or_frozen}', []).append(value)


        if (validation_data_loader != None and metrics_epoch_val[optimized_metric] >= best_model[optimized_metric] and metrics_epoch_val['accuracy'] >= best_model['accuracy_condition']):
            print(f">> New best {optimized_metric} found. Previous {best_model[optimized_metric]}, current {metrics_epoch_val[optimized_metric]}")
            best_model[optimized_metric] = metrics_epoch_val[optimized_metric]
            best_model['accuracy_condition'] = metrics_epoch_val['accuracy']
            best_model['dict'] = model.state_dict()
            best_model['epoch'] = epoch
            torch.save(best_model['dict'], f'{save_intermediate}_best.pth')
            print(f"Intermediate best model {save_intermediate}_best.pth saved. Epoch {best_model['epoch']}, {optimized_metric} {best_model[optimized_metric]}")


        # Save the model at each epoch
        print(f">> Epoch {epoch}, manual save.")
        torch.save(model.state_dict(), f'{save_intermediate}_epoch{epoch}.pth')

    # Plot the different metrics for training and validation
    # if validation_data_loader != None:
    for key, value in metrics_per_epoch_training.items():
        if(validation_data_loader != None):
            plot_val_and_train(training_metric_per_epoch=value,
                                val_metric_per_epoch=metrics_per_epoch_val[key],
                                metric_name=key,
                                writer=writer,
                                best_epoch=nb_epoch,
                                best_lr=learning_rate,
                                tensorboard_output_name=tensorboard_output_name)
        else:
            plot_val_and_train(training_metric_per_epoch=value,
                                val_metric_per_epoch=None,
                                metric_name=key,
                                writer=writer,
                                best_epoch=nb_epoch,
                                best_lr=learning_rate,
                                tensorboard_output_name=tensorboard_output_name)


    if (validation_data_loader != None):
        report_tensorboard(tensorboard_output_name, f"Best model chosen at epoch {best_model['epoch']}, with accuracy of {best_model['accuracy_condition']} and AUC of {best_model[optimized_metric]}.", writer)

    # Save last layer
    m = ReproductedModel(dropout=dropout, input_dimension=INPUT_NB_CHANNEL) # get a new instance
    m.load_state_dict(best_model['dict'])
    f = open(f'{output_directory}/last_layer.pckl', 'wb')
    pickle.dump(m.last_layer, f)
    f.close()

    return metrics_per_epoch_training


def test_model(model, test_data_loader, device, loss_function, writer=None, plot_name=''):
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
            Loss: {metrics_test['loss']}
            Accuracy: {metrics_test['accuracy']}
            Precision:{metrics_test['precision']}
            Recall:   {metrics_test['recall']}
            F1 score: {metrics_test['f1score']}
            Specificity: {metrics_test['specificity']}
            AUC: {metrics_test['auc']}
            --------------------
            Running time: {end-start}

            """

            print(summary_text)

            report_tensorboard(plot_name, summary_text, writer)

        return metrics_test

###### END TRAINING AND TESTING FUNCTIONS ######

##### MAIN TRAINING FUNCTIONS

def main_train_model(models_dict, tensorboard_dict, hyperparameters_dict, paths_dict, fixed_parameters_dict, batchsize, save_intermediate=''):
    """
    Description:
        - This function trains the models_dict given with the given hyperparameters
    Params:
        - models_dict:  {'model_to_load': ...., 'model_to_output': ....'} dict that contains the path to the .pth file storing the parameters of the model to load and the output name to save the trained model.
          If given, load this model, otherwise initialise one.
        - tensorboard_dict: {'directory': .... , 'output_name': ....} dict that contains the directory in which tensorboard outputs have to be written and their output names
        - hyperparameters_dict: {'epochs': ..., 'lrs': ....} dict that contains the number of epochs and the learning rate needed for training
        - paths_dict: {'training': ..., 'validation':...} dict that contains the path of the training and validation sets
        - fixed_parameters_dict: fixed_parameters_dict={'loss_function': ..., 'cuda_device': ..., 'dropout': ..., 'optimized_metric': ..., 'save_last_layer':...} dict containing parameters needed for the training:
            - loss function: PyTorch loss function used to compute the loss based on the neural network output
            - cuda_device: device used to run the model, i.e 'cuda' or 'cpu'
            - dropout: dropout rate (between 0 and 1). Corresponds to the probability of turning off each neuron during the training. This has a regularization effect, consequently, it avoids overfitting
            - optimized_metric: choose the best intermediate model to keep with respect to this metric: auc, accuracy, precision, recall, f1-score, specificity
            - save_last_layer: boolean that tells if the last layer has to be saved. In our transfer learning pipeline, the last layer of the model trained on the first dataset only is saved.
        - batchsize: number of training samples per batch
        - save_intermediate: string path that is used to save the best intermediate model during training. If not given, intermediate model is not saved
    Returns:
        - No return value
    """
    # Initialize Tensorboard
    writer = SummaryWriter(tensorboard_dict['directory'])

    # Get available device
    device = fixed_parameters_dict['cuda_device'] if torch.cuda.is_available() else 'cpu'

    print("> Before Data loading")

    # Load all data as training data
    train_data_loader, val_data_loader = load_data(train_data_path=paths_dict['training'],
                                                    validation_data_path=paths_dict['validation'],
                                                    batch_size=batchsize)

    print("> Data loaded")


    # Creation of the baseline model
    print(f'Input channels: {INPUT_NB_CHANNEL}')
    model = ReproductedModel(dropout=fixed_parameters_dict['dropout'], input_dimension=INPUT_NB_CHANNEL)

    # Initialize the model with another one
    if (models_dict['model_to_load'] != ''):
        # check that the model to load has the same number dimension input as the one given in args
        input_dimension_model_to_load = torch.load(models_dict['model_to_load'])['features.0.weight'].size()[1]
        if(INPUT_NB_CHANNEL != input_dimension_model_to_load ):
            print(f"The given number of channel ({INPUT_NB_CHANNEL}) does not match the number of input channel of the model to load ({input_dimension_model_to_load})")
            exit(0)
        else:
            model.load_state_dict(torch.load(models_dict['model_to_load']))

    if fixed_parameters_dict['freeze']:
        if fixed_parameters_dict['attach_DM']:
            # Instead of initializing it randomly, attach decision maker resulting from DS1/Full
            print(">> Attaching decision maker")
            f = open(f"{paths_dict['last_layer']}", 'rb')
            last_l = pickle.load(f)
            f.close()
            model.last_layer = last_l
            print(">> Done")

        else:
            model.last_layer = nn.Sequential(
                nn.Linear(256, 64),
                nn.ELU(),

                nn.Dropout(p=fixed_parameters_dict['dropout']),

                nn.Linear(64, 16),
                nn.ELU(),

                nn.Linear(16, 2),
                nn.Softmax(1)
            )

            print("> Last layer reset")

        params = list()
        for i in range(7):
            params += list(model.last_layer[i].parameters())

    else:
        params = model.parameters()

    print("> Model created")

    max_hp_lr = float(hyperparameters_dict['lrs'])
    max_hp_epoch = int(hyperparameters_dict['epochs'])

    # Train
    print(f">>> Training {models_dict['model_to_output']} with hyperparameters NbEpoch{max_hp_epoch}/LearningRate{max_hp_lr}")
    metrics_per_epoch_training = train_model(model=model,
                                            train_data_loader=train_data_loader,
                                            learning_rate=max_hp_lr,
                                            dropout=hyperparameters_dict['dropout'],
                                            nb_epoch=max_hp_epoch,
                                            loss_function=fixed_parameters_dict['loss_function'],
                                            optimizer=optim.Adam(params, lr=max_hp_lr),
                                            device=device,
                                            writer=writer,
                                            tensorboard_output_name=tensorboard_dict['output_name'],
                                            output_directory=paths_dict['output'],
                                            optimized_metric=fixed_parameters_dict['optimized_metric'],
                                            save_intermediate=save_intermediate,
                                            validation_data_loader=val_data_loader,
                                            test_reference=fixed_parameters_dict['test_reference'],
                                            last_layer_path=paths_dict['last_layer'],
                                            freeze=fixed_parameters_dict['freeze'])

    # Plot training values
    plot_tensorboard(plot_name=f"{tensorboard_dict['output_name']}/NbEpoch{max_hp_epoch}/LearningRate{max_hp_lr}/Training", nb_epochs=max_hp_epoch, dict_metrics=metrics_per_epoch_training, writer=writer)
##### END MAIN TRAINING FUNCTIONS
if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script is a reproduction of the model used in article Computer-Aided Diagnosis of Prostate Cancer Using a Deep Convolutional Neural Network From Multiparametric MRI')

    parser.add_argument('--trainingset',
                        help='training folder path',
                        required=True,
                        type=str)

    parser.add_argument('--validationset',
                        help='validation folder path',
                        required=False,
                        default=None,
                        type=str)

    parser.add_argument('--reference_training',
                        help='reference dataset training folder path',
                        required=False,
                        type=str,
                        default='no_path')

    parser.add_argument('--reference_validation',
                        help='reference dataset validation folder path',
                        required=False,
                        type=str,
                        default='no_path')

    parser.add_argument('--last_layer',
                        help='path to the an existing last layer (global performance)',
                        required=False,
                        type=str,
                        default='no_path')

    parser.add_argument('--batchsize',
                        help='batch size',
                        required=True,
                        type=int)

    parser.add_argument('--nbepochs',
                        help='number of epochs',
                        required=True,
                        type=int)

    parser.add_argument('--lr',
                        help='learning rate',
                        required=False,
                        type=float,
                        default=0.001)

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
                        help='pth file to load as initial model (has to be an instance of ReproductedModel)',
                        required=False,
                        type=str,
                        default='')

    parser.add_argument('--dropout',
                        help='dropout probability',
                        required=False,
                        type=float,
                        default=0.3)

    parser.add_argument('--inputchannel',
                        help='number of channel of the input',
                        required=True,
                        type=int,
                        choices=[1,3])

    parser.add_argument('--optimizedmetric',
                        help="intermediate model with the best --optimized_metric will be saved, choices=['auc', 'accuracy', 'precision', 'recall', 'f1score', 'specificity']",
                        required=False,
                        type=str,
                        choices=['auc', 'accuracy', 'precision', 'recall', 'f1score', 'specificity'],
                        default='auc')

    parser.add_argument('--freeze',
                        help="True to freeze the feature extractor, False otherwise",
                        required=False,
                        type=str,
                        choices=['True', 'False'],
                        default='False')

    parser.add_argument('--attach_DM',
                        help="True to attach a previously trained decision maker, False otherwise",
                        required=False,
                        type=str,
                        choices=['True', 'False'],
                        default='False')

    parser.add_argument('--tensorboard_output_name',
                        help='output name for Tensorboard',
                        required=True,
                        type=str,
                        choices=['DS1/Full', 'DS2/Frozen', 'DS2/Full', 'DS3/Frozen', 'DS3/Full', 'DS4/Frozen', 'DS4/Full'])

    parser.add_argument('--outputdirectory',
                        help='root of the output directory used by Tensorboard and to save the models',
                        required=True,
                        type=str,
                        default='./runs')

    args = parser.parse_args()

    # Save the stdout output
    BruteLogger.save_stdout_to_file(path=f"{args.outputdirectory}/stdout_logs")

    # Loss functions
    loss_functions_dict = {
    'L1Loss': nn.L1Loss(),
    'MSELoss': nn.MSELoss(),
    'CrossEntropyLoss': nn.CrossEntropyLoss()
    }

    # global_performance_dict_validation = {loss: {dataset1_frozen: [], dataset1_full: [], dataset2_frozen: [], ...}, ...}
    global_performance_dict_validation = {}
    global_performance_dict_training = {}

    test_reference = False

    if args.reference_training != 'no_path' and args.reference_validation != 'no_path':
        dataset_of_reference_training, dataset_of_reference_val = load_data(train_data_path = args.reference_training,
                                                    validation_data_path= args.reference_validation,
                                                    batch_size=args.batchsize)
        print(f'>> Reference dataset loaded')
        test_reference = True

    if args.last_layer == 'no_path':
        print(">>> You did not specify any path to a last layer (global performance).\n    If it is not the first step of the transfer learning, please specify one!")

    INPUT_NB_CHANNEL = int(args.inputchannel)

    # Train the model
    main_train_model(
    models_dict={'model_to_load': args.modeltoload, 'model_to_output': f'{args.outputdirectory}/trainedBaseline.pth'},
    tensorboard_dict={'directory': f"{args.outputdirectory}/{args.tensorboard_output_name.split('/')[0]}", 'output_name': args.tensorboard_output_name},
    hyperparameters_dict={'epochs': args.nbepochs, 'lrs': args.lr, 'dropout': args.dropout},
    paths_dict={'training': args.trainingset, 'validation': args.validationset, 'output': args.outputdirectory, 'last_layer': args.last_layer},
    fixed_parameters_dict={'loss_function': loss_functions_dict[args.lossfunction],
                            'cuda_device': args.cudadevice,
                            'dropout': args.dropout,
                            'optimized_metric': args.optimizedmetric,
                            'test_reference': test_reference,
                            'freeze': True if args.freeze=='True' else False,
                            'attach_DM': True if args.attach_DM=='True' else False
                            },
    batchsize=args.batchsize,
    save_intermediate=f'{args.outputdirectory}/trainedBaselineInter')

    f = open(f'{args.outputdirectory}/global_performance_dict_training.pckl', 'wb')
    print(global_performance_dict_training)
    pickle.dump(global_performance_dict_training, f)
    f.close()

    f = open(f'{args.outputdirectory}/global_performance_dict_validation.pckl', 'wb')
    print(global_performance_dict_validation)
    pickle.dump(global_performance_dict_validation, f)
    f.close()
