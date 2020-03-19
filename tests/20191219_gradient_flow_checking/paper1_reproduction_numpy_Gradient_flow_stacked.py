
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
from matplotlib.lines import Line2D
from pathlib import Path

###### UTIL ######
def plot_grad_flow(named_parameters, index, epoch, output_directory):
    """
    Description: Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        (Adapted from RoshanRane https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7)
    Params:
        - named_parameters: name of each parameter of the model (used for x-axis labels)
        - index: ith batch
        - epoch: ith epoch
    Returns:
        - No return value
    """
    # Create path
    output_path = Path(output_directory) / "Gradients_flow"
    if (not output_path.exists()):
        output_path.mkdir(parents=True)

    ave_grads = []
    max_grads= []
    layers = []
    counter = 0
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            counter += 1
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(f"{output_path}/epoch_{epoch}_batch{index}.png")
    plt.close()

def new_loader(path):
    """
    Description:
        - Load an image, convert it to a tensor with one single channel and send the latter to cuda/cpu
    Params:
        - path: takes an image path
    Returns:
        - A tensor with one channel instead of 3 (default loader automatically converts to RGB)
    """
    # Load the data into a numpy array
    np_array = np.load(path)

    # # Add an extra dimension to the tensor
    # wrap = np.array([np_array])

    # Send the tensor to the GPU/CPU depending on what device is available
    tensor = torch.from_numpy(np_array).to('cuda' if torch.cuda.is_available() else 'cpu')

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
    # Create figure
    fig = plt.figure()

    # Plot the training and validation values
    plt.plot(training_metric_per_epoch)
    plt.plot(val_metric_per_epoch)

    # plot training loss before and after backpropagation
    plt.legend([f'Training {metric_name}', f'Validation {metric_name}'], loc='upper left')

    # Add legend, title and axes name
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
    """
    Description:
        - Makes a written report of the given text on Tensorboard
    Params:
        - plot_name: name of the tensorboard plot
        - text: text that will be added on Tensorboard
        - writer: tensorboard writer object
    Returns:
        - No return value
    """
    writer.add_text(plot_name, text)
###### END UTIL ######

###### MODEL ######
class Flatten(nn.Module):
    """Flattens the output of the previous layer"""
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ReproductedModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super(ReproductedModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1),
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

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.last_layer(x)
        return x
###### END MODEL ######

###### TRAINING AND TESTING FUNCTIONS ######
def train_model(model, train_data_loader, learning_rate, nb_epoch, loss_function, optimizer, device, writer, tensorboard_output_name, optimized_metric, validation_data_loader=None, save_intermediate=''):
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
    # Send model to GPU if available
    model.to(device)

    # Metrics per epoch
    metrics_per_epoch_training = {}

    if validation_data_loader != None:
        metrics_per_epoch_val = {}

    losses_before_backpropagation_per_epoch = []

    # Best model
    best_model = {optimized_metric: 0, 'dict': None, 'epoch': -1, 'accuracy_condition': 0}

    for epoch in range(nb_epoch):
        # {'TP': [TP_batch_0, TP_batch_1, ...], 'TN': [], 'FP': [], 'FN': []}
        confusion_matrix_dict = {}

        # list of predictions to compute AUC (float)
        predictions_per_batch = {'labels': [], 'predictions_positive_class': []}

        # model.train() tells your model that you are training the model.
        # So effectively layers like dropout, batchnorm etc. which behave
        # differently on the train and test procedures.
        model.train()

        # For each batch
        for index, (inputs, labels) in enumerate(train_data_loader):

            # Send data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Feed the model and get output
            output = model(inputs)

            # Compute loss
            loss = loss_function(output, labels)

            # Backprop
            # computes dloss/dx for every parameter x which has requires_grad=True. These are accumulated into x.grad for every parameter x
            loss.backward()

            # Plot the gradient value for each batch at each epoch
            plot_grad_flow(model.named_parameters(), index, epoch, args.outputdirectory)

            # Optimizer.step is performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) and the update rule.
            optimizer.step()

            # Every time a variable is back propagated, the gradient will be accumulated instead of being replaced.
            # (This makes it easier for rnn, because each module will be back propogated through several times.)
            # zero_grad() sets all gradients to zero
            optimizer.zero_grad()


        # End of an epoch:
        metrics_epoch_training = test_model(model=model, test_data_loader=train_data_loader, device=device, loss_function=loss_function)

        # Add all metrics for this epoch to the corresponding list
        for key, value in metrics_epoch_training.items():
            metrics_per_epoch_training.setdefault(key, []).append(value)

        # if we want to see the performance of the current model (i.e at each epoch) on the validation set
        if(validation_data_loader != None):
            metrics_epoch_val = test_model(model=model, test_data_loader=validation_data_loader, device=device, loss_function=loss_function)

            # Add all validation metrics for this epoch to the corresponding list
            for key, value in metrics_epoch_val.items():
                metrics_per_epoch_val.setdefault(key, []).append(value)

        print(f"> Epoch {epoch}")
        print(f">>Train: {metrics_epoch_training}")
        print(f">>Val: {metrics_epoch_val}")

        if (save_intermediate != '' and metrics_epoch_val[optimized_metric] >= best_model[optimized_metric] and metrics_epoch_val['accuracy'] >= best_model['accuracy_condition'] ):
            print(f">> New best {optimized_metric} found. Previous {best_model[optimized_metric]}, current {metrics_epoch_val[optimized_metric]}")
            best_model[optimized_metric] = metrics_epoch_val[optimized_metric]
            best_model['accuracy_condition'] = metrics_epoch_val['accuracy']
            best_model['dict'] = model.state_dict()
            best_model['epoch'] = epoch
            torch.save(best_model['dict'], f'{save_intermediate}.pth')
            print(f"Intermediate best model {save_intermediate}.pth saved. Epoch {best_model['epoch']}, {optimized_metric} {best_model[optimized_metric]}")

    # Plot the different metrics for training and validation
    if validation_data_loader != None:
        for key, value in metrics_per_epoch_training.items():
            plot_val_and_train(training_metric_per_epoch=value,
                                val_metric_per_epoch=metrics_per_epoch_val[key],
                                metric_name=key,
                                writer=writer,
                                best_epoch=nb_epoch,
                                best_lr=learning_rate,
                                tensorboard_output_name=tensorboard_output_name)

    report_tensorboard(tensorboard_output_name, f"Best model chosen at epoch {best_model['epoch']}, with accuracy of {best_model['accuracy_condition']} and AUC of {best_model[optimized_metric]}.", writer)

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
def main_train_model(models_dict, tensorboard_dict, hyperparameters_dict, paths_dict, fixed_parameters_dict, batchsize, save_intermediate='', validation_loss_display=False):
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

    # Creation of the baseline model (one single model for all CV + hyperparameters tuning to have the same random initialization each time)
    model = ReproductedModel(dropout=fixed_parameters_dict['dropout'])

    # If the initialization is not random
    if (models_dict['model_to_load'] != ''):
        model.load_state_dict(torch.load(models_dict['model_to_load']))

    params = model.parameters()

    print("> Model created")

    max_hp_lr = float(hyperparameters_dict['lrs'])
    max_hp_epoch = int(hyperparameters_dict['epochs'])

    # Load all data as training data
    train_data_loader, val_data_loader = load_data(train_data_path=paths_dict['training'],
                                                    validation_data_path=paths_dict['validation'],
                                                    batch_size=batchsize)
    print("> Data loaded")

    # Train
    print(f">>> Training {models_dict['model_to_output']} with hyperparameters NbEpoch{max_hp_epoch}/LearningRate{max_hp_lr}")
    metrics_per_epoch_training = train_model(model=model,
                                            train_data_loader=train_data_loader,
                                            learning_rate=max_hp_lr,
                                            nb_epoch=max_hp_epoch,
                                            loss_function=fixed_parameters_dict['loss_function'],
                                            optimizer=optim.Adam(params, lr=max_hp_lr),
                                            device=device,
                                            writer=writer,
                                            tensorboard_output_name=tensorboard_dict['output_name'],
                                            optimized_metric=fixed_parameters_dict['optimized_metric'],
                                            save_intermediate=save_intermediate,
                                            validation_data_loader=val_data_loader)

    # plot training values
    plot_tensorboard(plot_name=f"{tensorboard_dict['output_name']}/NbEpoch{max_hp_epoch}/LearningRate{max_hp_lr}/Training", nb_epochs=max_hp_epoch, dict_metrics=metrics_per_epoch_training, writer=writer)
##### END MAIN TRAINING FUNCTIONS
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='This script is a reproduction of the model used in article Computer-Aided Diagnosis of Prostate Cancer Using a Deep Convolutional Neural Network From Multiparametric MRI')

    parser.add_argument('--trainingSet1',
                        help='training dataset 1 training folder path',
                        required=True,
                        type=str)

    parser.add_argument('--validationSet1',
                        help='validation set 1 folder path',
                        required=True,
                        type=str)

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

    parser.add_argument('--cuda-device',
                        help='cuda device',
                        required=False,
                        type=str,
                        default='cuda')

    parser.add_argument('--modeltoload',
                        help='pth file to load as initial model',
                        required=False,
                        type=str,
                        default='')

    parser.add_argument('--dropout',
                        help='dropout probability',
                        required=False,
                        type=float,
                        default=0.3)

    parser.add_argument('--optimized-metric',
                        help="intermediate model with the best --optimized_metric will be saved, choices=['auc', 'accuracy', 'precision', 'recall', 'f1score', 'specificity']",
                        required=False,
                        type=str,
                        choices=['auc', 'accuracy', 'precision', 'recall', 'f1score', 'specificity'],
                        default='auc')

    parser.add_argument('--outputdirectory',
                        help='root of the output directory used by Tensorboard and to save the models',
                        required=True,
                        type=str,
                        default='./runs')

    args = parser.parse_args()

    # Loss functions
    loss_functions_dict = {
    'L1Loss': nn.L1Loss(),
    'MSELoss': nn.MSELoss(),
    'CrossEntropyLoss': nn.CrossEntropyLoss()
    }

    # Train the model
    main_train_model(
    models_dict={'model_to_load': args.modeltoload, 'model_to_output': f'{args.outputdirectory}/trainedBaseline.pth'},
    tensorboard_dict={'directory': f'{args.outputdirectory}/DS1', 'output_name': 'DS1/Full'},
    hyperparameters_dict={'epochs': args.nbepochs, 'lrs': args.lr},
    paths_dict={'training': args.trainingSet1, 'validation': args.validationSet1},
    fixed_parameters_dict={'loss_function': loss_functions_dict[args.lossfunction], 'cuda_device': args.cuda_device, 'dropout': args.dropout, 'optimized_metric': args.optimized_metric},
    batchsize=args.batchsize,
    save_intermediate=f'{args.outputdirectory}/trainedBaselineInter',
    validation_loss_display=True)
