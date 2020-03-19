#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import numpy as np
from PIL import Image
import os
from pathlib import Path

# Input directory
training0 = Path("./mnist_png2/training/0")
training1 = Path("./mnist_png2/training/1")
testing0 = Path("./mnist_png2/testing/0")
testing1 = Path("./mnist_png2/testing/1")

# Output directory
training0Out = Path("./numpy_mnist/training/0")
training1Out = Path("./numpy_mnist/training/1")
testing0Out = Path("./numpy_mnist/testing/0")
testing1Out = Path("./numpy_mnist/testing/1")

# Get training 0
training0list = list()
for (dirpath, dirnames, filenames) in os.walk(training0):
    training0list += [os.path.join(training0, file) for file in filenames]

# Get training 1
training1list = list()
for (dirpath, dirnames, filenames) in os.walk(training1):
    training1list += [os.path.join(training1, file) for file in filenames]

# Get testing 0
testing0list = list()
for (dirpath, dirnames, filenames) in os.walk(testing0):
    testing0list += [os.path.join(testing0, file) for file in filenames]

# Get training 0
testing1list = list()
for (dirpath, dirnames, filenames) in os.walk(testing1):
    testing1list += [os.path.join(testing1, file) for file in filenames]

# Training 0
for image in training0list:
    im = Image.open(image)
    np_im = np.array(im)
    np.save(os.path.join(training0Out, Path(image).stem), np_im)

# Training 1
for image in training1list:
    im = Image.open(image)
    np_im = np.array(im)
    np.save(os.path.join(training1Out, Path(image).stem), np_im)

# Testing 1
for image in testing0list:
    im = Image.open(image)
    np_im = np.array(im)
    np.save(os.path.join(testing0Out, Path(image).stem), np_im)

# Testing 1
for image in testing1list:
    im = Image.open(image)
    np_im = np.array(im)
    np.save(os.path.join(testing1Out, Path(image).stem), np_im)
