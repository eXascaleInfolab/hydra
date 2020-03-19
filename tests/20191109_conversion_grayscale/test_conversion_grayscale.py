#########################################
#                                       #
#  Julien clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def conversionGrayscale16To8bits():
    """
    Description: test the image conversion method from 16-bit images to 8-bit images
    Params:
        - No Params
    Returns:
        - No return value
    """
    # Create a 16-bit gray gradient
    gradient16bits = np.zeros((512,512), dtype='uint16')

    # Image dimensions
    width = height =  512

    # 16 bits grayscaled pixel values are inserted in the images
    for y in range(0, height):
        v = 65535 * y / height
        for x in range(0, width):
            gradient16bits[y][x] = round((v + 65535 * x / width) / 2)

    # Get the 16 bits PIL image
    img16 = Image.fromarray(gradient16bits)

    # Convert gradient to 8-bit gradient
    ratio = np.max(gradient16bits) / 256 ;
    gradient8bits = (gradient16bits / ratio).astype('uint8')
    img8 = Image.fromarray(gradient8bits)

    # Plot both gradients
    fig, (_, _) = plt.subplots(1, 2)
    fig.suptitle('16 bits to 8 bits conversion - Gray gradient')

    sp = plt.subplot(121)
    sp.title.set_text('Data type: {}'.format(gradient16bits.dtype))
    plt.imshow(img16, cmap='gray', vmin=0, vmax=65635)

    sp = plt.subplot(122)
    sp.title.set_text('Data type: {}'.format(gradient8bits.dtype))
    plt.imshow(img8, cmap='gray', vmin=0, vmax=255)

    plt.show()

if __name__ == "__main__":

    conversionGrayscale16To8bits()
