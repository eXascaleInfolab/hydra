#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################
import importlib
import normalizeDicom as n
import pydicom
from PIL import Image

def dicomToPng(dicomPath, outputName, outputFolder='./', outputInstanceNumber=False, flip=False):
    """
    Description: Converts a dicom file into a .png file
    Params:
        - dicomPath path to the dicom file
        - outputName: output name
        - outputFolder: output folder
    Returns:
        - no return value
    """
    # Load DICOM file
    ds=pydicom.read_file(dicomPath,force=True)

    # Bit representation
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    # Grayscale + normalization
    image = n.get_PIL_image(ds, flip)

    # InstanceNumber given -> add it to filename in order to sort files in the correct order
    if outputInstanceNumber:
        # Save as png
        image.save("{}Instance{}_{}.png".format(outputFolder, ds.InstanceNumber, outputName))
    else:
        # Save as png
        image.save("{}/{}.png".format(outputFolder, outputName))
