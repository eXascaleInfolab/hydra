#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import pydicom
from glob import glob
import os
from pathlib import Path
import argparse

def get_corresponding_slice(dicom_path, instance_number):
    """
    Description: this function returns the slice corresponding to the given instance number for the given DICOM folder
    Params:
        - dicom_path: DICOM folder path
        - instance_number: instance number
    Returns:
        - No return value
    """
    # Create a path object (cross-platform Mac/Windows)
    path = Path(dicom_path)

    print(f"\nLooking for the instance_number {instance_number} in {dicom_path}...\n\n")

    # Look for the slice
    for root, _, files in os.walk(path):
        for file in files:
            ds=pydicom.read_file(os.path.join(root, file), force=True)

            # if the instance number of the current slice corresponds to the instance_number desired
            if ds.InstanceNumber == instance_number:
                print(f">> RESULT: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Get the slice corresponding to the given instance number for the given DICOM folder')

    parser.add_argument('--dicom-path',
                        help='DICOM folder path',
                        required=True,
                        type=str
                        )

    parser.add_argument('--instance-number',
                        help='instance number',
                        required=True,
                        type=int)

    args = parser.parse_args()

    get_corresponding_slice(args.dicom_path, args.instance_number)
