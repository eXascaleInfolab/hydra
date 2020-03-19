#########################################
#                                       #
#  Julien Clément and Johan Jobin       #
#  University of Fribourg               #
#  2019                                 #
#  Master's thesis                      #
#                                       #
#########################################

import argparse

# Run the test with: cat config.txt | xargs -L 1 -P 3 python test.py
    # -L: indicate the number of lines that have to be given at most as input to the xargs_test.py
    # -P: run at most 3 processes at a time
parser = argparse.ArgumentParser(description='Just a test to understand xargs')
parser.add_argument('--input1', help='Input file name', required=True)
parser.add_argument('--input2', help='Input file name', required=True)
parser.add_argument('--input3', help='Input file name', required=True)
args = parser.parse_args()

print(args.input1)
print(args.input2)
print(args.input3)
print("End")
