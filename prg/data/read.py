import numpy as np
import sys

def read_input(input_file):
    """
    Reads filename from arguments else asks user for input if no filename was given.
    So far only works with numpy specific file formats: .npy and .dat.

    Return:
        Numpy array containing data.
    """
    # Check file type, read file and return np array
    if input_file[-3:] == "npy":
        return np.load(f"./{input_file}")
    elif input_file[-3:] == "dat":
        # Assuming .dat files contain at each row a binary string, eg. '00..01100'
        return np.genfromtxt(f"./{input_file}", delimiter=1, dtype=np.int8)  
    else:
        print("Make sure input file has extensions .npy or .dat")
        exit()
