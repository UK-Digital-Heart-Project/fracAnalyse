import numpy as np

def normalize01(f):
    """
    Normalize the input array to the range [0,1]

    Parameters:
    f (numpy array): Input array

    Returns:
    numpy array: Normalized array
    """
    fmin = np.min(f)
    fmax = np.max(f)
    normalized_f = (f - fmin) / (fmax - fmin)
    return normalized_f
