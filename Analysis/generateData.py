import numpy as np

"""
Generate random data and save it as a .npy-file
"""

def generateData(size):
    """
    :param size: length of output array
    :return: numpy array with random numbers between 0 and 1, shape: (size, 2)
    """
    x = np.random.rand(size, 1)
    y = np.random.rand(size, 1)
    return np.c_[x, y]

if __name__ == '__main__':
    # simple example of how the code works
    # Generate data
    data = generateData(1000)
    np.save('data_for_part_1.npy', data)

    # Load generated data
    data1 = np.load('data_for_part_1.npy')
