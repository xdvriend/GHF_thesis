"""
Testing the transformation functions
====================================

Simple tests to check whether or not the functions return the correct value.
"""
from hf.utilities import transform as tr
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_tensor_transform():
    """
    Test whether tensor transformation happens correctly.
    """
    # Create a test tensor
    ss = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    ss[i, j, k, l] = 1

    # Create a transformation matrix
    t = np.array([[2, 3],
                  [3, 4]])

    # Set up the tensor transformation function
    x = tr.tensor_basis_transform(ss, t)

    # Set up the control
    control = np.zeros_like(ss)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if (i + j + k + l) == 0:
                        control[i, j, k, l] = 625.0
                    elif (i + j + k + l) == 1:
                        control[i, j, k, l] = 875.0
                    elif (i + j + k + l) == 2:
                        control[i, j, k, l] = 1225.0
                    elif (i + j + k + l) == 3:
                        control[i, j, k, l] = 1715.0
                    elif (i + j + k + l) == 4:
                        control[i, j, k, l] = 2401.0

    assert np.allclose(x, control)
