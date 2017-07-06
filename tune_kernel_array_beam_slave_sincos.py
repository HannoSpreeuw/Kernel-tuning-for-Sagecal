#!/usr/bin/env python
import numpy as np
import kernel_tuner
from collections import OrderedDict

def power_bit_length(x):
    return 2**(x-1).bit_length()

def tune():

    with open('predict_model_snippet.cu', 'r') as f:
        kernel_string = f.read()

    # problem_size = (4096, 2048)
    Nelem = 500
    # N: no of stations
    N     =  61
    K     = 1e4
    T     =   1
    F     =   1

    problem_size = 100000

    ThreadsPerBlock = 8

    grid_size = int(2 * (problem_size + ThreadsPerBlock - 1)/ThreadsPerBlock)

    # size = numpy.prod(problem_size)

    r1, r2, r3 = np.random.rand(3).astype(np.float32)
    x, y, z    = np.random.rand(3, Nelem).astype(np.float32)
    tar        = np.empty(2).astype(np.float32)
    blockDim_2 = power_bit_length(Nelem)

    args = [np.int32(Nelem), r1, r2, r3, x, y, z, tar, np.int32(blockDim_2)]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range (5,11)]
    tune_params["use_cub"] = [0, 1]

    grid_div_x = []

    return kernel_tuner.tune_kernel("kernel_array_beam_slave_sincos", kernel_string, problem_size,
        args, tune_params, grid_div_x=grid_div_x,
        verbose = True)


if __name__ == "__main__":
    print(tune())
