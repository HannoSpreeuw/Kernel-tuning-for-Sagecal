#!/usr/bin/env python
import numpy as np
import kernel_tuner
from collections import OrderedDict

def power_bit_length(x):
    return 2**(int(x-1).bit_length())

def get_input_data(Nelem):
    # r1, r2, r3 = np.random.rand(3).astype(np.float32)
    r1, r2, r3 = np.linspace(0.1,1,3, endpoint=False).astype(np.float32)
    # x, y, z    = np.random.rand(3, Nelem).astype(np.float32)
    x, y, z = np.linspace(0.1,1,3*Nelem, endpoint=False).astype(np.float32).reshape(3, Nelem)
    tar        = np.empty(2).astype(np.float32)
    return np.int32(Nelem), r1, r2, r3, x, y, z, tar


def call_reference_kernel(Nelem, r1, r2, r3, x, y, z, tar):
    with open('predict_model_snippet.cu', 'r') as f:
        kernel_string = f.read()
    blockDim_2 = np.int32(power_bit_length(Nelem))
    args = [np.int32(Nelem), r1, r2, r3, x, y, z, tar, blockDim_2]
    params = {"block_size_x": int(Nelem)}
    reference = kernel_tuner.run_kernel("kernel_array_beam_slave_sincos_original", kernel_string, 1,
        args, params, grid_div_x=[])
    return reference[7]


def test_manual_kernel():
    with open('predict_model_snippet.cu', 'r') as f:
        kernel_string = f.read()
    Nelem = 500
    args = get_input_data(Nelem)
    reference = call_reference_kernel(*args)

    params = {"block_size_x": 32}
    answer = kernel_tuner.run_kernel("sincos_manual", kernel_string, 1,
        args, params, grid_div_x=[])
    answer = answer[7]

    assert np.allclose(reference, answer, atol=1e-6)


def test_cub_kernel():
    with open('predict_model_snippet.cu', 'r') as f:
        kernel_string = f.read()
    Nelem = 500
    args = get_input_data(Nelem)
    reference = call_reference_kernel(*args)

    params = {"block_size_x": 192}
    answer = kernel_tuner.run_kernel("sincos_cub", kernel_string, 1,
        args, params, grid_div_x=[])
    answer = answer[7]

    assert np.allclose(reference, answer, atol=1e-6)


def tune_kernels():

    with open('predict_model_snippet.cu', 'r') as f:
        kernel_string = f.read()

    # problem_size = (4096, 2048)
    Nelem = 500
    args = get_input_data(Nelem)
    # N: no of stations
    N     =  61
    K     = 1e4
    T     =   1
    F     =   1

    # Tell the Kernel Tuner how the grid dimensions are to be computed
    problem_size = 1
    grid_div_x = []

    # Compute reference answer using the original kernel
    reference = call_reference_kernel(*args)
    answer = [None, None, None, None, None, None, None, reference]

    # Tune the kernel with the manual reduction loop
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range (5,11)]
    manual_kernel, _ = kernel_tuner.tune_kernel("sincos_manual", kernel_string, problem_size,
        args, tune_params, grid_div_x=grid_div_x, verbose=True, answer=answer)

    # Tune the kernel that uses CUB for reductions
    tune_params["block_size_x"] = [32*i for i in range (1,33)]
    cub_kernel, _ = kernel_tuner.tune_kernel("sincos_cub", kernel_string, problem_size,
        args, tune_params, grid_div_x=grid_div_x, verbose=True, answer=answer)

    return manual_kernel, cub_kernel

if __name__ == "__main__":
    test_manual_kernel()
    test_cub_kernel()
    tune_kernels()
