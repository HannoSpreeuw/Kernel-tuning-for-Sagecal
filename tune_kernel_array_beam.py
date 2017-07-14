#!/usr/bin/env python

from collections import OrderedDict
import os

import numpy as np
from kernel_tuner import tune_kernel, run_kernel

def get_kernel_path():
    """ get path to the kernels as a string """
    return str(os.path.dirname(os.path.realpath(__file__)))+'/'


def generate_input_data(N, T, K, F):
    """ Generate random input data for calling kernel_array_beam
    N: no of stations
    T: no of time slots
    K: no of sources
    F: no of frequencies
    freqs: frequencies Fx1
    longitude, latitude: Nx1 station locations
    time_utc: Tx1 time
    Nelem: Nx1 array of no. of elements
    x,y,z: N*Nelemsx1 array of station locations
    ra,dec: Kx1 source positions
    beam: output beam values NxTxKxF values
    ph_ra0,ph_dec0: beam pointing direction
    ph_freq0: beam referene freq
    """

    freqs = np.random.randn(F).astype(np.float32)
    longitude = np.random.randn(N).astype(np.float32)
    latitude = np.random.randn(N).astype(np.float32)
    time_utc = np.random.randn(T).astype(np.float64)
    Nelem = (500+5.*np.random.randn(N)).astype(np.int32)

    print(Nelem)
    TotalElem = np.sum(Nelem)
    print(TotalElem)

    x, y, z = np.random.randn(3, TotalElem).astype(np.float32)
    ra, dec = np.random.randn(2, K).astype(np.float32)
    beam = np.zeros(N*T*K*F).astype(np.float32)
    ph_ra0, ph_dec0, ph_freq0 = np.random.randn(3).astype(np.float32)

    return (np.int32(N), np.int32(T), np.int32(K), np.int32(F), freqs, longitude, latitude,
           time_utc, Nelem, x, y, z, ra, dec, ph_ra0, ph_dec0, ph_freq0, beam, np.int32(TotalElem))

def run():
    N = 61
    T = 200
    K = 150
    F = 10

    problem_size = N*T*K*F
    cp = ["-rdc=true", "-arch=sm_52", "-I"+get_kernel_path(), "-maxrregcount=32"]

    args = generate_input_data(N, T, K, F)

    problem_size = N*T*K*F

    params = {"block_size_x": 128, "use_kernel": 0}
    run_kernel("kernel_tuner_host_array_beam", [get_kernel_path()+"predict_model.cu"], problem_size, args, params,
                lang="C", compiler_options=cp)


def call_reference_kernel(N, T, K, F, args, cp):

    problem_size = N*T*K*F

    params = {"block_size_x": 32, "use_kernel": 1}
    answer = run_kernel("kernel_tuner_host_array_beam", [get_kernel_path()+"predict_model.cu"], problem_size, args, params,
                lang="C", compiler_options=cp)
    ref = [None for _ in answer]
    ref[17] = answer[17]
    return ref


def tune():

    N = 61
    T = 200
    K = 150
    F = 10

    problem_size = N*T*K*F
    cp = ["-rdc=true", "-arch=sm_52", "-I"+get_kernel_path(), "-maxrregcount=32"]

    args = generate_input_data(N, T, K, F)

    ref = call_reference_kernel(N, T, K, F, args, cp)

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,11)]
    tune_params["use_kernel"] = [0]

    #restrict = ["use_kernel == 0 or block_size_x<=64"]
    results, env = tune_kernel("kernel_tuner_host_array_beam", [get_kernel_path()+"predict_model.cu"], problem_size, args, tune_params,
                lang="C", compiler_options=cp, verbose=True, answer=ref)

    return results



if __name__ == "__main__":

    tune()
    #run() #useful for running this script using NVVP
