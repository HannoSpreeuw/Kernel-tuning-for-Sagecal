#!/usr/bin/env python

from collections import OrderedDict
import os

import numpy as np
from kernel_tuner import tune_kernel, run_kernel

def get_kernel_path():
    """ get path to the kernels as a string """
    return str(os.path.dirname(os.path.realpath(__file__)))+'/'

cp = ["-rdc=true", "-arch=sm_52", "-I"+get_kernel_path()] #, "-maxrregcount=32"] #, "-Xptxas=-v"]


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

    elem_per_station = 500
    max_elem = 512
    Nelem = (elem_per_station+5.*np.random.randn(N)).astype(np.int32)
    Nelem = np.array([i if i <= max_elem else max_elem for i in Nelem]).astype(np.int32)
    TotalElem = np.sum(Nelem)

    #it's important that TotalElem is always the same to get our
    #measurements right across multiple runs of this script
    target_total = N*500
    while TotalElem < N*elem_per_station:
        too_few = min(N*elem_per_station - TotalElem, N)
        for i in range(too_few):
            if Nelem[i] < max_elem:
                Nelem[i] = Nelem[i]+1
        TotalElem = np.sum(Nelem)
    while TotalElem > N*elem_per_station:
        too_many = min(TotalElem - N*elem_per_station, N)
        for i in range(too_many):
            Nelem[i] = Nelem[i]-1
        TotalElem = np.sum(Nelem)
    print(Nelem)
    print(TotalElem)
    assert TotalElem == N*elem_per_station


    x, y, z = (1e7 * np.random.randn(3, TotalElem)).astype(np.float32)
    ra, dec = (2 * np.pi * np.random.randn(2, K)).astype(np.float32)
    beam = (1e6 * np.random.randn(N*T*K*F)).astype(np.float32)
    ph_ra0, ph_dec0, ph_freq0 = (2 * np.pi * np.random.randn(3)).astype(np.float32)

    return (np.int32(N), np.int32(T), np.int32(K), np.int32(F), freqs, longitude, latitude,
           time_utc, Nelem, x, y, z, ra, dec, ph_ra0, ph_dec0, ph_freq0, beam, np.int32(TotalElem))

def run():
    N = 61
    T = 200
    K = 150
    F = 10

    args = generate_input_data(N, T, K, F)

    problem_size = (T*K*F, N)

    #ref = call_reference_kernel(N, T, K, F, args, cp)

    params = {"block_size_x": 256, "use_kernel": 0, "use_shared_mem": 1}
    ans = run_kernel("kernel_tuner_host_array_beam", [get_kernel_path()+"predict_model.cu"], problem_size, args, params,
                lang="C", compiler_options=cp + ['-Xptxas=-v'])


    if False: #debugging
        print(ref[17][:20])
        print(ans[17][:20])

        ref = ref[17]
        ans = ans[17]

        refp = ref.reshape(T*K, N*F)
        ansp = ans.reshape(T*K, N*F)

        from matplotlib import pyplot
        pyplot.imshow(refp)
        pyplot.show()
        pyplot.imshow(ansp)
        pyplot.show()

        pyplot.imshow(refp-ansp)
        pyplot.show()

        err = ref-ans
        print(err[np.absolute(err)>1e-6])

        assert np.allclose(ref, ans, atol=1e-6)


def call_reference_kernel(N, T, K, F, args, cp):

    problem_size = (T*K*F, N)

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

    problem_size = (T*K*F, N)

    args = generate_input_data(N, T, K, F)

    ref = call_reference_kernel(N, T, K, F, args, cp)

    print(ref[17][:20])

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,11)]
    tune_params["use_kernel"] = [0]
    tune_params["use_shared_mem"] = [0, 1]

    #restrict = ["use_kernel == 0 or block_size_x<=64"]
    results, env = tune_kernel("kernel_tuner_host_array_beam", [get_kernel_path()+"predict_model.cu"], problem_size, args, tune_params,
                lang="C", compiler_options=cp, verbose=True, answer=ref)

    return results



if __name__ == "__main__":

    tune()
    #run() #useful for running this script using NVVP
