#!/usr/bin/env python

import itertools
from collections import OrderedDict
import os

import numpy as np
import pylab as pyl
from kernel_tuner import tune_kernel, run_kernel

def get_kernel_path():
    """ get path to the kernels as a string """
    return str(os.path.dirname(os.path.realpath(__file__)))+'/'

cp = ["-rdc=true", "-I"+get_kernel_path(), "-Xptxas=-v"] #, "-maxrregcount=32"] #, "-Xptxas=-v"]


def generate_input_data(B, N, T, K, F):
    """ generate random input to calculate coherencies
    B: total baselines
    N: no of stations
    T: no of time slots
    K: no of sources
    F: no of frequencies
    u,v,w: Bx1 uvw coords
    barr: Bx1 array of baseline/flag info
    freqs: Fx1 frequencies
    beam: NxTxKxF beam gain
    ll,mm,nn : Kx1 source coordinates
    sI: Kx1 source flux at reference freq
    stype: Kx1 source type info
    sI0: Kx1 original source referene flux
    f0: Kx1 source reference freq for calculating flux
    spec_idx,spec_idx1,spec_idx2: Kx1 spectra info
    exs: Kx1 array of pointers to extended source info
    deltaf,deltat: freq/time smearing integration interval
    dec0: phace reference dec
    coh: coherency Bx8 values, all K sources are added together
    dobeam: enable beam if >0
    """

    u,v,w = (np.random.randn(3, B)).astype(np.float32)

    #you would think that the baseline_t struct is 9 bytes, but the compiler pads it to 12
    #baseline = np.dtype([('int1', np.int32, 1), ('int2', np.int32, 1), ('flag', np.uint8, (1))])
    baseline = np.dtype([('int1', '<u4'), ('int2', '<u4'), ('flag', '<u4')])
    barr = np.zeros(B).astype(baseline)
    baselines = np.array(list(itertools.combinations(range(N), r=2)))
    barr['int1'] = list(baselines[:,0]) * T
    barr['int2'] = list(baselines[:,1]) * T
    barr['flag'] = (1.2*np.random.rand(B)).astype(np.uint32)

    eps = 1e-6 #used to avoid zero values

    freqs = eps+np.absolute(np.random.randn(F).astype(np.float32))
    beam = np.random.randn(N*K*T*F).astype(np.float32)

    #source info
    ll,mm,nn = (np.random.randn(3, K)).astype(np.float32)
    sI = eps+np.absolute(np.random.randn(K).astype(np.float32))
    sI0 = eps+np.absolute(np.random.randn(K).astype(np.float32))
    f0 = eps+np.absolute(np.random.randn(K).astype(np.float32))
    spec_idx,spec_idx1,spec_idx2 = eps+np.absolute(1e-5*np.random.randn(3, K).astype(np.float32))

    #for the moment assume only point sources
    stype = np.zeros(K).astype(np.uint8)
    #stype = (5.0*np.random.rand(K)).astype(np.uint8)
    exs = np.zeros(1).astype(np.int64) #this should be an array of pointers to exinfo_ structs

    deltaf, deltat = eps+np.absolute(np.random.randn(2).astype(np.float32))
    dec0 = np.random.randn(1).astype(np.float32)

    #the output
    coh = np.zeros(8*B*F).astype(np.float32)

    dobeam = np.int32(1)

    return (np.int32(B), np.int32(N), np.int32(T), np.int32(K), np.int32(F), u,v,w, barr, freqs, beam,
            ll,mm,nn, sI, stype, sI0, f0, spec_idx,spec_idx1,spec_idx2, exs, deltaf, deltat, dec0, coh, dobeam)



def call_reference_kernel(N, B, T, K, F, args):

    problem_size = B

    params = {'block_size_x': 32, "use_kernel": 1}
    answer = run_kernel("kernel_coherencies", get_kernel_path()+"predict_model.cu",
                               problem_size, args, params, compiler_options=cp)

    return answer


def tune(number_of_frequencies):

    N = 61
    T = 20 
    K = 150
    F = number_of_frequencies
    B = (N)*(N-1)//2 * T

    print('N', N, 'B', B, 'T', T, 'K', K, 'F', F)

    args = generate_input_data(B, N, T, K, F)

    problem_size = B

    tune_params = OrderedDict()
    tune_params['block_size_x'] = [2**i for i in range(5,10)]

    print("First call the reference kernel")
    ref = call_reference_kernel(N, B, T, K, F, args)
    answer = [None for _ in args]
    answer[-2] = ref[-2]

    tolerance = 1e-2
    verbosity = False
    print("Next, we call the modified kernel, with (use_kernel = 1) and without (use_kernel = 0) the slave kernel")
    print("With slave kernel:")
    # tune_kernel("kernel_coherencies", get_kernel_path()+"predict_model.cu",
    #                           problem_size, args, {'block_size_x': [32], 'use_kernel': [1]}, compiler_options=cp, verbose=True, answer=answer, atol=tolerance)

    tune_params['use_kernel'] = [1]
    results, env = tune_kernel("kernel_coherencies", get_kernel_path()+"predict_model.cu",
                               problem_size, args, tune_params, compiler_options=cp, verbose=verbosity, answer=answer, atol=tolerance)

    min_time_with_slave = min([item['time'] for item in results])

    print("Without slave kernel:")
    tune_params['use_kernel'] = [0]
    results, env = tune_kernel("kernel_coherencies", get_kernel_path()+"predict_model.cu",
                               problem_size, args, tune_params, compiler_options=cp, verbose=verbosity, answer=answer, atol=tolerance)

    min_time_without_slave = min([item['time'] for item in results])

    return min_time_with_slave/min_time_without_slave


if __name__ == "__main__":
    
    min_frequencies  = 1
    max_frequencies  = 100
    number_measurements = 10
    numbersoffrequencies = np.logspace(np.log10(min_frequencies), np.log10(max_frequencies), number_measurements, dtype=np.int32) 
    accelerations = np.empty(number_measurements, dtype=np.float32)

    for counter, number_of_frequencies in enumerate(numbersoffrequencies):
        accel = tune(number_of_frequencies) 
        print("Acceleration by abandoning the slave kernel = {0:.2f}".format(accel))
        print()
        print()
        accelerations[counter] = accel
    
    np.save("numbersoffrequencies", numbersoffrequencies)
    np.save("accelerations-from-varying-number-of-frequencies", accelerations)
     
    pyl.plot(numbersoffrequencies, accelerations, 'ro')
    pyl.show()
