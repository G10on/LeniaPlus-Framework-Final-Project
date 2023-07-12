from datetime import datetime
import time
import timeit
from functools import partial
# import subprocess
# import sys
# import cv2

# import cupy

import jax
import jax.numpy as jnp
import jax.scipy as jsp
# import jax.scipy as jsp
import numba as nb
from numba import jit, njit
from numba import int32, float64
from numba.typed import List
from numba.experimental import jitclass

# import pygame as pg
import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
import typing as t

# import os


class Kernel():

    # Initialization of N of kernels, size and parameters
    def __init__(self,
                 connection_matrix=None,
                 ker_params=None
                 ) -> None:

        self.generate_random_parameters(connection_matrix, ker_params)

    # Generation based on Notebook implementation of Lenia and Flow Lenia
    def generate_random_parameters(self,
                                 connection_matrix=None,
                                 new_kernel_parameters=None,
                                 seed=101
                                 ) -> None:

        rand_gen = np.random.RandomState(seed)

        if connection_matrix is None:

            connection_matrix = np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])

        self.n_kernels = int(connection_matrix.sum())

        if new_kernel_parameters is None:

            self.spaces = {
                "r": {'low': .2, 'high': 1., 'mut_std': .2, 'shape': None},
                "B": {'low': .001, 'high': 1., 'mut_std': .2, 'shape': (3,)},
                "w": {'low': .01, 'high': .5, 'mut_std': .2, 'shape': (3,)},
                "a": {'low': .0, 'high': 1., 'mut_std': .2, 'shape': (3,)},
                "m": {'low': .05, 'high': .5, 'mut_std': .2, 'shape': None},
                "s": {'low': .001, 'high': .18, 'mut_std': .01, 'shape': None},
                "h": {'low': .01, 'high': 1., 'mut_std': .2, 'shape': None},
                'R': {'low': 2., 'high': 25., 'mut_std': .2, 'shape': None},
            }

            new_kernel_parameters = {}
            for k in 'rmsh':
                new_kernel_parameters[k] = np.round(rand_gen.uniform(
                    self.spaces[k]['low'], self.spaces[k]['high'], self.n_kernels
                ), 4)
            for k in 'Baw':
                new_kernel_parameters[k] = np.round(rand_gen.uniform(
                    self.spaces[k]['low'], self.spaces[k]['high'], (
                        self.n_kernels, 3)
                ), 4)

        self.kernel_parameters = {}

        for k in 'rmshBaw':
            self.kernel_parameters[k] = new_kernel_parameters[k]

        self.kernel_parameters['R'] = 13

        self.conn_from_matrix(connection_matrix)


    # Method from Notebook implementation of Lenia and Flow Lenia
    def conn_from_matrix(self, connection_matrix):
        
        connection_shape = connection_matrix.shape[0]
        channels_0 = []
        channels_1 = [[] for _ in range(connection_shape)]
        i = 0

        for s in range(connection_shape):
            for t in range(connection_shape):
                n = connection_matrix[s, t]
                if n:
                    channels_0 = channels_0 + [s]*n
                    channels_1[t] = channels_1[t] + list(range(i, i + n))
                i += n

        self.kernel_parameters['C'] = channels_0
        self.kernel_parameters['T'] = channels_1


    def new_params(self, data):

        max_rings = 0

        for k in 'Baw':
            temp = max(len(sublist) for sublist in data[k])
            if temp > max_rings:
                max_rings = temp

        for k in 'Baw':
            for B in data[k]:
                temp = [0] * (max_rings - len(B))
                B.extend(temp)

        for k in 'rmshBaw':
            self.kernel_parameters[k] = np.array(data[k], dtype=np.float64)

        for k in 'C':
            self.kernel_parameters[k] = np.array(data[k], dtype=np.int64)

        self.kernel_parameters['T'] = data['T']

        self.n_kernels = len(self.kernel_parameters['r'])
    

    def sigmoid(self, x):
        return 0.5 * (np.tanh(x / 2) + 1)








class FlowLeniaKernel(Kernel):

    def __init__(self,
                 connection_matrix=None,
                 ker_params=None
                 ) -> None:

        super().__init__(connection_matrix, ker_params)
        
        self.kernel_function = lambda x, a, w, b: (
                b * np.exp(- (x[..., None] - a)**2 / w)
            ).sum(-1)    


    # Compilation method based on Flow Lenia Notebook
    def compile_kernels(self, SX, SY):

        midX = SX >> 1
        midY = SY >> 1

        distances = [ np.linalg.norm(np.mgrid[-midX:midX, -midY:midY], axis=0) /
        ((self.kernel_parameters['R']+15) * self.kernel_parameters['r'][k]) for k in range(self.n_kernels) ]

        kernels = np.dstack([self.sigmoid(-(D-1)*10) * self.kernel_function(D, self.kernel_parameters["a"][k], self.kernel_parameters["w"][k], self.kernel_parameters["B"][k])
                       for k, D in zip(range(self.n_kernels), distances)])

        normalized_kernels = kernels / np.sum(kernels, axis=(0, 1), keepdims=True)
        fourier_kernels = np.fft.fft2(np.fft.fftshift(normalized_kernels, axes=(0, 1)), axes=(0, 1))

        return fourier_kernels


    def new_params(self, data):

        max_rings = 0
        for k in 'Baw':
            temp = max(len(sublist) for sublist in data[k])
            if temp > max_rings:
                max_rings = temp

        for k in 'Baw':
            for B in data[k]:
                temp = [0] * (max_rings - len(B))
                B.extend(temp)

        for k in 'rmshBaw':
            self.kernel_parameters[k] = np.array(data[k], dtype=np.float64)

        for k in 'C':
            self.kernel_parameters[k] = np.array(data[k], dtype=np.int64)

        self.kernel_parameters['T'] = data['T']

        self.n_kernels = len(self.kernel_parameters['r'])









class LeniaKernel(Kernel):

    # Initialization of N of kernels, size and parameters

    def __init__(self,
                 connection_matrix=None,
                 ker_params=None
                 ) -> None:

        super().__init__(connection_matrix, ker_params)

        self.kernel_function = lambda x, m, s: np.exp(-((x-m)/s)**2 / 2)

    # Compilation method based on Lenia Notebook
    def compile_kernels(self, SX, SY):

        midX = SX >> 1
        midY = SY >> 1

        D_norm = np.linalg.norm(np.ogrid[-midX: midX, -midY: midY])

        distances = [D_norm /
              self.kernel_parameters['R'] * len(self.kernel_parameters['B'][k]) / self.kernel_parameters['r'][k] for k in range(self.n_kernels)]

        kernels = [(D < len(self.kernel_parameters['B'][k])) * np.asarray(self.kernel_parameters['B'][k])[np.minimum(D.astype(int),
            len(self.kernel_parameters['B'][k])-1)] * self.kernel_function(D % 1, 0.5, 0.15) for D, k in zip(distances, range(self.n_kernels))]

        normalized_kernels = [K / np.sum(K) for K in kernels]
        fourier_kernels = [np.fft.fft2(np.fft.fftshift(K)) for K in normalized_kernels]

        return fourier_kernels


    def new_params(self, data):

        for k in 'rmsh':
            self.kernel_parameters[k] = np.array(data[k], dtype=np.float64)

        for k in 'C':
            self.kernel_parameters[k] = np.array(data[k], dtype=np.int64)

        self.kernel_parameters['B'] = data['B']
        self.kernel_parameters['a'] = data['a']
        self.kernel_parameters['w'] = data['w']
        self.kernel_parameters['T'] = data['T']

        self.n_kernels = len(self.kernel_parameters['r'])
