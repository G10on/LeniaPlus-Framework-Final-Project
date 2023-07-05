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


class KernelParametersGeneral():

    # Initialization of N of kernels, size and parameters
    def __init__(self,
                 connection_matrix=None,
                 ker_params=None
                 ) -> None:

        self.generateRandomParameters(connection_matrix, ker_params)

    def generateRandomParameters(self,
                                 connection_matrix=None,
                                 ker_params=None,
                                 seed=101
                                 ) -> None:

        rand_gen = np.random.RandomState(seed)

        self.ker_f = lambda x, a, w, b: (
            b * np.exp(- (x[..., None] - a)**2 / w)).sum(-1)

        if connection_matrix is None:

            connection_matrix = np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])

        self.n_kernels = int(connection_matrix.sum())

        if ker_params is None:

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

            ker_params = {}
            for k in 'rmsh':
                ker_params[k] = np.round(rand_gen.uniform(
                    self.spaces[k]['low'], self.spaces[k]['high'], self.n_kernels
                ), 4)
            for k in 'Baw':
                ker_params[k] = np.round(rand_gen.uniform(
                    self.spaces[k]['low'], self.spaces[k]['high'], (
                        self.n_kernels, 3)
                ), 4)

        self.kernels = {}

        for k in 'rmshBaw':
            self.kernels[k] = ker_params[k]

        self.kernels.update({
            'R': rand_gen.uniform(self.spaces['R']['low'], self.spaces['R']['high'])
        })
        self.kernels['R'] = 13

        # CHANGE WAY TO INPUT CONNECTIONS MATRIX
        self.conn_from_matrix(connection_matrix)

    # Return kernels compiled

    # Connection matrix where M[i, j] = number of kernels from channel i to channel j
    def conn_from_matrix(self, connection_matrix):
        C = connection_matrix.shape[0]
        c0 = []
        c1 = [[] for _ in range(C)]
        i = 0
        for s in range(C):
            for t in range(C):
                n = connection_matrix[s, t]
                if n:
                    c0 = c0 + [s]*n
                    c1[t] = c1[t] + list(range(i, i + n))
                i += n

        self.kernels['C'] = c0
        self.kernels['T'] = c1

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
            self.kernels[k] = np.array(data[k], dtype=np.float64)

        for k in 'C':
            self.kernels[k] = np.array(data[k], dtype=np.int64)

        self.kernels['T'] = data['T']
        self.kernels['R'] = 13

        self.n_kernels = len(self.kernels['r'])








class KernelParameters(KernelParametersGeneral):

    # Initialization of N of kernels, size and parameters
    def __init__(self,
                 connection_matrix=None,
                 ker_params=None
                 ) -> None:

        super().__init__(connection_matrix, ker_params)
        self.ker_f = lambda x, a, w, b: (
                b * np.exp(- (x[..., None] - a)**2 / w)
            ).sum(-1)

    def compile_kernels(self, SX, SY):

        midX = SX >> 1
        midY = SY >> 1

        r = self.kernels['r'] * (self.kernels['R'])
        D = np.linalg.norm(np.mgrid[-midX: midX, -midY: midY], axis=0)
        # Ds = [D / r[k] for k in range(self.n_kernels)]

        Ds = [ np.linalg.norm(np.mgrid[-midX:midX, -midY:midY], axis=0) /
        ((self.kernels['R']+15) * self.kernels['r'][k]) for k in range(self.n_kernels) ]

        def sigmoid(x):
            return 0.5 * (np.tanh(x / 2) + 1)

        def ker_f(x, a, w, b): return (
            b * np.exp(- (x[..., None] - a)**2 / w)).sum(-1)

        K = np.dstack([sigmoid(-(D-1)*10) * ker_f(D, self.kernels["a"][k], self.kernels["w"][k], self.kernels["B"][k])
                       for k, D in zip(range(self.n_kernels), Ds)])

        nK = K / np.sum(K, axis=(0, 1), keepdims=True)
        fK = np.fft.fft2(np.fft.fftshift(nK, axes=(0, 1)), axes=(0, 1))

        return fK


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
            self.kernels[k] = np.array(data[k], dtype=np.float64)

        for k in 'C':
            self.kernels[k] = np.array(data[k], dtype=np.int64)

        self.kernels['T'] = data['T']
        self.kernels['R'] = 13

        self.n_kernels = len(self.kernels['r'])









class LeniaKernelParameters(KernelParametersGeneral):

    # Initialization of N of kernels, size and parameters

    def __init__(self,
                 connection_matrix=None,
                 ker_params=None
                 ) -> None:

        super().__init__(connection_matrix, ker_params)

        self.ker_f = lambda x, m, s: np.exp(-((x-m)/s)**2 / 2)

    def compile_kernels(self, SX, SY):

        midX = SX >> 1
        midY = SY >> 1

        new_B = []

        def sigmoid(x):
            return 0.5 * (np.tanh(x / 2) + 1)

        r = self.kernels['r'] * (self.kernels['R'])
        D_norm = np.linalg.norm(np.ogrid[-midX: midX, -midY: midY])

        # The correct
        # Ds = [D / r[k] for k in range(self.n_kernels)]

        Ds = [D_norm /
              self.kernels['R'] * len(self.kernels['B'][k]) / self.kernels['r'][k] for k in range(self.n_kernels)]

        # ker_f = lambda x, a, w, b : (b * np.exp( - (x[..., None] - a)**2 / w)).sum(-1)

        # The correct
        # Ks = [sigmoid(-(D-1)*10) * self.ker_f(D, self.kernels["a"][k], self.kernels["w"][k], self.kernels["B"][k])
        #    for k, D in zip(range(self.n_kernels), Ds)]

        # The test
        Ks = [(D < len(self.kernels['B'][k])) * np.asarray(self.kernels['B'][k])[np.minimum(D.astype(int),
                                                                                            len(self.kernels['B'][k])-1)] * self.ker_f(D % 1, 0.5, 0.15) for D, k in zip(Ds, range(self.n_kernels))]

        # The correct
        # K = np.dstack(Ks)
        # nK = K / np.sum(K, axis=(0, 1), keepdims=True)
        # fK = np.fft.fft2(np.fft.fftshift(nK, axes=(0, 1)), axes=(0, 1))

        # The test
        nKs = [K / np.sum(K) for K in Ks]
        fKs = [np.fft.fft2(np.fft.fftshift(K)) for K in nKs]
        # print(len(fKs), fKs[0].shape)
        # fKs = np.asarray(fKs).transpose((1, 2, 0))

        return fKs


    def new_params(self, data):

        max_rings = 0

        for k in 'rmsh':
            self.kernels[k] = np.array(data[k], dtype=np.float64)

        for k in 'C':
            self.kernels[k] = np.array(data[k], dtype=np.int64)

        self.kernels['B'] = data['B']
        self.kernels['a'] = data['a']
        self.kernels['w'] = data['w']
        self.kernels['T'] = data['T']
        self.kernels['R'] = 13

        self.n_kernels = len(self.kernels['r'])
