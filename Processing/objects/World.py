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

import os








class World():

    k = np.array([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
    ])

    # Initialize world
    def __init__(self,
                 A = None,
                 seed = 101
                 ) -> None:
        
        self.new_world(A, seed)
        

    # Set/generate new world
    def new_world(self, 
                  A = None,
                  seed = 101
                  ) -> None:
        
        self.A = A
        self.seed = seed
        # self.rand_gen = np.random.RandomState(self.seed)
        self.sX = 128
        self.sY = self.sX
        self.numChannels = 3
        self.theta = 3
        self.dd = 7
        self.dt = 0.2
        self.sigma = 0.65

        if A is None:
            self.generateWorld()

        # self.dA = self.compute_gradient(self.A)
        # self.fA = self.compute_fftA(self.A)

    def generateWorld(self):
        # Generate random world
        rand_gen = np.random.RandomState(self.seed)
        init_size = self.sX // 2
        self.A = np.zeros((self.sX, self.sY, self.numChannels))
        self.A[self.sX//2-init_size//2:self.sX//2+init_size//2, self.sY//2-init_size//2:self.sY//2+init_size//2, :] = rand_gen.rand(init_size, init_size, self.numChannels)

    
    # VECTORIZE OR VRAM CUPY?
    @staticmethod
    @jax.jit
    def compute_gradient(H):

        # @jax.jit
        return jnp.concatenate((World.sobel(H, World.k)[:, :, None, :],
                                World.sobel(H, World.k.transpose())[:, :, None, :]),
                                axis = 2)
    

    @staticmethod
    @jax.jit
    def sobel(A, k):
        return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], k, mode = 'same') 
                        for c in range(A.shape[-1])])


    # Return world state
    # def get_world(self):
    #     #  RETURN REFERENCE INSTEAD OF COPY
    #     return self.A


    # Return FFT of world
    @staticmethod
    # @jax.jit(static_argnums = ["A"])
    def compute_fftA(A):
        #  RETURN REFERENCE INSTEAD OF COPY
        return np.fft.fft2(A, axes=(0,1))  # (x,y,c)


    # Statistics about the world
    def get_total_mass(self) -> float64:
        
        return self.A.sum()

