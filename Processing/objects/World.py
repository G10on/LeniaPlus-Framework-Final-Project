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
                 A = None
                 ) -> None:
        
        self.new_world(A)
        

    # Set/generate new world
    def new_world(self, 
                  A = None,
                  seed = 101
                  ) -> None:
        
        if A is None:
            rand_gen = np.random.RandomState(seed)
            self.A = rand_gen.rand(128, 128, 1)
        
        
        self.seed = seed
        self.A = A
        self.sX = self.A.shape[0]
        self.sY = self.A.shape[1]
        self.numChannels = self.A.shape[2]
        self.theta = 3
        self.dd = 7
        self.dt = 0.2
        self.sigma = 0.65

        # self.dA = self.compute_gradient(self.A)
        # self.fA = self.compute_fftA(self.A)

    
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

