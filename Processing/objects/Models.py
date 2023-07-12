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

from objects import World, Kernel


class Model():

    def __init__(self,
                 world: World.World,
                 ) -> None:

        self.set_world_and_kParams(world)

    # Set world and/or kernel parameters

    def set_world_and_kParams(self,
                              world: World.World
                              ):

        self.world = world
        self.kernel_parameters = self.getNewKParams()
        

        self.x, self.y = np.arange(self.world.sX), np.arange(self.world.sY)
        X, Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((Y, X)) + .5

        self.rollxs = []
        self.rollys = []

        for dx in range(-self.world.dd, self.world.dd + 1):
            for dy in range(-self.world.dd, self.world.dd + 1):
                self.rollxs.append(dx)
                self.rollys.append(dy)

        self.rollxs = np.array(self.rollxs)
        self.rollys = np.array(self.rollys)

        self.sobel_k = np.array([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
        ])

    def compile(self):

        self.fourier_kernels = self.kernel_parameters.compile_kernels(
            self.world.sX, self.world.sY)

        self.gradient_func = self.compile_gradient_func()
        self.flow_func = self.compile_flow_function()
        self.next_gen_func = self.compile_next_gen_func()

    def step(self):

        new_world = self.next_gen_func(self.world.A)
        self.world.A = new_world

        return new_world

    def compile_next_gen_func(self):

        # Based on Notebook implementation of Flow Lenia
        def next_gen_func(world: np.asarray):

            fourier_world = jnp.fft.fft2(world, axes=(0, 1))

            fourier_world_kernel = fourier_world[:, :,
                                                 self.kernel_parameters.kernel_parameters['C']]

            potential_distribution = jnp.real(jnp.fft.ifft2(
                self.fourier_kernels * fourier_world_kernel, axes=(0, 1)))

            affinity = self.growth_function(
                potential_distribution,
                self.kernel_parameters.kernel_parameters['m'],
                self.kernel_parameters.kernel_parameters['s']
            ) * self.kernel_parameters.kernel_parameters['h']

            H = jnp.dstack([affinity[:, :, self.kernel_parameters.kernel_parameters['T'][c]].sum(axis=-1)
                           for c in range(self.world.numChannels)])

            flow_distribution = self.gradient_func(H)

            d_world = self.gradient_func(world.sum(axis=-1, keepdims=True))

            alpha = jnp.clip(
                (world[:, :, None, :] / self.world.theta)**2, .0, 1.)

            flow_distribution = flow_distribution * \
                (1 - alpha) - d_world * alpha

            moved_coordinates = self.pos[..., None] + \
                self.world.dt * flow_distribution

            new_world = self.flow_func(
                self.rollxs,
                self.rollys,
                world,
                moved_coordinates
            ).sum(axis=0)

            return new_world

        return jax.jit(next_gen_func)

    def compile_gradient_func(self):

        @jax.jit
        def sobel(A, k):
            return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], k, mode='same')
                               for c in range(A.shape[-1])])

        def compute_gradient(H):

            return jnp.concatenate((sobel(H, self.sobel_k.transpose())[:, :, None, :],
                                    sobel(H, self.sobel_k)[:, :, None, :]),
                                   axis=2)

        return jax.jit(compute_gradient)

    def compile_flow_function(self):

        # Based on Notebook implementation of Flow Lenia
        @partial(jax.vmap, in_axes=(0, 0, None, None))
        def flow_function(
            x: int,
            y: int,
            A: jnp.ndarray,
            mus: jnp.ndarray
        ):
            rollA = jnp.roll(A, (x, y), axis=(0, 1))
            dpmu = jnp.absolute(
                self.pos[..., None] - jnp.roll(mus, (x, y), axis=(0, 1)))
            sz = .5 - dpmu + self.world.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(
                1, 2 * self.world.sigma)), axis=2) / (4 * self.world.sigma**2)
            nA = rollA * area
            return nA

        return jax.jit(flow_function)

    def getKParams(self):

        return self.kernel_parameters







class FlowLeniaModel(Model):
    

    def __init__(self,
                world:World.World,
                ) -> None:
        
        super().__init__(world)
        self.growth_function = jax.jit(lambda x, m, s: (
            jnp.exp(-((x - m) / s)**2 / 2)) * 2 - 1)
    

    def compile(self):

        self.fourier_kernels = self.kernel_parameters.compile_kernels(self.world.sX, self.world.sY)

        self.gradient_func = self.compile_gradient_func()
        self.flow_func = self.compile_flow_function()
        self.next_gen_func = self.compile_next_gen_func()


    def compile_next_gen_func(self):

        # Based on Notebook implementation of Flow Lenia
        def next_gen_func(world : np.asarray):
            
            fourier_world = jnp.fft.fft2(world, axes=(0,1))
        
            fourier_world_kernel = fourier_world[:, :, self.kernel_parameters.kernel_parameters['C']]

            potential_distribution = jnp.real(jnp.fft.ifft2(self.fourier_kernels * fourier_world_kernel, axes=(0,1)))

            affinity = self.growth_function(
                potential_distribution, 
                self.kernel_parameters.kernel_parameters['m'], 
                self.kernel_parameters.kernel_parameters['s']
            ) * self.kernel_parameters.kernel_parameters['h']

            H = jnp.dstack([ affinity[:, :, self.kernel_parameters.kernel_parameters['T'][c]].sum(axis=-1)
                           for c in range(self.world.numChannels) ])
            
            flow_distribution = self.gradient_func(H)

            d_world = self.gradient_func(world.sum(axis = -1, keepdims = True))
        
            alpha = jnp.clip((world[:, :, None, :] / self.world.theta)**2, .0, 1.)
        
            flow_distribution = flow_distribution * (1 - alpha) - d_world * alpha
            
            moved_coordinates = self.pos[..., None] + self.world.dt * flow_distribution
            
            new_world = self.flow_func(
                self.rollxs,
                self.rollys,
                world,
                moved_coordinates
            ).sum(axis = 0)

            return new_world
        
        return jax.jit(next_gen_func)
    
   
    def compile_flow_function(self):
        
        # Based on Notebook implementation of Flow Lenia
        @partial(jax.vmap, in_axes = (0, 0, None, None))
        def flow_function(
            x : int, 
            y : int, 
            A : jnp.ndarray, 
            mus : jnp.ndarray
        ):
            rollA = jnp.roll(A, (x, y), axis=(0, 1))
            dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (x, y), axis = (0, 1)))
            sz = .5 - dpmu + self.world.sigma
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2 * self.world.sigma)) , axis = 2) / (4 * self.world.sigma**2)
            nA = rollA * area
            return nA
        
        return jax.jit(flow_function)

    
    def getNewKParams(self):

        return Kernel.FlowLeniaKernel()







class LeniaModel(Model):

    def __init__(self, world:World.World) -> None:
        
        super().__init__(world)
        self.growth_function = jax.jit(lambda x, m, s: (
            jnp.exp( -((jnp.absolute(x - m))**2) / (2*s*s) )))


    def compile(self):
        
        self.fourier_kernels = self.kernel_parameters.compile_kernels(self.world.sX, self.world.sY)
        self.gradient_func = self.compile_gradient_func()
        self.next_gen_func = self.compile_next_gen_func()


    
    def compile_next_gen_func(self):

        # Based on Notebook implementation of Flow Lenia 
        def next_gen_func(A : np.asarray):

            fourier_world = [ jnp.fft.fft2(A[:,:,c]) for c in range(self.world.numChannels) ]
            
            potential_distribution = [ np.real(jnp.fft.ifft2(fourier_kernel * fourier_world[c0])) for fourier_kernel, c0 in zip(self.fourier_kernels, self.kernel_parameters.kernel_parameters["C"]) ]

            growth_distribution = [ self.growth_function(u, self.kernel_parameters.kernel_parameters['m'][k], self.kernel_parameters.kernel_parameters['s'][k]) * 2 - 1 for u, k in zip(potential_distribution, range(len(self.kernel_parameters.kernel_parameters['m']))) ]

            Hs = [sum(self.kernel_parameters.kernel_parameters['h'][k] * g for g, k in zip(growth_distribution, range(len(self.kernel_parameters.kernel_parameters['m'])))
                      if k in self.kernel_parameters.kernel_parameters['T'][c1]) for c1 in range(A.shape[2])]
            
            world_channels = [ jnp.clip(A[:,:,cA] + 1/self.world.dt * H, 0, 1) for cA,H in zip(range(A.shape[2]),Hs) ]
            
            new_world = jnp.dstack(world_channels)

            return new_world
        
        return jax.jit(next_gen_func)
 
   
    def getNewKParams(self):

        return Kernel.LeniaKernel()
    




