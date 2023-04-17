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

import pygame as pg
import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
import typing as t

import os

from models import Models
from objects import World, KernelParameters, ModelParameters
# import models.FlowLeniaModel as FlowLeniaModel
# from models.LeniaModel import LeniaModel












class Model2():

    def __init__(
            self, 
            C,
            c0, # : List(List()),
            c1,
            m : np.ndarray,
            s : np.ndarray,
            h : np.ndarray,
            pos,
            dt,
            theta_A,
            g_func = jax.jit(lambda x, m, s: jnp.exp(-((x - m) / s)**2 / 2) * 2 - 1), # : t.Callable,
    ) -> None:
        self.C = C
        self.c0 = c0
        self.c1 = c1
        self.m = m
        self.s = s
        self.h = h
        self.pos = pos
        self.dt = dt
        self.theta_A = theta_A
        self.g_func = g_func
        # self.compute_G = self

        self.G_func = self.compile_G_func()
        self.gradient_func = self.compile_gradient_func()
        self.mus_func = self.compile_mus_func()
        self.flow_func = self.compile_flow_function()


    

    
    # @jax.jit
    def compile_G_func(
            self
            # g_func, # : t.Callable,
            # c2, # : List(List()),
            # m : np.ndarray,
            # s : np.ndarray,
            # h : np.ndarray
    ):

        # A = jnp.empty((2**8, 2**8, 3))
        #  IN LINUX, PASS ALL NP TO JNP
        def from_U_compute_H(U
                                # g_func, # : t.Callable,
                                # c2, # : List(List()),
                                # m : np.ndarray,
                                # s : np.ndarray,
                                # h : np.ndarray,
                                # C : int
                                ):

            G = self.g_func(U, self.m, self.s) * self.h  # (x,y,k)
            H = np.dstack([ G[:, :, self.c1[c]].sum(axis=-1)
                           for c in range(self.C) ])  # (x,y,c)
            return H
        
        # return jax.jit(f, static_argnames = ["A"])
        return jax.jit(from_U_compute_H)

        
        # fAk = fA[:, :, c0]  # (x,y,k)

        # U = np.real(np.fft.ifft2(fK * fAk, axes=(0,1)))  # (x,y,k)


    def compile_gradient_func(
            self
    ):
    
        @jax.jit
        def sobel(A):
            return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], k, mode = 'same') 
                            for c in range(A.shape[-1])])
        

        # @jax.jit
        def compute_gradient(H):

            # @jax.jit
            return jnp.concatenate((sobel(H, k)[:, :, None, :],
                                    sobel(H, k.transpose())[:, :, None, :]),
                                    axis = 2)
        
        return jax.jit(compute_gradient)
    
    
    def compile_mus_func(
            self
    ):

        # @jax.jit
        def compute_mus(A, dA, F):
            alpha = np.clip((A[:, :, None, :] / self.theta_A)**2, .0, 1.)
        
            F = F * (1 - alpha) - dA * alpha
            
            mus = self.pos[..., None] + self.dt * F #(x, y, 2, c) : target positions (distribution centers)

            return mus
        
        return jax.jit(compute_mus)


    def compile_flow_function(
            self
    ):
        
        @partial(jax.vmap, in_axes = (0, 0, None, None))
        # @jax.jit
        def step_flow(x : int, y : int, A : jnp.ndarray, mus : jnp.ndarray):
            rollA = jnp.roll(A, (x, y), axis=(0, 1))
            # rollA = np.roll(A, (x, y), axis=(0, 1))
            # dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (x, y), axis = (0, 1))) # (x, y, 2, c)
            dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (x, y), axis = (0, 1))) # (x, y, 2, c)
            # sz = .5 - dpmu + self.sigma #(x, y, 2, c)
            sz = .5 - dpmu + self.sigma #(x, y, 2, c)
            # area = jnp.prod(np.clip(sz, 0, min(1, 2 * self.sigma)) , axis = 2) / (4 * self.sigma**2) # (x, y, c)
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2 * self.sigma)) , axis = 2) / (4 * self.sigma**2) # (x, y, c)
            nA = rollA * area
            return nA
        
        return jax.jit(step_flow)






class Model3():

    # def borrar(
            # self, 

            # C,
            # c0, # : List(List()),
            # c1,
            # m : np.ndarray,
            # s : np.ndarray,
            # h : np.ndarray,
            # pos,
            # dt,
            # theta_A,
            # g_func = jax.jit(lambda x, m, s: jnp.exp(-((x - m) / s)**2 / 2) * 2 - 1), # : t.Callable,
    # ) -> None:
    #     self.C = C
    #     self.c0 = c0
    #     self.c1 = c1
    #     self.m = m
    #     self.s = s
    #     self.h = h
    #     self.pos = pos
    #     self.dt = dt
    #     self.theta_A = theta_A
    #     self.g_func = g_func
    #     # self.compute_G = self

        # self.G_func = self.compile_G_func()
        # self.gradient_func = self.compile_gradient_func()
        # self.mus_func = self.compile_mus_func()
        # self.flow_func = self.compile_flow_function()
    

    def __init__(self,
                world:World,
                k_params
                ) -> None:
        
        self.set_world_and_kParams(world, k_params)
    

    # Set world and/or kernel parameters
    def set_world_and_kParams(self,
                              world:World,
                              k_params,
                              dd = 5,
                              dt = 0.2,
                              sigma = 0.65,
                              theta_A = 3,
                              g_func = jax.jit(lambda x, m, s: (jnp.exp(-((x - m) / s)**2 / 2)) * 2 - 1)
    ):
        
        
        self.world = world
        self.k_params = k_params
        self.dd = dd
        self.dt = dt
        self.sigma = sigma
        self.theta_A = theta_A
        self.C = self.world.A.shape[-1]
        self.g_func = g_func
        self.fK = self.k_params.compile_kernels(self.world.A.shape[0], self.world.A.shape[1])

        self.x, self.y = np.arange(world.A.shape[0]), np.arange(world.A.shape[1])
        X, Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((Y, X)) + .5 #(SX, SY, 2)

        self.rollxs = []
        self.rollys = []
        
        for dx in range(-self.dd, self.dd + 1):
            for dy in range(-self.dd, self.dd + 1):
                self.rollxs.append(dx)
                self.rollys.append(dy)
        self.rollxs = np.array(self.rollxs)
        self.rollys = np.array(self.rollys)

        self.sobel_k = np.array([
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
        ])


        self.H_func = self.compile_H_func()
        self.gradient_func = self.compile_gradient_func()
        self.mus_func = self.compile_mus_func()
        self.flow_func = self.compile_flow_function()
    


    def step(self):

        fA = np.fft.fft2(self.world.A, axes=(0,1))  # (x,y,c)
        
        fAk = fA[:, :, self.k_params.c0]  # (x,y,k)

        U = np.real(np.fft.ifft2(self.fK * fAk, axes=(0,1)))  # (x,y,k)

        H = self.H_func(U)
        F = self.gradient_func(H)
        dA = self.gradient_func(self.world.A.sum(axis = -1, keepdims = True))
        
        mus = self.mus_func(
            self.world.A,
            dA,
            F
        )
        
        nA = self.flow_func(
            self.rollxs,
            self.rollys,
            self.world.A,
            mus
        ).sum(axis = 0)

        self.world.new_world(nA)

        return nA

    
    
    # @jax.jit
    def compile_H_func(
            self
            # g_func, # : t.Callable,
            # c2, # : List(List()),
            # m : np.ndarray,
            # s : np.ndarray,
            # h : np.ndarray
    ):

        # A = jnp.empty((2**8, 2**8, 3))
        #  IN LINUX, PASS ALL NP TO JNP
        def from_U_compute_H(U : np.asarray
                                # g_func, # : t.Callable,
                                # c2, # : List(List()),
                                # m : np.ndarray,
                                # s : np.ndarray,
                                # h : np.ndarray,
                                # C : int
                                ):

            G = self.g_func(
                U, 
                self.k_params.kernels['m'], 
                self.k_params.kernels['s']
            ) * self.k_params.kernels['h']  # (x,y,k)

            H = jnp.dstack([ G[:, :, self.k_params.c1[c]].sum(axis=-1)
                           for c in range(self.C) ])  # (x,y,c)
            return H
        
        # return jax.jit(f, static_argnames = ["A"])
        return jax.jit(from_U_compute_H)

        
        # fAk = fA[:, :, c0]  # (x,y,k)

        # U = np.real(np.fft.ifft2(fK * fAk, axes=(0,1)))  # (x,y,k)


    def compile_gradient_func(
            self
    ):
    
        @jax.jit
        def sobel(A, k):
            return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], k, mode = 'same') 
                            for c in range(A.shape[-1])])
        

        # @jax.jit
        def compute_gradient(H):

            # @jax.jit
            return jnp.concatenate((sobel(H, self.sobel_k)[:, :, None, :],
                                    sobel(H, self.sobel_k.transpose())[:, :, None, :]),
                                    axis = 2)
        
        return jax.jit(compute_gradient)
    
    
    def compile_mus_func(
            self
    ):

        # @jax.jit
        def compute_mus(A, dA, F):
            alpha = jnp.clip((A[:, :, None, :] / self.theta_A)**2, .0, 1.)
        
            F = F * (1 - alpha) - dA * alpha
            
            mus = self.pos[..., None] + self.dt * F #(x, y, 2, c) : target positions (distribution centers)

            return mus
        
        return jax.jit(compute_mus)


    def compile_flow_function(
            self
    ):
        
        @partial(jax.vmap, in_axes = (0, 0, None, None))
        # @jax.jit
        def step_flow(
            x : int, 
            y : int, 
            A : jnp.ndarray, 
            mus : jnp.ndarray
        ):
            rollA = jnp.roll(A, (x, y), axis=(0, 1))
            # rollA = np.roll(A, (x, y), axis=(0, 1))
            # dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (x, y), axis = (0, 1))) # (x, y, 2, c)
            dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (x, y), axis = (0, 1))) # (x, y, 2, c)
            # sz = .5 - dpmu + self.sigma #(x, y, 2, c)
            sz = .5 - dpmu + self.sigma #(x, y, 2, c)
            # area = jnp.prod(np.clip(sz, 0, min(1, 2 * self.sigma)) , axis = 2) / (4 * self.sigma**2) # (x, y, c)
            area = jnp.prod(jnp.clip(sz, 0, min(1, 2 * self.sigma)) , axis = 2) / (4 * self.sigma**2) # (x, y, c)
            nA = rollA * area
            return nA
        
        return jax.jit(step_flow)







# JITCLASS?
class System():
    
    world = None
    k_params = None
    C = None
    fK = None
    g_func = njit()(lambda x, m, s: np.exp(-((x - m) / s)**2 / 2) * 2 - 1)
    theta_A = 3
    dd = 20
    dt = 0.5
    sigma = 0.61
    x = None
    y = None
    pos = None
    rollxs = []
    rollys = []
    k = np.array([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
    ])
    
    spec = [
    ('fA', np.ndarray),          # an array field
    ('fAk', np.ndarray),               # a simple scalar field
    ('U', np.ndarray)
    ]


    # Initialization of values of world and kernel parameters
    def __init__(self,
                world:World,
                k_params,
                # m_params
                ) -> None:
        
        self.set_world_and_kParams(world, k_params)
    

    # Set world and/or kernel parameters
    def set_world_and_kParams(self,
                              world:World,
                              k_params,
                            #   m_params
                              ):
        
        
        self.world = world
        self.k_params = k_params
        # self.m_params = m_params

        # Model = getattr(Models, m_params["version"])
        Model = getattr(Models, version)
        self.model = Model(
            world = self.world,
            k_params = self.k_params,
            # m_params = self.m_params
        )


        # self.C = self.world.A.shape[-1]
        # self.fK = self.k_params.compile_kernels(self.world.A.shape[0], self.world.A.shape[1])

        # self.x, self.y = np.arange(world.A.shape[0]), np.arange(world.A.shape[1])
        # X, Y = np.meshgrid(self.x, self.y)
        # self.pos = np.dstack((Y, X)) + .5 #(SX, SY, 2)

        # for dx in range(-self.dd, self.dd + 1):
        #     for dy in range(-self.dd, self.dd + 1):
        #         self.rollxs.append(dx)
        #         self.rollys.append(dy)
        # self.rollxs = np.array(self.rollxs)
        # self.rollys = np.array(self.rollys)

    
    def step(self):

        return self.model.step()









version = "FlowLeniaModel"

sX = sY = 2**7
nC = 3
SCALE = 800 // sX
init_size = sX >> 2

connections = np.array([
        # [1, 2, 2],
        # [2, 1, 2],
        # [2, 2, 1]
            
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
            
            # [3, 1, 4],
            # [2, 2, 4],
            # [1, 5, 1]

        # [5, 0, 5],
        # [0, 0, 0],
        # [5, 0, 5]
        ])
connections = connections * 6
A0 = np.zeros((sX, sY, nC))
A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = np.random.rand(init_size, init_size, nC)

A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = np.ones((init_size, init_size, nC)) * 1

# A0 = np.ones((A0.shape)) * 0.2

world = World.World(A0)

# model = Model(1, 2, 3, 4, 5)
# func = model.compile_G_function()

# print(func(3))
# print("FINN")



k_params = KernelParameters.KernelParameters(connections, (40, 40))

# Model = getattr(Models, version)

system = System(
    world = world,
    k_params = k_params,
    # m_params = m_params
)


pg.init()
display = pg.display.set_mode((sX * SCALE, sY * SCALE))

running = True
total_mass = world.A.sum()
# print("Initial mass:", total_mass)
tm = .0
i = 0
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    st = time.process_time()
    # result = model.next_step()
    system.step()
    et = time.process_time()
    tm += et - st
    i += 1
    # print(tm, "CPU seconds until step n", i)
    current_mass = world.A.sum()
    # print(world.A.sum())
    if (total_mass != current_mass) : 
        # print("Total mass changed:", current_mass - total_mass)
        total_mass = current_mass
    # print(A[:, :, 1, None].shape)
    # surf = np.stack((A[:, :, :], A[:, :, 1, None]), axis = -1)
    surf = pg.surfarray.make_surface(np.kron(np.uint8(world.A.clip(0, 1) * 255.0), np.ones((SCALE, SCALE, 1))))
    display.blit(surf, (0, 0))

    pg.display.update()

pg.quit()










