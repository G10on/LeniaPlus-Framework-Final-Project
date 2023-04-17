from datetime import datetime
import time
import timeit
from functools import partial
import subprocess
import sys
import cv2

# import cupy

import jax
import jax.numpy as jnp
import jax.scipy as jsp
from numba import jit, njit
from numba import int32, float64
from numba.typed import List
import numba

import pygame as pg
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import typing as t

import os


def NumpyArrayToVideo(
        filename = datetime.now().strftime("output_%m-%d_%H-%M-%S.mp4"), 
        evolution = np.random.randint(0, 255, (10, 32, 32, 3), dtype = np.uint8),
        fps = 1
):

    n_instances, width, height, C = evolution.shape
    sec = n_instances // fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video = cv2.VideoWriter(filename, fourcc, float(fps), (width, height), C)
    # ------------------
    
    for instance_frame in range(n_instances):
        # img = np.uint8(evolution[instance_frame].clip(0, 1) * 255)
        # img = np.random.randint(0,255, (height, width, C), dtype = np.uint8)
        img = evolution[instance_frame]
        video.write(img)
    
    video.release()








class World():

    A = None
    dA = None
    fA = None

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
                  A# = np.random.rand(64, 64, 1)
                  ) -> None:
        
        self.A = A
        self.dA = self.compute_gradient(self.A)

    
        # VECTORIZE OR VRAM CUPY?
    # @njit(numba.float64[:, :, :](numba.float64[:, :]))
    def compute_gradient(self, H):
        
        # @jax.jit
        return np.concatenate((self.sobel(H, self.k)[:, :, None, :], self.sobel(H, self.k.transpose())[:, :, None, :]),
                                axis = 2)
    

    def sobel(self, A, k):
        return np.dstack([sp.signal.convolve2d(A[:, :, c], k, mode = 'same') 
                        for c in range(A.shape[-1])])


    # Return world state
    # def get_world(self):
    #     #  RETURN REFERENCE INSTEAD OF COPY
    #     return self.A


    # Return FFT of world
    # def get_FFT(self):
    #     #  RETURN REFERENCE INSTEAD OF COPY
    #     return self.fA


    # Statistics about the world
    def get_total_mass(self) -> float:
        
        return self.A.sum()



class KernelParameters():

    c0 = []
    c1 = []

    # INITIALIZE CALLING ANOTHER SETTER, LIKE THE OTHERS AND CHOOSE PARAMETER VALUES
    # Initialization of N of kernels, size and parameters
    def __init__(self,
                 connection_matrix,
                 k_shape = (30, 30)
                #  R = 15,
                 ) -> None:
        
        self.n_kernels = int(connection_matrix.sum())
        
        self.spaces = {
            "r" : {'low' : .2, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            "b" : {'low' : .001, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "w" : {'low' : .01, 'high' : .5, 'mut_std' : .2, 'shape' : (3,)},
            "a" : {'low' : .0, 'high' : 1., 'mut_std' : .2, 'shape' : (3,)},
            "m" : {'low' : .05, 'high' : .5, 'mut_std' : .2, 'shape' : None},
            "s" : {'low' : .001, 'high' : .18, 'mut_std' : .01, 'shape' : None},
            "h" : {'low' : .01, 'high' : 1., 'mut_std' : .2, 'shape' : None},
            'T' : {'low' : 10., 'high' : 50., 'mut_std' : .1, 'shape' : None},
            'R' : {'low' : 2., 'high' : 25., 'mut_std' : .2, 'shape' : None},
            'init' : {'low' : 0., 'high' : 1., 'mut_std' : .2, 'shape' : k_shape}
        }

        self.kernels = {}
        for k in 'rmsh':
            self.kernels[k] = np.random.uniform(
                self.spaces[k]['low'], self.spaces[k]['high'], self.n_kernels
            )
        for k in "awb":
            self.kernels[k] = np.random.uniform(
                self.spaces[k]['low'], self.spaces[k]['high'], (self.n_kernels, 3)
            )
        
        self.kernels.update({
            'T' : np.random.uniform(self.spaces['T']['low'], self.spaces['T']['high']),
            'R' : np.random.uniform(self.spaces['R']['low'], self.spaces['R']['high']),
            'init' : np.random.rand(*k_shape)
        })

        # CHANGE WAY TO INPUT CONNCTIONS MATRIX
        self.conn_from_matrix(connection_matrix)


    # Return kernels compiled
    def compile_kernels(self, SX, SY):
        
        midX = SX >> 1
        midY = SY >> 1

        r = self.kernels['r'] * self.kernels['R']
        D = np.linalg.norm(np.mgrid[-midX : midX, -midY : midY], axis=0)
        Ds = [ D / r[k] for k in range(self.n_kernels) ]

        def sigmoid(x):
            return 0.5 * (np.tanh(x / 2) + 1)

        ker_f = lambda x, a, w, b : (b * np.exp( - (x[..., None] - a)**2 / w)).sum(-1)

        K = np.dstack([sigmoid(-(D-1)*10) * ker_f(D, self.kernels["a"][k], self.kernels["w"][k], self.kernels["b"][k]) 
                  for k, D in zip(range(self.n_kernels), Ds)])
        
        nK = K / np.sum(K, axis=(0,1), keepdims=True)
        fK = np.fft.fft2(np.fft.fftshift(nK, axes=(0,1)), axes=(0,1))

        return fK


    # Connection matrix where M[i, j] = number of kernels from channel i to channel j
    def conn_from_matrix(self, connection_matrix):
        C = connection_matrix.shape[0]
        self.c1 = [[] for _ in range(C)]
        # self.c1 = List([List([]) for _ in range(C)])
        i = 0
        for s in range(C):
            for t in range(C):
                n = connection_matrix[s, t]
                if n:
                    self.c0 = self.c0 + [s]*n
                    self.c1[t] = self.c1[t] + list(range(i, i + n))
                i += n
        
        # self.c1 = np.asarray(self.c1, dtype=object)
        # self.c1 = numba.typed.List(self.c1)
        # temp = List()
        # for l in self.c1:
        #     temp2 = List()
        #     for i in l:
        #         temp2.append(i)
        #     temp.append(l)
        
        # self.c1 = temp


# JITCLASS?
class System():
    
    world = None
    k_params = None
    C = None
    fK = None
    g_func = njit()(lambda x, m, s: np.exp(-((x - m) / s)**2 / 2) * 2 - 1)
    theta_A = 3
    dd = 5
    dt = 0.2
    sigma = 0.65
    x = None
    y = None
    pos = None
    rollxs = []
    rollys = []
    k = np.array([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
    ], dtype=np.float64)
    
    spec = [
    ('fA', np.ndarray),          # an array field
    ('fAk', np.ndarray),               # a simple scalar field
    ('U', np.ndarray)
    ]


    # Initialization of values of world and kernel parameters
    def __init__(self,
                world:World,
                k_params
                ) -> None:
        
        self.set_world_and_kParams(world, k_params)
    

    # Set world and/or kernel parameters
    def set_world_and_kParams(self,
                              world:World,
                              k_params):
        
        
        self.world = world
        self.k_params = k_params
        self.C = self.world.A.shape[-1]
        self.fK = self.k_params.compile_kernels(self.world.A.shape[0], self.world.A.shape[1])

        self.x, self.y = np.arange(world.A.shape[0]), np.arange(world.A.shape[1])
        X, Y = np.meshgrid(self.x, self.y)
        self.pos = np.dstack((Y, X)) + .5 #(SX, SY, 2)

        for dx in range(-self.dd, self.dd + 1):
            for dy in range(-self.dd, self.dd + 1):
                self.rollxs.append(dx)
                self.rollys.append(dy)
        self.rollxs = np.array(self.rollxs)
        self.rollys = np.array(self.rollys)


    # @njit([
    # ('fAk', float64[:, :, :]),
    # ])
    def next_step(self):

        fA = np.fft.fft2(self.world.A, axes=(0,1))  # (x,y,c)
        fAk = fA[:, :, self.k_params.c0]  # (x,y,k)
        # fAk = self.world.A[:, :, self.k_params.c0]  # (x,y,k)

        U = np.real(np.fft.ifft2(self.fK * fAk, axes=(0,1)))  # (x,y,k)

        # U = np.dstack([jsp.signal.convolve2d(fAk[:, :, c], self.fK[c], mode = 'same') 
                            # for c in range(len(self.k_params.c0))])

        G = self.from_U_compute_G(U, 
                                  self.g_func,
                                #   self.k_params.c1,
                                  self.k_params.kernels['m'],
                                  self.k_params.kernels['s'],
                                  self.k_params.kernels['h'],
                                #   self.C
                                  )
        
        H = np.dstack([ G[:, :, self.k_params.c1[c]].sum(axis=-1) for c in range(self.C) ])  # (x,y,c)

        #-------------------------------FLOW------------------------------------------

        F = self.compute_gradient(H) #(x, y, 2, c)
    
        alpha = np.clip((self.world.A[:, :, None, :] / self.theta_A)**2, .0, 1.)
        
        F = F * (1 - alpha) - self.world.dA * alpha
        
        mus = self.pos[..., None] + self.dt * F #(x, y, 2, c) : target positions (distribution centers)

        nA = System.step_flow(self.rollxs, self.rollys, self.world.A, mus, self.pos).sum(axis = 0)
        # nA = np.zeros(self.world.A.shape, dtype=np.float64)
        # temp = np.dstack((self.x, self.y))[0]
        # for x, y in temp:
        #     nA += self.step_flow(x, y, self.world.A, mus, self.pos)

        self.world.new_world(nA)
        
        # return self.world.A
    

    @staticmethod
    # @njit(["[np.ndarray(np.float64)(np.ndarray(np.ndarray(np.float64)), t.List[int], np.ndarray(np.ndarray(np.float64)))]"])
    @njit(fastmath = True)
    def from_U_compute_G(U : np.ndarray, 
                            g_func, # : t.Callable,
                            # c2, # : List(List()),
                            m : np.ndarray,
                            s : np.ndarray,
                            h : np.ndarray,
                            # C : int
                            ) -> np.ndarray:

        G = g_func(U, m, s) * h  # (x,y,k)

        # H = np.dstack([ G[:, :, c1[c]].sum(axis=-1) for c in range(C) ])  # (x,y,c)
        # H = np.empty((U.shape), dtype=np.float64)
        # H = G[:, :, c1[0]].sum(axis=-1)
        # c1 = List()
        # for l in c2:
        #     temp2 = List()
        #     for i in l:
        #         temp2.append(i)
        #     c1.append(l)
        
        # self.c1 = tem
        # H = np.zeros(G.shape)
        # for ki in range(len(c1[0])):
        #     H += G[:, :, c1[0][ki]] #.sum(axis=-1)
        # for c in range(1, C):
        #     temp = np.zeros(G.shape)
        #     for ki in range(len(c1[c])):
        #         temp += G[:, :, c1[c][ki]] #.sum(axis=-1)
        #     H = np.dstack((H, temp))

        return G



    def set_growth_func(self, new_func):
        
        # TRY TO COMPILE IN JIT
        if new_func != None:
            self.g_func = new_func


    # VECTORIZE OR VRAM CUPY?
    # @njit(numba.float64[:, :, :](numba.float64[:, :]))
    def compute_gradient(self, H):
        
        # @jax.jit
        return np.concatenate((self.sobel(H, self.k)[:, :, None, :], self.sobel(H, self.k.transpose())[:, :, None, :]),
                                axis = 2)
    

    def sobel(self, A, k):
        return np.dstack([sp.signal.convolve2d(A[:, :, c], k, mode = 'same') 
                        for c in range(A.shape[-1])])

    
    @partial(jax.vmap, in_axes = (0, 0, None, None, None))
    @jax.jit
    # def step_flow(self, x, y, A, mus):
    #     return self.step_flow_once(x, y, A, mus)
    
    # @staticmethod
    # @njit
    def step_flow(x : int, y : int, A : jnp.ndarray, mus : jnp.ndarray, pos : jnp.ndarray):
        sigma = 0.65
        rollA = jnp.roll(A, (x, y), axis=(0, 1))
        # rollA = np.roll(A, (x, y), axis=(0, 1))
        # dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (x, y), axis = (0, 1))) # (x, y, 2, c)
        dpmu = jnp.absolute(pos[..., None] - jnp.roll(mus, (x, y), axis = (0, 1))) # (x, y, 2, c)
        # sz = .5 - dpmu + self.sigma #(x, y, 2, c)
        sz = .5 - dpmu + sigma #(x, y, 2, c)
        # area = jnp.prod(np.clip(sz, 0, min(1, 2 * self.sigma)) , axis = 2) / (4 * self.sigma**2) # (x, y, c)
        area = jnp.prod(jnp.clip(sz, 0, min(1, 2 * sigma)) , axis = 2) / (4 * sigma**2) # (x, y, c)
        nA = rollA * area
        return nA










sX = sY = 2**9
nC = 3
SCALE = 800 // sX
init_size = sX >> 1

connections = np.array([
        # [1, 2, 2],
        # [2, 1, 2],
        # [2, 2, 1]
            # [2, 2, 2],
            # [2, 2, 2],
            # [2, 2, 2]
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
connections = connections * 5

A0 = np.zeros((sX, sY, nC))
A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = np.random.rand(init_size, init_size, nC)

world = World(A0)
k_params = KernelParameters(connections, (40, 40))
system = System(
    world = world,
    k_params = k_params
)


pg.init()
display = pg.display.set_mode((sX * SCALE, sY * SCALE))

running = True
# total_mass = world.A.sum()
# print("Initial mass:", total_mass)
tm = .0
i = 0
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    st = time.process_time()
    result = system.next_step()
    et = time.process_time()
    tm += et - st
    i += 1
    print(tm, "CPU seconds until step n", i)
    # current_mass = world.A.sum()
    # if (total_mass != current_mass) : 
    #     print("Total mass changed:", current_mass - total_mass)
    #     total_mass = current_mass
    # print(A[:, :, 1, None].shape)
    # surf = np.stack((A[:, :, :], A[:, :, 1, None]), axis = -1)
    surf = pg.surfarray.make_surface(np.kron(np.uint8(world.A.clip(0, 1) * 255.0), np.ones((SCALE, SCALE, 1))))
    display.blit(surf, (0, 0))

    pg.display.update()

pg.quit()













