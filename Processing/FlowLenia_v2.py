from datetime import datetime
import time
import timeit
from functools import partial
import subprocess
import sys
# import cv2

import jax
import jax.numpy as jnp
# import jax.scipy as jsp
from numba import jit, njit
import numba

import pygame as pg
import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
import typing as t

import os

np.random.seed(1)

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


"""
  returns step function State x Params(compiled) --> State
  
  Params :
    SX, SY : world size
    nb_k : number of kernels
    C : number of channels
    c0 : list[int] of size nb_k where c0[i] is the source channel of kernel i
    c1 : list[list[int]] of size C where c1[i] is the liste of kernel indexes targetting channel i
    dt : float integrating timestep
    dd : max distance to look for when moving matter
    sigma : half size of the Brownian motion distribution
    n : scaling parameter for alpha
    theta_A : critical mass at which alpha = 1
  Return :
    callable : array(SX, SY, C) x params --> array(SX, SY, C)
  """

# configuration = {}
# configuration["conf_1"] = {
#     "name":"Configuration 1",
#     "R":12,
#     "T":2,
#     "kernels":[
#         {"b":[1],"m":0.272,"s":0.0595,"h":0.138,"r":0.91,"c0":0,"c1":0},
#         {"b":[1],"m":0.349,"s":0.1585,"h":0.48,"r":0.62,"c0":0,"c1":0},
#         {"b":[1,1/4],"m":0.2,"s":0.0332,"h":0.284,"r":0.5,"c0":0,"c1":0},
#         {"b":[0,1],"m":0.114,"s":0.0528,"h":0.256,"r":0.97,"c0":1,"c1":1},
#         {"b":[1],"m":0.447,"s":0.0777,"h":0.5,"r":0.72,"c0":1,"c1":1},
#         {"b":[5/6,1],"m":0.247,"s":0.0342,"h":0.622,"r":0.8,"c0":1,"c1":1},
#         {"b":[1],"m":0.21,"s":0.0617,"h":0.35,"r":0.96,"c0":2,"c1":2},
#         {"b":[1],"m":0.462,"s":0.1192,"h":0.218,"r":0.56,"c0":2,"c1":2},
#         {"b":[1],"m":0.446,"s":0.1793,"h":0.556,"r":0.78,"c0":2,"c1":2},
#         {"b":[11/12,1],"m":0.327,"s":0.1408,"h":0.344,"r":0.79,"c0":0,"c1":1},
#         {"b":[3/4,1],"m":0.476,"s":0.0995,"h":0.456,"r":0.5,"c0":0,"c1":2},
#         {"b":[11/12,1],"m":0.379,"s":0.0697,"h":0.67,"r":0.72,"c0":1,"c1":0},
#         {"b":[1],"m":0.262,"s":0.0877,"h":0.42,"r":0.68,"c0":1,"c1":2},
#         {"b":[1/6,1,0],"m":0.412,"s":0.1101,"h":0.43,"r":0.82,"c0":2,"c1":0},
#         {"b":[1],"m":0.201,"s":0.0786,"h":0.278,"r":0.82,"c0":2,"c1":1}
#     ]
# }

# # globals().update(configuration["conf_1"])




# # INITIALIZE PARAMETERS
# sz_X = 2**8            # Height
# sz_Y = sz_X         # Width
# mid_X = sz_X >> 1
# mid_Y = sz_Y >> 1
# C = 3               # N of channels
# n_Ks = 20            # N of kernels
# R = 20            # Maximum Radiosize (n of cells)

# r = np.random.uniform(.001, 1., n_Ks) * R
# a = np.random.uniform(.001, 1., (n_Ks, 3))
# b = np.random.uniform(.001, 1., (n_Ks, 3))

# # INITIALIZE WORLD A
# A = np.random.rand(sz_X, sz_Y, C)


# # INITIALIZE KERNELS
# px, py = np.arange(mid_Y), np.arange(mid_Y)
# X, Y = np.meshgrid(px, py)
# pos = np.dstack((Y, X))

# D = np.linalg.norm(np.mgrid[-mid_X : mid_X, -mid_Y : mid_Y], axis=0)
# Ds = [ D / r[k] for k in range(n_Ks) ]

# def createKernels(a, w, b)

# K = np.dstack([sigmoid(-(D-1)*10) * ker_f(D, kernels["a"][k], kernels["w"][k], kernels["b"][k]) 
#                   for k, D in zip(range(n_Ks), Ds)])





# def way1():
#       Ds = [ np.linalg.norm(np.mgrid[-mid_X : mid_X, -mid_Y : mid_Y], axis=0) / 
#         (R * r[k]) for k in range(n_Ks) ]
#     # K = np.dstack([D for D in Ds2])
#     # print(K.shape)
#     # print(len(Ds2[0][0]))
#     # 1.3523329999999998
#     # 4.067824699999999

# def way2():
#     r2 = r * R
#     D = np.linalg.norm(np.mgrid[-mid_X : mid_X, -mid_Y : mid_Y], axis=0)
#     Ds = [ D / r2[k] for k in range(n_Ks) ]
#     # print(Ds.shape)
#     # Ds = D.
#     # 0.7595743000000001
#     # 

# n = 100
# print(timeit.timeit(stmt='way1()', globals=globals(), number=n))
# print(timeit.timeit(stmt='way2()', globals=globals(), number=n))

# pass








class Model():

    def __init__(self,
                 worldConfig,
                 kernelConfig,
                 growthConfig) -> None:
      
      self.worldConfig = worldConfig
      self.kernelConfig = kernelConfig
      self.growthConfig = growthConfig
    
    
    def setWorldConfig(self):
        pass
    

    def getWolrldConfig(self):
        pass


    def setKernelConfig(self):
        pass
    

    def getKernelConfig(self):
        pass
    

    def setGrowthConfig(self):
        pass
    

    def getGrowthConfig(self):
        pass
    






class World():

    A = None
    fA = None

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
        self.fA = np.fft.fft2(A, axes=(0,1))  # (x,y,c)


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
                 k_shape = (40, 40)
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

        print(self.kernels)


    # Return kernels compiled
    def compile_kernels(self, SX, SY):
        
        midX = SX >> 1
        midY = SY >> 1

        # r = self.kernels['r'] * (self.kernels['R'] + 15)
        # D = np.linalg.norm(np.mgrid[-midX : midX, -midY : midY], axis=0)
        # Ds = [ D / r[k] for k in range(self.n_kernels) ]

        Ds = [ np.linalg.norm(np.mgrid[-midX:midX, -midY:midY], axis=0) / 
        ((self.kernels['R']+15) * self.kernels['r'][k]) for k in range(self.n_kernels) ]  # (x,y,k)

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
        i = 0
        for s in range(C):
            for t in range(C):
                n = connection_matrix[s, t]
                if n:
                    self.c0 = self.c0 + [s]*n
                    self.c1[t] = self.c1[t] + list(range(i, i + n))
                i += n



class System():
    
    world = None
    k_params = None
    C = None
    fK = None
    g_func = lambda self, x, m, s: np.exp(-((x - m) / s)**2 / 2) * 2 - 1
    theta_A = 3
    dd = 5
    dt = 0.2
    sigma = 0.65
    pos = None
    rollxs = []
    rollys = []
    k = np.array([
            [1., 0., -1.],
            [2., 0., -2.],
            [1., 0., -1.]
    ])


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

        x, y = np.arange(world.A.shape[0]), np.arange(world.A.shape[1])
        X, Y = np.meshgrid(x, y)
        self.pos = np.dstack((Y, X)) + .5 #(SX, SY, 2)

        for dx in range(-self.dd, self.dd + 1):
            for dy in range(-self.dd, self.dd + 1):
                self.rollxs.append(dx)
                self.rollys.append(dy)
        self.rollxs = np.array(self.rollxs)
        self.rollys = np.array(self.rollys)


    def next_step(self):
        
        fAk = self.world.fA[:, :, self.k_params.c0]  # (x,y,k)

        U = np.real(np.fft.ifft2(self.fK * fAk, axes=(0,1)))  # (x,y,k)

        G = self.g_func(U, self.k_params.kernels['m'], self.k_params.kernels['s']) * self.k_params.kernels['h']  # (x,y,k)

        H = np.dstack([ G[:, :, self.k_params.c1[c]].sum(axis=-1) for c in range(self.C) ])  # (x,y,c)

        #-------------------------------FLOW------------------------------------------

        F = self.compute_gradient(H) #(x, y, 2, c)

        C_grad = self.compute_gradient(self.world.A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)
    
        alpha = np.clip((self.world.A[:, :, None, :] / self.theta_A)**2, .0, 1.)
        
        F = F * (1 - alpha) - C_grad * alpha
        
        mus = self.pos[..., None] + self.dt * F #(x, y, 2, c) : target positions (distribution centers)

        self.world.new_world(self.step_flow(self.rollxs, self.rollys, self.world.A, mus).sum(axis = 0))
        
        # return self.world.A
    

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

    

    @partial(jax.vmap, in_axes = (None, 0, 0, None, None))
    def step_flow(self, rollx, rolly, A, mus):
        rollA = jnp.roll(A, (rollx, rolly), axis = (0, 1))
        dpmu = jnp.absolute(self.pos[..., None] - jnp.roll(mus, (rollx, rolly), axis = (0, 1))) # (x, y, 2, c)
        sz = .5 - dpmu + self.sigma #(x, y, 2, c)
        area = jnp.prod(np.clip(sz, 0, min(1, 2 * self.sigma)) , axis = 2) / (4 * self.sigma**2) # (x, y, c)
        nA = rollA * area
        return nA










sX = sY = 2**7
nC = 3
SCALE = 800 // sX
init_size = sX >> 1

connections = np.array([
        [1, 2, 2],
        [2, 1, 2],
        [2, 2, 1]
            # [5, 0, 5],
            # [0, 0, 0],
            # [5, 0, 5]
        ])

A0 = np.zeros((sX, sY, nC))
A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = np.random.rand(init_size, init_size, nC)

# A0 = np.ones(A0.shape) * 0.5

world = World(A0)
k_params = KernelParameters(connections)
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

surf = pg.surfarray.make_surface(np.kron(np.uint8(world.A.clip(0, 1) * 255.0), np.ones((SCALE, SCALE, 1))))
display.blit(surf, (0, 0))

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    st = time.process_time()
    result = system.next_step()
    et = time.process_time()
    tm += et - st
    i += 1
    # print(tm, "CPU seconds until step n", i)
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













