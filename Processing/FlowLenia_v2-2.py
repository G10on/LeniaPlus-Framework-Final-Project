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
            # 'T' : np.random.uniform(self.spaces['T']['low'], self.spaces['T']['high']),
            # 'R' : np.random.uniform(self.spaces['R']['low'], self.spaces['R']['high']),
            'T' : 10,
            'R' : 15,
            'init' : np.random.rand(*k_shape)
        })

        # CHANGE WAY TO INPUT CONNCTIONS MATRIX
        self.conn_from_matrix(connection_matrix)

        self.kernels = {'r': ([0.76634743, 0.47018081, 0.76636333, 0.35054701, 0.23478423,
       0.71698058, 0.88654508, 0.28661219, 0.80361973, 0.69048362,
       0.99168588, 0.9185199 , 0.90960559, 0.83702397, 0.62473215,
       0.84609996, 0.73287811, 0.78068426, 0.96204027, 0.75884059,
       0.22738918, 0.88760476, 0.74450353]), 'm': ([0.30578526, 0.37004121, 0.27404792, 0.17520817, 0.06223769,
       0.05378856, 0.2041672 , 0.18132879, 0.29821052, 0.20422586,
       0.41066778, 0.20527192, 0.05053646, 0.27669673, 0.09058977,
       0.34516137, 0.47707771, 0.43209836, 0.0800518 , 0.15124545,
       0.35701054, 0.13294558, 0.13123462]), 's': ([0.1273104 , 0.01979493, 0.00742021, 0.13996285, 0.08992389,
       0.0166036 , 0.15798597, 0.17900029, 0.10288983, 0.01190856,
       0.08613584, 0.00754177, 0.0785249 , 0.05049254, 0.14196005,
       0.07105219, 0.0661709 , 0.03776983, 0.04210903, 0.10605671,
       0.08949068, 0.03092427, 0.14983486]), 'h': ([0.20744405, 0.77932709, 0.52149892, 0.21021571, 0.90559317,
       0.4627338 , 0.61516257, 0.10015862, 0.09261503, 0.55949839,
       0.0173419 , 0.25118523, 0.18248384, 0.94326852, 0.73124612,
       0.43777952, 0.41598905, 0.93721234, 0.47642252, 0.46731499,
       0.58491531, 0.16351237, 0.65308021]), 'a': ([[0.41102545, 0.91163713, 0.91265759],
       [0.3173253 , 0.47706789, 0.10049852],
       [0.60277243, 0.20014341, 0.23484231],
       [0.45624065, 0.46619683, 0.22814414],
       [0.12310821, 0.58694277, 0.3125004 ],
       [0.0216743 , 0.10321514, 0.81731222],
       [0.02908126, 0.75066423, 0.30436572],
       [0.40506781, 0.52089929, 0.68720172],
       [0.3095452 , 0.62751229, 0.80990549],
       [0.40909965, 0.97233232, 0.64727088],
       [0.78846933, 0.00611307, 0.21690812],
       [0.57577345, 0.95340035, 0.06044492],
       [0.9670974 , 0.48219557, 0.28936779],
       [0.45337985, 0.70276921, 0.21035793],
       [0.13665509, 0.26819293, 0.81669858],
       [0.88737626, 0.10632503, 0.41537837],
       [0.69614979, 0.43995325, 0.92906265],
       [0.89481983, 0.65347796, 0.95595388],
       [0.23222683, 0.77652963, 0.70485578],
       [0.48641917, 0.35177859, 0.15594575],
       [0.63231226, 0.69078628, 0.15436898],
       [0.14196202, 0.70460984, 0.76014889],
       [0.64781681, 0.18162512, 0.11611443]]), 'w': ([[0.25958396, 0.23408195, 0.47596519],
       [0.21719383, 0.24782565, 0.06505422],
       [0.48033336, 0.15253509, 0.29911478],
       [0.43575662, 0.06340645, 0.4154093 ],
       [0.1760248 , 0.31565952, 0.49166452],
       [0.31859949, 0.45941052, 0.13682372],
       [0.0753521 , 0.23120467, 0.28584471],
       [0.01972678, 0.49579604, 0.18464371],
       [0.03345485, 0.09522233, 0.44790606],
       [0.20252579, 0.36812501, 0.08881347],
       [0.31058824, 0.38406085, 0.35629509],
       [0.41023314, 0.11951324, 0.09173535],
       [0.4412097 , 0.18657235, 0.27601308],
       [0.41063081, 0.09044755, 0.22790408],
       [0.19587172, 0.08777261, 0.37353615],
       [0.49762478, 0.02564848, 0.02599246],
       [0.1070137 , 0.38425391, 0.49888224],
       [0.2934873 , 0.11720942, 0.42628028],
       [0.22847716, 0.05197931, 0.37196289],
       [0.05047364, 0.0567179 , 0.46546752],
       [0.40931395, 0.49198576, 0.10936856],
       [0.36037639, 0.22606052, 0.38418538],
       [0.15186424, 0.19420158, 0.37825103]]), 'b': ([[0.67987203, 0.14778322, 0.02230689],
       [0.20746478, 0.47594238, 0.90156335],
       [0.01926032, 0.07181395, 0.52827832],
       [0.01370842, 0.99856507, 0.49013051],
       [0.82388616, 0.9669976 , 0.24216188],
       [0.22089031, 0.96999942, 0.34116525],
       [0.20776791, 0.17323825, 0.30790824],
       [0.42688787, 0.11607953, 0.37588191],
       [0.78273642, 0.38657138, 0.30169999],
       [0.71555726, 0.90253242, 0.79517908],
       [0.62902226, 0.16469695, 0.74919975],
       [0.59279327, 0.28430671, 0.48160149],
       [0.7770337 , 0.22757876, 0.07628881],
       [0.67347186, 0.41190637, 0.43331628],
       [0.88760838, 0.05479054, 0.27137366],
       [0.17110844, 0.10362615, 0.60058912],
       [0.29941361, 0.61968778, 0.93871287],
       [0.74297581, 0.90381906, 0.04803833],
       [0.85247177, 0.10174394, 0.15638686],
       [0.73739656, 0.08422347, 0.51559688],
       [0.10837292, 0.4866398 , 0.31235345],
       [0.25842798, 0.74578272, 0.71790413],
       [0.78436448, 0.50016297, 0.92471911]]), 'T': 44.70092412253771, 'R': 18.297515664732828, 'init': ([[0.67210471, 0.43734521, 0.27182378, ..., 0.36218003, 0.89516175,
        0.02337612],
       [0.34673577, 0.97932563, 0.02639627, ..., 0.37993803, 0.17095071,
        0.51737923],
       [0.82548041, 0.4002493 , 0.39591735, ..., 0.16970434, 0.28008008,
        0.31439595],
       ...,
       [0.51802666, 0.40278894, 0.77316723, ..., 0.38830724, 0.58342473,
        0.01509933],
       [0.00826243, 0.7430714 , 0.32965287, ..., 0.72732346, 0.89497832,
        0.53893476],
       [0.78029378, 0.25481559, 0.13421271, ..., 0.88331053, 0.46084115,
        0.36253365]])}

        # print(self.kernels)


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
        # [1, 2, 2],
        # [2, 1, 2],
        # [2, 2, 1]
            [3, 1, 4],
            [2, 2, 4],
            [1, 5, 1]
            # [5, 0, 5],
            # [0, 0, 0],
            # [5, 0, 5]
        ])

A0 = np.zeros((sX, sY, nC))
A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = np.random.rand(init_size, init_size, nC)

A0 = np.ones(A0.shape) * 0.5

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













