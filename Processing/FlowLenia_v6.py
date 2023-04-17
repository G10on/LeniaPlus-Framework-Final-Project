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

from models import Models
from objects import World, KernelParameters, ModelParameters
# import models.FlowLeniaModel as FlowLeniaModel
# from models.LeniaModel import LeniaModel

import eel







# JITCLASS?
class System():
    
    world = None
    k_params = None
    C = None
    fK = None
    theta_A = 3
    dd = 20
    dt = 5
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
                version,
                params,
                # C = None,
                # theta_A = 3,
                # dd = 20,
                # dt = 5,
                # sigma = 0.61,
                ) -> None:
        
        self.set_world_and_kParams(world, 
                                k_params, 
                                version,
                                params,
                                # C,
                                # theta_A,
                                # dd,
                                # dt,
                                # sigma,
        )
    

    # Set world and/or kernel parameters
    def set_world_and_kParams(self,
                            world:World,
                            k_params,
                            version,
                            params
                            # C = None,
                            # fK = None,
                            # theta_A = 3,
                            # dd = 20,
                            # dt = 5,
                            # sigma = 0.61,
    ) -> None:
        
        
        self.world = world
        self.k_params = k_params
        # self.m_params = m_params

        # Model = getattr(Models, m_params["version"])
        Model = getattr(Models, version)
        self.model = Model(
            world = self.world,
            k_params = self.k_params,
            params = params,
            # m_params = self.m_params
        )

    
    def step(self):

        return self.model.step()





sX, sY, world, system = None, None, None, None

eel.init("web")


@eel.expose
def compile_version(version,
):
    global sX, sY, world, system
    params = eel.getParameters()()
    size = params["size"]
    seed = params["seed"]
    numChannels = params["numChannels"]
    ker_params = params["kernel_params"]
    # SCALE = 800 // sX
    sX = sY = size
    rand_gen = np.random.RandomState(seed)
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
    connections = connections * 1
    A0 = np.zeros((sX, sY, numChannels))
    A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = rand_gen.rand(init_size, init_size, numChannels)

    # A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = np.ones((init_size, init_size, nC)) * 1

    # A0 = np.ones((A0.shape)) * 0.7

    world = World.World(A0)

    # model = Model(1, 2, 3, 4, 5)
    # func = model.compile_G_function()

    # print(func(3))
    # print("FINN")



    
    for k in ker_params.keys():
        ker_params[k] = np.array(ker_params[k], dtype=np.float64)
    
    # print(ker_params)

    k_params = KernelParameters.KernelParameters(connections,
                                                 ker_params)

    # Model = getattr(Models, version)

    system = System(
        world = world,
        k_params = k_params,
        version = version,
        params = params,
        # m_params = m_params
    )

@eel.expose
def step():
    
    # a = np.ones((2, 2)) * 40
    # b = np.ones((2, 2)) * 150
    # c = np.ones((2, 2)) * 210

    global system
    system.step()

@eel.expose
def getWorld():

    scl = 256 // sX
    
    a = np.uint8(world.A.clip(0, 1) * 255.0)
    b = np.ones((scl, scl, 1))
    res = np.kron(a, b)
    res = np.dstack((res, np.ones((sX * scl, sY * scl), dtype=np.int8) * 255))

    return res.flatten().tolist()

eel.start("index.html", mode="chrome-app")







