from datetime import datetime
import time
import timeit
from functools import partial
# import subprocess
# import sys
import cv2

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
import pickle







# JITCLASS?
class System():
    
    # world = None
    # k_params = None
    # C = None
    # fK = None
    # theta_A = 3
    # dd = 20
    # dt = 5
    # sigma = 0.61
    # x = None
    # y = None
    # pos = None
    # rollxs = []
    # rollys = []
    # k = np.array([
    #         [1., 0., -1.],
    #         [2., 0., -2.],
    #         [1., 0., -1.]
    # ])
    
    # spec = [
    # ('fA', np.ndarray),          # an array field
    # ('fAk', np.ndarray),               # a simple scalar field
    # ('U', np.ndarray)
    # ]


    # Initialization of values of world and kernel parameters
    def __init__(self,
                world:World = None,
                k_params = None,
                version = None,
                ) -> None:
        
        self.set_world_and_kParams(world, 
                                k_params,
                                version,
        )
    

    # Set world and/or kernel parameters
    def set_world_and_kParams(self,
                            world:World = None,
                            k_params = None,
                            version = None,
    ) -> None:
        
        
        self.world = world
        self.k_params = k_params
        self.version = version

        if self.world is None:
            self.world = World.World()
        
        if self.k_params is None:
            self.k_params = KernelParameters.KernelParameters()
        
        if self.version is None:
            self.version = "FlowLeniaModel"

        # self.m_params = m_params

        # Model = getattr(Models, m_params["version"])
        # Model = getattr(Models, self.version)
        # self.model = Model(
        #     world = self.world,
        #     k_params = self.k_params,
        #     # params = params,
        #     # m_params = self.m_params
        # )
        self.compile()

    def compile(self):

        self.k_params.compile_kernels(self.world.sX, self.world.sY)
        Model = getattr(Models, self.version)
        self.model = Model(
            world = self.world,
            k_params = self.k_params,
            # params = params,
            # m_params = self.m_params
        )
    
    def generateRandomParams(self):
        self.k_params.generateRandomParameters(seed = self.world.seed)

    
    def step(self):

        return self.model.step()





sX, sY, world, system = None, None, None, None

# world = World()
# kernel_params = KernelParameters()
system = System()

eel.init("web")



@eel.expose
def getParameters():

    data = {}

    world = system.world
    k_params = system.k_params

    data["version"] = system.version
    data["seed"] = world.seed
    data["size"] = world.sX
    data["numChannels"] = world.numChannels
    data["theta"] = world.theta
    data["dd"] = world.dd
    data["dt"] = world.dt
    data["sigma"] = world.sigma

    # print(k_params.ker_params)

    for k in 'CrmshBawT':
        temp = k_params.kernels[k]
        if type(temp) is np.ndarray:
            temp = temp.tolist()
        data[k] = temp

    # data.update(k_params.kernels)

    return data

@eel.expose
def setParameters(data):

    global system

    setNewParameters(data)
    system.compile()

# REDUCE DUPLICATE CODE FROM SETPARAMETERS FUNCTION!!!
@eel.expose
def generateKernel(data):
    
    global system
    
    setNewParameters(data)
    system.generateRandomParams()
    system.compile()

def setNewParameters(data):
    
    global system

    world = system.world
    k_params = system.k_params

    system.version = data["version"]

    # Pass through method for world and kernel params!!!
    world.seed = data["seed"]
    world.sX = data["size"]
    world.sY = data["size"]
    world.numChannels = data["numChannels"]
    world.theta = data["theta"]
    world.dd = data["dd"]
    world.dt = data["dt"]
    world.sigma = data["sigma"]

    world.generateWorld()

    max_rings = 0
    for k in 'Baw':
        temp = max(len(sublist) for sublist in data[k])
        if temp > max_rings:
            max_rings = temp
        
    for k in 'Baw':
        for B in data[k]:
            temp = [0.2] * (max_rings - len(B))
            B.extend(temp)
    
    for k in 'rmshBaw':
        k_params.kernels[k] = np.array(data[k], dtype=np.float64)
    
    for k in 'CT':
        k_params.kernels[k] = np.array(data[k], dtype=np.int64)
    
    k_params.n_kernels = len(k_params.kernels['r'])
    


@eel.expose
def compile_version(data,
):
    global sX, sY, world, system

    # params = eel.getKernelParamsFromWeb()()
    # size = params["size"]
    # seed = params["seed"]
    # numChannels = params["numChannels"]
    # ker_params = params["kernel_params"]
    # # SCALE = 800 // sX
    # sX = sY = size
    # rand_gen = np.random.RandomState(seed)
    # init_size = sX >> 2

    # connections = np.array([
    #         # [1, 2, 2],
    #         # [2, 1, 2],
    #         # [2, 2, 1]
                
    #             [1, 1, 1],
    #             [1, 1, 1],
    #             [1, 1, 1]
                
    #             # [3, 1, 4],
    #             # [2, 2, 4],
    #             # [1, 5, 1]

    #         # [5, 0, 5],
    #         # [0, 0, 0],
    #         # [5, 0, 5]
    #         ])
    # connections = connections * 1
    # A0 = np.zeros((sX, sY, numChannels))
    # A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = rand_gen.rand(init_size, init_size, numChannels)

    # # A0[sX//2-init_size//2:sX//2+init_size//2, sY//2-init_size//2:sY//2+init_size//2, :] = np.ones((init_size, init_size, nC)) * 1

    # # A0 = np.ones((A0.shape)) * 0.7

    # world = World.World(A0)

    # # model = Model(1, 2, 3, 4, 5)
    # # func = model.compile_G_function()

    # # print(func(3))
    # # print("FINN")



    
    # max_B = max(len(sublist) for sublist in ker_params['B'])
    # for B in ker_params['B']:
    #     temp = [0] * (max_B - len(B))
    #     B.extend(temp)

    # # print(ker_params)

    # for k in ker_params.keys():
    #     ker_params[k] = np.array(ker_params[k], dtype=np.float64)
    

    # k_params = KernelParameters.KernelParameters(connections,
    #                                              ker_params)

    # # Model = getattr(Models, version)

    # system = System(
    #     world = world,
    #     k_params = k_params,
    #     params = params,
    #     version = version,
    #     # m_params = m_params
    # )

@eel.expose
def step():
    
    # a = np.ones((2, 2)) * 40
    # b = np.ones((2, 2)) * 150
    # c = np.ones((2, 2)) * 210

    global system
    system.step()

@eel.expose
def getWorld():

    scl = 512 // system.world.sX
    
    a = np.uint8(system.world.A.clip(0, 1) * 255.0)
    b = np.ones((scl, scl, 1))
    res = np.kron(a, b)
    res = np.dstack((res, np.ones((system.world.sX * scl, system.world.sY * scl), dtype=np.int8) * 255))

    return res.flatten().tolist()



@eel.expose
def saveParameterState():

    data = getParameters()
    with open('LeniaParameters.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(os.path.abspath('LeniaParameters.pkl'))

@eel.expose
def loadParameterState():

    # load the dictionary from the file
    with open('LeniaParameters.pkl', 'rb') as f:
        data = pickle.load(f)
        setParameters(data)



eel.start("index.html", mode="chrome-app")



@eel.expose
def saveNStepsToVideo(nSteps):

    # global sX, sY, world, system
    obs = np.zeros((nSteps, *world.A.shape))
    obs[0] = world.A
    
    params = eel.getParameters()()
    size = params["size"]
    seed = params["seed"]
    numChannels = params["numChannels"]
    ker_params = params["kernel_params"]
    # SCALE = 800 // sX
    sX = sY = size
    rand_gen = np.random.RandomState(seed)
    init_size = sX >> 2

