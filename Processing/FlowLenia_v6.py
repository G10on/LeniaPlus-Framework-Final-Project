from datetime import datetime
import os
import time
import timeit
from functools import partial
# import subprocess
# import sys
# import cv2

# import cupy

# import jax
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

from models import Models
from objects import World, KernelParameters, ModelParameters
# import models.FlowLeniaModel as FlowLeniaModel
# from models.LeniaModel import LeniaModel

import eel
import pickle


# from jax.config import config
# config.parse_flags_with_absl()
# config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", False)
# config.update("jax_debug_nans", False)
# config.update("jax_disable_pmap", False)
# config.update("jax_platform_name", "cpu")
# config.update("jax_allow_unregistered_dialects", True)





# JITCLASS?
class System():
    
    # Initialization of values of world and kernel parameters
    def __init__(self,
                world:World = None,
                # k_params = None,
                version = None,
                ) -> None:
        
        self.set_world_and_kParams(world, 
                                # k_params,
                                version,
        )
    

    # Set world and/or kernel parameters
    def set_world_and_kParams(self,
                            world:World = None,
                            version = None,
    ) -> None:
        
        
        # self.world = world

        # data = {"world": None, 
        #         "version": version}
        self.setData()
        self.compile()
        return

        self.version = version

        if self.world is None:
            self.world = World.World()
        
        # if self.k_params is None:
        #     self.k_params = KernelParameters.KernelParameters()
        
        if self.version is None:
            self.version = "FlowLeniaModel"

        Model = getattr(Models, self.version)
        self.model = Model(
            world = self.world,
            # params = params,
            # m_params = self.m_params
        )
        self.k_params = self.model.getKParams()
        
        # self.compile()

    def compile(self):

        # Model = getattr(Models, self.version)
        # self.model = Model(
        #     world = self.world,
        #     # params = params,
        #     # m_params = self.m_params
        # )
        # self.k_params = self.model.getKParams()
        # self.k_params.compile_kernels(self.world.sX, self.world.sY)
        self.model.compile()
    
    def generateRandomParams(self):
        self.k_params.generateRandomParameters(seed = self.world.seed)

    
    def step(self):

        return self.model.step()
    
    def getData(self):

        data = {}

        data["version"] = self.version
        data["seed"] = self.world.seed
        data["size"] = self.world.sX
        data["numChannels"] = self.world.numChannels
        data["theta"] = self.world.theta
        data["dd"] = self.world.dd
        data["dt"] = self.world.dt
        data["sigma"] = self.world.sigma

        # print(k_params.ker_params)

        for k in 'CrmshBawT':
            temp = self.k_params.kernels[k]
            if type(temp) is np.ndarray:
                temp = temp.tolist()
            data[k] = temp

        # data.update(k_params.kernels)

        return data

    def setData(self, data = None):

        print("START of data to set")
        # print(data)
        print("END  of data to set")

        if data == None:

            # data = getParameters()!!!

            self.world = World.World()
            # self.world.generateWorld()
            self.version = "FlowLeniaModel"
            Model = getattr(Models, self.version)
            self.model = Model(
                world = self.world,
                # params = params,
                # m_params = self.m_params
            )
            self.k_params = self.model.getKParams()
            # self.system.compile()
            return

        # if self.k_params is None:
        #     self.k_params = KernelParameters.KernelParameters()


        self.version = data["version"]
        # self.world = data["world"]

        # if data["world"] is None:
        #     self.world = World.World()
        
        # if self.k_params is None:
        #     self.k_params = KernelParameters.KernelParameters()

        # Model = getattr(Models, self.version)
        # self.model = Model(
        #     world = self.world,
        #     # params = params,
        #     # m_params = self.m_params
        # )
        # self.k_params = self.model.getKParams()

        # world = self.world
        # k_params = self.k_params


        # Pass through method for kernel params!!!

        self.world.new_world(data, 
                        # seed = data["seed"], 
                        # size = data["size"], 
                        # numChannels = data["numChannels"], 
                        # theta = data["theta"], 
                        # dd = data["dd"], 
                        # dt = data["dt"], 
                        # sigma = data["sigma"]
                        )

        # max_rings = 0
        # for k in 'Baw':
        #     temp = max(len(sublist) for sublist in data[k])
        #     if temp > max_rings:
        #         max_rings = temp
            
        # for k in 'Baw':
        #     for B in data[k]:
        #         temp = [0] * (max_rings - len(B))
        #         B.extend(temp)
        
        # for k in 'rmshBaw':
        #     k_params.kernels[k] = np.array(data[k], dtype=np.float64)
        
        # for k in 'C':
        #     k_params.kernels[k] = np.array(data[k], dtype=np.int64)

        # k_params.kernels['T'] = data['T']
        # k_params.kernels['R'] = 13
        
        # k_params.n_kernels = len(k_params.kernels['r'])

        Model = getattr(Models, self.version)
        self.model = Model(
            world = self.world,
            # params = params,
            # m_params = self.m_params
        )
        self.k_params = self.model.getKParams()

        self.k_params.new_params(data, 
                            # data['m'],
                            # data['s'],
                            # data['h'],
                            # data['B'],
                            # data['a'],
                            # data['w'],
                            # data['C'],
                            # data['T'],
                            # data['R']
                            )
        # self.k_params.compile_kernels(self.world.sX, self.world.sY)
        # self.system.compile()

    # Getter and setters for world and k_params!!!



# world = World()
# kernel_params = KernelParameters()
system = System()

eel.init("web")



@eel.expose
def getParameters():

    data = system.getData()

    return data

@eel.expose
def setParameters(data):

    global system

    system.setData(data)
    system.compile()

# REDUCE DUPLICATE CODE FROM SETPARAMETERS FUNCTION!!!
@eel.expose
def generateKernel(data):
    
    global system
    
    system.setData(data)
    system.generateRandomParams()
    system.compile()


@eel.expose
def step():

    global system
    system.step()

@eel.expose
def getWorld():

    # scl = 512 // system.world.sX
    
    # a = jnp.uint8(system.world.A.clip(0, 1) * 255.0)
    # b = np.ones((scl, scl, 1))
    # res = np.kron(a, b)
    # res = np.dstack((res, np.ones((system.world.sX * scl, system.world.sY * scl), dtype=np.int8) * 255))

    res = jnp.dstack((system.world.A.clip(0, 1), np.ones((system.world.sX, system.world.sY))))
    res = jnp.uint8(res * 255.0).tolist()

    # res = res.flatten().tolist()
    return res






def sample(new_data):

    pattern = {}
    pattern["aquarium"] = {"name":"Tessellatium gyrans","R":12,"T":2,"kernels":[
  {"b":[1],"m":0.272,"s":0.0595,"h":0.138,"r":0.91,"c0":0,"c1":0},
  {"b":[1],"m":0.349,"s":0.1585,"h":0.48,"r":0.62,"c0":0,"c1":0},
  {"b":[1,1/4],"m":0.2,"s":0.0332,"h":0.284,"r":0.5,"c0":0,"c1":0},
  {"b":[0,1],"m":0.114,"s":0.0528,"h":0.256,"r":0.97,"c0":1,"c1":1},
  {"b":[1],"m":0.447,"s":0.0777,"h":0.5,"r":0.72,"c0":1,"c1":1},
  {"b":[5/6,1],"m":0.247,"s":0.0342,"h":0.622,"r":0.8,"c0":1,"c1":1},
  {"b":[1],"m":0.21,"s":0.0617,"h":0.35,"r":0.96,"c0":2,"c1":2},
  {"b":[1],"m":0.462,"s":0.1192,"h":0.218,"r":0.56,"c0":2,"c1":2},
  {"b":[1],"m":0.446,"s":0.1793,"h":0.556,"r":0.78,"c0":2,"c1":2},
  {"b":[11/12,1],"m":0.327,"s":0.1408,"h":0.344,"r":0.79,"c0":0,"c1":1},
  {"b":[3/4,1],"m":0.476,"s":0.0995,"h":0.456,"r":0.5,"c0":0,"c1":2},
  {"b":[11/12,1],"m":0.379,"s":0.0697,"h":0.67,"r":0.72,"c0":1,"c1":0},
  {"b":[1],"m":0.262,"s":0.0877,"h":0.42,"r":0.68,"c0":1,"c1":2},
  {"b":[1/6,1,0],"m":0.412,"s":0.1101,"h":0.43,"r":0.82,"c0":2,"c1":0},
  {"b":[1],"m":0.201,"s":0.0786,"h":0.278,"r":0.82,"c0":2,"c1":1}],
  "cells":[
  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.04,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0.49,1.0,0,0.03,0.49,0.49,0.28,0.16,0.03,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0.6,0.47,0.31,0.58,0.51,0.35,0.28,0.22,0,0,0,0,0], [0,0,0,0,0,0,0.15,0.32,0.17,0.61,0.97,0.29,0.67,0.59,0.88,1.0,0.92,0.8,0.61,0.42,0.19,0,0,0], [0,0,0,0,0,0,0,0.25,0.64,0.26,0.92,0.04,0.24,0.97,1.0,1.0,1.0,1.0,0.97,0.71,0.33,0.12,0,0], [0,0,0,0,0,0,0,0.38,0.84,0.99,0.78,0.67,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.95,0.62,0.37,0,0], [0,0,0,0,0.04,0.11,0,0.69,0.75,0.75,0.91,1.0,1.0,0.89,1.0,1.0,1.0,1.0,1.0,1.0,0.81,0.42,0.07,0], [0,0,0,0,0.44,0.63,0.04,0,0,0,0.11,0.14,0,0.05,0.64,1.0,1.0,1.0,1.0,1.0,0.92,0.56,0.23,0], [0,0,0,0,0.11,0.36,0.35,0.2,0,0,0,0,0,0,0.63,1.0,1.0,1.0,1.0,1.0,0.96,0.49,0.26,0], [0,0,0,0,0,0.4,0.37,0.18,0,0,0,0,0,0.04,0.41,0.52,0.67,0.82,1.0,1.0,0.91,0.4,0.23,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.04,0,0.05,0.45,0.89,1.0,0.66,0.35,0.09,0], [0,0,0.22,0,0,0,0.05,0.36,0.6,0.13,0.02,0.04,0.24,0.34,0.1,0,0.04,0.62,1.0,1.0,0.44,0.25,0,0], [0,0,0,0.43,0.53,0.58,0.78,0.9,0.96,1.0,1.0,1.0,1.0,0.71,0.46,0.51,0.81,1.0,1.0,0.93,0.19,0.06,0,0], [0,0,0,0,0.23,0.26,0.37,0.51,0.71,0.89,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.42,0.06,0,0,0], [0,0,0,0,0.03,0,0,0.11,0.35,0.62,0.81,0.93,1.0,1.0,1.0,1.0,1.0,0.64,0.15,0,0,0,0,0], [0,0,0,0,0,0,0.06,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0.05,0.09,0.05,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.02,0.28,0.42,0.44,0.34,0.18,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.34,1.0,1.0,1.0,1.0,1.0,0.91,0.52,0.14,0], [0,0,0,0,0,0,0,0,0,0,0,0,0.01,0.17,0.75,1.0,1.0,1.0,1.0,1.0,1.0,0.93,0.35,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.22,0.92,1.0,1.0,1.0,1.0,1.0,1.0,0.59,0.09], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.75,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.71,0.16], [0,0,0,0,0,0,0,0,0,0,0,0,0.01,0.67,0.83,0.85,1.0,1.0,1.0,1.0,1.0,1.0,0.68,0.17], [0,0,0,0,0,0,0,0,0,0,0,0,0.21,0.04,0.12,0.58,0.95,1.0,1.0,1.0,1.0,1.0,0.57,0.13], [0,0,0,0,0,0,0,0,0,0,0,0.07,0,0,0,0.2,0.64,0.96,1.0,1.0,1.0,0.9,0.24,0.01], [0,0,0,0,0,0,0,0,0,0,0.13,0.29,0,0,0,0.25,0.9,1.0,1.0,1.0,1.0,0.45,0.05,0], [0,0,0,0,0,0,0,0,0,0,0.13,0.31,0.07,0,0.46,0.96,1.0,1.0,1.0,1.0,0.51,0.12,0,0], [0,0,0,0,0,0,0,0,0.26,0.82,1.0,0.95,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.3,0.05,0,0,0], [0,0,0,0,0,0,0,0,0.28,0.74,1.0,0.95,0.87,1.0,1.0,1.0,1.0,1.0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0.07,0.69,1.0,1.0,1.0,1.0,1.0,0.96,0.25,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0.4,0.72,0.9,0.83,0.7,0.56,0.43,0.14,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0.04,0.25,0.37,0.44,0.37,0.24,0.11,0.04,0,0,0,0], [0,0,0,0,0,0,0,0,0,0.19,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.75,0.4,0.15,0,0,0,0], [0,0,0,0,0,0,0,0,0.14,0.48,0.83,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.4,0,0,0,0], [0,0,0,0,0,0,0,0,0.62,0.78,0.94,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.64,0,0,0,0], [0,0,0,0,0,0,0,0.02,0.65,0.98,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.78,0,0,0,0], [0,0,0,0,0,0,0,0.15,0.48,0.93,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.79,0.05,0,0,0], [0,0,0,0,0,0,0.33,0.56,0.8,1.0,1.0,1.0,0.37,0.6,0.94,1.0,1.0,1.0,1.0,0.68,0.05,0,0,0], [0,0,0,0,0.35,0.51,0.76,0.89,1.0,1.0,0.72,0.15,0,0.29,0.57,0.69,0.86,1.0,0.92,0.49,0,0,0,0], [0,0,0,0,0,0.38,0.86,1.0,1.0,0.96,0.31,0,0,0,0,0.02,0.2,0.52,0.37,0.11,0,0,0,0], [0,0,0.01,0,0,0.07,0.75,1.0,1.0,1.0,0.48,0.03,0,0,0,0,0,0.18,0.07,0,0,0,0,0], [0,0.11,0.09,0.22,0.15,0.32,0.71,0.94,1.0,1.0,0.97,0.54,0.12,0.02,0,0,0,0,0,0,0,0,0,0], [0.06,0.33,0.47,0.51,0.58,0.77,0.95,1.0,1.0,1.0,1.0,0.62,0.12,0,0,0,0,0,0,0,0,0,0,0], [0.04,0.4,0.69,0.88,0.95,1.0,1.0,1.0,1.0,1.0,0.93,0.68,0.22,0.02,0,0,0.01,0,0,0,0,0,0,0], [0,0.39,0.69,0.91,1.0,1.0,1.0,1.0,1.0,0.85,0.52,0.35,0.24,0.17,0.07,0,0,0,0,0,0,0,0,0], [0,0,0.29,0.82,1.0,1.0,1.0,1.0,1.0,1.0,0.67,0.29,0.02,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0.2,0.51,0.77,0.96,0.93,0.71,0.4,0.16,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0.08,0.07,0.03,0,0,0,0,0,0,0,0,0,0,0,0,0]]]
}
    
    entity = pattern["aquarium"]
    test = np.asarray(entity["cells"]).transpose((1, 2, 0))
    SX = new_data["size"]
    SY = SX
    C = new_data["numChannels"]

    # print("SUBARRAY:", [sublst['b'] for sublst in entity["kernels"]])
    lst_B = [sublst['b'] for sublst in entity["kernels"]]
    max_rings = max(len(sublist) for sublist in lst_B)

    # for B in lst_B:
    #     temp = [0.0] * (max_rings - len(B))
    #     B.extend(temp)
        # B.reverse()

    # for k in "ms":
        
        # temp = []
        # for kernel in entity["kernels"]:
        #     temp.append(kernel[k] * np.ones(max_rings))

        
    c1 = [[] for _ in range(C)]
    for kn in range(len(entity["kernels"])):
        c1[entity["kernels"][kn]["c1"]].append(kn)
    new_world = jnp.zeros((SX, SY, C))
    # print(new_world[SX//2-(test.shape[0])//2 :SX//2+(test.shape[0])//2, SY//2-(test.shape[1])//2:SY//2+(test.shape[1])//2, :].shape, test.shape
    new_world = new_world.at[SX//2-(test.shape[0] + 1)//2 :SX//2+(test.shape[0])//2, SY//2-(test.shape[1])//2:SY//2+(test.shape[1] + 1)//2, :].set(test)
    
    
    for k in "rmsh":
        temp = []
        for kernel in entity["kernels"]:
            temp.append(kernel[k])
        new_data[k] = np.asarray(temp, dtype=np.float64)

    # new_data["dt"] = entity['T'] * 0.05
    new_data["world"] = new_world
    new_data['B'] = lst_B
    # print(new_data['B'])
    new_data['a'] = 0.5 * np.ones((len(entity["kernels"]), max_rings))
    new_data['a'] = new_data['a'].tolist()
    new_data['w'] = 0.15 * np.ones((len(entity["kernels"]), max_rings))
    new_data['w'] = new_data['w'].tolist()
    new_data['C'] = [kernel["c0"] for kernel in entity["kernels"]]
    new_data['T'] = c1
    new_data['version'] = "LeniaModel"

    return new_data









@eel.expose
def saveParameterState():

    data = getParameters()
    # data["world"] = system.world.A
    
    data = sample(data)

    with open('LeniaParameters.pkl', 'wb') as f:

        pickle.dump(data, f)
    
    print(os.path.abspath('LeniaParameters.pkl'))

@eel.expose
def loadParameterState():

    # load the dictionary from the file
    with open('LeniaParameters.pkl', 'rb') as f:
        data = pickle.load(f)
    
    setParameters(data)
    # system.world.A = data["world"]



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

