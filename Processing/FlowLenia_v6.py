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
        print(data)
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
    pattern["emitter"] = {"name":"Smooth glider gun","R":13,"T":2,"kernels":[
  {"b":[1],"m":0.184,"s":0.0632,"h":0.076,"r":0.56,"c0":0,"c1":0},
  {"b":[1],"m":0.1,"s":0.1511,"h":0.516,"r":0.76,"c0":0,"c1":0},
  {"b":[1],"m":0.246,"s":0.047,"h":0.554,"r":0.5,"c0":0,"c1":0},
  {"b":[1/12,1],"m":0.1,"s":0.0553,"h":0.294,"r":0.84,"c0":1,"c1":1},
  {"b":[1],"m":0.324,"s":0.0782,"h":0.594,"r":0.97,"c0":1,"c1":1},
  {"b":[5/6,1],"m":0.229,"s":0.0321,"h":0.612,"r":0.98,"c0":1,"c1":1},
  {"b":[1],"m":0.29,"s":0.0713,"h":0.396,"r":0.87,"c0":2,"c1":2},
  {"b":[1],"m":0.484,"s":0.1343,"h":0.244,"r":0.96,"c0":2,"c1":2},
  {"b":[1],"m":0.592,"s":0.1807,"h":0.562,"r":0.93,"c0":2,"c1":2},
  {"b":[1],"m":0.398,"s":0.1411,"h":0.36,"r":0.89,"c0":0,"c1":1},
  {"b":[1],"m":0.388,"s":0.1144,"h":0.192,"r":0.67,"c0":0,"c1":2},
  {"b":[1,11/12,0],"m":0.312,"s":0.0697,"h":0.462,"r":0.58,"c0":1,"c1":0},
  {"b":[1],"m":0.327,"s":0.1036,"h":0.608,"r":1.0,"c0":1,"c1":2},
  {"b":[1],"m":0.471,"s":0.1176,"h":0.394,"r":0.8,"c0":2,"c1":0},
  {"b":[1,1/12],"m":0.1,"s":0.0573,"h":0.14,"r":0.62,"c0":2,"c1":1}],
  "cells":[
  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0.15,0.48,0.19,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0.61,1.00,1.00,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0.12,0.60,1.00,1.00,1.00,1.00,0,0,0,0,0,0.19,0.61,0.11,0,0,0,0,0,0], [0,0,0,0,0,0.36,0,0,0,0,0,1.00,1.00,1.00,0.72,0.40,0,0,0,0.91,1.00,0.61,0.26,0,0,0,0,0], [0,0,0,0,0.34,0.76,0.10,0,0,0,0,0,0.96,1.00,0.96,0.83,0.88,0.72,0.86,1.00,1.00,0.87,0.44,0.05,0,0,0,0], [0,0,0,0.12,0.49,0.89,0.16,0,0,0,0,0,0.52,0.96,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.50,0.03,0,0,0,0], [0,0,0,0.58,0.82,1.00,0.70,0,0,0,0,0,0,0.37,0.34,0.06,0,0.49,1.00,1.00,1.00,1.00,0.57,0.01,0,0,0,0], [0,0,0.16,0.17,0.16,0.53,0.63,0.82,0.71,0.16,0,0,0,0,0,0,0,0,0,0.52,0.45,0.07,0.37,0.05,0,0,0,0], [0,0,0,0,0,0,0.14,1.00,1.00,1.00,0.43,0,0,0.35,0.03,0,0,0,0,0,0.03,0,0,0.08,0,0,0,0], [0,0,0,0,0,0,0,1.00,1.00,1.00,1.00,1.00,1.00,0.22,0,0,0,0,0,0,0,0.07,0,0.18,0,0,0,0], [0,0,0.25,0,0,0,0,0.01,1.00,1.00,0.81,0.40,0.25,0,0,0,0,0,0,0,0,0.31,0.53,0.48,0,0,0,0], [0,0,0.63,0,0,0,0,0,0.40,1.00,0.14,0.18,0.17,0,0,0,0,0,0.07,0,0.46,1.00,1.00,0.94,0.23,0,0,0], [0,0,0.97,1.00,0,0,0,0,0,1.00,0.19,0.13,0.09,0,0,0,0,0.91,1.00,0.97,1.00,1.00,1.00,0.99,0.61,0,0,0], [0,0.22,1.00,1.00,1.00,0.58,0,0,0,0.64,0.10,0.10,0,0,0,0,0,0.92,0.73,0.73,0.88,1.00,1.00,0.27,0.36,0.11,0,0], [0,1.00,1.00,1.00,1.00,1.00,0.78,0,0.06,0.34,0,0,0,0,0,0,0,0,0,0.29,0.70,1.00,1.00,0,0,0.11,0,0], [0,0.39,0,0.78,1.00,1.00,0.88,0,0,0,0,0,0,0,0,0,0,0,0,0,0.60,1.00,0.82,0,0.01,0.02,0,0], [0,0,0,0,0.59,0.85,0.52,0,0,0,0,0,0,0,0,0,0,0,0,0,0.55,1.00,0.62,0.05,0.21,0,0,0], [0,0,0,0,0.45,0.86,1.00,0,0,0,0,0,0.48,0.93,0,0,0,0,0,0,0.92,1.00,0.74,0.37,0.10,0,0,0], [0,0,0,0,0,0.75,1.00,0.33,0,0,0,0,1.00,1.00,0.22,0,0,0,0,0.29,1.00,1.00,0.49,0.18,0,0,0,0], [0,0,0,0,0,0.88,1.00,1.00,0.27,0,0,0,0.61,1.00,0.54,0.27,0,0,0.38,1.00,1.00,0.60,0.24,0.02,0,0,0,0], [0,0,0.43,0,0.95,1.00,1.00,1.00,0.12,0,0,0,0.66,1.00,0.84,0.75,0.78,0.90,1.00,1.00,0.65,0.27,0.14,0,0,0,0,0], [0,0,0,0.97,1.00,1.00,1.00,1.00,0,0,0.07,0.43,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.58,0.27,0.17,0,0,0,0,0,0], [0,0,0,0.09,0.48,0.87,1.00,1.00,0,0,0,0,1.00,1.00,1.00,0.81,0.33,0.29,0.49,0.30,0.12,0,0,0,0,0,0,0], [0,0,0,0,0.14,0.22,0.21,0.19,0.12,0,0,0.45,0.98,1.00,0.42,0,0,0.09,0.28,0.05,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0.19,0.60,0.50,0.14,0.16,0.19,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0.06,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0.08,0.21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0.81,0.98,0.85,0.67,0.45,0.69,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0.35,0,0,0,0,0.64,1.00,1.00,1.00,1.00,1.00,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0.06,0.36,0,0,0,0,0.28,0.50,0.46,0.91,1.00,1.00,0.82,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0.12,0.13,0.06,0.05,0.05,0.03,0,0.36,0.56,0.60,1.00,1.00,1.00,0.62,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0.18,0.35,0.35,1.00,0.07,0,0.47,0.77,1.00,1.00,1.00,1.00,0.14,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0.16,0.38,0,0,0.37,0.29,0.34,0.90,1.00,1.00,1.00,1.00,0.34,0,0,0,0,0,0], [0,0,0,0,0,0.05,0,0,0,0.11,1.00,0,0,0,0,0.50,1.00,1.00,1.00,1.00,1.00,0.43,0,0,0,0,0,0], [0,0,0,0,0,0,0.86,0,0,0.01,0.19,0.82,0,0,0,0.52,1.00,1.00,1.00,1.00,0.97,0.48,0,0,0,0,0,0], [0,0,0,0,0,0,0.73,0.73,0.30,0.32,0,0.21,0.10,0,0.07,0.98,1.00,1.00,1.00,1.00,0.85,0.50,0,0,0,0,0,0], [0,0,0,0,0,0,0.30,1.00,0.26,0.44,0.57,0.36,0.49,0.69,1.00,1.00,1.00,1.00,1.00,1.00,0.86,0.45,0,0,0,0,0,0], [0,0,0,0,0,0,0.12,0.95,0.71,0.26,0.55,0.78,0.93,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.27,0,0,0,0,0,0], [0,0,0,0,0,0,0,0.83,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.93,0.06,0,0,0,0,0,0], [0,0,0,0,0,0,0,0.38,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.53,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0.04,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.68,0.05,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0.56,0.85,0.95,0.97,0.92,0.82,0.82,0.69,0.32,0.02,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0.03,0.12,0.11,0.04,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],
  [[0,0,0,0,0,0,0.01,0.02,0.03,0.04,0.06,0.08,0.09,0.10,0.10,0.10,0.10,0.08,0.05,0.03,0.02,0.01,0,0,0,0,0,0], [0,0,0,0,0.01,0.01,0.03,0.05,0.08,0.11,0.13,0.15,0.17,0.18,0.20,0.22,0.21,0.19,0.15,0.10,0.06,0.04,0.02,0.01,0,0,0,0], [0,0,0,0.01,0.02,0.04,0.08,0.12,0.17,0.20,0.22,0.24,0.26,0.28,0.30,0.33,0.33,0.31,0.26,0.20,0.14,0.09,0.06,0.03,0.01,0,0,0], [0,0,0.01,0.03,0.05,0.09,0.15,0.22,0.28,0.31,0.32,0.32,0.33,0.34,0.37,0.40,0.42,0.41,0.36,0.30,0.23,0.17,0.11,0.07,0.03,0.01,0,0], [0,0.01,0.03,0.06,0.10,0.16,0.24,0.31,0.37,0.40,0.39,0.37,0.36,0.37,0.39,0.42,0.44,0.44,0.42,0.38,0.33,0.26,0.19,0.13,0.07,0.03,0.01,0], [0.01,0.02,0.05,0.10,0.16,0.23,0.30,0.38,0.42,0.43,0.43,0.41,0.39,0.39,0.40,0.42,0.43,0.43,0.42,0.41,0.40,0.35,0.28,0.20,0.12,0.06,0.02,0], [0.01,0.04,0.09,0.15,0.23,0.29,0.34,0.39,0.42,0.43,0.44,0.45,0.45,0.45,0.45,0.45,0.44,0.43,0.42,0.43,0.43,0.42,0.37,0.29,0.19,0.10,0.04,0.01], [0.03,0.07,0.13,0.21,0.28,0.32,0.35,0.39,0.42,0.45,0.48,0.51,0.53,0.54,0.53,0.52,0.50,0.48,0.46,0.45,0.46,0.47,0.45,0.37,0.26,0.15,0.07,0.03], [0.04,0.10,0.19,0.27,0.33,0.35,0.36,0.39,0.44,0.50,0.57,0.62,0.64,0.65,0.63,0.61,0.59,0.56,0.54,0.52,0.51,0.52,0.52,0.45,0.34,0.21,0.11,0.05], [0.06,0.14,0.25,0.34,0.39,0.39,0.39,0.42,0.48,0.56,0.65,0.71,0.74,0.74,0.72,0.69,0.66,0.63,0.60,0.58,0.56,0.56,0.56,0.51,0.41,0.27,0.15,0.07], [0.09,0.19,0.30,0.39,0.42,0.43,0.44,0.48,0.55,0.64,0.69,0.73,0.75,0.77,0.77,0.76,0.73,0.69,0.66,0.63,0.60,0.58,0.58,0.55,0.46,0.32,0.19,0.09], [0.11,0.21,0.33,0.41,0.43,0.44,0.48,0.55,0.63,0.69,0.69,0.66,0.66,0.69,0.73,0.76,0.76,0.73,0.71,0.67,0.63,0.59,0.58,0.56,0.49,0.36,0.22,0.11], [0.13,0.22,0.33,0.40,0.43,0.45,0.50,0.60,0.69,0.72,0.66,0.59,0.56,0.58,0.64,0.70,0.73,0.73,0.71,0.68,0.64,0.59,0.57,0.56,0.51,0.40,0.25,0.13], [0.14,0.23,0.32,0.37,0.41,0.45,0.52,0.63,0.73,0.74,0.67,0.57,0.52,0.52,0.57,0.65,0.71,0.72,0.71,0.68,0.63,0.59,0.56,0.56,0.53,0.43,0.27,0.14], [0.14,0.24,0.31,0.35,0.38,0.43,0.51,0.62,0.73,0.77,0.71,0.60,0.54,0.53,0.57,0.65,0.71,0.73,0.72,0.68,0.64,0.59,0.57,0.57,0.55,0.44,0.28,0.15], [0.15,0.24,0.32,0.35,0.37,0.41,0.50,0.60,0.70,0.77,0.76,0.67,0.60,0.58,0.62,0.69,0.75,0.76,0.74,0.70,0.65,0.60,0.58,0.58,0.55,0.45,0.28,0.14], [0.15,0.25,0.33,0.37,0.38,0.41,0.48,0.57,0.66,0.74,0.77,0.73,0.68,0.66,0.70,0.75,0.77,0.77,0.76,0.72,0.66,0.61,0.59,0.59,0.55,0.43,0.26,0.13], [0.15,0.26,0.35,0.40,0.41,0.41,0.46,0.54,0.62,0.69,0.74,0.74,0.72,0.72,0.75,0.77,0.77,0.76,0.75,0.72,0.66,0.61,0.59,0.59,0.53,0.39,0.23,0.11], [0.13,0.25,0.36,0.42,0.43,0.42,0.45,0.51,0.58,0.65,0.70,0.72,0.72,0.72,0.75,0.77,0.76,0.74,0.73,0.70,0.64,0.60,0.59,0.56,0.48,0.34,0.19,0.09], [0.11,0.22,0.33,0.41,0.43,0.42,0.43,0.48,0.55,0.61,0.67,0.69,0.69,0.70,0.72,0.74,0.74,0.73,0.70,0.66,0.61,0.58,0.57,0.52,0.42,0.28,0.15,0.06], [0.07,0.17,0.28,0.36,0.40,0.41,0.42,0.46,0.52,0.58,0.63,0.66,0.66,0.66,0.67,0.69,0.70,0.69,0.65,0.61,0.58,0.56,0.54,0.47,0.35,0.21,0.11,0.04], [0.04,0.11,0.20,0.29,0.36,0.39,0.42,0.45,0.51,0.55,0.60,0.62,0.62,0.61,0.62,0.63,0.64,0.63,0.61,0.58,0.56,0.54,0.48,0.39,0.27,0.15,0.07,0.02], [0.02,0.07,0.13,0.21,0.30,0.36,0.41,0.46,0.51,0.55,0.57,0.59,0.58,0.57,0.57,0.58,0.59,0.59,0.59,0.57,0.54,0.49,0.41,0.30,0.19,0.10,0.04,0.01], [0.01,0.03,0.08,0.15,0.23,0.31,0.38,0.44,0.51,0.55,0.57,0.58,0.56,0.55,0.55,0.56,0.57,0.58,0.57,0.54,0.48,0.41,0.31,0.21,0.12,0.05,0.02,0], [0,0.02,0.05,0.09,0.15,0.23,0.30,0.38,0.46,0.52,0.55,0.56,0.56,0.55,0.56,0.56,0.57,0.56,0.52,0.46,0.38,0.29,0.20,0.12,0.06,0.02,0,0], [0,0.01,0.02,0.05,0.09,0.15,0.21,0.28,0.36,0.42,0.46,0.49,0.50,0.51,0.52,0.52,0.50,0.47,0.41,0.34,0.26,0.18,0.11,0.06,0.02,0.01,0,0], [0,0,0.01,0.02,0.04,0.08,0.12,0.17,0.23,0.28,0.32,0.35,0.38,0.39,0.40,0.39,0.37,0.32,0.26,0.20,0.14,0.09,0.05,0.02,0.01,0,0,0], [0,0,0,0,0.01,0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.23,0.24,0.25,0.23,0.21,0.18,0.14,0.10,0.06,0.04,0.02,0,0,0,0,0], [0,0,0,0,0,0.01,0.02,0.03,0.05,0.07,0.09,0.11,0.12,0.12,0.12,0.11,0.10,0.08,0.06,0.04,0.02,0.01,0,0,0,0,0,0]]]
}
    
    
    entity = pattern["emitter"]
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

