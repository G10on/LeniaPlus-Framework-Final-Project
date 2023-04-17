import time
import pygame as pg

from functools import partial
import subprocess
import sys
# import cv2

import numba as nb

import jax
import jax.numpy as jnp
# import jax.scipy as np

# import jax.numpy as np
import numpy as np
import scipy as sp
# import matplotlib.pyplot as plt
import typing as t

import os


np.random.seed(1)



kx = np.array([
                [1., 0., -1.],
                [2., 0., -2.],
                [1., 0., -1.]
])
ky = np.transpose(kx)
def sobel_x(A):
  """
  A : (x, y, c)
  ret : (x, y, c)
  """
  return np.dstack([sp.signal.convolve2d(A[:, :, c], kx, mode = 'same') 
                    for c in range(A.shape[-1])])
def sobel_y(A):
  return np.dstack([sp.signal.convolve2d(A[:, :, c], ky, mode = 'same') 
                    for c in range(A.shape[-1])])
  
# @jax.jit
def sobel(A):
  return np.concatenate((sobel_y(A)[:, :, None, :], sobel_x(A)[:, :, None, :]),
                         axis = 2)




def conn_from_matrix(mat):
  C = mat.shape[0]
  c0 = []
  c1 = [[] for _ in range(C)]
  i = 0
  for s in range(C):
    for t in range(C):
      n = mat[s, t]
      if n:
        c0 = c0 + [s]*n
        c1[t] = c1[t] + list(range(i, i+n))
      i+=n
  return c0, c1




def rollout(step_fn, c_params, A, steps, verbose = False):
  global display
    
  obs = np.zeros((steps+1, *A0.shape))
  obs[0] = A
  rnge = range(steps)

  running = True
  tm = .0
  i = 0
  # total_mass = A.sum()
  # print("Initial mass:", total_mass)
  
  surf = pg.surfarray.make_surface(np.kron(np.uint8(A.clip(0, 1) * 255.0), np.ones((SCALE, SCALE, 1))))
  display.blit(surf, (0, 0))

  while running:
    for event in pg.event.get():
      if event.type == pg.QUIT:
        running = False

    st = time.process_time()
    A = step_fn(A, c_params)
    et = time.process_time()
    tm += et - st
    i += 1
    # print(tm, "CPU seconds until step n", i)
    

    # current_mass = A.sum()
    # if (total_mass != current_mass) : 
    #   print("Total mass changed:", current_mass - total_mass)
    #   total_mass = current_mass
    # print(A[:, :, 1, None].shape)
    # surf = np.stack((A[:, :, :], A[:, :, 1, None]), axis = -1)
    surf = pg.surfarray.make_surface(np.kron(np.uint8(A.clip(0, 1) * 255.0), np.ones((SCALE, SCALE, 1))))
    display.blit(surf, (0, 0))

    pg.display.update()

  pg.quit()

  # for t in rnge :
  #   A = step_fn(A, c_params)
  #   obs[t+1] = A
  return obs




class Rule_space :
  #-----------------------------------------------------------------------------
  def __init__(self, nb_k, init_shape = (40, 40)):
    self.nb_k = nb_k
    self.init_shape = init_shape
    self.kernel_keys = 'r b w a m s h'.split()
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
        'init' : {'low' : 0., 'high' : 1., 'mut_std' : .2, 'shape' : self.init_shape}
    }
  #-----------------------------------------------------------------------------
  def sample(self):
    kernels = {}
    for k in 'rmsh':
      kernels[k] = np.random.uniform(
          self.spaces[k]['low'], self.spaces[k]['high'], self.nb_k
      )
    for k in "awb":
      kernels[k] = np.random.uniform(
          self.spaces[k]['low'], self.spaces[k]['high'], (self.nb_k, 3)
      )
    return {
        'kernels' : kernels, 
        # 'T' : np.random.uniform(self.spaces['T']['low'], self.spaces['T']['high']),
        # 'R' : np.random.uniform(self.spaces['R']['low'], self.spaces['R']['high']),
        'T' : 10,
        'R' : 15,
        # 'init' : np.random.rand(*self.init_shape) 
    }
  #-----------------------------------------------------------------------------


# @nb.njit(fastmath = True)
def sigmoid(x):
  return 0.5 * (np.tanh(x / 2) + 1)

ker_f = lambda x, a, w, b : (b * np.exp( - (x[..., None] - a)**2 / w)).sum(-1)

bell = lambda x, m, s: np.exp(-((x-m)/s)**2 / 2)

def growth(U, m, s):
  return bell(U, m, s)*2-1




# @nb.njit(fastmath = True)
def compile_kernel_computer(SX, SY, nb_k, params : dict[str, any]):
  # def sigmoid(x):
  #   return 0.5 * (np.tanh(x / 2) + 1)
  """return a jit compiled function taking as input lenia raw params and returning computed kernels (compiled params)"""
  mid = SX // 2
  """Compute kernels and return a dic containing fft kernels, T and R"""
  kernels = params['kernels']
  kernels = {'r': ([0.76634743, 0.47018081, 0.76636333, 0.35054701, 0.23478423,
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

  Ds = [ np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / 
        ((params['R']+15) * kernels['r'][k]) for k in range(nb_k) ]  # (x,y,k)
#   t0 = -(Ds[0]-1)*10
#   t1 = sigmoid(t0)
#   print(kernels["b"][0])
#   kernels["b"][0][0] = 0.0
#   kernels["b"][0][1] = 0.0
#   kernels["b"][0][2] = kernels["b"][0][1]
#   t2 = ker_f(Ds[0], kernels["a"][0], kernels["w"][0], kernels["b"][0])
  K = np.dstack([sigmoid(-(D-1)*10) * ker_f(D, kernels["a"][k], kernels["w"][k], kernels["b"][k]) 
                  for k, D in zip(range(nb_k), Ds)])
  nK = K / np.sum(K, axis=(0,1), keepdims=True)
  fK = np.fft.fft2(np.fft.fftshift(nK, axes=(0,1)), axes=(0,1))

  compiled_params = {k : kernels[k] for k in kernels.keys()}
  for k in ['R', 'T']:
    compiled_params[k] = params[k]
  compiled_params['fK'] = fK
  return compiled_params




def build_system(SX : int, SY : int, nb_k : int, C : int, c0 : t.Iterable = None, 
                 c1 : t.Iterable = None, dt : float = .5, dd : int = 5, 
                 sigma : float = .65, n = 2, theta_A = None) -> t.Callable:
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

  if theta_A is None:
    theta_A = C

  if c0 is None or c1 is None :
    c0 = [0] * nb_k
    c1 = [[i for i in range(nb_k)]]
  
  xs = np.concatenate([np.arange(SX) for _ in range(SY)])
  ys = np.arange(SY).repeat(SX)

  x, y = np.arange(SX), np.arange(SY)
  X, Y = np.meshgrid(x, y)
  pos = np.dstack((Y, X)) + .5 #(SX, SY, 2)

  rollxs = []
  rollys = []
  for dx in range(-dd, dd+1):
    for dy in range(-dd, dd+1):
      rollxs.append(dx)
      rollys.append(dy)
  rollxs = np.array(rollxs)
  rollys = np.array(rollys)


  @partial(jax.vmap, in_axes = (0, 0, None, None))
  def step_flow(rollx, rolly, A, mus):
    rollA = jnp.roll(A, (rollx, rolly), axis = (0, 1))
    dpmu = jnp.absolute(pos[..., None] - jnp.roll(mus, (rollx, rolly), axis = (0, 1))) # (x, y, 2, c)
    sz = .5 - dpmu + sigma #(x, y, 2, c)
    area = jnp.prod(np.clip(sz, 0, min(1, 2*sigma)) , axis = 2) / (4 * sigma**2) # (x, y, c)
    nA = rollA * area
    return nA


  def step(A, params):
    """
    Main step
    A : state of the system (SX, SY, C)
    params : compiled paremeters (dict) must contain m, s, h and fK (computed kernels fft)
    """
    #---------------------------Original Lenia------------------------------------
    fA = np.fft.fft2(A, axes=(0,1))  # (x,y,c)

    fAk = fA[:, :, c0]  # (x,y,k)

    U = np.real(np.fft.ifft2(params['fK'] * fAk, axes=(0,1)))  # (x,y,k)

    G = growth(U, params['m'], params['s']) * params['h']  # (x,y,k)

    H = np.dstack([ G[:, :, c1[c]].sum(axis=-1) for c in range(C) ])  # (x,y,c)

    #-------------------------------FLOW------------------------------------------

    F = sobel(H) #(x, y, 2, c)

    C_grad = sobel(A.sum(axis = -1, keepdims = True)) #(x, y, 2, 1)
  
    alpha = np.clip((A[:, :, None, :]/theta_A)**n, .0, 1.)
    
    F = F * (1 - alpha) - C_grad * alpha
    
    mus = pos[..., None] + dt * F #(x, y, 2, c) : target positions (distribution centers)

    nA = step_flow(rollxs, rollys, A, mus).sum(axis = 0)
    
    return nA

  return step





# SX = SY = 2**3 # World dimensions
# C = 1 # Number of channels
# nb_k = 1 # Number of kernels
# c0 = None ; c1 = None # kernel connections : by default all kernels are from channel 0 to channel 0 in 1-channel Lenia
# init_sz = 2
# rule_space = Rule_space(nb_k, init_shape= (init_sz, init_sz)) 

# step_fn = build_system(SX, SY, nb_k, C, c0, c1, dd = 1, dt = .3, sigma = .65)

# params = rule_space.sample()
# c_params = compile_kernel_computer(SX, SY, nb_k, params) # Kernel computer : raw params --> compiled params
# steps = 100
# A0 = np.zeros((SX, SY, C))
# A0[SX//2-init_sz//2:SX//2+init_sz//2, SY//2-init_sz//2:SY//2+init_sz//2, 0] = params['init']

# obs = rollout(step_fn, c_params, A0, steps, verbose = True)
# print(obs.shape)



#@title 2-channel {vertical-output : true}

DIM = 7
SX = SY = 2**DIM # World dimensions
C = 3 # Number of channels
init_size = SX >> 1

connection_matrix = np.array([
    # [1, 2, 2],
    # [2, 1, 2],
    # [2, 2, 1]
            [3, 1, 4],
            [2, 2, 4],
            [1, 5, 1]
    # [5, 0, 5],
    # [0, 0, 0],
    # [5, 0, 5]
]) # Connection matrix where M[i, j] = number of kernels from channel i to channel j

A0 = np.zeros((SX, SY, C))
A0[SX//2-init_size//2:SX//2+init_size//2, SY//2-init_size//2:SY//2+init_size//2, :] = np.random.rand(init_size, init_size, C)

A0 = np.ones(A0.shape) * 0.5

c0, c1 = conn_from_matrix(connection_matrix)
nb_k = int(connection_matrix.sum())
rule_space = Rule_space(nb_k) 

step_fn = build_system(SX, SY, nb_k, C, c0, c1, dd = 5, dt = 0.2, sigma = .65)

params = rule_space.sample()
c_params = compile_kernel_computer(SX, SY, nb_k, params)
steps = 10
# A0[:, :, 2] = np.zeros((SX, SY))


SCALE = 800 // SX
pg.init()
display = pg.display.set_mode((SX * SCALE, SY * SCALE))

obs = rollout(step_fn, c_params, A0, steps, verbose = True)




# ------------------
# width = SX
# height = SY
# fps = 10
# sec = steps // fps
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
 
# video = cv2.VideoWriter('test9.mp4', fourcc, float(fps), (width, height), C)
# # ------------------
 
# for frame_count in range(obs.shape[0]):
#     img = np.uint8(obs[frame_count].clip(0, 1) * 255)
#     # img = np.random.randint(0,255, (height, width, C), dtype = np.uint8)
#     video.write(img)
 
# video.release()