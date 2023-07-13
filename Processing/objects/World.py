import jax.numpy as jnp
import numpy as np


class World():

    # Initialize world
    def __init__(self,
                 A=None,
                 seed=101
                 ) -> None:

        data = {"world":None,
                "seed":seed,
                "size":128,
                "numChannels":3,
                "theta":3,
                "dd":7,
                "dt":0.2,
                "sigma":0.65,
        }
        self.new_world(data)

    # Set/generate new world
    def new_world(self, data
                  ) -> None:
            
        self.seed = data["seed"]
        self.sX = data["size"]
        self.sY = self.sX
        self.numChannels = data["numChannels"]
        self.theta = data["theta"]
        self.dd = data["dd"]
        self.dt = data["dt"]
        self.sigma = data["sigma"]

        try:
            layout = jnp.zeros((self.sX, self.sY, self.numChannels))

            layout = layout.at[self.sX//2-(data["world"].shape[0] + 1)//2 :self.sX//2+(data["world"].shape[0])//2, self.sY//2-(data["world"].shape[1])//2:self.sY//2+(data["world"].shape[1] + 1)//2, :].set(data["world"])
            
            self.A = layout
        except Exception:
            self.generateWorld()
        
        self.A_initial = self.A

    def generateWorld(self):
        
        # Generate random world
        rand_gen = np.random.RandomState(self.seed)
        init_size = self.sX // 2
        self.A = np.zeros((self.sX, self.sY, self.numChannels))
        self.A[self.sX//2-init_size//2:self.sX//2+init_size//2, self.sY//2-init_size //
               2:self.sY//2+init_size//2, :] = rand_gen.rand(init_size, init_size, self.numChannels)
        

    