import base64
from datetime import datetime
import os
import random
import sys
from functools import partial


import jax.numpy as jnp
import numpy as np
import scipy as sp

from scipy import ndimage

from skimage.feature import hog
from skimage import exposure
from skimage.measure import moments, moments_central, moments_normalized, moments_hu


import eel
import pickle

from objects import World, Models

# import time
# import timeit



sample_images_path = "web/images/samples"
sample_parameters_path = "web/sample_parameters"





class System():
    
    # Initialization of values of world and kernel parameters
    def __init__(self) -> None:
        
        self.initialize_world_kernel()
    

    # Set world and kernel parameters. New dictionaries to 
    # store detection information
    def initialize_world_kernel(self) -> None:
        
        self.set_world_kernel_parameters()
        self.compile()
        self.centroids = {}
        self.individuals_initial_state = {}
        self.survival_scores = {}
        self.previous_survival_scores = {}
        self.reproduction_scores = {}
        self.morphology_scores = {}
        self.next_id = 0

    def compile(self):

        self.model.compile()
    
    def generate_random_kernel_parameters(self):

        self.kernel_parameters.generate_random_parameters(seed = self.world.seed, num_channels = self.world.numChannels)

    
    def step(self):

        return self.model.step()
    
    def get_world_matrix(self):

        return self.world.A
    
    def get_initial_world(self):

        return self.world.A_initial
    
    def set_initial_world(self, A):

        self.world.A_initial = A

    def get_world_kernel_parameters(self):

        data = {}

        data["version"] = self.version
        data["seed"] = self.world.seed
        data["size"] = self.world.sX
        data["numChannels"] = self.world.numChannels
        data["theta"] = self.world.theta
        data["dd"] = self.world.dd
        data["dt"] = self.world.dt
        data["sigma"] = self.world.sigma

        for k in 'CrmshBawT':
            temp_nparray = self.kernel_parameters.kernel_parameters[k]
            if type(temp_nparray) is np.ndarray:
                temp_nparray = temp_nparray.tolist()
            
            data[k] = temp_nparray

        return data

    def set_world_kernel_parameters(self, data = None):

        if data == None:

            self.world = World.World()
            self.version = "LeniaModel"

            Model = getattr(Models, self.version)
            self.model = Model(world = self.world)

            self.kernel_parameters = self.model.getKParams()
            return

        self.version = data["version"]
        self.world.new_world(data)

        Model = getattr(Models, self.version)
        self.model = Model(world = self.world)

        self.kernel_parameters = self.model.getKParams()
        self.kernel_parameters.new_params(data)

    def compute_individuals_information(self):

        # Minimum size of individual to being detected
        min_size = 10 * 2

        # Sum the values along the last axis
        summed_world = np.sum(self.world.A, axis=-1)

        # Threshold the array
        mask = np.where(summed_world > 0, True, False)

        # Label the connected regions
        self.labels, num_features = ndimage.label(mask)

        # Count the number of non-zero values in each group
        counts = np.bincount(self.labels.ravel())

        # Find the bounding box of each region
        slices = ndimage.find_objects(self.labels)

        # Calculate length of each bounding box
        individuals_bbox = [tuple(slice_interval.stop - slice_interval.start 
                         for slice_interval in slice) 
                         for slice in slices]

        # Filter out regions that don't meet the minimum size threshold
        individuals_bbox = [(length[0] / self.world.A.shape[0], length[1] / self.world.A.shape[1]) 
                   for i, length in enumerate(individuals_bbox) 
                   if np.bincount(self.labels.ravel())[i+1] >= min_size]

        # Find the center of mass of each connected region
        new_centroids = np.array([ndimage.center_of_mass(summed_world, labels=self.labels, index=i)
                        for i in range(1, num_features+1) if counts[i] >= min_size])
        
        new_centroids = (new_centroids / self.world.A.shape[0]).tolist()
        
        temp_centroids = {}
        
        # Matchin previous detections with new detections 
        # to track individuals
        for k, v in self.centroids.items():

            if len(new_centroids) == 0:
                break

            # Convert the list of tuples to a NumPy array
            new_centroids_array = np.array(new_centroids)
            
            # Calculate the Euclidean distance between the target point and each point in the list
            distances = np.linalg.norm(new_centroids_array[:, None] - v[:2], axis=2)
            
            # Find the index of the nearest point
            nearest_index = np.argmin(distances)
            
            if distances[nearest_index] < individuals_bbox[nearest_index][0] * 1.5:

                temp_centroids[k] = [
                    new_centroids[nearest_index][0], 
                    new_centroids[nearest_index][1], 
                    individuals_bbox[nearest_index][0], 
                    individuals_bbox[nearest_index][1]]
                
                # Remove the nearest point from the list
                new_centroids.pop(nearest_index)
                individuals_bbox.pop(nearest_index)

        self.centroids = temp_centroids


        # Now only new detections remain in new_centroids
        for point, dimension in zip(new_centroids, individuals_bbox):
            
            bbox = [
                point[0], 
                point[1], 
                dimension[0], 
                dimension[1] ]
            
            self.centroids[self.next_id] = bbox
            self.individuals_initial_state[self.next_id] = self.get_individual_array_from_bbox(bbox)

            self.survival_scores[self.next_id] = 0
            self.previous_survival_scores[self.next_id] = 0
            self.reproduction_scores[self.next_id] = 0
            self.next_id += 1

        return self.centroids

    def get_individual_array_from_bbox(self, bbox):

        # Calculate the start and end indices for slicing
        start_row = int((bbox[0] - bbox[3] / 2) * self.world.A.shape[0])
        end_row = int((bbox[0] + bbox[3] / 2) * self.world.A.shape[0])
        start_col = int((bbox[1] - bbox[2] / 2) * self.world.A.shape[1])
        end_col = int((bbox[1] + bbox[2] / 2) * self.world.A.shape[1])

        start_row = max(start_row, 0)
        end_row = min(end_row, self.world.A.shape[0])
        start_col = max(start_col, 0)
        end_col = min(end_col, self.world.A.shape[1])

        # Slice the array and store the subarray in the result dictionary
        individual_array = self.world.A[start_row:end_row, start_col:end_col, :]
        
        return individual_array
    
    
    def resize_individual(self, array, new_shape):

        # Create the coordinates for the original array
        curr_shape = array.shape
        coords = [np.linspace(0, curr_dim-1, curr_dim) for curr_dim in curr_shape]
        
        # Create the interpolator
        interpolator = sp.interpolate.RegularGridInterpolator(coords, array, method='linear', bounds_error=False, fill_value=0)
        
        # Create the coordinates for the interpolated array
        new_coords = [np.linspace(0, new_dim-1, new_dim) for new_dim in new_shape]
        interp_coords = np.meshgrid(*new_coords, indexing='ij')
        points = np.stack(interp_coords, axis=-1)

        # Perform interpolation
        interpolated = interpolator(points)
        
        return interpolated
    

    def compute_survival_score(self):

        density_scores = {}

        for k, bbox in self.centroids.items():

            indiv = self.get_individual_array_from_bbox(bbox)

            total_mass = np.sum(indiv)
            total_volume = np.count_nonzero(indiv)
            
            if total_volume == 0: total_volume = jnp.asarray([1])
            density_scores[k] = (total_mass / total_volume).item()
            
            self.survival_scores[k] = density_scores[k] - self.previous_survival_scores[k]
            self.previous_survival_scores[k] = density_scores[k]

        return density_scores

    def compute_reproduction_score(self):

        for k, bbox in self.centroids.items():

            indiv = self.get_individual_array_from_bbox(bbox)

            keys = [key for key in self.centroids.keys() if key != k]

            if len(keys) == 0: continue

            neighbour = self.get_individual_array_from_bbox(self.centroids[random.choice(keys)])
            
            neighbour_resized = self.resize_individual(neighbour, indiv.shape)

            similarity_score = self.calculate_similarity(indiv, neighbour_resized)
            
            self.reproduction_scores[k] = (self.reproduction_scores[k] + similarity_score) / 2

        common_keys = set(self.centroids.keys()) & set(self.reproduction_scores.keys())

        self.reproduction_scores = {key: self.reproduction_scores[key] for key in common_keys}

        return self.reproduction_scores

    def compute_morphology_score(self):

        scores = {}

        for k, bbox in self.centroids.items():

            indiv = self.get_individual_array_from_bbox(bbox)
            
            original = self.individuals_initial_state[k]
            resized_target = self.resize_individual(indiv, original.shape)
            
            similarity_score = self.calculate_similarity(original, resized_target)

            scores[k] = similarity_score

        self.morphology_scores = scores

        return self.morphology_scores


    def calculate_similarity(self, image1, image2):

        # Convert arrays to JAX DeviceArrays
        image1 = jnp.asarray(image1)
        image2 = jnp.asarray(image2)

        # Check if the standard deviation is zero
        if jnp.std(image1) == 0 or jnp.std(image2) == 0:
            
            return 0.0  # Return a default similarity value when there is zero variance
        
        # Normalize the arrays
        arr1_norm = (image1 - jnp.mean(image1)) / jnp.std(image1)
        arr2_norm = (image2 - jnp.mean(image2)) / jnp.std(image2)
        
        # Compute the cross-correlation
        cross_corr = jnp.correlate(arr1_norm.flatten(), arr2_norm.flatten(), mode='same')

        # Compute the similarity
        similarity = jnp.max(cross_corr) / (jnp.linalg.norm(arr1_norm) * jnp.linalg.norm(arr2_norm))

        score = similarity.item()
        if type(score) != float:
            score = 0.0
    
        return score
    
    def get_individuals_ID(self):

        # print(list(self.centroids.keys()))

        return list(self.centroids.keys())
    
    def get_all_stats_from_key(self, key):
        
        stats = {}

        try:
            stats["survival"] = self.survival_scores[key]
            stats["reproduction"] = self.reproduction_scores[key]
            stats["morphology"] = self.morphology_scores[key]
        except KeyError:
            stats = {}

        return stats






system = System()

eel.init("web")



@eel.expose
def get_parameters_from_python():

    data = system.get_world_kernel_parameters()

    return data

@eel.expose
def set_parameters_in_python(data):

    data["world"] = system.get_initial_world()

    system.set_world_kernel_parameters(data)
    system.compile()


@eel.expose
def generate_kernel(data):
        
    system.set_world_kernel_parameters(data)
    system.generate_random_kernel_parameters()
    system.compile()


@eel.expose
def step():

    system.step()

@eel.expose
def get_world(visible_channels):

    Ac = system.world.A.clip(0, 1)[:, :, visible_channels]
    alpha = jnp.ones((system.world.sX, system.world.sY))
    while Ac.shape[-1] < 3:
        Ac = jnp.dstack((Ac, alpha * 0))
    res = jnp.dstack((Ac, alpha))
    res = jnp.uint8(res * 255.0).tolist()
    
    return res





@eel.expose
def save_parameter_state(imageData=None, filename="screenshot"):

    filepath_image = sample_images_path + '/' + filename + ".png"
    filepath_parameter = sample_parameters_path + '/' + filename + ".pickle"
    
    data = get_parameters_from_python()
    data["world"] = system.get_world_matrix()
    
    if imageData is None:
        with open(filepath_parameter, 'wb') as f:
            pickle.dump(data, f)
        return
    
    # Remove the "data:image/png;base64," prefix from the base64-encoded string
    image = imageData.split(",")[1]

    # Decode the base64-encoded string to bytes
    imageBytes = base64.b64decode(image)

    # Save the bytes to a file
    with open(filepath_image, "wb") as f:
        f.write(imageBytes)
    
    with open(filepath_parameter, 'wb') as f:
        pickle.dump(data, f)

@eel.expose
def load_parameter_state(filename = "screenshot"):

    filepath_parameter = sample_parameters_path + '/' + filename + ".pickle"

    # load the dictionary from the file
    with open(filepath_parameter, 'rb') as f:
        data = pickle.load(f)
    
    system.set_initial_world(data["world"])
    set_parameters_in_python(data)

@eel.expose
def delete_sample(filename = "screenshot"):

    filepath_image = sample_images_path + '/' + filename + ".png"
    filepath_parameter = sample_parameters_path + '/' + filename + ".pickle"

    if os.path.exists(filepath_parameter) and os.path.exists(filepath_image):
         os.remove(filepath_parameter)
         os.remove(filepath_image)


@eel.expose
def get_sample_names():
    names = [os.path.splitext(filename)[0] for filename in os.listdir(
        sample_images_path) if os.path.isfile(os.path.join(sample_images_path, filename))]
    return list(names)


@eel.expose
def set_sample(name, new_data):

    load_parameter_state(name)




@eel.expose
def get_coordinates_from_python():

    centers = system.compute_individuals_information()

    for k, v in centers.items():
        centers[k] = [v[0], #/ world_size,
                     v[1], #/ world_size,
                     v[2], # / world_size,
                     v[3] # / world_size
                     ]
    
    return centers





@eel.expose
def get_global_survival_stats():

    data = system.compute_survival_score()
    
    return data

@eel.expose
def get_global_reproduction_stats():

    data = system.compute_reproduction_score()
    
    return data

@eel.expose
def get_global_morphology_stats():

    data = system.compute_morphology_score()
    return data


@eel.expose
def get_individuals_ID_from_python():

    return system.get_individuals_ID()

@eel.expose
def get_all_stats_from_python(id):

    return system.get_all_stats_from_key(id)




@eel.expose
def shutdown():
    sys.exit()


eel.start("index.html", mode="chrome-app", shtudown_delay = 2.0)

