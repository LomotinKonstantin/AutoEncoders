from SAE import SAE, load_images_to_dict, add_zero_mask_noise
import tensorflow as tf 
from time import time

print("Starting")
t = time()

dataset_path = "./kagglecatsanddogs_3367a/PetImages/Cat"
resol = (128, 128)

X = load_images_to_dict(dataset_path, last=5, resol=resol, flatten=True)

print("Dataset loaded in {} sec".format(time() - t))
t = time()

input_size = X["r"][0].shape[0]

sae = SAE(input_size, [4096, 1024, 4096])

#sae.fit(add_zero_mask_noise(X["r"], percent=0.1), 300, tf.train.RMSPropOptimizer(0.05), greedy=False)

print("Fitting finished in {} sec".format(time() - t))