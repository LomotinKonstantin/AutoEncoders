# coding: utf-8

# In[1]:

from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import tensorflow as tf
from skimage.transform import resize
import os

# get_ipython().magic('matplotlib inline')

gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))


# In[2]:

def add_noise(X, stddev=0.1):
    res = X.copy()
    res += np.random.normal(size=X.shape, scale=stddev)
    return res


def imshow(vector):
    if len(vector.shape) != 3:
        vector = vector.reshape(resol)
    plt.imshow(vector)


# In[3]:

dataset_path = "./kagglecatsanddogs_3367a/PetImages/Cat"
resol = (128, 128)


def load_images_to_dict(path, first=0, last=5000, resol=(256, 256)):
    res = {
        "r": [],
        "g": [],
        "b": []
    }
    for n, file in enumerate(os.listdir(path)[first:last]):
        if file[-4:] != ".jpg":
            continue
        clear_output(True)
        print("Loading {}".format(file))
        print("%.2f" % (n / (last - first) * 100))
        full_dp = os.path.join(dataset_path, file)
        img = misc.imread(full_dp)
        img = resize(img, resol)
        try:
            res["r"].append(img[:, :, 0])
            res["g"].append(img[:, :, 1])
            res["b"].append(img[:, :, 2])
        except Exception:
            continue
    for k in res.keys():
        res[k] = np.array(res[k])
    return res


# In[4]:

X = load_images_to_dict(dataset_path, last=2000, resol=resol)

# In[7]:

input_size = resol[0] * resol[1]

# In[8]:

for k in X.keys():
    X[k] = X[k].reshape(X[k].shape[0], input_size)

# In[9]:

latent_size = 8000

w_hid = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[input_size, latent_size]), dtype="float32")
b_hid = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[latent_size]), dtype="float32")

w_r_out = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[latent_size, input_size]), dtype="float32")
b_r_out = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[input_size]), dtype="float32")

# w_g_out = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[latent_size, input_size]), dtype="float32")
# b_g_out = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[input_size]), dtype="float32")
#
# w_b_out = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[latent_size, input_size]), dtype="float32")
# b_b_out = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[input_size]), dtype="float32")

X_in = tf.placeholder("float32", shape=[None, input_size])

# In[10]:

learning_rate = 0.05

latent_layer = tf.nn.sigmoid(tf.matmul(X_in, w_hid) + b_hid)

out_layer_r = tf.nn.sigmoid(tf.matmul(latent_layer, w_r_out) + b_r_out)
# out_layer_g = tf.nn.sigmoid(tf.matmul(latent_layer, w_g_out) + b_g_out)
# out_layer_b = tf.nn.sigmoid(tf.matmul(latent_layer, w_b_out) + b_b_out)

# loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=X_in_clear, labels=out_wo_activ))
loss_r = tf.losses.mean_squared_error(X_in, out_layer_r)
# loss_g = tf.losses.mean_squared_error(X_in, out_layer_g)
# loss_b = tf.losses.mean_squared_error(X_in, out_layer_b)

# optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, var_list=[w_hid, b_hid, w_out, b_out])
optimizer_r = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_r)


# optimizer_g = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_g)
# optimizer_b = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_b)

def train_epoch(X):
    optimizer_r.run({X_in: X}, session=s)
    #     optimizer_g.run({X_in: X}, session=s)
    #     optimizer_b.run({X_in: X}, session=s)
    return loss_r.eval({X_in: X}, s)


def latent_repr(X):
    return latent_layer.eval({X_in: X}, s)


def predict(X_r, X_g, X_b):
    channels = (out_layer_r.eval({X_in: X_r}, s).reshape(resol),
                out_layer_r.eval({X_in: X_g}, s).reshape(resol),
                out_layer_r.eval({X_in: X_b}, s).reshape(resol))
    return np.stack(channels, axis=2)


# In[ ]:

# imshow(add_noise(X["g"][1], 0.05))


# In[ ]:

batch_size = 1000
epochs = 300

s.run(tf.global_variables_initializer())

for i in range(epochs):
    out_batch = X["r"][np.random.choice(range(X["r"].shape[0]), size=[batch_size], replace=False)]
    in_batch = add_noise(out_batch, 0.05)
    #     in_batch = out_batch

    loss_r = train_epoch(in_batch)
    # clear_output(True)
    print("Epoch: {}\nLoss R: {}".format(i + 1, loss_r))

imshow(predict(add_noise(X["b"][10], 0.1)))
