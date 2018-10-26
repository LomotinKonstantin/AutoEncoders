import os
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
from IPython.display import display, clear_output
import tensorflow as tf
from skimage.transform import resize


def add_gaussian_noise(X, stddev=0.1):
    res = X.copy()
    res += np.random.normal(size=X.shape, scale=stddev)
    return res

def add_zero_mask_noise(X, percent=0.3):
    """
    X - ndarray of vectors
    """
    res = X.copy()
    for i, x in enumerate(X):
        mask_indices = np.random.choice(np.arange(len(x)), 
                                        int(percent * len(x)),
                                        replace=False)
        res[i][mask_indices] = 0
    return res

def show_vector(vector):
    if len(vector.shape) != 3:
            vector = vector.reshape(resol)
    plt.imshow(vector)


def assemble_image(r, g, b):
    return np.stack([r, g, b], axis=2)

def assemble_image_from_vectors(r, g, b, resol):
    return np.stack([r.reshape(resol), g.reshape(resol), b.reshape(resol)], axis=2)


def load_images_to_dict(path, first=0, last=5000, resol=(256, 256), flatten=True):
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
        print("{}%".format(n / (last-first) * 100))
        full_dp = os.path.join(path, file)
        img = misc.imread(full_dp)
        img = resize(img, resol)
        try:
            for i, color in enumerate(["r", "g", "b"]):
                if flatten:
                    img_data = img[:,:,i].reshape([img.shape[0] * img.shape[1]])
                else:
                    img_data = img[:,:,i]
                res[color].append(img_data)
#             res["r"].append(img[:,:,0])
#             res["g"].append(img[:,:,1])
#             res["b"].append(img[:,:,2])
        except Exception:
            continue
    for k in res.keys():
        res[k] = np.array(res[k])
    return res
            


class SAE:
    
    def __init__(self, input_size, hidden_layer_sizes, ini_stddev=0.2, session=None):
        """
        Simple Stacked AutoEncoders
        
        Parameters:
            input_size: size of the input vector
            hidden_layer_sizes: list of the numbers of neurons in hidden layers (excluding output layer)
            ini_stdev: standard deviation for gaussian weights & biases initialization
            session: TensorFlow session
        """
        assert len(hidden_layer_sizes) > 0, "Invalid hidden layer sizes. List of ints expected"
#         assert np.all(hidden_layer_sizes[i] == hidden_layer_sizes[len(hidden_layer_sizes) - 1 - i] 
#                       for i in range(len(hidden_layer_sizes))), "Hidden layer sizes must be symmetric"
        # Configuring TF
        if session is None:
            gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.1)
            self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        else: 
            self.session = session
        self.hidden_layer_sizes = hidden_layer_sizes
        # Creating AE
        self.weights = []
        self.biases = []
        # First layer:
        first_w = tf.Variable(initial_value=np.random.normal(scale=ini_stddev, size=[input_size, hidden_layer_sizes[0]]), 
                            dtype="float32", name="w_0")
        first_b = tf.Variable(initial_value=np.random.normal(scale=ini_stddev, size=[hidden_layer_sizes[0]]), 
                              dtype="float32", name="b_0")
        self.session.run(first_b.initializer)
        self.session.run(first_w.initializer)
        self.weights.append(first_w)
        self.biases.append(first_b)
        for i in range(0, len(hidden_layer_sizes) - 1):
            next_layer_size = hidden_layer_sizes[i + 1]
            w = tf.Variable(initial_value=np.random.normal(scale=ini_stddev, size=[hidden_layer_sizes[i], next_layer_size]), 
                            dtype="float32", name="w_{}".format(i + 1))
            b = tf.Variable(initial_value=np.random.normal(scale=ini_stddev, size=[next_layer_size]), 
                            dtype="float32", name="b_{}".format(i + 1))
            self.weights.append(w)
            self.biases.append(b)
        last_w = tf.Variable(initial_value=np.random.normal(scale=ini_stddev, size=[hidden_layer_sizes[-1], input_size]), 
                             dtype="float32", name="w_last")
        last_b = tf.Variable(initial_value=np.random.normal(scale=ini_stddev, size=[input_size]), 
                             dtype="float32", name="b_last")
        #self.session.run(last_b.initializer)
        #self.session.run(last_w.initializer)
        self.weights.append(last_w)
        self.biases.append(last_b)
        
        # Creating layers
        self.input_tensor = tf.placeholder("float32", shape=[None, input_size])
        self.layers = []
        self.output_layer = None
        first_layer = tf.nn.sigmoid(tf.matmul(self.input_tensor, first_w) + first_b)
        self.layers.append(first_layer)
        self.output_layer = first_layer
        for i in range(1, len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            layer = tf.nn.sigmoid(tf.matmul(self.output_layer, w) + b)
            self.layers.append(layer)
            self.output_layer = layer
            self.session.run(self.weights[i].initializer)
            self.session.run(self.biases[i].initializer)
        self.session.run(tf.global_variables_initializer())
        self.session.run(tf.local_variables_initializer())
#         print(self.hidden_layer_sizes)
#         print(self.layers)
        
        
    def __train_epoch(self, X, optimizer, loss):
        optimizer.run({self.input_tensor: X}, session=self.session)
        return loss.eval({self.input_tensor: X}, session=self.session)

    def __get_batch(X, size):
        ind = np.random.choice(np.arange(X.shape[0]), replace=False, size=size)
        return X[ind]

    def __train_greedy(self, X, n_epochs, optimizer_obj, batch_size=None, min_delta=1e-5):
        # Training layer-wise
        print("Starting greedy training")
        for i in range(len(self.layers[:-1])):
            report_str = "Training layer {} out of {}".format(i + 1, len(self.layers))
            layer = self.layers[i]
            fake_w = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[self.hidden_layer_sizes[i], X.shape[1]]), 
                                 dtype="float32", name="fake_w_{}".format(i))
            fake_b = tf.Variable(initial_value=np.random.normal(scale=0.2, size=[X.shape[1]]), 
                                 dtype="float32", name="fake_b_{}".format(i))
            fake_output_layer = tf.nn.sigmoid(tf.matmul(layer, fake_w.initialized_value()) + fake_b.initialized_value())
            loss = tf.losses.mean_squared_error(X, fake_output_layer)
            optimizer = optimizer_obj.minimize(loss, var_list=[self.weights[i], self.biases[i], fake_w, fake_b])
            self.session.run(fake_w.initializer)
            self.session.run(fake_b.initializer)
            prev_loss = .0
            for epoch in range(n_epochs):
                if batch_size is not None:
                    X_train = self.__get_batch(X, batch_size)
                else:
                    X_train = X
                curr_loss = self.__train_epoch(X_train, optimizer, loss)
                delta = abs(curr_loss - prev_loss)
                if delta <= min_delta:
                    break
                clear_output(True)
                if epoch % 100 == 0:
                    print(report_str)
                    print("Epoch: {}\nLoss: {}".format(epoch + 1, curr_loss))
            if i == len(self.layers[:-2]):
                self.weights[i + 1] = fake_w
                self.biases[i + 1] = fake_b
    
    def __print_variables(self):
        for i in tf.trainable_variables():
            print(i, self.session.run(i).shape)
    
    def __train(self, X, n_epochs, optimizer_obj, batch_size=None, min_delta=1e-5):
        print("Starting non-greedy training")
        loss = tf.losses.mean_squared_error(X, self.output_layer)
        optimizer = optimizer_obj.minimize(loss)
        self.__print_variables()
        for epoch in range(n_epochs):
            if batch_size is not None:
                X_train = self.__get_batch(X, batch_size)
            else:
                X_train = X
            curr_loss = self.__train_epoch(X_train, optimizer, loss)
            delta = abs(curr_loss - prev_loss)
            if delta <= min_delta:
                break
#            clear_output(True)
            if epoch % 100 == 0:
                print("Epoch: {}\nLoss: {}".format(epoch + 1, curr_loss))

    def fit(self, X, n_epochs, optimizer_obj, batch_size=None, min_delta=1e-5, greedy=True):
        """
        Train the SDAE

        Parameters:
            X: ndarray of training vectors
            optimizer_obj: algorithm for optimization (need to pass the object from tensorflow.train)
            batch_size: for stochastic training
        """
        if greedy:
            self.__train_greedy(X, n_epochs, optimizer_obj, batch_size, min_delta)
        else:
            self.__train(X, n_epochs, optimizer_obj, batch_size, min_delta)


    def predict(self, sample):
        return self.out_layer.eval({X: sample}, self.session)
        