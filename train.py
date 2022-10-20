from urllib.request import urlretrieve
from sklearn.preprocessing import OneHotEncoder
from os.path import isfile, isdir
from tqdm import tqdm
import tensorflow as tf
import tarfile
import helper
import numpy as np
import pickle

## Get the Data

cifar10_dataset_folder_path = 'cifar-10-batches-py'
epochs = 30
batch_size = 128
keep_probability = 0.7
save_model_path = './image_classification'

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

if not isfile('cifar-10-python.tar.gz'):
    with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
        urlretrieve(
            'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
            'cifar-10-python.tar.gz',
            pbar.hook)

if not isdir(cifar10_dataset_folder_path):
    with tarfile.open('cifar-10-python.tar.gz') as tar:
        
        import os
        
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
        tar.close()




def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    x_norm = np.empty(x.shape, dtype = np.float32)
    
    for i in range(x.shape[0]):
        i_min = np.amin(x[i])
        i_max = np.amax(x[i])
        x_norm[i] = (x[i] - i_min) / (i_max - i_min)
    
    return x_norm


enc = OneHotEncoder()
enc.fit(np.arange(10).reshape(-1,1))

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    
    x_arr = np.array(x).reshape(-1,1)
    x_new = enc.transform(x_arr).toarray()
    
    return x_new


# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)


# Load the Preprocessed Validation data
valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))



def neural_net_image_input(image_shape):
    """
    Return a Tensor for a bach of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """

    input_shape = list(image_shape)
    input_shape[:0] = [None]
    
    x = tf.placeholder (tf.float32, input_shape, name = 'x')
    
    return x

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    y = tf.placeholder(tf.float32, [None, n_classes], name = 'y')
    return y

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """

    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
        
    return keep_prob 

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """

    
    weights_shape = conv_ksize + (int (x_tensor.shape[3]),) + (conv_num_outputs,)
    
    weights = tf.Variable(tf.truncated_normal(weights_shape, stddev = 0.05) )
    bias = tf.Variable(0.05 * tf.ones(conv_num_outputs))
    
    output = tf.nn.conv2d(x_tensor, 
                          filter = weights,
                          strides = (1,) + conv_strides + (1,),
                          padding = 'SAME')
    
    output = tf.nn.bias_add(output, bias)
    
    output =tf.nn.relu(output)
    
    output = tf.nn.max_pool(output, 
                            ksize = (1,) + pool_ksize  + (1,),
                            strides = (1,) + pool_strides + (1,),
                            padding = 'SAME')
    
    return output




def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """

    t_shape = x_tensor.get_shape().as_list()
    
    flat_shape = (-1, t_shape[1] * t_shape[2] * t_shape[3])
    
    x_flat = tf.reshape (x_tensor, flat_shape)
    
    return x_flat


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """

    weights_shape = (x_tensor.get_shape().as_list()[1], num_outputs)
    
    weights = tf.Variable (tf.truncated_normal(weights_shape, stddev = 0.05))
    
    bias = tf.Variable(0.05 * tf.ones(num_outputs))
    
    fc = tf.add(tf.matmul(x_tensor, weights), bias)
    
    fc = tf.nn.relu(fc)
    
    return fc


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """

    weights_shape = (x_tensor.get_shape().as_list()[1], num_outputs)
    
    weights = tf.Variable(tf.truncated_normal(weights_shape, stddev = 0.05))
    bias = tf.Variable(0.05 * tf.ones(num_outputs))
    
    out = tf.add(tf.matmul(x_tensor, weights), bias)
    
    return out



def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """

    c1 = conv2d_maxpool(x, 
                        conv_num_outputs = 32,
                        conv_ksize = (3,3),
                        conv_strides = (1,1),
                        pool_ksize = (2,2),
                        pool_strides = (2,2) )
    
    c2 = conv2d_maxpool(c1, 
                        conv_num_outputs = 64,
                        conv_ksize = (3,3),
                        conv_strides = (1,1),
                        pool_ksize = (2,2),
                        pool_strides = (2,2) )
    
    c3 = conv2d_maxpool(c2, 
                        conv_num_outputs = 128,
                        conv_ksize = (3,3),
                        conv_strides = (1,1),
                        pool_ksize = (2,2),
                        pool_strides = (2,2) )

    c3_flat = flatten(c3)
    fc1 = fully_conn(c3_flat, num_outputs = 1000)
    fc1 = tf.nn.dropout(fc1, keep_prob = keep_prob)
    fc2 = fully_conn(fc1, num_outputs = 400)
    fc2 = tf.nn.dropout(fc2, keep_prob = keep_prob)
    
    logits = output(fc2, 10)

    return logits


##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

##############################
## Train the Neural Network ##
##############################


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    
    session.run (optimizer,
                 feed_dict = {
                     keep_prob : keep_probability,
                     x : feature_batch,
                     y : label_batch})
    


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """

    train_loss, train_accuracy = session.run((cost, accuracy),
                                             feed_dict = {
                                                 keep_prob : keep_probability,
                                                 x : feature_batch,
                                                 y : label_batch})
    
    valid_loss, valid_accuracy = session.run((cost, accuracy),
                                             feed_dict = {
                                                 keep_prob : 1.0,
                                                 x : valid_features,
                                                 y : valid_labels})    

    print(("train loss : {:.2f}, train accuracy : {:.2f}, validation loss : {:.2f}, validation accuracy : {:.2f}").format(train_loss, train_accuracy, valid_loss, valid_accuracy))
    
##############################
## Train the Neural Network ##
##############################


print('Training...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())
    
    # Training cycle
    for epoch in range(epochs):
        # Loop over all batches
        n_batches = 5
        for batch_i in range(1, n_batches + 1):
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                
            print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
            
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
            
    # Save Model
    saver = tf.train.Saver()
    save_path = saver.save(sess, save_model_path)
