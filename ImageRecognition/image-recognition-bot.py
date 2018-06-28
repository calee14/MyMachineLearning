'''
Image Recognition

This notebook will create a convolutional neural network to classify images in either the mnist or cifar-10 datasets.

'''

# Tensorflow and numpy to create the neural network
import tensorflow as tf
import numpy as np

# Matplotlib to plot info to show our results
import matplotlib.pyplot as plt

# OS to load files and save checkpoints
import os

%matplotlib inline

'''
Loading the data
---

This code will load the dataset that you'll use to train and test the model.

The code provided will load the mnist or cifar data from files, you'll need to add the code that processes it into a format your neural network can use.

MNIST
---

Run this cell to load mnist data.

'''
# Load MNIST data from tf examples

image_height = 28
image_width = 28

color_channels = 1

model_name = "mnist"

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

category_names = list(map(str, range(10)))

# TODO: Process mnist data
# MNIST data has shape of (55000, 784) meaning there are 55000 lists of 784 pixels
# The neual network will need the 784 pixels to be shaped in a multi-demensional list 
# shaped like (iamage_height, image_width, color_channels)
print(train_data.shape)

# Numpy's functions can transform lists into whatever shape you need
train_data = np.reshape(train_data, (-1, image_height, image_width, color_channels))

# The train_data should now be in the shape of (55000, 28, 28, 1)
print(train_data.shape)

# Use Numpy to reshape the eval data the same way
eval_data = np.reshape(eval_data, (-1, image_height, image_width, color_channels))

'''
CIFAR-10
---

Run this cell to load cifar-10 data

'''
# Load cifar data from file
# Cifar data is in bytes with values between 0 and 255 which needs to be converted to floats 
# with values between 0.0 and 1.0
# The cifar data is organized in a list of all each red green and blue pixel 
# ex: [r1, r2, ..., g1, g2, ... b1, b2, ...]

image_height = 32
image_width = 32

color_channels = 3

model_name = "cifar"

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

cifar_path = '/home/student/Desktop/cifar-10-batches-py/'

train_data = np.array([])
train_labels = np.array([])

# Load all the data batches.
for i in range(1,2):
    data_batch = unpickle(cifar_path + 'data_batch_' + str(i))
    train_data = np.append(train_data, data_batch[b'data'])
    train_labels = np.append(train_labels, data_batch[b'labels'])


# Load the eval batch.
eval_batch = unpickle(cifar_path + 'test_batch')

eval_data = eval_batch[b'data']
eval_labels = eval_batch[b'labels'] 

# Load the english category names.
category_names_bytes = unpickle(cifar_path + 'batches.meta')[b'label_names']
category_names = list(map(lambda x: x.decode("utf-8"), category_names_bytes))

# TODO: Process Cifar data
# Process the Cifar training and eval data
def process_data(data):
    # Run a function to convert the array of floats to values between 0.0 and 1.0
    float_data = np.array(data, dtype=float) / 255.0
    
    # Use the reshape function as the mnist processing
    reshaped_data = np.reshape(float_data, (-1, color_channels, image_height, image_width))
    
    # The incorrect image
    transposed_data = np.transpose(reshaped_data, [0, 2, 3, 1])
    
    # return the tranposed data from the function
    return transposed_data

train_data = process_data(train_data)

eval_data = process_data(eval_data)

'''
Once the data is processed, you have a few variables for the data itself and info about its shape:

### Model Info

- **image_height, image_width** - The height and width of the processed images
- **color_channels** - the number of color channels in the image. This will be either 1 for grayscale or 3 for rgb.
- **model_name** - either "cifar" or "mnist" - if you need to handle anything differently based on the model, check this variable.
- **category_names** - strings for each category name (used to print out labels when testing results)

### Training Data

- **train_data** - the training data images
- **train_labels** - the labels for the training data - the "answer key"

### Evaluation Data

- **eval_data** - Image data for evaluation. A different set of images to test your network's effectiveness.
- **eval_labels** - the answer key for evaluation data.

Building the Neural Network Model
--

Next, you'll build a neural network with the following architecture:

- An input placeholder that takes one or more images.
- 1st Convolutional layer with 32 filters and a kernel size of 5x5 and same padding
- 1st Pooling layer with a 2x2 pool size and stride of 2
- 2nd Convolutional layer with 64 filters and a kernel size of 5x5 and same padding
- 2nd Pooling layer with a 2x2 pool size and stride of 2
- Flatten the pooling layer
- A fully connected layer with 1024 units
- A dropout layer with a rate of 0.4
- An output layer with an output size equal to the number of labels.

'''
# TODO: The neural network
# Convolutional Neural Network helps maintain a good amount of neurons and maintain positional relationships in the data
class ConvNet:
    
    def __init__(self, image_height, image_width, channels, num_classes):
        
        # Create input_layer as a placeholder which will be used to feed data into the network
        # This will be our placeholder tensor for image input.
        # Data type will be float32 and the shape of the data will be [None, image_height, image_width, channels]
        # The first dimension of the shape, None, allows you to pass any sized list of images to the network at once
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, image_height, image_width, channels], name="inputs")
        print(self.input_layer.shape)
        
        # Convolutional layers are primary tools of of a convolutional neural network
        # These layers use small clusters of neurons called filters that are moved accross the image and activate based on the pixels they see
        # These clusters learn to recognize features in the data.
        # By adjusting the size of the filters in the layer you adjust the area of the image the filters look at.
        # Higher filter count will allow recognizing a wider range of features
        
        # The function tf.layers.conv2d is a function provided by tensor flow to easily make convolutional layers
        
        # Create our first convolutional layer which is connected to the input placeholder with 32 5x5 filters
        # Padding insures that the all filters are the same size. The neurons will use the ReLu activation function
        conv_layer_1 = tf.layers.conv2d(self.input_layer, filters = 32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        print(conv_layer_1.shape)
        
        # Pooling layers help manage the growth of the complexity by simplifying and shrinking the data set
        # These layers move across the data with a specified stride, simplifying the contents of each filter into a single value
        # This shrinks the size of they layer's output based on the filter's size
        # This also helps reduce the network's translation variance - how sensitive the network is to an object's exact position in an image
        # Pooling layers are usually 2x2 filters with the stride of two. This results in simplifying the data with out too much loss of specificity in the image
        
        # Use the tensorflow function to create pooling layers for 2d convolutions
        # The filter will reduce the height and width of the previous layer by half
        pooling_layer_1 = tf.layers.max_pooling2d(conv_layer_1, pool_size=[2,2], strides=2)
        print(pooling_layer_1.shape)
        
        # Core of convolutional networks are built from a series of convolutional and pooling layers. 
        
        # Create a conv2d function to create another convolutional layer with 64 5x5 filters
        conv_layer_2 = tf.layers.conv2d(pooling_layer_1, filters = 64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        print(conv_layer_2.shape)
        
        # Create a pooling layer with 2x2 filters and with a stride of 2
        pooling_layer_2 = tf.layers.max_pooling2d(conv_layer_2, pool_size=[2, 2], strides=2)
        print(pooling_layer_2.shape)
        
        # At the end of of the convolutional layers you'll need to set up some neurons to help make you final classification decision. 
        # This will be a standard fully-connected layer of neurons. In order to connect these layers, you'll first need to flatten the 2d images's filters
        
        # Use the tensor flow function, flatten, on the last pooling layer
        flattened_pooling = tf.layers.flatten(pooling_layer_2)
        # Use tf.layers.dense to add 1024 units of fully connected ReLu neurons to the flattened layer
        dense_layer = tf.layers.dense(flattened_pooling, 1024, activation=tf.nn.relu)
        print(dense_layer.shape)
        
        # A dropout layer takes a percentage of all the neurons in the input and deactivates them at random. 
        # This random dropout of neurons forces more of the network to adapt to the task.
        # Without a dropout layer, larger networks run the risk of growing over-dependent on a small set of competent neurons rather than the whole network learning task
        
        # Add a dropout layer to the network
        dropout = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        
        # Output layer makes the output based on the possible classifications
        outputs = tf.layers.dense(dropout, num_classes)
        print(outputs.shape)
        
        # tf.argmax will find the index of the highest weight element in a tensor
        self.choice = tf.argmax(outputs, axis=1)
        # tf.nn.softmax will return decimal probabilities of each element
        self.probability = tf.nn.softmax(outputs)
        
        # Create a placeholder for labels
        # Feed values into this placeholder to give the network some answers to grade against
        self.labels = tf.placeholder(dtype=tf.float32, name="labels")
        # tf.metrics.accuracy creates accuracy variables to store accuracy of network
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels, self.choice)
        
        # Loss functions are the most widely used functions for classification problems
        # The function softmax_cross_entropy needs a set of logits (the weights of each output) and a one-hot encoding of the correct output
        
        # Create a one hot label. A one hot encoding is a list full of only 0's and 1's. Ex: 3 = [0,0,0,1,0,0,0,0,0,0]
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, dtype=tf.int32), depth=num_classes)
        # Create the loss operation using one_hot_labels
        self.loss = tf.losses.softmax_cross_entropy(one_hot_labels, logits=outputs)
        
        # Make a gradient descent optimizer to learn from the loss given by the loss function
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        # Create a training operation with the optimizer
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

'''
The Training Process
---

The cells below will set up and run the training process.

- Set up initial values for batch size, training length.
- Process data into batched datasets to feed into the network.
- Run through batches of training data, update weights, save checkpoints.

'''
# TODO: initialize variables

# Initialize variabels to control the training
# These variables will control how long the training loop will run and how many images will be passed into the network at each step of training
training_steps = 1200
batch_size = 500

# Set the path to save the Neural network
path = "/home/student/Desktop/" + model_name + "-cnn/"

# Boolean to check if the network should load past checkpoints from previous training
# By loading previous checkpoints the bot should be more accurate at recognizing images
# Variable that can be toggled to set whether a past trained model should be loaded
load_checkpoint = True

# Create an empty numpy array for the graphs values
performance_graph = np.array([])

# TODO: implement the training loop

# Tensor flow uses the dataset class to easily manage large chunks of data. 
# Reset the default tensorflow graph
tf.reset_default_graph()

# The dataset class can be created from existing tensors or lists
dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
# The datasets can be shuffled to randomize the data.
# This avoids the network developing any idiosyncrasies based on the exact order of the input data.
dataset = dataset.shuffle(buffer_size=train_labels.shape[0])
# The neural network can receive the entire dataset at one time but that would take tons of memory and processing power
# Datasets can use batches to pass small amounts of the data in at a time.
# Smaller batches produce noisier training data but train faster
# Batch the dataset into batches based on the batch_size
dataset = dataset.batch(batch_size)
# The dataset class provides an easy way to make sure you never run out of training data while looping through training steps
# Make the dataset loop repeatedly
dataset = dataset.repeat()

# Datasets use a tensor flow operator called an iterator to get the next element of the data.
# We use an initializable iterator to go through each element of the dataset

# Create an iterator
dataset_iterator = dataset.make_initializable_iterator()
# Save the get_next operation to a variable
next_element = dataset_iterator.get_next()

# Create an instance of the convolutional network class with the parameters that was set up when loading the data
cnn = ConvNet(image_height,image_width,color_channels,10)

# The Saver class allows the wights and variables in tensor flow graphs to be saved between runs.
# This will let you keep you training progress between runs

# Declare a saver
# The max_to_keep parameter controls the number of checkpoints to save. 
# Complex neural networks can take lots of space to save, so keeping this number low will save lots of disk space
saver = tf.train.Saver(max_to_keep=2)

# Create a folder to save checkpoints if the directory doesn't exists already
if not os.path.exists(path):
    os.makedirs(path)
     

# Start a tensor flow session and initialize initial wights and variables for the network
with tf.Session() as sess:
     
    # Now either load from a checkpoint or initialize wights. 
    # check we're set to load a checkpoint
    if load_checkpoint:
        # load a checkpoint
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        # run a global initialize if we're on a new network
        # This sets up the initial weights for the network, either with a pre-existing checkpoint or with new values.
        sess.run(tf.global_variables_initializer())
     
    # Initialize local variables - this is needed for hte accuracy metrics calculation
    
    # Run the local initializer
    sess.run(tf.local_variables_initializer())
    # Initialize the iterator
    sess.run(dataset_iterator.initializer)
    # Create a for loop based on the training_steps variable
    for step in range(training_steps):
        # By using the next_element operation it returns the batch of data to store in the variable
        current_batch = sess.run(next_element)
        
        # Split the resulting batch into the image inputs and image labels
        batch_inputs = current_batch[0]
        batch_labels = current_batch[1]
        
        # Run the networks training operation and the accuracy operation
        # Inside the feed_dict, feed the images into the input_layer, and the labels into labels placeholder
        sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict={cnn.input_layer:batch_inputs, cnn.labels:batch_labels})
        
        # Create an if statement that updates the list every 10 steps
        if step % 10 == 0:
            performance_graph = np.append(performance_graph, sess.run(cnn.accuracy))
            
        # Periodically save the model's progress and print its current accuracy
        # Every 100 training steps after the first step print model's current accuracy and save the network weights to a file
        if step % 100 == 0 and step > 0:
            # Get current accuracy by running accuracy from the neural network
            current_acc = sess.run(cnn.accuracy)
            # Print the current step and accuracy
            print("Accuracy at step " + str(step) + ": " + str(current_acc))
            print("Saving checkpoint")
            # Use the Saver class to save a checkpoint
            # Function below will save a checkpoint of the model to the path we specified
            saver.save(sess, path + model_name, step)
    
    # Save one last checkpoint once the training has ended
    print("Saving final checkpoint for training session.")
    saver.save(sess, path + model_name, step)

'''
Evaluating Performance
---

These cells will evaluate the performance of your network!

'''
# TODO: Display graph of performance over time

# Use the matplotlib module to plot the graph
plt.plot(performance_graph)
# Make adjustments to make the graph easier to read
# This will set a white background and label each axis for readability
plt.figure().set_facecolor('white')
plt.xlabel('Steps')
plt.ylabel('Accuracy')

# TODO: Run through the evaluation data set, check accuracy of model
# The real test comes when the network attempts to classify images that it has never seen. 
# This is called the evaluation data and it consists of a sample of images totally separate from the training data
# Load the saved neural network weights then feed in the eval data
with tf.Session() as sess:
    # load the saved checkpoint the same way training loop
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)
    
    # Initialize the accuracy metrics variables then feed the evaluation data and labels into the network to test results
    # Initialize local variables for accuracy
    sess.run(tf.local_variables_initializer())
    
    # Create a for loop that goes through each pair of image label
    for image, label in zip(eval_data, eval_labels):
        # Run the network's accuracy operation on each pair of image and label
        sess.run(cnn.accuracy_op, feed_dict={cnn.input_layer:[image], cnn.labels:label})
    # Print the accuracy after the loop
    print(sess.run(cnn.accuracy))

# TODO: Get a random set of images and make guesses for each
# Get a random set of images cell
# Start a tf session and load the saved checkpoint
with tf.Session() as sess:
    # load the saved checkpoint
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)
    
    # Get a random selection of 10 indexes from the eval data
    indexes = np.random.choice(len(eval_data), 10, replace=False)
    
    # This will give a list of random data points to pull from.
    # The code for arranging and displaying the images in matplotlib is a bit fiddly, so we've provided it below
    # Code below should show a few selected images from the dataaset and what the network guessed for them
    rows = 5
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(5,5))
    fig.patch.set_facecolor('white')
    image_count = 0
    
    for idx in indexes:
        image_count += 1
        sub = plt.subplot(rows, cols, image_count)
        img = eval_data[idx]
        if model_name == "mnist":
            img = img.reshape(28, 28)
        plt.imshow(img)
        guess = sess.run(cnn.choice, feed_dict={cnn.input_layer:[eval_data[idx]]})
        if model_name == "mnist":
            guess_name = str(guess[0])
            actual_name = str(eval_labels[idx])
        else:
            guess_name = category_names[guess[0]]
            actual_name = category_names[eval_labels[idx]]
        sub.set_title("G: " + guess_name + " A: " + actual_name)
    plt.tight_layout()
        
    
