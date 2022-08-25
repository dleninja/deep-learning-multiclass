"""
Deep Learning using Tensorflow on Kaggle OCT Dataset

This python script is to demonstrate an example of how
to build a neural network in tensorflow.

Source: https://www.tensorflow.org/datasets/keras_example

In this script, the modification is for GPU based training.

Author: github.com/dleninja
"""
# ....................................................................
# Step 0. Load the relevant libraries
# ....................................................................
import tensorflow as tf
#
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
#
from sklearn.utils.class_weight import compute_class_weight
#
from pathlib import Path
import numpy as np
#
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print('Tensorflow Version: ',tf.__version__)
print('tf.keras Version  : ',tf.keras.__version__)
# ....................................................................
# ////////////////////////////////////////////////////////////////////
# ....................................................................
# Step 1a. Hyperparameters and Dataset paths
# ....................................................................
img_width, img_height, img_depth = 512, 512, 3
# Data organization - Training Data
train_dir = Path('OCT2017/train/')
val_dir = Path('OCT2017/val/')
#
# Training parameters
epochs = 100                    # For fitting the model
freq = 10                       # For saving the model
batch_size = 64                 # For fitting the model/ data augmentation parameters
batch_size_valid = 4           # For fitting the model/ data augmentation parameters
num_classes = 4                 # For the model architecture
#
# Learning rate
learning_rate = 0.00001         # For fitting the model
#
act_type = 'softmax'
class_mode = 'categorical'
loss_fun = 'categorical_crossentropy'
color_mode = 'rgb'
# ....................................................................
# Step 1b. Setup Data augmentation generators
# ....................................................................
# ImageDataGenerator
data_gen_args = dict(
    rescale = 1./255,
    brightness_range = [0.8,1.0],
    horizontal_flip = True,
    vertical_flip = True,
    rotation_range = 80.,
    zoom_range = 0.2,
    height_shift_range = 1.0,
    width_shift_range = 1.0,
    fill_mode = 'reflect')
#
datagen = ImageDataGenerator(
    **data_gen_args)
#
val_datagen = ImageDataGenerator(
    rescale = 1./255)
#
# Train data generator
train_generator = datagen.flow_from_directory(
    directory = train_dir,
    target_size = (img_width, img_height),
    class_mode = class_mode,
    color_mode = color_mode,
    batch_size = batch_size,
    shuffle = True)
#
# Validation data generator
valid_generator = val_datagen.flow_from_directory(
    directory = val_dir,
    target_size = (img_width, img_height),
    class_mode = class_mode,
    color_mode = color_mode,
    batch_size = batch_size_valid,
    shuffle = True)
# ....................................................................
# Step 1c. Training hyperparameters derived from the dataset
# ....................................................................
class_weights = compute_class_weight(
    class_weight = 'balanced',
    classes = np.unique(train_generator.classes),
    y = train_generator.classes
)
class_weights = dict(enumerate(class_weights))
#
nb_train_samples = len(train_generator.classes)
nb_valid_samples = len(valid_generator.classes)
#
steps_per_epoch_param = nb_train_samples // batch_size
validation_steps_param = nb_valid_samples // batch_size_valid
# ....................................................................
# ////////////////////////////////////////////////////////////////////
# ....................................................................
# Step 2a. Create a network model
# ....................................................................
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    """
    This is a basic model for a convolutional neural network (CNN).
    We leverage the Keras application for EfficientNetB3

    Architecture
    1. Input layer
    	We define the image size, unlike the previous MNIST dataset,
    	the input is an unflattened image
    2. Pre-trained Model
    	The EfficientNetB3 model with ImageNet pretrained weights.
    3. Flatten Layer
    	We convert the matrix output into a vector that is fully
    	connected to the following layer
	4. Dense Layer/Output Layer
		The output layer is a Densely connected layer that takes in all
		outputs from the feature layer.
    """
    input_layer = Input(shape = (img_width, img_height, img_depth))
    #
    efficient_model = efficientnet.EfficientNetB3(
            input_tensor = input_layer,
            include_top = False,
            weights = 'imagenet',
        )
    #
    flatten_layer = Flatten()(efficient_model.layers[-1].output)
    output_layer = Dense(num_classes, activation = act_type)(flatten_layer)
    #
    model = Model(input_layer, output_layer)
    #
    model.compile(
        loss = loss_fun,
        optimizer = Adam(lr = learning_rate),
        metrics = ['accuracy'])
# ....................................................................
# Step 2b. Train a model
# ....................................................................
model.fit(
	train_generator,
	epochs = epochs,
	steps_per_epoch = steps_per_epoch_param,
	validation_data = valid_generator,
	validation_steps = validation_steps_param,
	class_weight = class_weights
)