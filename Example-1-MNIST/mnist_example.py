"""
Deep Learning using Tensorflow on MNIST Dataset

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
import tensorflow_datasets as tfds
# ////////////////////////////////////////////////////////////////////
# ....................................................................
# Step 1a. Load the dataset
# ....................................................................
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
# ....................................................................
# Step 1b. Building a training pipeline
# ....................................................................
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
# ....................................................................
# Step 1c. Build an evaluation pipeline
# ....................................................................
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
# ....................................................................
# ////////////////////////////////////////////////////////////////////
# ....................................................................
# Step 2a. Create a network model
# ....................................................................
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    '''
    This is a basic model for an artificial neural network (ANN).

    Architecture
    1. Input layer
        The input layer is simply the image, which has been flatten from
        a 28 x 28 pixels image into a 784 x 1 pixels.
    2. Dense layer
        A layer of n number of neurons, e.g., 128 with an ReLU activation function
    3. Output layer
        The output layer is similarly a dense layer, however with 10 neurons for the
        10 classes of digits on the MNIST dataset.
    '''
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    #
    model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
# ....................................................................
# Step 2b. Train a model
# ....................................................................
model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
)
