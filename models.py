## @package models
#  @author Alexandre Potvin-Demers
#  @date 04-05-2020
#  @brief Contains functions which create models based on certain available datasets.
#  @details Contains functions which create models based on certain available datasets. As for now,
#  all models created are sequentials models.

import tensorflow as tf
import tensorflow_datasets as tfds
from itertools import cycle

## @function generator
#  @brief A generator function for iterating through the dataset dictionary objects.
#  @details A generator function for iterating through the dataset dictionary objects. Since a
#  tf.data.dataset object is not subscriptable, it is passed to the model.fit and model.evaluate
#  functions through a generator function. This function gathers the data from the dictionary and
#  formats it so that it can be accepted by the model methods.
#  @param dataset The dataset to iterate through
#  @returns Returns a tuple of length 2 containing, in order, the input and output of the data
#  point.
def _generator(dataset):
    for elem in dataset:
        yield (elem.get('image'), elem.get('label'))
    return


def generator(dataset):
    for elem in cycle(dataset):
        yield (elem.get('image'), elem.get('label'))

## @function newModelMNIST
#  @brief Creates a new model from the available MNIST dataset in TensorFlow.
#  @details Creates a new model from the available MNIST dataset in TensorFlow. In order to do so,
#  the dataset is loaded as four different numpy arrays, which are used to train the model, and
#  then evaluate its metrics.
#  @param epochNum The number of epochs to run during the training of the model
#  @return Returns a tuple of length 3 containing, in order, the loss of the model, the accuracy of
#  the model and the model itself.
def newModelMNIST(p_epochNum, p_batchSize):

    datasets = tfds.load(name='mnist', as_supervised=True)

    mnist_train = datasets['train'].batch(p_batchSize).repeat()
    mnist_test = datasets['test'].batch(p_batchSize).repeat()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(mnist_train, epochs=p_epochNum, steps_per_epoch=(60000//p_batchSize))

    metrics = model.evaluate(mnist_test, verbose=2, steps=(60000//p_batchSize))

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    return (metrics[0], metrics[1], probability_model)


## @function newModelEMNIST
#  @brief Creates a new model from the available EMNIST dataset in TensorFlow.
#  @details Creates a new model from the available MNIST dataset in TensorFlow. In order to do so,
#  the dataset is loaded as two different tf.data.dataset objects, which are used to train the
#  model, and then evaluate its metrics.
#  @param epochNum The number of epochs to run during the training of the model
#  @param p_batchSize The number of data points in one batch
#  @return Returns a tuple of length 3 containing, in order, the loss of the model, the accuracy of
#  the model and the model itself.
def newModelEMNIST(p_epochNum, p_batchSize):

    datasets = tfds.load(name='emnist/digits', as_supervised=True)

    emnist_train = datasets['train'].batch(p_batchSize).repeat()
    emnist_test = datasets['test'].batch(p_batchSize).repeat()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(emnist_train, epochs=p_epochNum, steps_per_epoch=(240000//p_batchSize))

    metrics = model.evaluate(emnist_test, verbose=2, steps=(240000//p_batchSize))

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    return (metrics[0], metrics[1], probability_model)
