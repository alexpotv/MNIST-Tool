## @package models
#  @author Alexandre Potvin-Demers
#  @date 04-05-2020
#  @brief Contains functions which create models based on certain available datasets.
#  @details Contains functions which create models based on certain available datasets. As for now,
#  all models created are sequentials models.

import tensorflow as tf
import tensorflow_datasets as tfds
from itertools import cycle

"""
class multiGenerator(object):
    def __init__(self, gFunction, *args, **kwargs):
        self.__gFunction = gFunction
        self.__args = args
        self.__kwargs = kwargs
        self.__gObject = None
    def __iter__(self):
        return self
    def next(self):
        if self.__gObject is None:
            self.__gObject = self.__gFunction(*self.__args, **self.__kwargs)
        try:
            return self.__gObject.next()
        except StopIteration:
            self.__gObject = None
            raise
"""

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
def newModelMNIST(epochNum):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    predictions = model(x_train[:1]).numpy()

    tf.nn.softmax(predictions).numpy()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    loss_fn(y_train[:1], predictions).numpy()

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochNum)

    metrics = model.evaluate(x_test,  y_test, verbose=2)

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
#  @return Returns a tuple of length 3 containing, in order, the loss of the model, the accuracy of
#  the model and the model itself.
def newModelEMNIST(epochNum):

    datasets = tfds.load(name='emnist/digits', as_supervised=True)

    emnist_train = datasets['train']
    emnist_test = datasets['test']

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0

        return image, label

    emnist_train = emnist_train.map(scale).shuffle(10000).batch(240000)
    emnist_test = emnist_test.map(scale).batch(240000)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(emnist_train, epochs=epochNum, steps_per_epoch=240000)

    metrics = model.evaluate(emnist_test, verbose=2)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    return (metrics[0], metrics[1], probability_model)
