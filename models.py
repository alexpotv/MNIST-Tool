## @package models
#  @author Alexandre Potvin-Demers
#  @date 04-05-2020
#  @brief Contains functions which create models based on certain available datasets.
#  @details Contains functions which create models based on certain available datasets. As for now,
#  all models created are sequentials models.

import tensorflow as tf
import tensorflow_datasets as tfds
import data_setup

## @function new_model_MNIST
#  @brief Creates a new model from the available MNIST dataset in TensorFlow.
#  @details Creates a new model from the available MNIST dataset in TensorFlow. In order to do so,
#  the dataset is loaded as four different numpy arrays, which are used to train the model, and
#  then evaluate its metrics.
#  @return Returns a tuple of length 3 containing, in order, the loss of the model, the accuracy of
#  the model and the model itself.
def new_model_MNIST():
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

    model.fit(x_train, y_train, epochs=10)

    metrics = model.evaluate(x_test,  y_test, verbose=2)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    return (metrics[0], metrics[1], probability_model)


## @function new_model_EMNIST
#  @brief Creates a new model from the available EMNIST dataset in TensorFlow.
#  @details Creates a new model from the available MNIST dataset in TensorFlow. In order to do so,
#  the dataset is loaded as two different tf.data.dataset objects, which are used to train the
#  model, and then evaluate its metrics.
#  @return Returns a tuple of length 3 containing, in order, the loss of the model, the accuracy of
#  the model and the model itself.
def new_model_EMNIST():

  ## @function generator
  #  @brief A generator function for iterating through the dataset dictionary objects.
  #  @details A generator function for iterating through the dataset dictionary objects. Since a
  #  tf.data.dataset object is not subscriptable, it is passed to the model.fit and model.evaluate
  #  functions through a generator function. This function gathers the data from the dictionary and
  #  formats it so that it can be accepted by the model methods.
  #  @returns Returns a tuple of length 2 containing, in order, the input and output of the data
  #  point.
    def generator(dataset):
        for elem in dataset:
            yield (elem.get('image'), elem.get('label'))


    emnist_train = tfds.load('emnist/digits', split="train")
    emnist_test = tfds.load('emnist/digits', split="test")

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(generator(emnist_train), epochs=1)

    metrics = model.evaluate(generator(emnist_test), verbose=2)

    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    return (metrics[0], metrics[1], probability_model)
