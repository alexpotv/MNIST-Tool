import tensorflow as tf
import tensorflow_datasets as tfds
import data_setup

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


def new_model_EMNIST():
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
