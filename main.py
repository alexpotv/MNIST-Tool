import tensorflow as tf
import data_setup

# Creates a new MNIST Sequential model
# Returns a tuple containing the loss, accuracy and the model itself
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

model_test = new_model_MNIST()

print("Created model with the following metrics:")
print("Loss: ", model_test[0])
print("Accuracy: ", model_test[1])
model_test[2].save("models/MNIST_model.h5")
print("Model saved")
