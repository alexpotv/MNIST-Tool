import tensorflow as tf
import data_setup

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

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

test_0 = data_setup.prepare("images/0_28.jpg")
test_1 = data_setup.prepare("images/1_28.jpg")
test_2 = data_setup.prepare("images/2_28.jpg")
test_3 = data_setup.prepare("images/3_28.jpg")
test_4 = data_setup.prepare("images/4_28.jpg")
test_5 = data_setup.prepare("images/5_28.jpg")
test_6 = data_setup.prepare("images/6_28.jpg")
test_7 = data_setup.prepare("images/7_28.jpg")
test_8 = data_setup.prepare("images/8_28.jpg")
test_9 = data_setup.prepare("images/9_28.jpg")

# Prediction of the test image
prediction0 = probability_model.predict([test_0[:1]])[0]
prediction1 = probability_model.predict([test_1[:1]])[0]
prediction2 = probability_model.predict([test_2[:1]])[0]
prediction3 = probability_model.predict([test_3[:1]])[0]
prediction4 = probability_model.predict([test_4[:1]])[0]
prediction5 = probability_model.predict([test_5[:1]])[0]
prediction6 = probability_model.predict([test_6[:1]])[0]
prediction7 = probability_model.predict([test_7[:1]])[0]
prediction8 = probability_model.predict([test_8[:1]])[0]
prediction9 = probability_model.predict([test_9[:1]])[0]

print("The best guess for 0_28.jpg is: ", prediction0.argmax())
print("The best guess for 1_28.jpg is: ", prediction1.argmax())
print("The best guess for 2_28.jpg is: ", prediction2.argmax())
print("The best guess for 3_28.jpg is: ", prediction3.argmax())
print("The best guess for 4_28.jpg is: ", prediction4.argmax())
print("The best guess for 5_28.jpg is: ", prediction5.argmax())
print("The best guess for 6_28.jpg is: ", prediction6.argmax())
print("The best guess for 7_28.jpg is: ", prediction7.argmax())
print("The best guess for 8_28.jpg is: ", prediction8.argmax())
print("The best guess for 9_28.jpg is: ", prediction9.argmax())