import tensorflow as tf
import numpy as np


# Load MNIST dataset
data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()
# Normalize the data
x_train = x_train/255.0
x_test = x_test/255.0
# Input shape 28x28 pixel images
input_shape = (28, 28)
# Build model (One input layer, one dense layer and an output layer)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=input_shape),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit model with 5 epochs
model.fit(x_train, y_train, epochs=5)
# Predict for test images
prediction = model.predict(x_test)
# Print Test labels and predicted output for test images
for i in range(len(x_test)):
    print("Image {}: Actual: {}, Predicted: {}".format(str(i+1), str(y_test[i]), str(np.argmax(prediction[i]))))
