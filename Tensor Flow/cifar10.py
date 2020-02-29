import tensorflow as tf
import numpy as np


# Load CIFAR-10 dataset
data = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = data.load_data()
# Normalize the data
x_train = x_train/255.0
x_test = x_test/255.0
# Input shape 32x32 pixel images with RGB channels (Matrix size 32x32x3)
input_shape = (32, 32, 3)
# Build model (Two Convolution layers with Max Pooling and three Dense layers and then output layer)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.relu, input_shape=input_shape),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(512, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])
# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit model with 5 epochs
model.fit(x_train, y_train, epochs=15)
# Predict for test images
prediction = model.predict(x_test)
# Print Test labels and predicted output for test images
for i in range(len(x_test)):
    print("Image {}: Actual: {}, Predicted: {}".format(str(i+1), str(y_test[i]), str(np.argmax(prediction[i]))))
