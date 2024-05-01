# Python code to test our output against :
import tensorflow as tf
import numpy as np

# Define the input matrix
input_matrix = np.ones((2, 2))
# Reshape the input matrix to match the expected shape of the Conv2D layer
input_matrix = np.reshape(input_matrix, (1, 2, 2, 1))
print("input_matrix:\n",input_matrix)
# Add padding to the input matrix
input_matrix = tf.pad(input_matrix, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
print("input_matrix:\n",input_matrix)
# Define the custom filter
custom_filter = np.ones((2, 2, 1, 1))
# Define a function that returns the custom filter
def my_init(shape, dtype=None):
    return custom_filter
# Pass the custom filter to the conv_layer
conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2), strides=1, kernel_initializer=my_init)
# Apply the convolution operation
output = conv_layer(input_matrix)
# Print the output
print(output.numpy()[0, :, :, 0])