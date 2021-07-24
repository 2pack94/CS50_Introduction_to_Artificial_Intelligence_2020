import sys
import tensorflow as tf

# A convolutional neural network (CNN) uses convolution and pooling, usually for analyzing images.
# First, one or more filters are used to create feature maps that extract different features from the image.
# After pooling, all resulting images are flattened (all pixels are put into an array) and fed into a
# traditional ANN. The kernel matrix elements can be trained like the other weights of the ANN.

# Basic Image Classification: https://www.tensorflow.org/tutorials/keras/classification?hl=en
# CNN Tutorial: https://www.tensorflow.org/tutorials/images/cnn?hl=en

# Test the handwriting recognition of a TensorFlow neural network.

# Use MNIST handwriting dataset (contains pictures of black and white handwritten digits).
mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

# Create a convolutional neural network
model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel.
    # The handwritten digits are on a 28 x 28 pixel grid and have only 1 color channel.
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),

    # Max-pooling layer, using a 2x2 pool size.
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten images to a one-dimensional array
    tf.keras.layers.Flatten(),

    # Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),

    # Add an output layer with output units for all 10 digits
    # softmax: The output will be turned into a probability distribution
    tf.keras.layers.Dense(10, activation="softmax")
])

# Train neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(x_train, y_train, epochs=10)

# Evaluate neural network performance
model.evaluate(x_test, y_test, verbose=2)

# Optionally save model to file
# python3 handwriting.py model.h5
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")
