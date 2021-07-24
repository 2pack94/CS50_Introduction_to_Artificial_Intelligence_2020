import cv2
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Use a TensorFlow neural network to classify road signs based on an image of those signs.
# The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used.
# Download (188 MB): https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip
# Each numbered subdirectory contains a collection of images of a type/ category of traffic sign.

# NumPy Tutorial: https://numpy.org/devdocs/user/quickstart.html
# OpenCV Basic Operations: https://docs.opencv.org/4.5.2/d3/df2/tutorial_py_basic_ops.html
# Tensorflow Tutorial: https://www.tensorflow.org/guide/keras/sequential_model?hl=en

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = loadData(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = getModel()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def loadData(data_dir):
    """
    Loads image data from directory `data_dir`.
    `data_dir` has one directory named after each category, numbered 0 through NUM_CATEGORIES - 1.
    Inside each category directory is a number of image files.
    Returns tuple `(images, labels)`. `images` is a list of all of the images in the data directory,
    where each image is formatted as a numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3 (3 color channels).
    `labels` is a list of integer labels, representing the categories for each of the corresponding `images`.
    """
    images = []
    labels = []

    for dir in range(NUM_CATEGORIES):
        img_names = os.listdir(os.path.join(data_dir, str(dir)))
        for img_name in img_names:
            img = cv2.imread(os.path.join(data_dir, str(dir), img_name))
            # The images all have different sizes, so they need to be resized.
            # This will output a numpy ndarray with the desired dimensions.
            # Divide by 255 to have a value range of 0 to 1.
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255
            images.append(img)
            labels.append(dir)

    return (images, labels)


def getModel():
    """
    Returns a compiled convolutional neural network model.
    The `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer has `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer. Learn 32 filters using a 3x3 kernel.
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
        ),
        # Max-pooling layer, using a 2x2 pool size.
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        # add dropout
        tf.keras.layers.Dropout(0.2),
        # Flatten images to a one-dimensional array
        tf.keras.layers.Flatten(),
        # Add a hidden layer with 128 units
        tf.keras.layers.Dense(128, activation='relu'),
        # Add an output layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    print(model.summary())

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()


# Experimentation Results:

# Configuration 1 (Reference):
#     Conv2D: 32, (3, 3)
#     MaxPooling2D: pool_size=(2, 2)
#     Dropout: 0.2
#     Dense: 128
# Output:
#     training accuracy: 0.9888
#     testing accuracy: 0.9716

# Configuration 2 (2 convolutional layers):
#     Conv2D: 32, (3, 3)
#     MaxPooling2D: pool_size=(2, 2)
#     Conv2D: 64, (3, 3)
#     MaxPooling2D: (2, 2)
#     Dropout: 0.2
#     Dense: 128
# Output:
#     training accuracy: 0.9897
#     testing accuracy: 0.9830

# Configuration 3 (2 hidden layers):
#     Conv2D: 32, (3, 3)
#     MaxPooling2D: pool_size=(2, 2)
#     Dropout: 0.2
#     Dense: 128
#     Dense: 128
# Output:
#     training accuracy: 0.9846
#     testing accuracy: 0.9729

# Configuration 4 (256 units in the hidden layer):
#     Conv2D: 32, (3, 3)
#     MaxPooling2D: pool_size=(2, 2)
#     Dropout: 0.2
#     Dense: 256
# Output:
#     training accuracy: 0.9862
#     testing accuracy: 0.9651

# Configuration 5 (64 units in the hidden layer):
#     Conv2D: 32, (3, 3)
#     MaxPooling2D: pool_size=(2, 2)
#     Dropout: 0.2
#     Dense: 64
# Output:
#     training accuracy: 0.9816
#     testing accuracy: 0.9641

# Configuration 6 (0.5 dropout):
#     Conv2D: 32, (3, 3)
#     MaxPooling2D: pool_size=(2, 2)
#     Dropout: 0.5
#     Dense: 128
# Output:
#     training accuracy: 0.9646
#     testing accuracy: 0.9715

# Configuration 7 (no convolutional layers):
#     Dropout: 0.2
#     Dense: 128
# Output:
#     training accuracy: 0.8333
#     testing accuracy: 0.8836
