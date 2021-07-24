import sys
from PIL import Image, ImageFilter

# Computer Vision
# Inputting each RGB pixel value of a picture to a Neural Network would be infeasible.
# Image convolution is applying a filter that adds each pixel value of an image to its neighbors,
# weighted according to a kernel matrix. In this way features from the image can be extracted.
# A popular kernel matrix for edge detection looks like this:
# [[-1, -1, -1],
#  [-1,  8, -1],
#  [-1, -1, -1]]
# When a pixel is the same as its neighbors, they should cancel each other out and produce the output
# pixel value of 0. When the pixel are different, a high value is produced.
# To reduce the size of an image, pooling is used. Max-Pooling takes the highest value from a n x n
# pixel matrix and creates a new picture from this output.
# The convolution and pooling steps can be repeated multiple times to extract additional features
# and reduce the size of the input to the neural network.

# Ensure correct usage
if len(sys.argv) != 2:
    sys.exit("Usage: python filter.py filename")

# Open image
image = Image.open(sys.argv[1]).convert("RGB")

# Filter image according to edge detection kernel
filtered = image.filter(ImageFilter.Kernel(
    size=(3, 3),
    kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
    scale=1
))

# Show resulting image
filtered.show()
