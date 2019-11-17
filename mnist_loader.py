''' 
Python Functions to load MNIST image and training data. 
The data resides in four gzipped files stored at
http://yann.lecun.com/exdb/mnist/. The code below will
download these files if they do not already exist locally.

   - loadImages reads in image data
   - loadLabels reads the corresponding label data, one for each image.
   - mnist_load packs the downloaded image and label data into a combined 
     format to be read later by our neural network.
'''
import urllib.request
import gzip
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

def loadImages(src, n_images=None):
    '''
    loadImages reads image data from the MNIST image data files.

    It checks for the existence of a 'src' file and if it is not present,
    it downloads it. 

    Next, it opens the file and reads in the first 4 bytes of the data. 
    That data is actually a 32-bit (4 byte) integer and is known as a
    "magic number" (See description at the URL given above.) The magic 
    number tells us whether the file contains image data or label data. 
    If it detects anything but an image file, it prints an error
    message and quits. (We will load label data in another function.)

    It then reads the next 4 bytes, another 32-bit integer that tells us 
    the toal number of images stored in the file. It will load all of the images
    if n_images was not passed in, otherwise it will only read in n_images,
    after first checking that n_images <= total images.

    Inputs:
            src: The name of a source file from which to read
       n_images: (optional) A count of how many images to load. If not
                 present, then all images are loaded.
                           
    '''
    # This is where the data files are located
    URLROOT = 'http://yann.lecun.com/exdb/mnist/'
    # First check to see if the file exists and if not, download it. The line
    # below creates a boolean value based on the existence of the file.
    # 'True' means the file exists, 'False' means it not.
    SRCExists = os.path.exists(src) 
    if not SRCExists:
       print ('Downloading ' + URLROOT + src)
       gzfname, h = urllib.request.urlretrieve(URLROOT+src, src)
       print ('Done.')
    else:
       gzfname = src
    # Now that we have the source file, try reading it and throw 
    # exceptions (error flags) if something bad happens.
    try:
        # Open a gzip'd (compressed) data file
        with gzip.open(gzfname) as gz:
            # Read in the first 4 bytes of the file
            n = struct.unpack('I', gz.read(4))
            # Read magic number. The binary data was stored as 
            # most-significant-bytes first. Modern PCs read in data in 
            # least-significant-bytes first. So if the magic number (in hex
            # format) is 0x00000803 (2051 in decimal) then our Python code
            # will actually read in the last 4-bytes first, and then the first 
            # 4 bytes. That is, Python will load this value as 0x0803 followed
            # by 0x0000. So this is how we check for the correct magic number
            # value.
            if n[0] != 0x03080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries from the next 4 bytes (32-bits)
            n = struct.unpack('>I', gz.read(4))[0]
            # Check that we haven't been asked to read in more than n images
            if n_images == None:
                n_images = n   # Set the number of images to read to the number
                               # present
            elif n_images > n:
                raise Exception('Unable to read {0} entries from data file.'.format(n_images))
            # Now start loading images. The next 8 bytes (4 bytes at a time) tell
            # us the number of rows and the number of columns for each image
            n_rows = struct.unpack('>I', gz.read(4))[0]
            n_cols = struct.unpack('>I', gz.read(4))[0]
            # Make sure we are working with the data containing 28x28 pixels
            if n_rows != 28 or n_cols != 28:
                raise Exception('Invalid file: expected 28 rows & cols per img')
            # The remainder of the data file is one continuous stream of binary
            # data, where every 28x28=784 bytes is one separate image. We will
            # read this data into one single array of bytes
            res = np.frombuffer(gz.read(n_images * n_rows * n_cols), dtype = np.uint8)
    finally:
        # Reshape 'res' into an array of images. There will be 'n_images' images
        # each of size n_rows * n_cols
        return res.reshape((n_images, n_rows * n_cols))
 
def loadLabels(src, n_images=None):
    '''
    loadLabels is very much like loadImages.
    '''
    URLROOT = 'http://yann.lecun.com/exdb/mnist/'
    SRCExists = os.path.exists(src) 
    if not SRCExists:
        print ('Downloading ' + URLROOT + src)
        gzfname, h = urllib.request.urlretrieve(URLROOT+src, src)
        print ('Done.')
    else:
        gzfname = src
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number. Check that we have a "label" file
            if n[0] != 0x01080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            # Check that we haven't been asked to read in more than n images
            if n_images == None:
                n_images = n[0]    # Set the number of images to read to the
                                   # number present
            elif n_images > n[0]:
                raise Exception('Unable to read {0} entires from label file.'.format(n_images))
            # Read labels. The remaining file is just a stream of integers that
            # label the corresponding images from the image file.
            res = np.frombuffer(gz.read(n_images), dtype = np.uint8)
    finally: 
        # Return an array of these integers
        return res.reshape((n_images, 1))

def mnist_load(imageSrc, labelsSrc, n_images):
    '''
    Inputs:
       imageSrc: The name of a source file containing image data
       labelSrc: The name of a file containing image labels
           n_images: How many image samples to read
    '''
    images = loadImages(imageSrc, n_images)
    labels = loadLabels(labelsSrc, n_images)
    vectorized_labels = []
    for label in labels:
        vd = vectorized_digit(label)
        vectorized_labels.append(vd)
    return images, vectorized_labels

def vectorized_digit(j):
    '''
    Given a digit 'j', return a 10-element vector representing that digit.
    The vector is all zeros excpet for the jth element, which is 1.
    '''
    # Create a 10-element array of zeros.
    e = np.zeros((10, 1))
    # Set the jth element to 1.0
    e[j] = 1.0
    return e

# Code to test out the library
if __name__ == "__main__":

    # Training image and label data
    training_image_file = 'train-images-idx3-ubyte.gz'
    training_label_file = 'train-labels-idx1-ubyte.gz'
    nTrainingSamples = 60000

    # Test image and label data
    test_image_file = 't10k-images-idx3-ubyte.gz'
    test_label_file = 't10k-labels-idx1-ubyte.gz'
    nTestSamples = 10000

    print("Loading training data.")
    training_images, training_labels = mnist_load(training_image_file, training_label_file, nTrainingSamples)

    print("Loading test data.")
    testing_images, testing_labels = mnist_load(test_image_file, test_label_file, nTestSamples)

    # visualize a random sample from the dataset
    sample_number = 1
    plt.imshow(training_images[sample_number].reshape(28,28), cmap="gray_r")
    plt.axis('off')
    plt.show()
    print("Image Label: ", training_labels[sample_number])

    sample_number = 5
    plt.imshow(training_images[sample_number].reshape(28,28), cmap="gray_r")
    plt.axis('off')
    plt.show()
    print("Image Label: ", training_labels[sample_number])
