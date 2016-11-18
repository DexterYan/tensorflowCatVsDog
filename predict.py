import numpy as np
import os
import tflearn

from glob import glob
from skimage import color, io
from scipy.misc import imresize
from PIL import Image

import matplotlib.pyplot as plt
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

files_path = './test'
test_file_path = os.path.join(files_path, '*.jpg')

test_file = sorted(glob(test_file_path))
n_file = len(test_file)

size_image = 64

X = np.zeros((n_file, size_image, size_image, 3), dtype='float64')
count = 0

for idx, f in enumerate(test_file):
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        X[idx] = np.array(new_img)

    except:
        continue

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
acc = Accuracy(name="Accuracy")
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.0005, metric=acc)

# Wrap the network in a model object
model = tflearn.DNN(network)
model.load('model_cat_dog_final.tflearn')
count = 0
result = model.predict(X)

for idx, f in enumerate(test_file):
    predicted_class= np.argmax(result[idx])
    if predicted_class == 0:
        label = "cat"
    else :
        label = "dog"
    print('\nThis is a {}'.format(label))
    print(result[idx])
    image = Image.open(f)
    arr = np.asarray(image)
    plt.imshow(arr)
    plt.axis('off')
    plt.title('This is a {}'.format(label))
    plt.show()
