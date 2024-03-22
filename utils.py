
import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.utils import to_categorical


def get_label(dir_name, file_name, file_labels=None):
    if file_labels is None:
        return dir_name
    for label in file_labels:
        if label in file_name:
            return label

def load_data_from_path(base_dir, img_height, img_width, file_labels=None):
    dataset = []
    labels = []
    img_size = (img_height, img_width)
    for dir in os.scandir(base_dir):
        if not dir.is_dir():
            continue
        for file in os.scandir(dir.path):
            # load and rescale to be between 0 and 1
            img = load_img(file.path, target_size=img_size, color_mode='grayscale')
            dataset.append(img)
            labels.append(get_label(dir.name, file.name, file_labels))
    dataset = np.array(dataset, dtype='float32')
    labels = np.array(labels)
    return dataset, labels

def shuffle_dataset(dataset, labels):
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    return dataset[indices], labels[indices]

def load_data(path, img_height, img_width, all_labels, shuffle=False, file_labels=None):
    dataset, labels = load_data_from_path(path, img_height, img_width, file_labels)
    # TODO: check if needed or delete it: 
    dataset /= 255 # rescale to be between 0 and 1
    labels = np.array([all_labels.index(label) for label in labels]) # one-hot encoding
    if len(all_labels) > 2:
        labels = to_categorical(labels)
    if shuffle:
        dataset, labels = shuffle_dataset(dataset, labels)
    dataset = np.expand_dims(dataset, -1) # image should be (height, width, channels)
    labels = np.expand_dims(labels, -1)
    return dataset, labels
# ---------------------------------------------------------------------


# TODO: delete it
def show_img(image):
    # Display the image with matplotlib
    plt.imshow(image.numpy(), cmap='gray')
    plt.axis('off')  # Optionally, turn off the axis.
    plt.show()
'''
# how to use:
for images, labels in train_ds:
    show_img(images[0])
    break
'''    

    
# TODO: delete it
def show_histogram(image):
    # Display the histogram
    
    # show the values on the x-axis and the frequency on the y-axis
    plt.hist(np.array(image).ravel(), bins=256)
    plt.show()
'''
# how to use:
for images, labels in train_ds.take(1):
    normalized_image = normalization_layer(images[0][tf.newaxis, ...])
    show_histogram(normalized_image)
    break
'''