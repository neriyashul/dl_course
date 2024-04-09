
import os
import random
import shutil
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical



def get_label(dir_name, file_name, file_labels=[]):
    for label in file_labels:
        if label in file_name:
            return label
    return dir_name

def load_data_from_path(base_dir, img_height, img_width, file_labels=[]):
    dataset = []
    labels = []
    img_size = (img_height, img_width)
    dir_names = os.path.basename(base_dir)
    for file in os.scandir(base_dir):
        if file.name == '.DS_Store':
            continue
        if file.is_dir():
            dir_images, dir_labels = load_data_from_path(file.path, img_height, img_width, file_labels)
            dataset.extend(dir_images)
            labels.extend(dir_labels)
        else:
            img = img_to_array(load_img(file.path, target_size=img_size, color_mode='grayscale'))
            dataset.append(img)
            labels.append(get_label(dir_names, file.name, file_labels))
    dataset = np.array(dataset, dtype='float32')
    labels = np.array(labels)
    return dataset, labels

def shuffle_dataset(dataset, labels):
    indices = np.arange(dataset.shape[0])
    np.random.shuffle(indices)
    return dataset[indices], labels[indices]

def load_data(path, img_height, img_width, all_labels, shuffle=False, file_labels=[]):
    dataset, labels = load_data_from_path(path, img_height, img_width, file_labels)
    # TODO: check if needed or delete it: 
    dataset /= 255 # rescale to be between 0 and 1
    labels = np.array([all_labels.index(label) for label in labels]) # one-hot encoding
    if len(all_labels) > 2:
        labels = to_categorical(labels, len(all_labels))
    dataset = np.expand_dims(dataset, -1) # image should be (height, width, channels)
    labels = np.expand_dims(labels, -1)
    return dataset, labels
# ---------------------------------------------------------------------


# TODO: delete it
def show_img(image):
    # Display the image with matplotlib
    plt.imshow(np.array(image), cmap='gray')
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

def move_files(source_dir, dest_dir, num_to_move, filter_fn=lambda x: True):
    filenames = [f.name for f in os.scandir(source_dir) if f.is_file() and filter_fn(f.name)]
    np.random.shuffle(filenames)
    for filename in filenames[:num_to_move]:
        src_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(dest_dir, filename)
        shutil.move(src_path, dest_path)

def add_samples_to_validation(base_dir, num_to_move=100):
    source_folder = os.path.join(base_dir, 'train')
    destination_folder = os.path.join(base_dir, 'val')

    for dir_name in ['NORMAL', 'PNEUMONIA']:
        source_dir = os.path.join(source_folder, dir_name)
        dest_dir = os.path.join(destination_folder, dir_name)
        os.makedirs(dest_dir, exist_ok=True)

        if dir_name == 'NORMAL':
            move_files(source_dir, dest_dir, num_to_move * 2)
        else:
            move_files(source_dir, dest_dir, num_to_move, lambda x: 'bacteria' in x)
            move_files(source_dir, dest_dir, num_to_move, lambda x: 'virus' in x)
