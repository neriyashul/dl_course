import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam, SGD
from keras.losses import BinaryFocalCrossentropy
import random
import matplotlib.pyplot as plt

np.random.seed(100)
random.seed(100)
tf.random.set_seed(100)


def show_img(image):
    plt.imshow(np.array(image))
    plt.axis('off')  # Optionally, turn off the axis.
    plt.show()




base_dir = os.path.join(os.getcwd(), "chest_xray")
# base_dir = "/Users/neriya.shulman/content/chest_xray/chest_xray"

train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

def load_images(dir_path, image_size=(128, 128)):
    images = []
    files = os.listdir(dir_path)
    for i, filename in enumerate(files):
        if i % 100 == 0:
            print(f'Loading {os.path.basename(dir_path)} {i/len(files):.2}%')
        if filename == '.DS_Store':
            continue
        img_path = os.path.join(dir_path, filename)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img)
            
            images.append(img)
    images = np.array(images)
    images = images.astype('float32') / 255.0
    return images

train_images = load_images(os.path.join(train_dir, 'NORMAL'))
val_images = load_images(os.path.join(val_dir, 'NORMAL'))


import copy
def add_blank_pixels(train_images_origs):
    print("Add blank pixels")
    new_list = copy.deepcopy(train_images_origs)

    for idx in range(new_list.shape[0]):
        image_shape = new_list[idx].shape
        lst_base = np.random.choice(image_shape[0] * image_shape[1],
                               size=int((image_shape[0] * image_shape[1]) / 20),
                               replace=False)
        lst = [[x*3, x*3+1, x*3+2] for x in lst_base]
        new_list[idx].flat[np.array(lst).flatten()] = 0

    show_img(train_images_origs[-1])
    show_img(new_list[-1])
    return new_list


def set_partial_image(train_images_origs):
    print("Set partial images")
    new_list = copy.deepcopy(train_images_origs)
    for idx in range(new_list.shape[0]):
        image_shape = new_list[idx].shape
        new_list[idx][:, ::5] = (0.0, 0.0, 0.0)

    show_img(train_images_origs[-1])
    show_img(new_list[-1])
    return new_list


def reverse_pixels_values(train_images_origs):
    print("Reverse pixels values")
    new_list = copy.deepcopy(train_images_origs)
    new_list = 1 - new_list

    show_img(train_images_origs[-1])
    show_img(new_list[-1])
    return new_list

def add_augmentations(train_images_origs):
    new_list = add_blank_pixels(train_images_origs)
    new_list2 = set_partial_image(train_images_origs)
#    new_list3 = reverse_pixels_values(train_images_origs)
    print("Collect all augmentations into one array.")
    new_list = np.concatenate((copy.deepcopy(train_images_origs),
                               new_list,
                               new_list2,
#                               new_list3
                               ))
    train_image_copies = np.concatenate((copy.deepcopy(train_images_origs),
                                         copy.deepcopy(train_images_origs),
                                         copy.deepcopy(train_images_origs),
#                                         copy.deepcopy(train_images_origs)
                                         ))
    print(train_images_origs.shape, new_list.shape, train_image_copies.shape)
    return new_list, train_image_copies


train_images_to_practice, train_images_to_compare = add_augmentations(train_images)
#train_images_to_practice, train_images_to_compare = train_images, train_images

# input_shape = (64,64,3)
input_shape = train_images.shape[1:]
print('shape:', input_shape)
encoded_dim = 32

encoder = Sequential()
encoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# encoder.add(Flatten())
# encoder.add(Dense(encoded_dim, activation='relu'))

decoder = Sequential()
# decoder.add(Dense(16 * 16 * 16, activation='relu', input_dim=encoded_dim))
# decoder.add(Reshape((16, 16, 16)))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
decoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse', metrics=["accuracy", "recall", "precision"])

autoencoder.fit(train_images_to_practice, train_images_to_compare,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(val_images, val_images))

# Reconstruct the input images
reconstructed_images = autoencoder.predict(val_images)
show_img(reconstructed_images[0])






'''

BEWARE - NOT TESTED - DON'T KNOW IF IT WORKS!!!!!!

'''

# Calculate the reconstruction error
reconstruction_error = np.mean(np.square(val_images - reconstructed_images), axis=(1, 2, 3))

# Set a threshold for anomaly detection
threshold = np.percentile(reconstruction_error, 95)

# Classify images as normal or anomalous
anomalous_indices = np.where(reconstruction_error > threshold)[0]
normal_indices = np.where(reconstruction_error <= threshold)[0]


#
# # 6. find most anomalous data item
# N = len(norm_x)
# max_se = 0.0; max_ix = 0
# predicteds = autoenc.predict(norm_x)
# for i in range(N):
#   diff = norm_x[i] - predicteds[i]
#   curr_se = np.sum(diff * diff)
#   if curr_se > max_se:
#     max_se = curr_se; max_ix = i