### General Imports ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

### Autoencoder ###
import tensorflow as tf
import tensorflow.keras
from keras.optimizers import Adam
from keras.metrics import Recall, Precision

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from keras.callbacks import TensorBoard

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist


img_width, img_height, img_channels = (256, 256, 3)
img_dims = (img_width, img_height, img_channels)

######################################################################################################
def load_images(dir_path, image_size=(img_width, img_height)):
    images = []
    files = os.listdir(dir_path)
    for i, filename in enumerate(files):
        if i % 100 == 0:
            print(f'Loading {os.path.basename(dir_path)} {i / len(files):.2}%')
        if filename == '.DS_Store':
            continue
        img_path = os.path.join(dir_path, filename)
        if os.path.isfile(img_path):
            img = load_img(img_path, target_size=image_size)
            img = img_to_array(img)
            images.append(img)
    images = np.array(images)
    return images.astype('float32') / 255.0


# Load grayscale images from a directory
training_normal_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("train", "NORMAL"))
training_normal_images = load_images(training_normal_image_directory)
training_pneumonia_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("train", "PNEUMONIA"))
training_pneumonia_images = load_images(training_pneumonia_image_directory)

test_normal_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("test", "NORMAL"))
test_normal_images = load_images(test_normal_image_directory)
test_pneumonia_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("test", "PNEUMONIA"))
test_pneumonia_images = load_images(test_pneumonia_image_directory)
######################################################################################################
# X_train, X_test = training_normal_images, test_pneumonia_images
# print(X_train.shape)
#
# (X_train, _), (X_test, _) = fashion_mnist.load_data()
# #(X_train, _), (X_test, _) = mnist.load_data()
#
# plt.imshow(X_train[0,:,:], cmap='gray_r')
# plt.axis("off")
# plt.show()
#
#
# X_train = X_train.astype('float32') / 255.
# X_test = X_test.astype('float32') / 255.
#
# X_train = X_train.reshape((X_train.shape[0] , *img_dims))
# X_test = X_test.reshape((X_test.shape[0] , *img_dims))
#
# # plt.imshow(X_train[0,:,:], cmap='gray_r')
# # plt.axis("off")
# # plt.show()
################################################################################################
# Augmentation:
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
    return new_list


def set_partial_image(train_images_origs):
    print("Set partial images")
    new_list = copy.deepcopy(train_images_origs)
    for idx in range(new_list.shape[0]):
        image_shape = new_list[idx].shape
        new_list[idx][:, ::5] = (0.0, 0.0, 0.0)
    return new_list


def reverse_pixels_values(train_images_origs):
    print("Reverse pixels values")
    new_list = copy.deepcopy(train_images_origs)
    new_list = 1 - new_list
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

#train_images_to_practice, train_images_to_compare = add_augmentations(training_normal_images)

################################################################################################
train_images_to_practice, train_images_to_compare = training_normal_images, training_normal_images

input_img = Input(shape=(img_width, img_height, img_channels))

# Ecoding
x = Conv2D(8, (3, 3), padding='same', activation='relu')(input_img)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(16,(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(32,(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(64,(3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2), padding='same')(x)
x = Conv2D(128,(3, 3), padding='same', activation='relu')(x)
encoded = MaxPooling2D(pool_size=(2,2), padding='same')(x)

# Decoding
x = Conv2DTranspose(128,(3, 3), padding='same', activation='relu')(encoded)
x = UpSampling2D((2, 2))(encoded)
x = Conv2DTranspose(64,(3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(32,(3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(16,(3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8,(3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(img_channels,(3, 3), padding='same')(x)
decoded = Activation('sigmoid')(x)

autoencoder = Model(input_img, decoded)
#autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
autoencoder.compile(optimizer=Adam(learning_rate=0.0001),
                    loss='mean_squared_error',
                    metrics=['accuracy', Recall(), Precision()]
                    )
autoencoder.summary()

autoencoder_history = autoencoder.fit(train_images_to_practice, train_images_to_compare,
                                      epochs = 10, batch_size = 32, validation_split = 0.1,
                                      shuffle=True,
                                      callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min"),
#                                                 TensorBoard(log_dir=os.path.join(os.getcwd(), 'logs'))
                                                 ]
                                      )

encoder = Model(inputs=input_img, outputs=encoded)


def display_results(test_group, encoded_imgs, decoded_imgs):
    n = 10
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(test_group[i][:, :, 0].reshape(img_width, img_height))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Encoded images
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[i][:, :, 0].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(decoded_imgs[i][:, :, 0].reshape(img_width, img_height))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def display_train_normal():
    training_normal_images_enc = encoder.predict(training_normal_images)
    training_normal_images_dec = autoencoder.predict(training_normal_images)
#    display_results(training_normal_images, training_normal_images_enc, training_normal_images_dec)
    # print("")
    # print("--------------------------------------------------------")
    # print("The Training Normal bottleneck:")
    print(training_normal_images_enc.shape)
    # print(training_normal_images_enc)
    # print("--------------------------------------------------------")
    # print("")
    return training_normal_images_enc, training_normal_images_dec


def display_train_pneu():
    training_pneumonia_images_enc = encoder.predict(training_pneumonia_images)
    training_pneumonia_images_dec = autoencoder.predict(training_pneumonia_images)
#    display_results(training_pneumonia_images, training_pneumonia_images_enc, training_pneumonia_images_dec)
    # print("")
    # print("--------------------------------------------------------")
    # print("The Training Pneu bottleneck:")
    print(training_pneumonia_images_enc.shape)
    # print(training_pneumonia_images_enc)
    # print("--------------------------------------------------------")
    # print("")
    return training_pneumonia_images_enc, training_pneumonia_images_dec

def display_test_normal():
    test_normal_images_enc = encoder.predict(test_normal_images)
    test_normal_images_dec = autoencoder.predict(test_normal_images)
#    display_results(test_normal_images, test_normal_images_enc, test_normal_images_dec)
    # print("")
    # print("--------------------------------------------------------")
    # print("The Test Normal bottleneck:")
    print(test_normal_images_enc.shape)
    # print(test_normal_images_enc)
    # print("--------------------------------------------------------")
    # print("")
    return test_normal_images_enc, test_normal_images_dec


def display_test_pneu():
    test_pneumonia_images_enc = encoder.predict(test_pneumonia_images)
    test_pneumonia_images_dec = autoencoder.predict(test_pneumonia_images)
#    display_results(test_pneumonia_images, test_pneumonia_images_enc, test_pneumonia_images_dec)
    # print("")
    # print("--------------------------------------------------------")
    # print("The Test Pneu bottleneck:")
    print(test_pneumonia_images_enc.shape)
    # print(test_pneumonia_images_enc)
    # print("--------------------------------------------------------")
    # print("")
    return test_pneumonia_images_enc, test_pneumonia_images_dec


training_normal_images_enc, training_normal_images_dec = display_train_normal()
training_pneumonia_images_enc, training_pneumonia_images_dec = display_train_pneu()
test_normal_images_enc, test_normal_images_dec = display_test_normal()
test_pneumonia_images_enc, test_pneumonia_images_dec = display_test_pneu()

# def display_in_scatter():
#     ## Wrong execution, ignore.
#     plt.figure(figsize = (8,8))
#     plt.scatter(training_normal_images_enc[:, 0], training_normal_images_enc[:, 1], c = "black", alpha = 0.5, s = 3)
#     plt.scatter(training_pneumonia_images_enc[:, 0], training_pneumonia_images_enc[:, 1], c = "yellow", alpha = 0.5, s = 3)
#     plt.scatter(test_normal_images_enc[:, 0], test_normal_images_enc[:, 1], c = "red", alpha = 0.5, s = 3)
#     plt.scatter(test_pneumonia_images_enc[:, 0], test_pneumonia_images_enc[:, 1], c = "green", alpha = 0.5, s = 3)
#     plt.show()


def display_distances():
    training_normal_images_latent_mean_image = np.mean(training_normal_images_enc, axis=0)

    # Calculate the distance of each image from the mean image
    training_normal_images_distances = np.sqrt(np.sum((training_normal_images_enc - training_normal_images_latent_mean_image) ** 2, axis=(1, 2, 3)))
    training_normal_images_distances.sort()
    training_pneumonia_images_distances = np.sqrt(np.sum((training_pneumonia_images_enc - training_normal_images_latent_mean_image) ** 2, axis=(1, 2, 3)))
    training_pneumonia_images_distances.sort()
    test_normal_images_distances = np.sqrt(np.sum((test_normal_images_enc - training_normal_images_latent_mean_image) ** 2, axis=(1, 2, 3)))
    test_normal_images_distances.sort()
    test_pneumonia_images_distances = np.sqrt(np.sum((test_pneumonia_images_enc - training_normal_images_latent_mean_image) ** 2, axis=(1, 2, 3)))
    test_pneumonia_images_distances.sort()

    print(training_normal_images_distances.shape)
    print(training_pneumonia_images_distances.shape)
    print(test_normal_images_distances.shape)
    print(test_pneumonia_images_distances.shape)

    # Plot the distances
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(training_normal_images_distances)), training_normal_images_distances, color='blue', marker='o', linestyle='-', label="training_normal")
    plt.plot(np.arange(len(training_pneumonia_images_distances)), training_pneumonia_images_distances, color='red', marker='o', linestyle='-', label="training_pneumonia")
    plt.plot(np.arange(len(test_normal_images_distances)), test_normal_images_distances, color='green', marker='o', linestyle='-', label="test_normal")
    plt.plot(np.arange(len(test_pneumonia_images_distances)), test_pneumonia_images_distances, color='orange', marker='o', linestyle='-', label="test_pneumonia")
    plt.xlabel('Image Index')
    plt.ylabel('Distance from Mean Image')
    plt.title('Distances of Images from Mean Image')
    plt.grid(True)
    plt.show()


display_distances()


def display_loss_graph():
    print("display loss graph:")
    loss = autoencoder_history.history['loss']
    val_loss = autoencoder_history.history['val_loss']
    epochs = range(10)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


display_loss_graph()



#
# # Set a threshold for anomaly detection
# threshold = np.percentile(reconstruction_error, 95)
#
# # Classify images as normal or anomalous
# anomalous_indices = np.where(reconstruction_error > threshold)[0]
# normal_indices = np.where(reconstruction_error <= threshold)[0]
#
