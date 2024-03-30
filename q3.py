import os
import sys
from math import pi

import numpy as np
from PIL import Image
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.metrics import Recall, Precision, F1Score
from keras.src.layers import RandomRotation
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Dense, Normalization
from keras.callbacks import EarlyStopping
import tensorflow as tf

##################################
import numpy as np
import tensorflow as tf
import random
# Set seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
##################################


pic_width, pic_height = (128, 128)
model_filepath = "my_model4.keras"
degrees = 20


def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory, filename)).convert('L')  # Open image in grayscale mode
        img = img.resize((pic_width, pic_height))  # Resize images if needed
        images.append(np.array(img))
    return np.array(images).astype('float32') / 255.0


def display_few_exampls(model, images, no_of_samples=4):
    # Get the reconstructed images
    reconstructed_images = model.predict(images).squeeze()
    _, axs = plt.subplots(no_of_samples, 2, figsize=(5, 8))
    axs = axs.flatten()
    imgs = []
    for i in range(no_of_samples):
        imgs.append(images[i])
        imgs.append(reconstructed_images[i])
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()


# # Load grayscale images from a directory
training_normal_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("train", "NORMAL"))
training_normal_images = load_images(training_normal_image_directory)
training_pneumonia_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("train", "PNEUMONIA"))
training_pneumonia_images = load_images(training_pneumonia_image_directory)

test_normal_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("test", "NORMAL"))
test_normal_images = load_images(test_normal_image_directory)
test_pneumonia_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("test", "PNEUMONIA"))
test_pneumonia_images = load_images(test_pneumonia_image_directory)


norm_layer = Normalization(axis=None)
norm_layer.adapt(training_normal_images)

autoencoder = Sequential([
#    norm_layer,
    Input(shape=(pic_width, pic_height, 1)),
    RandomRotation(degrees*pi/180),
    Conv2D(64, kernel_size=3, strides=1, activation='relu', padding='same'),
    Dropout(rate=0.5),
    Conv2D(32, kernel_size=3, strides=1, activation='relu', padding='same'),
    Dropout(rate=0.5),
    Conv2D(16, kernel_size=3, strides=1, activation='relu', padding='same'),
    Dropout(rate=0.5),
    Conv2D(8, kernel_size=3, strides=1, activation='relu', padding='same'),
    Conv2DTranspose(8, kernel_size=3, strides=1, activation='relu', padding='same'),
    Dropout(rate=0.5),
    Conv2DTranspose(16, kernel_size=3, strides=1, activation='relu', padding='same'),
    Dropout(rate=0.5),
    Conv2DTranspose(32, kernel_size=3, strides=1, activation='relu', padding='same'),
    Dropout(rate=0.5),
    Conv2DTranspose(64, kernel_size=3, strides=1, activation='relu', padding='same'),
    Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
])

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy', Recall(), Precision()])
autoencoder.summary()

input_images_data_training, input_images_data_validation = np.split(training_normal_images, [int(0.8 * len(training_normal_images)), ])
history = autoencoder.fit(input_images_data_training, input_images_data_training,
                epochs=50,
                batch_size=32,
                shuffle=True,
                callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min")],
                validation_data=(input_images_data_validation, input_images_data_validation))

# print(history.history)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# Get train MAE loss.
training_normal_images_pred = autoencoder.predict(training_normal_images).squeeze()

# print("stats:")
# print(training_normal_images_pred[0])
# print(training_normal_images_pred[0].shape)
# print(training_normal_images[0])
# print(training_normal_images[0].shape)
# print(training_normal_images_pred[0] - training_normal_images[0])
# print(np.mean(np.abs(training_normal_images_pred - training_normal_images), axis=(1, 2)))

train_mae_loss = np.mean(np.abs(training_normal_images_pred - training_normal_images), axis=(1, 2))

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)
print(train_mae_loss.shape)

test_pneumonia_images_pred = autoencoder.predict(test_pneumonia_images).squeeze()
test_mae_loss = np.mean(np.abs(test_pneumonia_images_pred - test_pneumonia_images), axis=(1, 2))
test_mae_loss = test_mae_loss.reshape((-1))
plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

sys.exit(0)

print("Evaluate on test data")
results = autoencoder.evaluate(test_normal_images, test_normal_images, batch_size=128)
print("test results for normal:", results)
results = autoencoder.evaluate(test_pneumonia_images, test_pneumonia_images, batch_size=128)
print("test results for pneumonia:", results)

display_few_exampls(autoencoder, training_normal_images)
display_few_exampls(autoencoder, test_normal_images)
display_few_exampls(autoencoder, training_pneumonia_images)
display_few_exampls(autoencoder, test_pneumonia_images)

