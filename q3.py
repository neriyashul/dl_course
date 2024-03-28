import os
import numpy as np
from PIL import Image
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape
from keras.models import Model, load_model
from keras.metrics import Recall, Precision, F1Score
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Flatten, Dense, Normalization
import tensorflow as tf

pic_width, pic_height = (256, 256)
model_filepath = "my_model4.keras"


def save(model):
    model.save(model_filepath)


def load():
    return load_model(model_filepath)


def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = Image.open(os.path.join(directory, filename)).convert('L')  # Open image in grayscale mode
        img = img.resize((pic_width, pic_height))  # Resize images if needed
        images.append(np.array(img))
    return np.array(images).astype('float32') / 255.0


class AutoEncoder(Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = Sequential([
            Input(shape=(pic_width, pic_height, 1)),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
        ])

        self.decoder = Sequential([
            Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
            Conv2DTranspose(64, (3, 3), activation='relu', padding='same'),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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


autoencoder = AutoEncoder()
autoencoder.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy', Recall(), Precision()])
# input_images_data_training, input_images_data_validation = np.split(training_normal_images, [int(0.8 * len(training_normal_images)), ])
# history = autoencoder.fit(input_images_data_training, input_images_data_training,
#                 epochs=20,
#                 batch_size=32,
#                 shuffle=True,
#                 validation_data=(input_images_data_validation, input_images_data_validation))
history = autoencoder.fit(training_normal_images, training_normal_images,
                epochs=40,
                batch_size=128,
                shuffle=True,
                validation_data=(training_normal_images+training_pneumonia_images, training_normal_images+training_pneumonia_images))
save(autoencoder)

# print(history.history)

print("Evaluate on test data")
results = autoencoder.evaluate(test_normal_images, test_normal_images, batch_size=128)
print("test results for normal:", results)
results = autoencoder.evaluate(test_pneumonia_images, test_pneumonia_images, batch_size=128)
print("test results for pneumonia:", results)

#autoencoder = load()
#
display_few_exampls(autoencoder, training_normal_images)
display_few_exampls(autoencoder, test_normal_images)
display_few_exampls(autoencoder, training_pneumonia_images)
display_few_exampls(autoencoder, test_pneumonia_images)

