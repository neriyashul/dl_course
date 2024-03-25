import os
import numpy as np
from PIL import Image
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from matplotlib import pyplot as plt
from keras.losses import MeanAbsoluteError
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
            Input(shape=(28, 28, 1)),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            Conv2D(8, (3, 3), activation='relu', padding='same')
            # Input(shape=(pic_width, pic_height, 1)),
            # Conv2D(32, (3, 3), activation='relu', padding='same'),
            # MaxPooling2D((2, 2), padding='same'),
            # Conv2D(16, (3, 3), activation='relu', padding='same'),
            # MaxPooling2D((2, 2), padding='same')
        ])

        self.decoder = Sequential([
            Conv2DTranspose(8, kernel_size=3, activation='relu', padding='same'),
            Conv2DTranspose(16, kernel_size=3, activation='relu', padding='same'),
            Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
            # Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
            # UpSampling2D((2, 2)),
            # Conv2DTranspose(32, (3, 3), activation='relu', padding='same'),
            # UpSampling2D((2, 2)),
            # Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
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


def calculate_error(X, pred_scores):
    euclidean_sq = np.square(pred_scores - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


# # Load grayscale images from a directory
image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("train", "NORMAL"))
images = load_images(image_directory)

autoencoder = AutoEncoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
input_images_data_training, input_images_data_validation = np.split(images, [int(0.8 * len(images)), ])
autoencoder.fit(input_images_data_training, input_images_data_training,
                epochs=20,
                batch_size=32,
                shuffle=True,
                validation_data=(input_images_data_validation, input_images_data_validation))
save(autoencoder)

#autoencoder = load()
#
display_few_exampls(autoencoder, images)
# reconstructed_images = autoencoder.predict(images).squeeze()
# print(reconstructed_images.shape, type(reconstructed_images))
# print(calculate_error(images, images))
# print(calculate_error(images, reconstructed_images))
#
# encoded_imgs = autoencoder.encoder(images).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy().squeeze()
# print(calculate_error(images, decoded_imgs))


reconstructions = autoencoder.predict(images).squeeze()
train_loss = tf.keras.losses.mae(reconstructions, images)
print(train_loss.shape)
plt.hist(train_loss, bins=20)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Mean: ", np.mean(train_loss), ", Deviation: ", np.std(train_loss),", Threshold: ", threshold)


anomalous_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("train", "PNEUMONIA"))
anomalous_image = load_images(anomalous_image_directory)
print(anomalous_image.shape)

anomalous_reconstructions = autoencoder.predict(anomalous_image).squeeze()
test_loss = tf.keras.losses.mae(anomalous_reconstructions, anomalous_image)

plt.hist(test_loss, bins=20)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()


# illness_image_directory = os.path.join(os.path.join(os.getcwd(), "chest_xray"), os.path.join("train", "PNEUMONIA"))
# illness_images = load_images(image_directory)
# illness_reconstructed_images = autoencoder.predict(images).squeeze()
# print(make_decision(images, illness_reconstructed_images))


# def predict(model, data, threshold):
#     reconstructions = model(data)
#     loss = tf.keras.losses.mae(reconstructions, data)
#     return tf.math.less(loss, threshold)
#
# from keras.metrics import Recall, Precision, Accuracy
# def print_stats(predictions, labels):
#     print("Accuracy = {}".format(Accuracy(labels, predictions)))
#     print("Precision = {}".format(Precision(labels, predictions)))
#     print("Recall = {}".format(Recall(labels, predictions)))
#
#
# preds = predict(autoencoder, input_images_data_validation, threshold)
# print_stats(preds, input_images_data_validation)
#
