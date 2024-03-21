
from os import path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Normalization
from keras.preprocessing import image_dataset_from_directory
from keras.callbacks import EarlyStopping
from keras.metrics import Recall, Precision, F1Score
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array


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

# ---------------------------------------------------------------------

base_dir = "/Users/neriya.shulman/content/chest_xray"

batch_size = 256
img_height = 128
img_width = 128
max_epochs = 100

print("Image size: ", img_height, "x", img_width)
print("Batch size: ", batch_size)

import os
import tensorflow as tf
from keras.preprocessing.image import load_img#, img_to_array

import random
import time


def load_images_from_directory(normal_dir, pneumonia_dir, target_size, color_mode):
    catalog = {"BACTERIA":[], "VIRUS":[], "NORMAL":[] }
#    print(os.listdir(pneumonia_dir))
    for curr_file in os.listdir(pneumonia_dir):
        img = load_img(os.path.join(pneumonia_dir, curr_file), target_size=target_size, color_mode=color_mode)
        img_data = tf.keras.utils.img_to_array(img)
        if "virus" in curr_file:
            catalog["VIRUS"].append(img_data)
        else:
            catalog["BACTERIA"].append(img_data)

#    print(os.listdir(normal_dir))
    for curr_file in os.listdir(normal_dir):
        img = load_img(os.path.join(normal_dir, curr_file), target_size=target_size, color_mode=color_mode)
        img_data = tf.keras.utils.img_to_array(img)
        catalog["NORMAL"].append(img_data)

    images_and_labels = []
    for element in catalog["BACTERIA"]:
        images_and_labels.append([element, 1])
    for element in catalog["VIRUS"]:
        images_and_labels.append([element, 2])
    for element in catalog["NORMAL"]:
        images_and_labels.append([element, 0])

    random.shuffle(images_and_labels)
    images_arr = [x for [x, y] in images_and_labels]
    labels_arr = [y for [x, y] in images_and_labels]

    from keras.utils import to_categorical
    labels_arr = to_categorical(labels_arr, 3)

#    for idx in range(len(images_and_labels)):
#        print("label: ", labels_arr[idx])
#        img = tf.keras.utils.array_to_img(images_arr[idx])
#        img.show(labels_arr[idx])
#        time.sleep(4)

    col_data = tf.data.Dataset.from_tensor_slices((images_arr, labels_arr))
    col_data = col_data.batch(batch_size)
#    return col_data.prefetch(len(images_arr))
    return col_data


#
# def load_images_from_directory(normal_dir, pneumonia_dir, target_size, color_mode):
# #    images = []
# #    labels = []
#
#     # TODO: remove it
#     counter = 0
#
#     for filename in os.listdir(normal_dir):
#         img = load_img(os.path.join(normal_dir, filename), target_size=target_size, color_mode=color_mode)
#         img_array = img_to_array(img)
#         images.append(img_array)
#         labels.append('normal')
#
#         # TODO: remove it
#         counter += 1
#         if counter > 10:
#             break
#
#     for filename in os.listdir(pneumonia_dir):
#         if 'virus' in filename:  # adjust this as needed
#             img = load_img(os.path.join(pneumonia_dir, filename), target_size=target_size, color_mode=color_mode)
#             img_array = img_to_array(img)
#             images.append(img_array)
#             labels.append('virus')
#         else:
#             img = load_img(os.path.join(pneumonia_dir, filename), target_size=target_size, color_mode=color_mode)
#             img_array = img_to_array(img)
#             images.append(img_array)
#             labels.append('bacteria')
#
#         # TODO: remove it
#         counter += 1
#         if counter > 20:
#             break
#     dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#     dataset = dataset.batch(batch_size)
#     return dataset


normal_dir_path= path.join(base_dir, "train", "NORMAL")
pneumonia_dir_path = path.join(base_dir, "train", "PNEUMONIA")
train_ds = load_images_from_directory(normal_dir_path, pneumonia_dir_path, (img_height, img_width), 'grayscale')

normal_dir_path = path.join(base_dir, "val", "NORMAL")
pneumonia_dir_path = path.join(base_dir, "val", "PNEUMONIA")
validation_ds = load_images_from_directory(normal_dir_path, pneumonia_dir_path, (img_height, img_width), 'grayscale')

normal_dir_path = path.join(base_dir, "test", "NORMAL")
pneumonia_dir_path = path.join(base_dir, "test", "PNEUMONIA")
test_ds = load_images_from_directory(normal_dir_path, pneumonia_dir_path, (img_height, img_width), 'grayscale')

print("after load images")
# TODO: remove it
# train_ds = image_dataset_from_directory(
#    directory=path.join(base_dir, "train/"),
#    # validation_split=0.2,
#    # subset="training",
#    seed=123,
#    label_mode='categorical',  # 'binary' for multiclasses
#    color_mode='grayscale',
#    image_size=(img_height, img_width),
#    batch_size=batch_size,
#    shuffle=True)
#
# validation_ds = image_dataset_from_directory(
#    directory=path.join(base_dir, "val/"),
#    # validation_split=0.2,
#    # subset="validation",
#    seed=123,
#    label_mode='categorical',
#    color_mode='grayscale',
#    image_size=(img_height, img_width),
#    batch_size=batch_size)
#
# test_ds = image_dataset_from_directory(
#    directory=path.join(base_dir, "test/"),
#    label_mode='categorical',
#    color_mode='grayscale',
#    image_size=(img_height, img_width),
#    batch_size=batch_size)

# print('Number of classes: ', len(train_ds.class_names))
# print('Number of training samples: ', train_ds.cardinality().numpy())
# print('Number of total samples: ', len(train_ds) * batch_size)
# sys.exit(0)

#for images, labels in train_ds.take(1):
#    for image in images:
#        show_img(image)
#        break


# ---------------------------------------------------------------------
'''
Normalization layer
'''

# Create the normalization layer
normalization_layer = Normalization()

# Adapt the normalization layer with your dataset
# Use the map function to extract just the image data if your dataset also contains labels
normalization_layer.adapt(train_ds.map(lambda x, _: x))
# ---------------------------------------------------------------------

# Build the CNN Model

model = Sequential()

model.add(normalization_layer)

# Convolution layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 1)))
# Pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
model.add(Flatten())

# Full connection
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=3, activation='softmax'))  # Use 'softmax' for more than 2 classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=[
                  'categorical_accuracy',
                  Recall(),
                  Precision(),
                  F1Score(average="micro", threshold=0.5)
              ]
)

# add early stopping
early_stopping = EarlyStopping(
    monitor='val_loss', patience=3, verbose=1, mode='min'
)

model.fit(train_ds, epochs=max_epochs, validation_data=validation_ds, callbacks=[early_stopping])

results = model.evaluate(test_ds)

print("---------------------------------------------------")
print("Test results: ")
print(f"Loss: {results[0]}")
print(f"Accuracy: {results[1]*100:.2f}%")
print(f"Recall: {results[2]*100:.2f}%")
print(f"Precision: {results[3]*100:.2f}%")
print(f"F1 score: {results[4]*100:.2f}%")


# save the model
model.save('my_model2.keras')

# load the model
# model = tf.keras.models.load_model('cnn_model.h5')


