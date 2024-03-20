from os import path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Normalization
from keras.preprocessing import image_dataset_from_directory
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


# ---------------------------------------------------------------------


# TODO: delete it
def show_img(image):
    image_np = image.numpy()[:, :, 0]  # Convert to numpy array
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

        
batch_size = 32
img_height = 64
img_width = 64

print("Image size: ", img_height, "x", img_width)
print("Batch size: ", batch_size)

train_ds = image_dataset_from_directory(
    directory=path.join(base_dir, "train/"),
    # validation_split=0.2,
    # subset="training",
    # seed=123,
    label_mode='binary',  # 'categorical' for multiclasses
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True)

validation_ds = image_dataset_from_directory(
    directory=path.join(base_dir, "val/"),
    # validation_split=0.2,
    # subset="validation",
    # seed=123,
    label_mode='binary',  # or 'categorical'
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = image_dataset_from_directory(
    directory=path.join(base_dir, "test/"),
    label_mode='binary',  # or 'categorical'
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size)

print('Number of classes: ', len(train_ds.class_names))
print('Number of training samples: ', train_ds.cardinality().numpy())
print('Number of total samples: ', len(train_ds) * batch_size)

# ---------------------------------------------------------------------
'''
Normalization layer
'''

# Create the normalization layer
normalization_layer = Normalization(axis=-1)

# Adapt the normalization layer with your dataset
# Use the map function to extract just the image data if your dataset also contains labels
normalization_layer.adapt(train_ds.map(lambda x, y: x))
# ---------------------------------------------------------------------

# Build the CNN Model

model = Sequential()

model.add(normalization_layer)

model.fit(train_ds, epochs=10, validation_data=validation_ds)


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
model.add(Dense(units=1, activation='sigmoid'))  # Use 'softmax' for more than 2 classes

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_ds, epochs=10, validation_data=validation_ds)

loss, accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {accuracy*100:.2f}%")
