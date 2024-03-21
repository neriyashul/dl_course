from os import path
from keras.models import load_model
from keras.layers import Normalization
from keras.preprocessing import image_dataset_from_directory
from keras.metrics import Recall, Precision, F1Score
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


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

train_ds = image_dataset_from_directory(
    directory=path.join(base_dir, "train/"),
    # validation_split=0.2,
    # subset="training",
    seed=123,
    label_mode='binary',  # 'categorical' for multiclasses
    color_mode='grayscale',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True)

validation_ds = image_dataset_from_directory(
    directory=path.join(base_dir, "val/"),
    # validation_split=0.2,
    # subset="validation",
    seed=123,
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
normalization_layer = Normalization()

# Adapt the normalization layer with your dataset
# Use the map function to extract just the image data if your dataset also contains labels
from keras.models import load_model

normalization_layer.adapt(train_ds.map(lambda x, _: x))
# ---------------------------------------------------------------------


model_path = 'my_model.keras'
assert path.exists(model_path), "The model file does not exist"


from keras.models import load_model, Sequential

# Load the model
model = load_model(model_path)

# Create a new model without the last layer
new_model = Sequential(model.layers[:-1])

# Recompile the model after modification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision(), F1Score(threshold=0.5)])

print(new_model.summary())

for images, labels in test_ds.take(1):
    # print an embedding vector using the model without running the last layer (the sigmoid layer)
    result = new_model.predict(images)
    print(result)







# original_model = tf.keras.models.load_model(model_path)
# # model = tf.keras.models.Model(inputs=original_model.input, outputs=[l.output for l in original_model.layers[:-1]])
# model = tf.keras.models.Model(inputs=original_model.input, outputs=original_model.layers[-2].output)


# # # load the original model
# # original_model = tf.keras.models.load_model(model_path)

# # # call the original model to define the input shape
# # _ = original_model(tf.keras.Input(shape=(img_height, img_width, 1), name='unique_input_layer'))

# # # create a new model that outputs from the second last layer
# # model = tf.keras.models.Model(inputs=original_model.input, outputs=original_model.layers[-2].output)

# # load the model
# model = load_model('my_model.keras')

# print(model.summary())

# for images, labels in test_ds.take(1):
#     for image in images:
#         # print an embedding vector using the model without running the last layer (the sigmoid layer)
#         result = model.predict(image)
#         print(result)


# for images, labels in train_ds.take(1):
#     # print an embedding vector using the model without running the last layer (the sigmoid layer)
#     result = model.predict(images)
#     print(result)
