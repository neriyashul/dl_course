from os import path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Normalization
from keras.callbacks import EarlyStopping
from keras.metrics import Recall, Precision, F1Score
from matplotlib import pyplot as plt
import numpy as np
from utils import load_data


# ---------------------------------------------------------------------


# TODO: delete it
def show_img(image):
    # Display the image with matplotlib
    plt.imshow(image.numpy(), cmap='gray')
    plt.axis('off')  # Optionally, turn off the axis.
    plt.show()
'''
# how to use:
for images, labels in train_set:
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
for images, labels in train_set.take(1):
    normalized_image = normalization_layer(images[0][tf.newaxis, ...])
    show_histogram(normalized_image)
    break
'''


base_dir = "/Users/neriya.shulman/content/chest_xray"

batch_size = 256
img_height = 128
img_width = 128
max_epochs = 100

print("Image size: ", img_height, "x", img_width)
print("Batch size: ", batch_size)

all_labels = ['NORMAL', 'PNEUMONIA']
train_path = path.join(base_dir, "train/")
train_set, train_labels = load_data(train_path, img_height, img_width, all_labels, shuffle=True)

val_path = path.join(base_dir, "val/")
validation_set, validation_labels = load_data(val_path, img_height, img_width, all_labels)

test_path = path.join(base_dir, "test/")
test_set, test_labels = load_data(test_path, img_height, img_width, all_labels)

print('Number of classes: ', len(all_labels))
print('Number of training samples: ', len(train_set//batch_size))
print('Number of total samples: ', len(train_set))

# ---------------------------------------------------------------------
'''
Normalization layer
'''
# TODO check if needed or delete it: 
# norm_layer = Normalization(axis=None)
# norm_layer.adapt(train_set)
# ---------------------------------------------------------------------


# Build the CNN Model

model = Sequential()
# model.add(norm_layer)


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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision(), F1Score(threshold=0.5)])

# add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

model.fit(train_set, train_labels, epochs=max_epochs, batch_size=batch_size, 
          validation_data=(validation_set, validation_labels), callbacks=[early_stopping])

results = model.evaluate(test_set, test_labels, batch_size=batch_size)

print("---------------------------------------------------")
print("Test results: ")
print(f"Loss: {results[0]}")
print(f"Accuracy: {results[1]*100:.2f}%")
print(f"Recall: {results[2]*100:.2f}%")
print(f"Precision: {results[3]*100:.2f}%")
print(f"F1 score: {results[4]*100:.2f}%")



# save the model
model.save('my_model.keras')

# load the model
# model = tf.keras.models.load_model('cnn_model.h5')


