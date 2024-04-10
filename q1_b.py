import os
import keras
import random
import numpy as np
import tensorflow as tf
from os import path
from keras import layers
from keras.optimizers import Adam
from keras.layers import Dense, RandomBrightness, RandomRotation, RandomZoom, RandomContrast
from keras.callbacks import EarlyStopping
from utils import load_data, add_samples_to_validation, shuffle_dataset

# Set seeds
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

base_dir = os.path.join(os.getcwd(), "chest_xray")
# base_dir = "/Users/neriya.shulman/content/chest_xray"

img_height = 150
img_width = 150
max_epochs = 50
batch_size = 64
learning_rate=0.001

print("Image size: ", img_height, "x", img_width)
print("Batch size: ", batch_size)


def balance_data(normal_data, pneum_data):
    multiply_by = round(len(pneum_data) / len(normal_data))
    return np.concatenate([normal_data for _ in range(multiply_by)], axis=0)


def get_balanced_data(base_dir, file_labels=[]):
    print('get_balanced_data')
    normal_train_set, normal_train_labels = load_data(path.join(base_dir, "NORMAL"), img_height, img_width, all_labels)
    pneum_train_set, pneum_train_labels = load_data(path.join(base_dir, "PNEUMONIA"), img_height, img_width, all_labels, file_labels=file_labels)
    normal_train_set = balance_data(normal_train_set, pneum_train_set)
    normal_train_labels = balance_data(normal_train_labels, pneum_train_labels)
    
    train_set = np.concatenate([normal_train_set, pneum_train_set], axis=0)
    train_labels = np.concatenate([normal_train_labels, pneum_train_labels], axis=0)
    
    train_set, train_labels = shuffle_dataset(train_set, train_labels)
    return train_set, train_labels


all_labels = ['NORMAL', 'bacteria', 'virus']
train_path = os.path.join(base_dir, "train")
train_set, train_labels = get_balanced_data(train_path, file_labels=['bacteria', 'virus'])

val_path = path.join(base_dir, "val/")
validation_set, validation_labels = load_data(val_path, img_height, img_width, all_labels, file_labels=['bacteria', 'virus'])
if len(validation_set) < 20:
    add_samples_to_validation(base_dir)

test_path = path.join(base_dir, "test/")
test_set, test_labels = load_data(test_path, img_height, img_width, all_labels, file_labels=['bacteria', 'virus'])

print('Number of classes: ', len(all_labels))
print('Number of training samples: ', len(train_set)//batch_size)
print('Number of total samples: ', len(train_set))


model = keras.Sequential(
    [
        keras.Input(shape=(img_height, img_width, 1)),
        RandomRotation(0.35),
        RandomZoom(0.1),
        RandomBrightness(0.02, value_range=(0,1)),
        RandomContrast(0.02),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(1024, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(name='flatten'),
        layers.Dense(units=1024, activation='relu'),
        layers.Dense(units=526, activation='relu'),
        layers.Dropout(0.25),
        layers.Dense(units=256, activation='relu'),
        layers.Dense(units=128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(all_labels), activation='softmax'),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", "recall", "precision"])


early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history = model.fit(train_set, train_labels, epochs=max_epochs, batch_size=batch_size, 
        validation_data=(validation_set, validation_labels), callbacks=[early_stopping])

print(model.summary())


score = model.evaluate(test_set, test_labels, batch_size=batch_size)
# results = model.evaluate(test_set, test_labels, batch_size=batch_size)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Test recall:", score[2])
print("Test precision:", score[3])




# save the model
model.save('model_q1_b.keras')

# load the model
# model = tf.keras.models.load_model('cnn_model.h5')


