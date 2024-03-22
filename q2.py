import matplotlib.pyplot as plt
from os import path
from keras.models import load_model, Sequential
from keras.metrics import Recall, Precision, F1Score
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from utils import load_data

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

test_path = path.join(base_dir, "test/")
test_set, test_labels = load_data(test_path, img_height, img_width, all_labels)


model_path = 'my_model.keras'
assert path.exists(model_path), "The model file does not exist"
orig_model = load_model(model_path)

print(orig_model.evaluate(test_set, test_labels))

model = Sequential(orig_model.layers[:-2])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall(), Precision(), F1Score(threshold=0.5)])
print(model.summary())

embed_vectors = model.predict(test_set)

test_labels = test_labels.squeeze()

# ---------------------------------------------------------------------
'''
Calculate nearest neighbors for every image in the test set, 
predict the class of the image and print the accuracy of the model 
compared to the true label.
'''

train_embed_vectors = model.predict(train_set)

knn = NearestNeighbors(n_neighbors=5)
knn.fit(train_embed_vectors)

correct = 0
embed_vectors = model.predict(test_set)
distances, indices = knn.kneighbors(embed_vectors)

sum_of_neighbors = np.sum(train_labels[indices].squeeze(), axis=1)
predictions = np.where(sum_of_neighbors > 2.5, 1, 0)
correct = np.sum(predictions == test_labels)

print("Accuracy: ", correct / len(test_labels))



# ---------------------------------------------------------------------
'''
Visualize the embeddings using t-SNE
'''

pca = PCA(n_components=50)
pca_result = pca.fit_transform(embed_vectors)
tsne = TSNE(n_components=2, verbose=1)
tsne_results = tsne.fit_transform(pca_result)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_results[test_labels == 0, 0], tsne_results[test_labels == 0, 1], c='blue', label='Class 0')
plt.scatter(tsne_results[test_labels == 1, 0], tsne_results[test_labels == 1, 1], c='red', label='Class 1')
plt.legend()
plt.show()