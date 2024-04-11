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
from utils import load_data, shuffle_dataset
from scipy.stats import mode

base_dir = "/Users/neriya.shulman/content/chest_xray"

# all_labels = ['NORMAL', 'PNEUMONIA']
# file_labels=[]
# model_path = 'model_q1_a.keras'
all_labels = ['NORMAL', 'virus', 'bacteria']
file_labels=['virus', 'bacteria']
model_path = 'model_q1_b.keras'

assert path.exists(model_path), "The model file does not exist"
orig_model = load_model(model_path)


img_height = orig_model.input_shape[1]
img_width = orig_model.input_shape[2]
print("Image size: ", img_height, "x", img_width)


train_path = path.join(base_dir, "train/")
train_set, train_labels = load_data(train_path, img_height, img_width, all_labels, file_labels=file_labels)
train_set, train_labels = shuffle_dataset(train_set, train_labels)

test_path = path.join(base_dir, "test/")
test_set, test_labels = load_data(test_path, img_height, img_width, all_labels, file_labels=file_labels)


print(orig_model.evaluate(test_set, test_labels))

flatten_layer = orig_model.get_layer('flatten')
embed_index = orig_model.layers.index(flatten_layer)

model = Sequential(orig_model.layers[:embed_index+1])
model.compile(optimizer=orig_model.optimizer.name, loss=orig_model.loss, metrics=orig_model.metrics)
print(model.summary())


# ---------------------------------------------------------------------
'''
Calculate nearest neighbors for every image in the test set, 
predict the class of the image and print the accuracy of the model 
compared to the true label.
'''

train_embed_vectors = model.predict(train_set)
embed_vectors = model.predict(test_set)
test_labels = test_labels.squeeze()

knn = NearestNeighbors(n_neighbors=5)
knn.fit(train_embed_vectors)

correct = 0
distances, indices = knn.kneighbors(embed_vectors)

predictions, _ = mode(train_labels[indices].squeeze(), axis=1)
correct = np.sum(predictions == test_labels)

print("Accuracy: ", correct / len(test_labels))

print('---------------------------------------------------------------------')

# ---------------------------------------------------------------------
'''
Visualize the embeddings using t-SNE
'''
# TODO: fix it to multilables
tsne = TSNE(n_components=len(all_labels), verbose=1)
tsne_test_results = tsne.fit_transform(embed_vectors)
tsne_train_results = tsne.fit_transform(train_embed_vectors)


plt.figure(figsize=(10, 6))
labels = train_labels.squeeze()
for i, label in enumerate(all_labels):
    plt.scatter(tsne_train_results[labels == i, 0], tsne_train_results[labels == i, 1], label=f'train {label.lower()}', alpha=0.4)

for i, label in enumerate(all_labels):
    plt.scatter(tsne_test_results[predictions == i, 0], tsne_test_results[predictions == i, 1], label=f'test {label.lower()} prediction', alpha=0.4)

plt.legend()
plt.show()