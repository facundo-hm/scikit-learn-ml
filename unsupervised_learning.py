from typing import cast
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, load_digits
from sklearn.linear_model import LogisticRegression
import numpy as np
from utils import save_representative_imgs

### K-MEANS ###
blob_centers = np.array([
    [ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
    [-2.8,  2.8], [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(
    n_samples=2000, centers=blob_centers,
    cluster_std=blob_std, random_state=7)

k = 5
km = KMeans(n_clusters=k, n_init=10, random_state=42)
km.fit(X)
km.cluster_centers_

X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
km.predict(X_new)

# Speed up the algorithm by using mini-batches and moving
# the centroids just slightly at each iteration
mbkm = MiniBatchKMeans(n_clusters=5, random_state=42)
mbkm.fit(X)

### SEMI-SUPERVISED LEARNING ###
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = cast(
    list[np.ndarray],
    train_test_split(X, y, test_size=0.5, random_state=42))

labeled_k = 50
max_iter = 10000
lr = LogisticRegression(max_iter=max_iter)
lr.fit(X_train[:labeled_k], y_train[:labeled_k])
print(lr.score(X_test, y_test))

km = KMeans(n_clusters=labeled_k, random_state=42)
# Distance of the images in each cluster
X_dist = km.fit_transform(X_train)
# Idxs of the image with min distance in each cluste
representative_idx = np.argmin(X_dist, axis=0)
# Find the image closest to the centroid
X_representative = X_train[representative_idx]

save_representative_imgs(X_representative, labeled_k)
# Label representative images manually
y_representative = np.array([
    8, 0, 7, 3, 6, 1, 1, 2, 5, 5,
    2, 4, 8, 7, 4, 4, 9, 7, 2, 1,
    6, 9, 6, 0, 5, 9, 6, 5, 3, 7,
    7, 4, 6, 0, 1, 9, 8, 1, 8, 2,
    7, 2, 3, 1, 1, 7, 2, 3, 7, 8])

lr = LogisticRegression(max_iter=max_iter)
lr.fit(X_representative, y_representative)
print(lr.score(X_test, y_test))

y_train_propagated = np.empty(len(X_train), dtype=np.int64)
# Apply label propagation
for i in range(labeled_k):
    y_train_propagated[km.labels_ == i] = y_representative[i]

lr = LogisticRegression(max_iter=max_iter)
lr.fit(X_train, y_train_propagated)
print(lr.score(X_test, y_test))

percentile_closest = 99
X_cluster_dist = X_dist[np.arange(len(X_train)), km.labels_]
# print('X_train', X_train[:5])
# print('km.labels_', km.labels_[:5])
# print('X_dist', X_dist[:5])
# print('X_cluster_dist', X_cluster_dist[:5])

for i in range(labeled_k):
    in_cluster = (km.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1
    partially_propagated = (X_cluster_dist != -1)

X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg.score(X_test, y_test))