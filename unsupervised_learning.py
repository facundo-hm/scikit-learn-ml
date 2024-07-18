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
# Get the training instances distance to its closest cluster center
# by using the instance index and its corresponding label 
X_cluster_dist = X_dist[np.arange(len(X_train)), km.labels_]
closest_idxs = []

for i in range(labeled_k):
    # Get boolean of the instances in current iterated cluster 
    in_cluster = (km.labels_ == i)
    # Arrange instances distance by cluster
    cluster_dist = X_cluster_dist[in_cluster]

    # Get farthest distance from cluster center
    farthest_distance = np.percentile(cluster_dist, percentile_closest)
    closest_distance_idxs = np.argwhere(
        (X_cluster_dist <= farthest_distance) & in_cluster).flatten()

    closest_idxs.extend(closest_distance_idxs) 

X_train_partially_propagated = X_train[closest_idxs]
y_train_partially_propagated = y_train_propagated[closest_idxs]

log_reg = LogisticRegression(max_iter=max_iter)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
print(log_reg.score(X_test, y_test))
