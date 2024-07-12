from typing import cast
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, load_digits
from sklearn.linear_model import LogisticRegression
import numpy as np

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

n_labeled = 50
log_reg = LogisticRegression(max_iter=10_000)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
log_reg.score(X_test, y_test)

k = 50
km = KMeans(n_clusters=k, random_state=42)
X_dist = km.fit_transform(X_train)
representative_idx = np.argmin(X_dist, axis=0)
X_representative = X_train[representative_idx]
