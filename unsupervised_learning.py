from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

### K-Means ###
blob_centers = np.array([
    [ 0.2,  2.3], [-1.5 ,  2.3], [-2.8,  1.8],
    [-2.8,  2.8], [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(
    n_samples=2000, centers=blob_centers,
    cluster_std=blob_std, random_state=7)
print('X', X)

k = 5
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(X)

X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)
