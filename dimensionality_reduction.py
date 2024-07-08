from typing import cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np

np.random.seed(42)

mnist = fetch_openml('mnist_784', as_frame=False)
X_mnist, y_mnist = cast(np.ndarray, mnist.data), cast(np.ndarray, mnist.target)

X_train, X_test, y_train, y_test = cast(
    list[np.ndarray],
    train_test_split(X_mnist, y_mnist, test_size=0.2, random_state=42))

m = 60
angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi

# Create 3D data set
X = np.zeros((m, 3))
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5
# Add noise
X += 0.28 * np.random.randn(m, 3)
X = Rotation.from_rotvec(
    [np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X += [0.2, 0, 0.2]

### PCA ###
# Principal component analysis
X_centered = X - X.mean(axis=0)
# Obtain all the principal components
U, s, Vt = np.linalg.svd(X_centered)
# Extract the two unit vectors that define the first two PC
c1 = Vt[0]
c2 = Vt[1]

# Project the training set onto the hyperplane defined by
# the first two principal components
W2 = Vt[:2].T
X_2d = X_centered @ W2

# Reduce the dimensionality of the dataset down to two dimensions
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
# Indicate the proportion of the dataset’s variance that lies
# along each principal component
pca.explained_variance_ratio_

# Float 0.0 to 1.0 is the ratio of variance to preserve,
# and the number of components is defined during training
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
pca.n_components_

clf = make_pipeline(
    PCA(random_state=42),
    RandomForestClassifier(random_state=42))
# Find a good combination of hyperparameters for both PCA
# and the random forest classifier
param_distrib = {
    'pca__n_components': np.arange(10, 60),
    'randomforestclassifier__n_estimators': np.arange(50, 500)
}
rscv = RandomizedSearchCV(
    clf, param_distrib, n_iter=10, cv=3, random_state=42)
rscv.fit(X_train[:100], y_train[:100])
rscv.best_params_

# Split the training set into mini-batches and feed these in
# one mini-batch at a time
n_batches = 100
ipca = IncrementalPCA(n_components=154)

for X_batch in np.array_split(X_train, n_batches):
    ipca.partial_fit(X_batch)

X_reduced = ipca.transform(X_train)
