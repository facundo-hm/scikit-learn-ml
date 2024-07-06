from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
import numpy as np

np.random.seed(42)
m = 60
angles = (np.random.rand(m) ** 3 + 0.5) * 2 * np.pi  # uneven distribution

# Create 3D data set
X = np.zeros((m, 3))
X[:, 0], X[:, 1] = np.cos(angles), np.sin(angles) * 0.5
# Add noise
X += 0.28 * np.random.randn(m, 3)
X = Rotation.from_rotvec([np.pi / 29, -np.pi / 20, np.pi / 4]).apply(X)
X += [0.2, 0, 0.2]

# Reduce the dimensionality of the dataset down to two dimensions
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
print('X_2d', X_2d)
