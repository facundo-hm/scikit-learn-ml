from typing import cast
from sklearn import datasets
import numpy as np

mnist = datasets.fetch_openml('mnist_784', as_frame=False)
X, y = cast(np.ndarray, mnist.data), cast(np.ndarray, mnist.target)

print(X.shape)
print(y.shape)
