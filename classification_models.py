from typing import cast
from sklearn import datasets, model_selection, linear_model, metrics
import numpy as np

mnist = datasets.fetch_openml('mnist_784', as_frame=False)
X, y = cast(np.ndarray, mnist.data), cast(np.ndarray, mnist.target)

X_train, X_test, y_train, y_test = cast(
    list[np.ndarray],
    model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42))

y_train_binary = (y_train == '5')
y_test_binary = (y_test == '5')

sgd_clf = linear_model.SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_binary)

y_train_binary_pred = model_selection.cross_val_predict(
    sgd_clf, X_train, y_train_binary, cv=3)

binary_cm = metrics.confusion_matrix(
    y_train_binary, y_train_binary_pred)
