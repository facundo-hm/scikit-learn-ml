from typing import cast
from sklearn import datasets, model_selection, linear_model, metrics
import numpy as np

mnist = datasets.fetch_openml('mnist_784', as_frame=False)
X, y = cast(np.ndarray, mnist.data), cast(np.ndarray, mnist.target)

X_train, X_test, y_train, y_test = cast(
    list[np.ndarray],
    model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42))

y_train_binary: np.ndarray = (y_train == '5')
y_test_binary: np.ndarray = (y_test == '5')

sgd_clf = linear_model.SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_binary)

y_train_binary_pred = model_selection.cross_val_predict(
    sgd_clf, X_train, y_train_binary, cv=3)

binary_cm = metrics.confusion_matrix(
    y_train_binary, y_train_binary_pred)

# Get decision scores for each instance
y_train_binary_scores = model_selection.cross_val_predict(
    sgd_clf, X_train, y_train_binary, cv=3, method='decision_function')

precisions, recalls, thresholds = metrics.precision_recall_curve(
    y_train_binary, y_train_binary_scores)

# Make predictions by searching the lower threshold that give
# at least 90% precision
idx_90_precision = (precisions >= 0.90).argmax()
threshold_90_precision = thresholds[idx_90_precision]
print('threshold_90_precision', threshold_90_precision)

y_train_binary_pred_90 = (
    y_train_binary_scores >= threshold_90_precision)
precision_score_90 = metrics.precision_score(
    y_train_binary, y_train_binary_pred_90)
recall_90_precision = metrics.recall_score(
    y_train_binary, y_train_binary_pred_90)
print('precision_score_90', precision_score_90)
print('recall_90_precision', recall_90_precision)
