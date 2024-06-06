from typing import cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, f1_score, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = cast(np.ndarray, mnist.data), cast(np.ndarray, mnist.target)

X_train, X_test, y_train, y_test = cast(
    list[np.ndarray],
    train_test_split(X, y, test_size=0.2, random_state=42))

y_train_binary: np.ndarray = (y_train == '5')
y_test_binary: np.ndarray = (y_test == '5')

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train, y_train_binary)

y_binary_sgd_pred = cross_val_predict(
    sgd, X_train, y_train_binary, cv=3)

binary_cm = confusion_matrix(y_train_binary, y_binary_sgd_pred)

# Get decision scores for each instance
y_binary_sgd_scores = cross_val_predict(
    sgd, X_train, y_train_binary, cv=3, method='decision_function')

precisions, recalls, thresholds = precision_recall_curve(
    y_train_binary, y_binary_sgd_scores)

# Make predictions by searching the lower threshold that give
# at least 90% precision
idx_90_precision = (precisions >= 0.90).argmax()
threshold_90_precision = thresholds[idx_90_precision]

y_binary_pred_90 = (
    y_binary_sgd_scores >= threshold_90_precision)
precision_score_90 = precision_score(
    y_train_binary, y_binary_pred_90)
recall_90_precision = recall_score(
    y_train_binary, y_binary_pred_90)

forest = RandomForestClassifier(random_state=42)
y_binary_proba_forest = cross_val_predict(
    forest, X_train, y_train_binary, cv=2, method='predict_proba')

# Get estimated probabilities for the positive class
y_binary_scores_forest = y_binary_proba_forest[:, 1]

(precisions_forest,
recalls_forest,
thresholds_forest) = precision_recall_curve(
    y_train_binary, y_binary_scores_forest)

y_positive_pred_forest = y_binary_proba_forest[:, 1] >= 0.5
y_f1_score_forest = f1_score(y_train_binary, y_positive_pred_forest)
y_roc_score_forest = roc_auc_score(
    y_train_binary, y_binary_scores_forest)

# Perform multiclass classification with
# binary classifier (OvO strategy)
svm = SVC(random_state=42)
svm.fit(X_train[:2000], y_train[:2000])
svm_prediction = svm.predict([X_train[2000]])
svm_prediction_scores = svm.decision_function([X_train[2000]])

y_train_pred = cross_val_predict(sgd, X_train[:2000], y_train[:2000], cv=2)
sample_weight = (y_train_pred != y_train[:2000])
ConfusionMatrixDisplay.from_predictions(
    y_train[:2000],
    y_train_pred,
    sample_weight=sample_weight,
    normalize='true',
    values_format='.0%')
plt.savefig('sgd_confusion_matrix')
