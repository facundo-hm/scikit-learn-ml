from typing import cast
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    confusion_matrix, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, f1_score, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_mnist_digit

mnist = fetch_openml('mnist_784', as_frame=False)
X, y = cast(np.ndarray, mnist.data), cast(np.ndarray, mnist.target)

X_train_normal, X_test_normal, y_train_normal, y_test_normal = cast(
    list[np.ndarray],
    train_test_split(X, y, test_size=0.2, random_state=42))

data_scaler = StandardScaler()
X_scaled = data_scaler.fit_transform(X.astype('float64'))

X_train, X_test, y_train, y_test = cast(
    list[np.ndarray],
    train_test_split(X_scaled, y, test_size=0.2, random_state=42))

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

# Mesure model scores
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

# Plot errors normalized by row. It displays the perentage
# of the totals errors the model made by misclassifying each image.
# E.g. 36% of the errors the model made on images of 7s were
# misclassifications as 9s.
y_train_pred = cross_val_predict(
    sgd, X_train[:2000], y_train[:2000], cv=2)
sample_weight = (y_train_pred != y_train[:2000])
ConfusionMatrixDisplay.from_predictions(
    y_train[:2000],
    y_train_pred,
    sample_weight=sample_weight,
    normalize='true',
    values_format='.0%')
plt.savefig('./charts/sgd_confusion_matrix')

# Creates a multilabel array containing two target labels
# for each digit image
y_large_values = (y_train >= '7')
y_odd_values = (y_train.astype('int8') % 2 == 1)
y_multilabel = np.c_[y_large_values, y_odd_values]

knn = KNeighborsClassifier()
knn.fit(X_train, y_multilabel)
knn_prediction = knn.predict([X_train[10]])
print('knn_prediction', knn_prediction, y_train[10])

# Use single label classifier model to perform
# multilable classification
chain = ClassifierChain(SVC(), cv=2, random_state=42)
chain.fit(X_train[:2000], y_multilabel[:2000])
chain_prediction = chain.predict([X_train[2000]])
print('chain_prediction', chain_prediction, y_train[2000])

# Add noise to digits pixel intensities
np.random.seed(42)
train_noise = np.random.randint(0, 100, (len(X_train_normal), 784))
X_train_noise = X_train_normal + train_noise
test_noise = np.random.randint(0, 100, (len(X_test_normal), 784))
X_test_noise = X_test_normal + test_noise
y_train_noise = X_train_normal
y_test_noise = X_test_normal

# Multioutput classification, each label can be multiclass.
# The classifier’s output is multilabel (one label per pixel)
# and each label can have multiple values (pixel intensity
# ranges from 0 to 255).
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_noise, y_train_noise)
clean_digit = knn_clf.predict([X_test_noise[0]])
plot_mnist_digit(clean_digit, 'digit_prediction')
plot_mnist_digit(y_test_noise[0], 'digit_test')
