from typing import cast
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.datasets import make_moons
from pandas import DataFrame

iris = load_iris(as_frame=True)
iris_data = cast(DataFrame, iris.data)
iris_target = cast(DataFrame, iris.target)

X = iris_data[['petal length (cm)', 'petal width (cm)']].values
y = (iris_target == 2)

# Soft Margin Classification
# Find a good balance between keeping the street as large as
# possible and limiting the margin violations
svm = make_pipeline(
    StandardScaler(), LinearSVC(C=1, random_state=42, dual=True))
svm.fit(X, y)

X_test = [[5.5, 1.7], [5.0, 1.5]]
svm_predict = svm.predict(X_test)
# Measure the signed distance between each instance and
# the decision boundary
svm_scores = svm.decision_function(X_test)
print('svm_predict', svm_predict)
print('svm_scores', svm_scores)

# Nonlinear SVM Classification
# Create dataset for binary classification
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

svm_poly = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42, dual=True)
)
svm_poly.fit(X, y)
