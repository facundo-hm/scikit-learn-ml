from typing import cast
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import make_moons
from pandas import DataFrame

iris = load_iris(as_frame=True)
iris_data = cast(DataFrame, iris.data)
iris_target = cast(DataFrame, iris.target)

X = iris_data[['petal length (cm)', 'petal width (cm)']].values
y = (iris_target == 2)

### SOFT MARGIN CLASSIFICATION ###
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

### NONLINEAR SVM CLASSIFICATION ###
# Create dataset for binary classification
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
print('X_moons', X)
print('X_moons_poly', PolynomialFeatures(degree=2, include_bias=False).fit_transform(X))

svm_poly = make_pipeline(
    PolynomialFeatures(degree=3),
    StandardScaler(),
    LinearSVC(C=10, max_iter=10_000, random_state=42, dual=True)
)
svm_poly.fit(X, y)

### POLYNOMIAL KERNEL ###
# Implement the kernel trick which gets the same result as adding
# many polynomial features, even with a very high degree,
# without actually having to add them.
poly_kernel_svm_clf = make_pipeline(
    StandardScaler(),
    SVC(kernel='poly', degree=3, coef0=1, C=5))
poly_kernel_svm_clf.fit(X, y)
