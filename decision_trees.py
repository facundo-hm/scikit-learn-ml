from typing import cast
from sklearn.datasets import load_iris, make_moons
from sklearn.tree import (
    DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor)
from sklearn.model_selection import train_test_split
from pandas import DataFrame
import numpy as np

np.random.seed(42)

iris = load_iris(as_frame=True)
iris_data = cast(DataFrame, iris.data)
iris_target = cast(DataFrame, iris.target)
iris_target_names = cast(list, iris.target_names)

X = iris_data[['petal length (cm)', 'petal width (cm)']].values
y = iris_target

dtc_iris = DecisionTreeClassifier(max_depth=2, random_state=42)
dtc_iris.fit(X, y)

# Visualize the trained decision tree
export_graphviz(
    dtc_iris,
    out_file='./charts/iris_tree.dot',
    feature_names=['petal length (cm)', 'petal width (cm)'],
    class_names=iris_target_names,
    rounded=True,
    filled=True
)

dtc_iris_proba = dtc_iris.predict_proba([[5, 1.5]]).round(3)
dtc_iris_predict = dtc_iris.predict([[5, 1.5]])
print('dtc_iris_proba', dtc_iris_proba)
print('dtc_iris_predict', dtc_iris_predict)

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)

X_train, X_test, y_train, y_test = cast(
    list[np.ndarray],
    train_test_split(X, y, test_size=0.4, random_state=42))

# Compare tree with and without regularization
dtc_moons_1 = DecisionTreeClassifier(random_state=42)
dtc_moons_2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)

dtc_moons_1.fit(X_train, y_train)
dtc_moons_2.fit(X_test, y_test)

dtc_moons_1_score = dtc_moons_1.score(X_test, y_test)
dtc_moons_2_score = dtc_moons_2.score(X_test, y_test)
print('dtc_moons_1_score', dtc_moons_1_score)
print('dtc_moons_2_score', dtc_moons_2_score)

### REGRESSION ###
X = np.random.rand(200, 1) - 0.5
y = X ** 2 + 0.025 * np.random.randn(200, 1)

dtr = DecisionTreeRegressor(max_depth=2, random_state=42)
dtr.fit(X, y)
