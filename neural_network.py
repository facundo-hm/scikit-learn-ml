from typing import cast
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import Perceptron
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas import DataFrame

iris = load_iris(as_frame=True)

X_iris = iris.data[['petal length (cm)', 'petal width (cm)']].values
y_iris = (iris.target == 0)

X_iris_train, X_iris_test, y_iris_train, y_iris_test = cast(
    list[DataFrame],
    train_test_split(X_iris, y_iris, test_size=0.2, random_state=42))

perceptron = Perceptron(random_state=42)
perceptron.fit(X_iris_train, y_iris_train)

perceptron.score(X_iris_test, y_iris_test)

### REGRESSION MLP ###
housing = fetch_california_housing()

X, X_test, y, y_test = cast(
    list[np.ndarray],
    train_test_split(housing.data, housing.target, random_state=42))
X_train, X_valid, y_train, y_valid = cast(
    list[np.ndarray],
    train_test_split(X, y, random_state=42))

mlpr = MLPRegressor(hidden_layer_sizes=[50, 50, 50], random_state=42)
pipeline = make_pipeline(StandardScaler(), mlpr)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_valid)
root_mean_squared_error(y_valid, y_pred)

### CLASSIFICATION MLP ###
mlpc = MLPClassifier(
    hidden_layer_sizes=[10], max_iter= 700, random_state=42)
pipeline = make_pipeline(StandardScaler(), mlpc)
pipeline.fit(X_iris_train, y_iris_train)

mlpc.score(X_iris_test, y_iris_test)
