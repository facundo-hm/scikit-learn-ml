from typing import cast
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.linear_model import Perceptron
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
from pandas import DataFrame

iris = load_iris(as_frame=True)

X = iris.data[['petal length (cm)', 'petal width (cm)']].values
y = (iris.target == 0)

X_train, X_test, y_train, y_test = cast(
    list[DataFrame],
    train_test_split(X, y, test_size=0.2, random_state=42))

perceptron = Perceptron(random_state=42)
perceptron.fit(X, y)

perceptron.score(X_test, y_test)

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
