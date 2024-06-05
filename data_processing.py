from typing import cast
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import pairwise
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import (
    make_column_transformer, make_column_selector, ColumnTransformer)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import Bunch
import numpy as np
import pandas as pd

housing = cast(Bunch, fetch_california_housing(as_frame=True))
X = cast(pd.DataFrame, housing['data'])
y = cast(pd.Series, housing['target'])
housing_X_y: pd.DataFrame = pd.concat([X, y], axis=1)

housing_X_y.info()
print('housing_X_y', housing_X_y)

# Scale data and target values
data_scaler = StandardScaler()
target_scaler = StandardScaler()
X_scaled = data_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y.to_frame())

X['IncomeCat'] = pd.cut(
    X['MedInc'],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5])

X_train, X_test, y_train, y_test = cast(
    list[pd.DataFrame],
    train_test_split(
        X, y, test_size=0.2, stratify=X['IncomeCat'], random_state=42))

for X_set in (X, X_train, X_test):
    X_set.drop('IncomeCat', axis=1, inplace=True)

# Compute the standard correlation coefficient between each
# attribute and MedHouseVal
corr_matrix = housing_X_y.corr()
print(corr_matrix['MedHouseVal'].sort_values(ascending=False))

# Use custom transformer to create a feature that measure
# the geographic similarity between each district and San Francisco
sf_coords = 37.7749, -122.41
sf_transformer = FunctionTransformer(
    pairwise.rbf_kernel,
    kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil = sf_transformer.transform(X[['Latitude', 'Longitude']])

# Custom class transformer.
# Use KMeans clusterer in the fit() method to identify the main
# clusters in the training data.
# Use rbf_kernel() in thetransform() method to measure how similar
# each sample is to each cluster center.
class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(
        self, n_clusters=10, gamma=1.0, random_state: int = None
    ):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(
        self, X: np.ndarray, y=None, sample_weight: np.ndarray = None
    ):
        # Locate the clusters
        self.kmeans_ = KMeans(
            self.n_clusters, random_state=self.random_state)
        # Weight each district by its median house value
        self.kmeans_.fit(X, sample_weight=sample_weight)

        return self

    def transform(self, X: np.ndarray):
        # Measure the Gaussian RBF similarity between each district
        # and all 10 cluster centers
        return pairwise.rbf_kernel(
            X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [
            f'Cluster {i} similarity' for i in range(self.n_clusters)]

cluster_simil = ClusterSimilarity(
    n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(
    X[['Latitude', 'Longitude']], sample_weight=y)

# Create a transformation pipeline
num_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler())
column_selector = make_column_selector(
    dtype_include=np.number)
# Transform numerical columns
num_columns_transformer = make_column_transformer(
    (num_pipeline, column_selector)
)
X_num_columns_processed = num_columns_transformer.fit_transform(X)


log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler()
)
cluster_pipeline = ClusterSimilarity(
    n_clusters=10, gamma=1., random_state=42)
default_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    StandardScaler())

columns_transformer = ColumnTransformer([
    ('log', log_pipeline, ['MedInc', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup']),
    ('geo', cluster_pipeline, ['Latitude', 'Longitude'])],
    remainder=default_pipeline
)
# X_columns_processed = columns_transformer.fit_transform(X)
