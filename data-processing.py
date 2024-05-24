from sklearn import (
    datasets, model_selection, preprocessing,
    metrics, base, cluster, compose, pipeline,
    impute)
import numpy as np
import pandas as pd

housing = datasets.fetch_california_housing(as_frame=True)
X: pd.DataFrame = housing['data']
y: pd.Series = housing['target']
housing_X_y: pd.DataFrame = pd.concat([X, y], axis=1)

housing_X_y.info()
print('housing_X_y', housing_X_y)

# Scale data and target values
data_scaler = preprocessing.StandardScaler()
target_scaler = preprocessing.StandardScaler()
X_scaled = data_scaler.fit_transform(X)
y_scaled = target_scaler.fit_transform(y.to_frame())

X['IncomeCat'] = pd.cut(
    X['MedInc'],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5])

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.2, stratify=X['IncomeCat'], random_state=42)

for X_set in (X, X_train, X_test):
    X_set.drop('IncomeCat', axis=1, inplace=True)

# Compute the standard correlation coefficient between each
# attribute and MedHouseVal
corr_matrix = housing_X_y.corr()
print(corr_matrix['MedHouseVal'].sort_values(ascending=False))

# Use custom transformer to create a feature that measure
# the geographic similarity between each district and San Francisco
sf_coords = 37.7749, -122.41
sf_transformer = preprocessing.FunctionTransformer(
    metrics.pairwise.rbf_kernel,
    kw_args=dict(Y=[sf_coords], gamma=0.1))
sf_simil: np.ndarray = sf_transformer.transform(
    X[['Latitude', 'Longitude']])

# Custom class transformer.
# Use KMeans clusterer in the fit() method to identify the main
# clusters in the training data.
# Use rbf_kernel() in thetransform() method to measure how similar
# each sample is to each cluster center.
class ClusterSimilarity(base.BaseEstimator, base.TransformerMixin):
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
        self.kmeans_ = cluster.KMeans(
            self.n_clusters, random_state=self.random_state)
        # Weight each district by its median house value
        self.kmeans_.fit(X, sample_weight=sample_weight)

        return self

    def transform(self, X: np.ndarray):
        # Measure the Gaussian RBF similarity between each district
        # and all 10 cluster centers
        return metrics.pairwise.rbf_kernel(
            X, self.kmeans_.cluster_centers_, gamma=self.gamma)

cluster_simil = ClusterSimilarity(
    n_clusters=10, gamma=1., random_state=42)
similarities = cluster_simil.fit_transform(
    X[['Latitude', 'Longitude']], sample_weight=y)

# Create a transformation pipeline
num_pipeline = pipeline.make_pipeline(
    impute.SimpleImputer(strategy='median'),
    preprocessing.StandardScaler())
column_selector = compose.make_column_selector(
    dtype_include=np.number)
# Transform numerical columns
num_columns_transformer = compose.make_column_transformer(
    (num_pipeline, column_selector)
)
X_num_columns_processed = num_columns_transformer.fit_transform(X)


log_pipeline = pipeline.make_pipeline(
    impute.SimpleImputer(strategy='median'),
    preprocessing.FunctionTransformer(
        np.log, feature_names_out='one-to-one'),
    preprocessing.StandardScaler()
)
log_columns_transformer = compose.make_column_transformer(
    (log_pipeline, ['MedInc', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup']),
    remainder='passthrough'
)
X_prepared = log_columns_transformer.fit_transform(X)
print(X_prepared[:3])
print(log_columns_transformer.get_feature_names_out())
