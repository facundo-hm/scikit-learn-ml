from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from data_processing import (
    X_train, y_train, X_test, y_test, columns_transformer)

lin_reg = make_pipeline(columns_transformer, LinearRegression())
lin_reg.fit(X_train, y_train)
print('Bias term: ', lin_reg.intercept_, ', Weights: ', lin_reg.coef_)

lin_reg_predictions = lin_reg.predict(X_train)
lin_reg_rmse = root_mean_squared_error(y_train, lin_reg_predictions)
print('lin_reg_predictions', lin_reg_predictions[:5])
print('lin_reg_labels', y_train.iloc[:5].values)
print('lin_reg_rmse', lin_reg_rmse)

tree_reg = make_pipeline(
    columns_transformer,
    DecisionTreeRegressor(random_state=42))
tree_reg.fit(X_train, y_train)

tree_reg_predictions = tree_reg.predict(X_train)
tree_reg_rmse = root_mean_squared_error(y_train, tree_reg_predictions)
print('tree_reg_predictions', tree_reg_predictions[:5])
print('tree_reg_labels', y_train.iloc[:5].values)
print('tree_reg_rmse', tree_reg_rmse)

tree_reg_cross_val = make_pipeline(
    columns_transformer,
    DecisionTreeRegressor(random_state=42))
# Randomly split the training set into 10 nonoverlapping
# subsets called folds.
# Train and evaluate the decision tree model 10 times.
# Pick a different fold for evaluation every time and use
# the other 9 folds for training.
# Return an array containing the 10 evaluation scores.
tree_rmses = cross_val_score(
    tree_reg_cross_val,
    X_train,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=10)
print('tree_rmses', tree_rmses)

forest_reg_pipeline = Pipeline([
    ('columns_transformer', columns_transformer),
    ('random_forest', RandomForestRegressor(random_state=42))
])

forest_rmses = cross_val_score(
    forest_reg_pipeline,
    X_train,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=10)
print('forest_rmses', forest_rmses)

rnd_param = {
    'columns_transformer__geo__n_clusters': randint(low=3, high=50),
    'random_forest__max_features': randint(low=2, high=20)}

rnd_search = RandomizedSearchCV(
    forest_reg_pipeline,
    param_distributions=rnd_param,
    n_iter=10,
    cv=3,
    scoring='neg_root_mean_squared_error',
    random_state=42)
rnd_search.fit(X_train, y_train)
print('rnd_search.best_params_', rnd_search.best_params_)

forest_reg_model = rnd_search.best_estimator_
forest_reg_predictions = forest_reg_model.predict(X_test)

forest_reg_rmse = root_mean_squared_error(
    y_test, forest_reg_predictions, squared=False)
print('forest_reg_rmse', forest_reg_rmse)
