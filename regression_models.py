from sklearn import (
    linear_model, pipeline, metrics, tree, model_selection, ensemble)
from data_processing import X_train, y_train, columns_transformer

lin_reg = pipeline.make_pipeline(
    columns_transformer, linear_model.LinearRegression())
lin_reg.fit(X_train, y_train)

lin_reg_predictions = lin_reg.predict(X_train)
lin_reg_rmse = metrics.root_mean_squared_error(
    y_train, lin_reg_predictions)
print('lin_reg_predictions', lin_reg_predictions[:5])
print('lin_reg_labels', y_train.iloc[:5].values)
print('lin_reg_rmse', lin_reg_rmse)

tree_reg = pipeline.make_pipeline(
    columns_transformer,
    tree.DecisionTreeRegressor(random_state=42))
tree_reg.fit(X_train, y_train)

tree_reg_predictions = tree_reg.predict(X_train)
tree_reg_rmse = metrics.root_mean_squared_error(
    y_train, tree_reg_predictions)
print('tree_reg_predictions', tree_reg_predictions[:5])
print('tree_reg_labels', y_train.iloc[:5].values)
print('tree_reg_rmse', tree_reg_rmse)

tree_reg_cross_val = pipeline.make_pipeline(
    columns_transformer,
    tree.DecisionTreeRegressor(random_state=42))
# Randomly split the training set into 10 nonoverlapping
# subsets called folds.
# Train and evaluate the decision tree model 10 times.
# Pick a different fold for evaluation every time and use
# the other 9 folds for training.
# Return an array containing the 10 evaluation scores.
tree_rmses = model_selection.cross_val_score(
    tree_reg_cross_val,
    X_train,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=10)
print('tree_rmses', tree_rmses)

forest_reg = pipeline.make_pipeline(
    columns_transformer,
    ensemble.RandomForestRegressor(random_state=42))
forest_rmses = model_selection.cross_val_score(
    forest_reg,
    X_train,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=10)
print('forest_rmses', forest_rmses)