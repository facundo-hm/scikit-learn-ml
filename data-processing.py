from sklearn import datasets, model_selection
import numpy as np
import pandas as pd

housing = datasets.fetch_california_housing(as_frame=True)
housing_data: pd.DataFrame = housing['data']
housing_target: pd.DataFrame = housing['target']
housing_total_data: pd.DataFrame = pd.concat(
    [housing_data, housing_target], axis=1)

housing_total_data.info()
print(housing_data.describe())

print('housing_total_data', housing_total_data)
print('housing_data', housing_data)
print('housing_target', housing_target)

housing_data['IncomeCat'] = pd.cut(
    housing_data['MedInc'],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5])

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    housing_data,
    housing_target,
    test_size=0.2,
    stratify=housing_data['IncomeCat'],
    random_state=42)

for X_set in (X_train, X_test):
    X_set.drop('IncomeCat', axis=1, inplace=True)

# Compute the standard correlation coefficient between each
# attribute and MedHouseVal
corr_matrix = housing_total_data.corr()
print(corr_matrix['MedHouseVal'].sort_values(ascending=False))
