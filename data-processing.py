from sklearn import datasets, model_selection
import numpy as np
import pandas as pd

housing = datasets.fetch_california_housing(as_frame=True)
housing_data = housing['data']
housing_target = housing['target']

housing_data.info()

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

print('data train:\n', X_train)
print('target train:\n', y_train)
print('data test:\n', X_test)
print('target test:\n', y_test)
