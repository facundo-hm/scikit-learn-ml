from sklearn import datasets, model_selection

housing = datasets.fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    housing['data'], housing['target'], test_size=0.2, shuffle=False)

print('data train:\n', X_train)
print('target train:\n', y_train)
print('data test:\n', X_test)
print('target test:\n', y_test)
