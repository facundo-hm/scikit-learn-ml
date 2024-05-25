from sklearn import linear_model, pipeline
from data_processing import X, y, columns_transformer

lin_reg = pipeline.make_pipeline(
    columns_transformer, linear_model.LinearRegression())
lin_reg.fit(X, y)

predictions = lin_reg.predict(X)
print('predictions', predictions[:5])
print('labels', y.iloc[:5].values)
