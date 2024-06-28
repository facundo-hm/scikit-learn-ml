from typing import cast
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pandas import DataFrame

iris = load_iris(as_frame=True)
iris_data = cast(DataFrame, iris.data)
iris_target = cast(DataFrame, iris.target)
iris_target_names = cast(list, iris.target_names)

X = iris_data[['petal length (cm)', 'petal width (cm)']].values
y = iris_target

dtc = DecisionTreeClassifier(max_depth=2, random_state=42)
dtc.fit(X, y)

# Visualize the trained decision tree
export_graphviz(
    dtc,
    out_file='./charts/iris_tree.dot',
    feature_names=['petal length (cm)', 'petal width (cm)'],
    class_names=iris_target_names,
    rounded=True,
    filled=True
)

dtc_proba = dtc.predict_proba([[5, 1.5]]).round(3)
dtc_predict = dtc.predict([[5, 1.5]])
print('dtc_proba', dtc_proba)
print('dtc_predict', dtc_predict)
