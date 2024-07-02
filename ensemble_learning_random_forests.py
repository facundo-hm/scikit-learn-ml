from sklearn.datasets import make_moons
from sklearn.ensemble import (
    RandomForestClassifier, VotingClassifier, BaggingClassifier,
    AdaBoostClassifier, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

### VOTING CLASSIFIERS ###
vc = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42))])

vc.fit(X_train, y_train)

# Use hard voting.
# Predict the class that gets the most votes.
vc_predict = vc.predict(X_test[:1])
vc_estimators_predict = [
    clf.predict(X_test[:1]) for clf in vc.estimators_]
vc_score_h = vc.score(X_test, y_test)
print('vc_predict', vc_predict)
print('vc_estimators_predict', vc_estimators_predict)
print('vc_score_h', vc_score_h)

# Use soft voting.
# Predict the class with the highest class probability,
# averaged over all the individual classifiers.
vc.voting = 'soft'
vc.named_estimators['svc'].probability = True
vc.fit(X_train, y_train)
vc_score_s = vc.score(X_test, y_test)
print('vc_score_s', vc_score_s)

### BAGGING AND PASTING ###
# Use the same training algorithm for every predictor but train them
# on different random subsets of the training set.
# Bagging, sampling with replacement.
# Pasting, sampling without replacement.
bc = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    oob_score=True, n_jobs=-1, random_state=42)
bc.fit(X_train, y_train)
bc_score = bc.score(X_test, y_test)
print('bc_score', bc_score)
print('bc.oob_score_', bc.oob_score_)

### RANDOM FORESTS ###
rfc = RandomForestClassifier(
    n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rfc.fit(X_train, y_train)
rfc_score = rfc.score(X_test, y_test)
print('rfc_score', rfc_score)

### ADABOOST ###
# A new predictor uses the training instances that the predecessor
# underfit to focuse more and more on the hard cases.
ada = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=50,
    learning_rate=0.5, random_state=42, algorithm='SAMME')
ada.fit(X_train, y_train)
ada_score = ada.score(X_test, y_test)
print('ada_score', ada_score)

### GRADIENT TREE BOOSTING ###
np.random.seed(42)

X_quad = np.random.rand(100, 1) - 0.5
y_quad = 3 * X_quad[:, 0] ** 2 + 0.05 * np.random.randn(100)

dtr_1 = DecisionTreeRegressor(max_depth=2, random_state=42)
dtr_1.fit(X_quad, y_quad)

# Train a new predictor on the residual error made by the previous one.
y_quad_1 = y_quad - dtr_1.predict(X_quad)
dtr_2 = DecisionTreeRegressor(max_depth=2, random_state=43)
dtr_2.fit(X_quad, y_quad_1)

y_quad_2 = y_quad_1 - dtr_2.predict(X_quad)
dtr_3 = DecisionTreeRegressor(max_depth=2, random_state=44)
dtr_3.fit(X_quad, y_quad_2)

# Predict a new instance by adding up the predictions of all the trees.
X_quad_test = np.array([[-0.4], [0.], [0.5]])
dtr_pred = sum(
    dtr.predict(X_quad_test) for dtr in (dtr_1, dtr_2, dtr_3))
print('dtr_pred', dtr_pred)

### GRADIENT BOOSTING ###
# Fit the new predictor to the residual errors made by
# the previous predictor.
gbr = GradientBoostingRegressor(
    max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbr.fit(X_quad, y_quad)
gbr_pred = gbr.predict(X_quad_test)
print('gbr_pred', gbr_pred)
