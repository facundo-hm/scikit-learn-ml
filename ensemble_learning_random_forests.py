from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

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
