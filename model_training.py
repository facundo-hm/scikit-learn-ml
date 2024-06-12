from sklearn.preprocessing import add_dummy_feature
import numpy as np

# Normal equation.
# θ^ = (X⊺ X)-1 X⊺ y
# Find the value of θ that minimizes the MSE (mean square error).
np.random.seed(42)
# number of instances
m = 100
X = 2 * np.random.rand(m, 1)
# y = 4 + 3x₁ + Gaussian noise
y = 4 + 3 * X + np.random.randn(m, 1)
# add x0 = 1 to each instance
X = add_dummy_feature(X)
# Compute the inverse of a matrix and perform matrix multiplication.
theta_best = np.linalg.inv(X.T @ X) @ X.T @ y
print('theta_best', theta_best)
# Predict using θ^
X_test = np.array([[0], [2]])
X_test = add_dummy_feature(X_test)
y_predict = X_test @ theta_best
print('y_predict', y_predict)
