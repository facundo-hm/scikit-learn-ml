from sklearn.preprocessing import add_dummy_feature
import numpy as np

# Normal equation
# θ^ = (X⊺ X)-1 X⊺ y
# Find the value of θ that minimizes the MSE (mean square error)
np.random.seed(42)
# number of instances
m = 100
X = 2 * np.random.rand(m, 1)
# y = 4 + 3x₁ + Gaussian noise
y = 4 + 3 * X + np.random.randn(m, 1)
# Add x0 = 1 to each instance
X = add_dummy_feature(X)
# Compute the inverse of a matrix and perform matrix multiplication
theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print('theta_hat', theta_hat)
# Predict using θ^
X_test = np.array([[0], [2]])
X_test = add_dummy_feature(X_test)
y_predict = X_test @ theta_hat
print('y_predict', y_predict)

# Find theta hat by computing the pseudoinverse using
# a standard matrix factorization technique called singular
# value decomposition (SVD). Which is more efficient than
# computing the Normal equation.
theta_hat_pinv = np.linalg.pinv(X) @ y
print('pseudoinverse', theta_hat_pinv)

# Batch Gradient Descent
# learning rate
eta = 0.1
n_epochs = 1000
# randomly initialized model parameters
theta = np.random.randn(2, 1)

for _ in range(n_epochs):
    # Compute the gradient vector of the MSE cost function
    gradients = 2 / m * X.T @ (X @ theta - y)
    theta = theta - eta * gradients

# Stochastic Gradient Descent
n_epochs = 50
# Learning schedule hyperparameters
t0, t1 = 5, 50
# Determine the learning rate
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for iteration in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ (xi @ theta - yi)
        eta = learning_schedule(epoch * m + iteration)
        theta = theta - eta * gradients

print('theta', theta)