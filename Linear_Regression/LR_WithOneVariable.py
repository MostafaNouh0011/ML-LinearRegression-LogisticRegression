import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read data
path = 'Ex1Data.txt'
data = pd.read_csv(path, header = None, names=['Population', 'Profit'])

# show details of data
print('data \n ', data.head())  # show first 5 rows from data
print('data describtion \n', data.describe())

# draw data
data.plot(kind='scatter', x='Population', y='Profit', figsize=(5, 5))
plt.show()

# adding new column called ones before data
data.insert(0, 'Ones', 1)
print('New data \n', data.head())

# separate X (training data) from y (target variable)
cols = data.shape[1]  # number of columns
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

print('X data \n', X.head())
print('y data \n', y.head())

# Convert from data frames to matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0, 0]) 

print('X \n', X)
print('X.shape : ', X.shape)
print('y \n', y)
print('y.shape : ', y.shape)
print('Theta : ', theta)

# Cost function
def ComputeCost(X, y, theta):
    z = np.power(((X * theta.T) - y), 2)
    return sum(z) / (2*len(X))                  # J

print('compute cost (X, y, theta) : ', ComputeCost(X, y, theta))

# GD function
def GradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y
        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))
    
        theta = temp                        # update theta
        cost[i] = ComputeCost(X, y, theta)  # update cost
    
    return theta, cost

# intialize variables for learning rate and iterations
alpha = 0.01
iters = 1000

# perform gradient descent to fit the model parameters
g, cost = GradientDescent(X, y, theta, alpha, iters)

print('g : ', g)
print('cost \n', cost[0:50])
print('Compute cost after updating : ', ComputeCost(X, y, g))

# get best fit line
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# print('x \n', x)
# print('g \n', g)
f = g[0, 0] + (g[0, 1] * x)                   # h(X) = theta0 + theta1 * X        (Linear Equation)

# draw the Best Fit Line
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted profit vs. Population size')
plt.show()

# draw error graph
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('iteration')
ax.set_ylabel('cost')
ax.set_title('Error vs. Training Epoch')
plt.show()
