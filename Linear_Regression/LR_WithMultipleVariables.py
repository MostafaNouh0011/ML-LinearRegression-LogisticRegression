import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# read data 
path = 'Ex2Data.txt'
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

# show data
print('data2 \n', data.head())

# rescaling data
data = (data - data.mean()) / data.std()

print('data after rescaling \n ', data.head())  

print('describe data \n', data.describe())

# add ones column
data.insert(0, 'Ones', 1)

print('New data \n', data.head())

# separate x (training data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols-1]
y = data.iloc[:, cols-1:cols]

# print('-----------------------------------------------')
# print('X \n', X.head(10))
# print('y \n', y.head(10))
# print('-----------------------------------------------')

# convert to matrices and initialize theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix([0, 0, 0])

# print('X \n', X)
# print('X.shape : ', X.shape)
# print('y \n', y)
# print('y.shape : ', y.shape)
# print('theta \n', theta)
# print('theta.shape : ', theta.shape)
# print('-----------------------------------------------')


# Cost function
def ComputeCost(X, y, theta):
    z = np.power(((X * theta.T) - y), 2)
    return sum(z) / (2*len(X))                  # J

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
    
        theta = temp
        cost[i] = ComputeCost(X, y, theta)
    
    return theta, cost


# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100

# perform linear regression on the data set
g, cost = GradientDescent(X, y, theta, alpha, iters)

# get the cost (error) of the model
thiscost = ComputeCost(X, y, g)

print('g : ', g)
print('cost : ', cost[0:50])
print('compute cost : ', thiscost)

# get best fit line for Size vs. Price
x = np.linspace(data.Size.min(), data.Size.max(), 100)
# print('x \n', x)
# print('g : ', g)

f = g[0, 0] + (g[0, 1] * x)            # h(X) = theta0 + theta1 * X
print('f : ', f)

#draw best fit line for Size vs. Price
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Size, data.Price, label='Trianing data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')
plt.show()

#draw error graph 
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training epoch')
plt.show()
