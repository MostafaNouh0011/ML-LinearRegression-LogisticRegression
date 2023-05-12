import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize


path = 'E:\\Machine_learning\\4-Classification\\ex3data.mat'
data = loadmat(path)

print('Data \n', data)
print('X \n', data['X'])
print('X.shape \n', data['X'].shape)
print('y \n', data['y'])
print('y.shape \n', data['y'].shape)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / 2 * len(X)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))

    return np.sum(first - second) / (len(X)) + reg

def gradient(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    error = sigmoid(X * theta.T) - y

    grad = ((X.T * error) / len(X)).T + ((learningRate / len(X)) * theta)

    grad[0, 0] = np.sum(np.multiply(error, X[:, 0])) / len(X)

    return np.array(grad).ravel()

def one_vs_all(X, y, num_labels, learningRate):
    rows = X.shape[0]
    params = X.shape[1]

    all_theta = np.zeros((num_labels, params + 1))

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))

        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learningRate), method= 'TNC', jac=gradient)
        all_theta[i-1, :] = fmin.x

    return all_theta


rows = data['X'].shape[0]    #5000
params = data['X'].shape[1]  #400
# print('rows = ', rows)
# print('params = ', params)

all_theta = np.zeros((10, params + 1))

# print('all_theta \n', all_theta)
# print('all_theta.shape \n', all_theta.shape)


# Insert column to X
X = np.insert(data['X'], 0, values=np.ones(rows), axis=1)

theta = np.zeros(params + 1)

# print('theta \n', theta)
# print('theta.shape \n', theta.shape)


y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
# print('y_0 \n', y_0)


all_theta = one_vs_all(data['X'], data['y'], 10, 1)

print('theta shape ', all_theta.shape)
print('theta \n', all_theta)


def predict_all(X, all_theta):
    rows = X.shape[0]
    params = X.shape[1]
    num_labels = all_theta.shape[0]

    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    X = np.matrix(X)
    all_theta = np.matrix(all_theta)

    h = sigmoid(X * all_theta.T)

    h_argmax = np.argmax(h, axis=1)
    h_argmax = h_argmax + 1

    return h_argmax

y_pred = predict_all(data['X'], all_theta)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accurcy = (sum(map(int, correct))) / float(len(correct))
print('Accurcy {0} %'.format(accurcy * 100))