import numpy as np
import os


class BinaryCrossEntropy(object):

    def __init__(self):
        self.need_initialize = False

    def forward(self, activations, y):
        self.batch_size = activations.shape[0]
        self.n_classes = activations.shape[1]

        prob = np.log(activations)
        loss_matrix = np.zeros(prob.shape)
        indices = np.arange(y.shape[0])
        loss_matrix[indices, y] = 1
        self.gradient = -1 * y / activations
        loss_matrix = loss_matrix * prob
        return np.mean(np.sum(loss_matrix, 1), 0)

    def backward(self):
        return self.gradient.reshape((self.batch_size,
                                      self.n_classes))


class CrossEntropy(object):

    def __init__(self):
        self.need_initialize = False

    def forward(self, activations, y):
        log_act = np.log(activations)
        # print activations[0], "log_act", y[0]
        cross_entropy = -1 * np.mean(
            log_act[np.arange(y.shape[0]), y.flatten()])
        self.prob = activations
        self.y = y
        return cross_entropy

    def backward(self):
        loss_matrix = np.zeros(self.prob.shape)
        indices = np.arange(self.y.shape[0])
        loss_matrix[indices, self.y.flatten()] = 1
        gradient = -1 * loss_matrix / self.prob
        return gradient


class SquaredError(object):

    def __init__(self):
        self.need_initialize = False

    def forward(self, y_hat, y):
        self.batch_size = y_hat.shape[0]
        self.n_classes = y_hat.shape[1]
        self.loss_matrix = np.zeros(y_hat.shape)
        indices = np.arange(y.shape[0])
        self.loss_matrix[indices, y.flatten()] = 1
        self.loss_matrix = y_hat - self.loss_matrix
        return np.mean(np.sum(self.loss_matrix**4, 1), 0)

    def backward(self):
        gradient = self.loss_matrix
        return gradient
