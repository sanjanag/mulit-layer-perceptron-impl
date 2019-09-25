import numpy as np


class Adam(object):

    def __init__(self, lr=0.002, b1=0.1, b2=0.001, e=1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.e = e
        self.m = []
        self.v = []
        self.g = []
        self.i = 0.00

    def initialize(self, params):
        for layer in params:
            temp_m = []
            temp_v = []
            temp_g = []
            for weights in layer:
                temp_m.append(np.zeros(weights.shape))
                temp_v.append(np.zeros(weights.shape))
                temp_g.append(np.zeros(weights.shape))

            self.m.append(temp_m)
            self.v.append(temp_v)
            self.g.append(temp_g)

    def get_update(self, gradients):
        self.i += 1.00
        fix1 = 1. - (1. - self.b1)**self.i
        fix2 = 1. - (1. - self.b2)**self.i
        lr_t = self.lr * (np.sqrt(fix2) / fix1)
        for idx, layer in enumerate(self.m):
            for w_idx, _ in enumerate(layer):
                self.m[idx][w_idx] = (self.b1 *
                                      gradients[idx][w_idx]) +\
                                     ((1 - self.b1) * self.m[idx][w_idx])
                self.v[idx][w_idx] = (self.b1 *
                                      np.power(gradients[idx][w_idx], 2)) +\
                                     ((1 - self.b1) * self.v[idx][w_idx])
                self.g[idx][w_idx] = lr_t * (self.m[idx][w_idx] /
                                             (np.sqrt(self.v[idx][w_idx]) +
                                              self.e))
        return self.g


class Adagrad(object):

    def __init__(self, learning_rate=1e-2, fudge_factor=1e-6):
        self.learning_rate = learning_rate
        self.fudge_factor = fudge_factor
        self.g = []

    def initialize(self, params):
        for layer in params:
            temp_g = []
            for weights in layer:
                temp_g.append(np.zeros(weights.shape))
            self.g.append(temp_g)

    def get_update(self, gradients):
        for idx, layer in enumerate(self.g):
            for w_idx, _ in enumerate(layer):
                #print np.asarray(gradients[idx][w_idx]).shape
                self.g[idx][w_idx] += gradients[idx][w_idx]**2
                gradients[idx][w_idx] /= \
                    (self.fudge_factor + np.sqrt(self.g[idx][w_idx]))
                gradients[idx][w_idx] *= self.learning_rate
        return gradients
