import numpy as np
import os


class Linear(object):

    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.initialized = False
        self.resize = True
        self.need_initialize = True
        self.parameters = []
        self.grad_params = []
        self.layer_num = 0
        self.batch_size = 0

    def __str__(self):
        return "Linear | Layer no. " + str(self.layer_num) +\
            " with (in_dim, out_dim) = (  " + str(self.input_dim) +\
            ", " + str(self.output_dim) + ")\n"

    def initialize(self):
        self.initialized = True
        W = np.random.normal(loc=0.0,
                             scale=0.001,
                             size=(self.input_dim, self.output_dim))
        b = np.zeros((self.output_dim))
        self.parameters.extend([W, b])

    def forward(self, x, train=False):
        assert(self.initialized)
        self.inp = x
        [W, b] = self.parameters
        output = np.dot(x, W) + b
        return output

    def get_delta(self):
        layer_err = self.parameters[0].T
        return layer_err

    def dW(self):
        batch_size = self.inp.shape[0]
        if self.batch_size != batch_size:
            self.b_gradient = np.ones((batch_size, self.output_dim))
            self.bs = batch_size
        self.w_gradient = self.inp
        return [self.w_gradient, self.b_gradient]

    def update_parameters(self, update):
        for i in xrange(len(update)):
            self.parameters[i] -= update[i]


class Softmax(object):

    def __init__(self, input_dim=None):
        self.need_initialize = False
        self.resize = False
        self.layer_num = 0

    def __repr__(self):
        return "Softmax | Layer no. " + str(self.layer_num)

    def forward(self, x, train=False):
        activations = x
        max_act = np.max(activations, axis=1, keepdims=True)
        exponentiated = np.exp(activations - max_act)
        partition = np.sum(exponentiated, axis=1, keepdims=True)
        self.out = exponentiated / partition
        # print np.sum(self.out, axis=1)
        return self.out

    def get_delta(self):
        self.gradient = -self.out[..., None] * self.out[:, None, :]
        iy, ix = np.diag_indices_from(self.gradient[0])
        self.gradient[:, iy, ix] = self.out * (1. - self.out)
        return self.gradient

    def dW(self, delta):
        pass

    def update_parameters(self, update):
        for i in len(update):
            self.parameters[i] += update[i]


class Logistic(object):

    def __init__(self, input_dim=None):
        self.need_initialize = False
        self.resize = False
        self.layer_num = 0

    def __str__(self):
        return "Logistic | Layer no. " + str(self.layer_num) + '\n'

    def forward(self, x, train=False):
        out = 1.0 / (1 + np.exp(-1 * x))
        if train:
            self.gradient = out * (1 - out)
        return out

    def get_delta(self):
        return self.gradient

    def dW(self, delta):
        pass


class Tanh(object):

    def __init__(self, input_dim=None):
        self.need_initialize = False
        self.resize = False
        self.layer_num = 0

    def __str__(self):
        return "Tanh  | Layer no. " + str(self.layer_num) + '\n'

    def forward(self, x, train=False):
        out = np.tanh(x)
        if train:
            self.gradient = 1 - out**2
        return out

    def dW(self, delta):
        pass

    def get_delta(self):
        return self.gradient


class ReLU(object):

    def __init__(self, leak=0, input_dim=None):
        self.need_initialize = False
        self.leak = leak
        self.resize = False
        self.layer_num = 0

    def __str__(self):
        return "Relu | Layer no. " + str(self.layer_num) + '\n'

    def forward(self, x, train=False):
        if train:
            pass
        self.flag = x > self.leak
        return x * self.flag + self.leak * (1 - self.flag)

    def dW(self):
        pass

    def get_delta(self):
        return np.asarray(self.flag)


class LogSoftmax(object):

    def __init__(self, input_dim, n_classes):
        self.input_dim = input_dim
        self.output_dim = n_classes
        self.initialized = False
        self.need_initialize = True
        self.resize = True
        self.parameters = []
        self.layer_num = 0

    def __repr__(self):
        return "LogSoftmax | Layer no. " + str(self.layer_num) +\
            " with (in_dim, out_dim) = (  " + str(self.input_dim) +\
            ", " + str(self.output_dim) + ")" + os.linesep

    def initialize(self):
        self.initialized = True
        W = np.random.normal(size=(self.input_dim, self.output_dim))
        b = np.zeros((self.output_dim,))
        self.parameters.extend([W, b])

    def forward(self, x, train=False):
        assert(self.initialized)
        [W, b] = self.parameters
        activations = np.dot(x, W) + b
        max_act = np.max(activations)
        exponentiated = np.exp(activations - max_act)
        partition = np.sum(exponentiated)
        prob = activations - np.log(partition) - max_act
        return prob

    def update_parameters(self, update, reg=10e-6):
        for i in len(update):
            self.parameters[i] -= update[i] + reg * self.parameters[i]


if __name__ == '__main__':
    LinearLayer = Linear(2, 5)
    LinearLayer.initialize()
    input_vec = np.asarray([1, 2])
    layer1_out = LinearLayer.forward(input_vec)
    SigmLayer = Logistic()
    layer2_out = SigmLayer.forward(layer1_out)
    TanhLayer = Tanh()
    layer3_out = TanhLayer.forward(layer1_out)
    ReLULayer = ReLU()
    layer4_out = ReLULayer.forward(layer1_out)
    LogSoftmax = LogSoftmax(input_dim=5, n_classes=10)
    LogSoftmax.initialize()
    LogProbs = LogSoftmax.forward(layer4_out)
    print layer2_out, layer1_out, layer3_out, layer4_out
    print LogProbs
