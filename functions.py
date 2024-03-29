import cupyx
import numpy as np
from eustoma import utils, cuda
from eustoma.config import Config
from eustoma.core import as_variable, Function, Variable


def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * sphere(x, y) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * (
            30 + (2 * x - 3 * y) ** 2 *
            (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return z


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return reshape(gy, self.x_shape)


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return gy.transpose(gy, inv_axes)


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx


class MatMul(Function):
    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW


class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0 - x1
        y = (diff ** 2).sum() / len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2. / len(diff))
        gx1 = -gx0
        return gx0, gx1


class Square(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.square(x)

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        return xp.exp(x)

    def backward(self, gy):
        y = self.outputs[0]()  # weakref
        gx = y * gy
        return gx


class Log(Function):
    def __init__(self, base=None):
        self.base = base

    def forward(self, x):
        xp = cuda.get_array_module(x)
        if self.base is None:
            return xp.log(x)
        return xp.log(x) / xp.log(self.base)

    def backward(self, gy):
        x, = self.inputs
        if self.base is not None:
            gx = 1 / (x * log(self.base)) * gy
        else:
            gx = 1 / x * gy
        return gx


class Sin(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


class Cos(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -1 * sin(x)
        return gx


class Tanh(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW, gb


class GetItem(Function):
    def __init__(self, slices):
        self.slices = slices

    def forward(self, x):
        y = x[self.slices]
        return y

    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy):
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape, dtype=gy.dtype)

        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            cupyx.scatter_add(gx, self.slices, gy)
            # 新版本cupy没有scatter_add
            # xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.slices)


class SoftMax(Function):
    def __init__(self, axis=1):
        self.axis = axis

    def forward(self, x):
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = y * gy
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


class SoftmaxCrossEntropy(Function):
    def forward(self, pred, labels):
        N = pred.shape[0]
        log_z = utils.logsumexp(pred, axis=1)
        log_p = pred - log_z
        xp = cuda.get_array_module(labels.data)
        log_p = log_p[xp.arange(N), labels.ravel()]
        y = -log_p.sum() / xp.float32(N)
        return y

    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape
        gy *= 1 / N
        y = softmax(x)
        xp = cuda.get_array_module(t.data)
        t_one_hot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_one_hot) * gy
        return y


class Sigmoid(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


class ReLU(Function):
    def forward(self, x):
        xp = cuda.get_array_module(x.data)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


class Concat(Function):
    def forward(self, *x):
        xp = cuda.get_array_module(x[0])
        y = xp.concatenate(x)
        return y

    def backward(self, gys):
        gxs = []
        i = 0
        for x in self.inputs:
            gxs.append(gys[i:i + len(x)])
            i += len(x)
        gxs = tuple(gxs)
        return gxs


class Stack(Function):
    def forward(self, *x):
        xp = cuda.get_array_module(x[0])
        y = xp.stack(x)
        return y

    def backward(self, gys):
        gxs = []
        for gx in gys:
            gxs.append(gx)
        gxs = tuple(gxs)
        return gxs


def stack(x):
    return Stack()(*x)


def concat(x):
    return Concat()(*x)


def relu(x):
    return ReLU()(x)


def sigmoid(x):
    return Sigmoid()(x)


def softmax_cross_entropy(pred, labels):
    return SoftmaxCrossEntropy()(pred, labels)


def softmax(x, axis=1):
    return SoftMax(axis=axis)(x)


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


def sin(x):
    x = as_variable(x)
    return Sin()(x)


def cos(x):
    x = as_variable(x)
    return Cos()(x)


def tan(x):
    x = as_variable(x)
    return sin(x) / cos(x)


def tanh(x):
    x = as_variable(x)
    return Tanh()(x)


def square(x):
    f = Square()
    return f(x)


def sqrt(x):
    return pow(x, 0.5)


def exp(x):
    return Exp()(x)


def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)


def matmul(x, W):
    return MatMul()(x, W)


def log(x):
    return Log()(x)


def log2(x):
    return log(x) / log(Variable.repeat_like(2, x))


def log10(x):
    return log(x) / log(Variable.repeat_like(10, x))


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum(x, axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x, axes=None):
    if not isinstance(axes, list):
        axes = None
    return Transpose(axes)(x)


def linear(x, W, b=None):
    return Linear()(x, W, b)


def dropout(x, dropout_ratio=0.5):
    x = as_variable(x)
    if Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


def flatten(x):
    """Flattens the input. Does not affect the batch size."""
    return reshape(x, (x.shape[0], -1))
