import heapq
import weakref

import numpy as np

import eustoma
from eustoma.config import Config, using_config


def as_array(x):
    if np.isscalar(x):  # 如果是x是numpy.float或者numpy.int则转变成numpy.ndarray
        return np.array(x)
    return x


def as_variable(x):
    if isinstance(x, Variable):
        return x
    if isinstance(x, float) or isinstance(x, int):
        x = np.array(x)
    return Variable(x)


class Variable:
    __array_priority__ = 200  # 设定计算优先级

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} if not supported'.format(type(data)))

        self.name = name
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            self.grad = Variable.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(var):
            if var not in seen_set:
                heapq.heappush(funcs, (-var.generation, var.creator))
                seen_set.add(var)  # 避免重复计算
                # funcs.append(f)
                # seen_set.add(f)
                # funcs.sort(key=lambda x: x.generation)

        add_func(self)

        while funcs:
            f = heapq.heappop(funcs)[-1]
            gys = [output().grad for output in f.outputs]
            # gys = [output.grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is not None:
                        x.grad += gx  # 避免重复计算时重置导数
                    else:
                        x.grad = gx
                    if x.creator:
                        add_func(x)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'

    def __copy__(self):
        data = self.data.copy()
        return Variable(data)

    @staticmethod
    def ones_like(other):
        data = np.ones_like(other.data)
        return Variable(data)

    @staticmethod
    def repeat_like(x, others):
        new_var = Variable.ones_like(others) * x
        return new_var

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return eustoma.functions.reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return eustoma.functions.transpose(self, axes)

    @property
    def T(self):
        return eustoma.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return eustoma.functions.sum(self, axis, keepdims)


class Function:
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # 在forward中定义具体的计算步骤
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)

            self.inputs = inputs  # 保存输入的变量
            self.outputs = [weakref.ref(output) for output in outputs]
            # self.outputs = [output for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError

    def __eq__(self, other):
        return True

    def __hash__(self):
        return hash(str(id(self)))


class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = eustoma.functions.sum_to(gx0, self.x0_shape)
            gx1 = eustoma.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:
            gx0 = eustoma.functions.sum_to(gx0, x0.shape)
            gx1 = eustoma.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape != self.x1_shape:
            gx0 = eustoma.functions.sum_to(gx0, self.x0_shape)
            gx1 = eustoma.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape != x1.shape:
            gx0 = eustoma.functions.sum_to(gx0, x0.shape)
            gx1 = eustoma.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def add(x0, x1):
    x1 = as_variable(x1)
    return Add()(x0, x1)


def mul(x0, x1):
    x1 = as_variable(x1)
    return Mul()(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_variable(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_variable(x1)
    return Sub()(x1, x0)


def div(x0, x1):
    x1 = as_variable(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_variable(x1)
    return Div()(x1, x0)


def pow(x, c):
    if isinstance(c, Variable):
        c = c.data
    return Pow(c)(x)


def rpow(x, c):
    if isinstance(c, Variable):
        c = c.data
    # x = Variable.repeat_like(x, c)
    return Pow(x)(c)


class Parameter(Variable):
    pass


def setup_variable():
    Variable.__getitem__ = eustoma.functions.get_item
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__rpow__ = rpow
