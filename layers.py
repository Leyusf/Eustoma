import os.path

import numpy as np
import weakref

from eustoma import cuda
from eustoma.core import Parameter
import eustoma.functions as F
import eustoma.functions_conv as Conv
from eustoma.utils import pair


class Layer:
    def __init__(self):
        self._params = set()
        self.device = 'cpu'

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def to(self, device):
        if device == 'cpu' or device == 'cuda':
            for param in self.params():
                param.to(device)
            self.device = device
        else:
            raise TypeError('{} if not supported'.format(device))

    def _flatten_params(self, params_dict, parent_keys=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_keys + "/" + name if parent_keys else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        old_device = self.device
        self.to('cpu')
        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
        if old_device != self.device:
            self.to(old_device)

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
        if self.device != 'cpu':
            self.to(self.device)


class Sigmoid(Layer):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):
        return F.sigmoid(x)


class ReLU(Layer):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        return F.relu(x)


class Linear(Layer):
    """
    out_size: 输出的尺度
    nobias: 有无偏置量
    dtype: 数据类型
    in_size: 输入尺度（默认不指定，则在forward过程中自动生成合适大小的parameter）
    """

    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype
        self.W = Parameter(None, name="Weight")

        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='Bias')

    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I, O).astype(self.dtype) * np.sqrt((1 / I))
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name='Weight')
        if in_channels is not None:
            self._init_W()
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='Bias')

    def _init_W(self, xp=np):
        IC, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (IC * KH * KW))
        W_data = xp.random.randn(OC, IC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)
        y = Conv.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Pooling(Layer):
    def __init__(self, kernel_size, stride=1, pad=0):
        super(Pooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return Conv.pooling(x, self.kernel_size, self.stride, self.pad)


class AveragePooling(Layer):
    def __init__(self, kernel_size, stride=1, pad=0):
        super(AveragePooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        return Conv.average_pooling(x, self.kernel_size, self.stride, self.pad)


class Dropout(Layer):
    def __init__(self, dropout=0.5):
        super(Dropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        return F.dropout(x, self.dropout)


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return F.flatten(x)

