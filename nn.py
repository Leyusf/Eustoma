import numpy as np
import eustoma.functions_conv as Conv
import eustoma.functions as F
from eustoma.core import Parameter
from eustoma.layers import Layer, RNNBlock, GRUBlock, LSTMBlock, GeneralRNNTemplate
from eustoma import utils, cuda
from eustoma.utils import pair


class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, item):
        return self.layers[item]

    def __len__(self):
        return len(self.layers)


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


class MLP(Model):

    def __init__(self, input_sizes, activation=F.sigmoid):
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = []
        for i, out_size in enumerate(input_sizes):
            layer = Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class RNN(GeneralRNNTemplate):
    """
    RNN模型接受的输入格式是(batch, seq length, data)
    return: (all_states, last_state)
    """

    def __init__(self, hidden_size, num_layers=1, bidirectional=False, in_size=None):
        super(RNN, self).__init__(RNNBlock, hidden_size, num_layers, bidirectional, in_size)


class GRU(GeneralRNNTemplate):
    """
    GRU模型接受的输入格式是(batch, seq length, data)
    return: (all_states, last_state)
    """

    def __init__(self, hidden_size, num_layers=1, bidirectional=False, in_size=None):
        super(GRU, self).__init__(GRUBlock, hidden_size, num_layers, bidirectional, in_size)


class LSTM(GeneralRNNTemplate):
    """
    GRU模型接受的输入格式是(batch, seq length, data)
    return: (all_states, last_state)
    """

    def __init__(self, hidden_size, num_layers=1, bidirectional=False, in_size=None):
        super(LSTM, self).__init__(LSTMBlock, hidden_size, num_layers, bidirectional, in_size)


### 固定的模型结构
class VGGBlock:
    def __init__(self, num_conv, out_channels):
        """
        :param num_conv: 卷积层数量
        :param out_channels: 输出通道数
        """
        layers = []
        for _ in range(num_conv):
            layers.append(Conv2d(out_channels, kernel_size=3, stride=1, pad=1))
            layers.append(ReLU())
        layers.append(Pooling(kernel_size=2, stride=2))
        self.block = Sequential(*layers)


class VGG16(Model):
    def __init__(self):
        super(VGG16, self).__init__()
        output_channels = [16, 32, 64, 128]
        blocks = []
        for output_channel in output_channels:
            blocks.append(VGGBlock(2, output_channel).block)
        self.blocks = Sequential(*blocks)
        self.mlp = Sequential(
            Flatten(),
            Linear(128 * 5),
            ReLU(),
            Dropout(),
            Linear(128 * 5),
            ReLU(),
            Dropout(),
            Linear(10)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.mlp(x)
        return x
