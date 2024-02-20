import eustoma.functions as F
import eustoma.layers as L
from eustoma import utils


class Model(L.Layer):
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


class MLP(Model):

    def __init__(self, input_sizes, activation=F.sigmoid):
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = []
        for i, out_size in enumerate(input_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


class GeneralRNNTemplate(Model):
    """
    RNN模型模板接受的输入格式是(batch, seq length, data)
    return: (all_states, last_state)
    """

    def __init__(self, rnn_layer, hidden_size, num_layers, bidirectional, in_size):
        super(GeneralRNNTemplate, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.bidirectional = bidirectional
        self.in_size = in_size
        self.H = []
        self._H = []
        # 深度RNN
        rnn_layers = []
        for i in range(num_layers):
            rnn_layers.append(rnn_layer(hidden_size, in_size=in_size))

        self.net = Sequential(*rnn_layers)
        # 双向RNN
        if bidirectional:
            rnn_layers = []
            for i in range(num_layers):
                rnn_layers.append(rnn_layer(hidden_size, in_size=in_size))
            self.back_net = Sequential(*rnn_layers)

    def reset_state(self):
        self.H = []
        self._H = []
        for rnn in self.net:
            rnn.reset_state()
        if self.bidirectional:
            for rnn in self.back_net:
                rnn.reset_state()

    def forward(self, x):
        self.reset_state()
        x = F.transpose(x, [1, 0, 2])  # seq_len, batch, input_size
        for step_data in x:
            # 计算每个时间步的隐状态
            h = self.net(step_data)  # 每个隐状态的大小是 (batch_size, hidden_size)
            self.H.append(h)
        last_state = [self.H[-1]]
        self.H = F.stack(self.H)  # (seq_length, batch_size, hidden_size)
        self.H = F.transpose(self.H, [1, 0, 2])
        if self.bidirectional:
            for step_data in x[::-1]:
                h = self.back_net(step_data)
                self._H.append(h)
            last_state.append(self._H[-1])
            self._H = F.stack(self._H)
            self._H = F.transpose(self._H, [1, 0, 2])
        # 拼接成每个时间步上
        if self.bidirectional:
            self.H = F.transpose(self.H, [2, 1, 0])
            self._H = F.transpose(self._H, [2, 1, 0])
            self.H = F.concat([self.H, self._H])
            self.H = F.transpose(self.H, [2, 1, 0])
        last_sates = []
        for state in last_state:
            last_sates.append(F.transpose(state, [1, 0]))
        last_sates = F.concat(last_sates)
        last_sates = F.transpose(last_sates, [1, 0])
        return self.H, last_sates


class RNN(GeneralRNNTemplate):
    """
    RNN模型接受的输入格式是(batch, seq length, data)
    return: (all_states, last_state)
    """
    def __init__(self, hidden_size, num_layers=1, bidirectional=False, in_size=None):
        super(RNN, self).__init__(L._RNNLayer, hidden_size, num_layers, bidirectional, in_size)


class GRU(GeneralRNNTemplate):
    """
    RNN模型接受的输入格式是(batch, seq length, data)
    return: (all_states, last_state)
    """
    def __init__(self, hidden_size, num_layers=1, bidirectional=False, in_size=None):
        super(GRU, self).__init__(L._GRULayer, hidden_size, num_layers, bidirectional, in_size)


### 固定的模型结构
class VGGBlock:
    def __init__(self, num_conv, out_channels):
        """
        :param num_conv: 卷积层数量
        :param out_channels: 输出通道数
        """
        layers = []
        for _ in range(num_conv):
            layers.append(L.Conv2d(out_channels, kernel_size=3, stride=1, pad=1))
            layers.append(L.ReLU())
        layers.append(L.Pooling(kernel_size=2, stride=2))
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
            L.Flatten(),
            L.Linear(128 * 5),
            L.ReLU(),
            L.Dropout(),
            L.Linear(128 * 5),
            L.ReLU(),
            L.Dropout(),
            L.Linear(10)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.mlp(x)
        return x
