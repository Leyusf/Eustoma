from eustoma import utils
import eustoma.functions as F
import eustoma.layers as L


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
            L.Linear(128*5),
            L.ReLU(),
            L.Dropout(),
            L.Linear(128*5),
            L.ReLU(),
            L.Dropout(),
            L.Linear(10)
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.mlp(x)
        return x

