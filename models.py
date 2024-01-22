from eustoma import utils
import eustoma.functions as F
import eustoma.layers as L


class Model(L.Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):

    def __init__(self, input_sizes, activation=F.sigmoid):
        super(MLP, self).__init__()
        self.activation = activation
        self.layers = []
        for i, out_size in enumerate(input_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l'+str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
