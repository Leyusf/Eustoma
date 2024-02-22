import os.path
import weakref

import numpy as np

import eustoma
import eustoma.functions as F
from eustoma.core import Parameter


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


class GeneralRNNTemplate(Layer):
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

        self.net = eustoma.nn.Sequential(*rnn_layers)
        # 双向RNN
        if bidirectional:
            rnn_layers = []
            for i in range(num_layers):
                rnn_layers.append(rnn_layer(hidden_size, in_size=in_size))
            self.back_net = eustoma.nn.Sequential(*rnn_layers)

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
        last_H = [self.H[-1]]
        states = F.stack(self.H)  # (seq_length, batch_size, hidden_size)
        states = F.transpose(states, [1, 0, 2])  # b s h
        if self.bidirectional:
            for step_data in x[::-1]:
                h = self.back_net(step_data)
                self._H.append(h)
            last_H.append(self._H[-1])
            _states = F.stack(self._H)  # s b h
            # 拼接成每个时间步上
            states = F.transpose(states, [2, 1, 0])  # h s b
            _states = F.transpose(_states, [2, 0, 1])  # h s b
            states = F.concat([states, _states])  # h s b
            states = F.transpose(states, [2, 1, 0])  # b s h

        last_state = []
        for state in last_H:
            last_state.append(F.transpose(state, [1, 0]))  # h b
        last_state = F.concat(last_state)  # h b
        last_state = F.transpose(last_state, [1, 0])  # b h
        return states, last_state  # b s h, b h


class RNNBlock(Layer):
    def __init__(self, hidden_size, in_size=None):
        super(RNNBlock, self).__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.H = None

        # X->H
        self.x2h = eustoma.nn.Linear(hidden_size, in_size=in_size)
        # H -> H
        self.h2h = eustoma.nn.Linear(hidden_size, in_size=in_size)

    def reset_state(self):
        self.H = None

    # 无优化的RNN
    def forward(self, x):
        if self.H is None:
            self.H = []
            new_H = F.tanh(self.x2h(x))
        else:
            new_H = F.tanh(self.x2h(x) + self.h2h(self.H[-1]))
        self.H.append(new_H)
        return new_H


class GRUBlock(Layer):
    def __init__(self, hidden_size, in_size=None):
        super(GRUBlock, self).__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.H = None

        self.x2r = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.h2r = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.x2z = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.h2z = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.x2h = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.h2h = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)

        self.br = Parameter(np.zeros(hidden_size, dtype=np.float32), name='Bias')
        self.bz = Parameter(np.zeros(hidden_size, dtype=np.float32), name='Bias')
        self.bh = Parameter(np.zeros(hidden_size, dtype=np.float32), name='Bias')

    def reset_state(self):
        self.H = None

    def forward(self, x):
        if self.H is None:
            self.H = []
            Z = F.sigmoid(self.x2z(x) + self.bz)
            H_ = F.tanh(self.x2h(x) + self.bh)
            H = (1 - Z) * H_
        else:
            H = self.H[-1]
            R = F.sigmoid(self.x2r(x) + self.h2r(H) + self.br)
            Z = F.sigmoid(self.x2z(x) + self.h2z(H) + self.bz)
            H_ = F.tanh(self.x2h(x) + self.h2h(H * R) + self.bh)
            H = Z * self.H[-1] + (1 - Z) * H_
        self.H.append(H)
        return H


class LSTMBlock(Layer):
    def __init__(self, hidden_size, in_size=None):
        super(LSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.H = None
        self.C = 0

        self.x2i = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.h2i = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.x2f = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.h2f = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.x2o = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.h2o = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)
        self.x2c = eustoma.nn.Linear(hidden_size, in_size=in_size, nobias=True)

        self.bi = Parameter(np.zeros(hidden_size, dtype=np.float32), name='Bias')
        self.bf = Parameter(np.zeros(hidden_size, dtype=np.float32), name='Bias')
        self.bo = Parameter(np.zeros(hidden_size, dtype=np.float32), name='Bias')
        self.bc = Parameter(np.zeros(hidden_size, dtype=np.float32), name='Bias')

    def reset_state(self):
        self.H = None
        self.C = 0

    # 无优化的RNN
    def forward(self, x):
        if self.H is None:
            self.H = []
            IG = F.sigmoid(self.x2i(x) + self.bi)
            FG = F.sigmoid(self.x2f(x) + self.bf)
            OG = F.sigmoid(self.x2o(x) + self.bo)
        else:
            H = self.H[-1]
            IG = F.sigmoid(self.x2i(x) + self.h2i(H) + self.bi)
            FG = F.sigmoid(self.x2f(x) + self.h2f(H) + self.bf)
            OG = F.sigmoid(self.x2o(x) + self.h2o(H) + self.bo)
        hat_C = F.tanh(self.x2c(x) + self.bc)
        self.C = FG * self.C + IG * hat_C
        H = OG * F.tanh(self.C)
        self.H.append(H)

        return H