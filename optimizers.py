import numpy as np

from eustoma import cuda


class Optimizer:
    def __init__(self, target=None):
        self.target = target
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self

    def update(self):
        params = [p for p in self.target.params() if p.grad is not None]
        # 预处理
        for f in self.hooks:
            f(params)

        # 更新参数
        for p in params:
            self.update_one(p)

    def update_one(self, param):
        raise NotImplementedError

    def add_hook(self, f):
        self.hooks.append(f)


class SGD(Optimizer):
    def __init__(self, target, lr=0.01, momentum=0.9):
        super(SGD, self).__init__(target)
        self.lr = lr
        self.momentua = momentum
        self.vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentua
        v -= self.lr * param.grad.data
        param.data += v
