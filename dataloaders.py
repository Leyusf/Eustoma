import math
from eustoma import cuda
import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)
        self.to(device)
        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size: (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        xp = cuda.cupy if self.device == 'cuda' else np
        data = xp.array([sample[0] for sample in batch])
        labels = xp.array([sample[1] for sample in batch])

        self.iteration += 1
        return data, labels

    def next(self):
        return self.__next__()

    def to(self, device):
        if device == 'cpu' or device == 'cuda':
            self.device = device
        else:
            raise TypeError('{} if not supported'.format(device))


class SeqDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, device='cpu'):
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False,
                         device=device)

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [(i * jump + self.iteration) % self.data_size for i in
                       range(self.batch_size)]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.device == 'cuda' else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t