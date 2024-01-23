import numpy as np

class Dataset:
    def __init__(self, train=True, transform=None, label_transform=None):
        self.train = train
        self.transform = transform
        self.label_transform = label_transform
        # 默认不ransform
        if self.transform is None:
            self.transform = lambda x: x
        if self.label_transform is None:
            self.label_transform = lambda x: x
        
        self.data = None
        self.label = None
        self.prepare()
    
    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        return self.transform(self.data[index]), self.label_transform(self.label[index])
    
    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


class LargeDataset(Dataset):
    def __getitem__(self, index):
        data = np.load('data/{}.npy'.format(index))
        label = np.load('label/{}.npy'.format(index))
        return data, label

    def __len__(self):
        return 1000000

def get_spiral(train=True):
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int64)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)