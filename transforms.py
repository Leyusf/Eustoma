from typing import Any
import numpy as np
from numpy import float32
try:
    import Image
except ImportError:
    from PIL import Image
from eustoma.utils import pair

class Compose:
    '''
    一系列的transforms
    '''
    def __init__(self, transforms=[]):
        self.transforms = transforms
    
    def __call__(self, img):
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img

class Convert:
    '''
    图片颜色模式转换
    '''
    def __init__(self, mode="RGB"):
        self.mode = mode
    
    def __call__(self, img):
        if self.mode == 'BGR':
            img = img.convert('RGB')
            r, g, b = img.split()
            img = Image.merge('RGB', (b, g, r))
            return img
        return img.convert(self.mode)


class Resize:
    '''
    改变图片大小
    '''
    def __init__(self, size, mode=Image.BILINEAR):
        self.size = pair(size)
        size.mode = mode
    
    def __call__(self, img):
        return img.resize(self.size, self.mode)


class CenterCrop:
    '''
    中心裁剪
    '''
    def __init__(self, size):
        self.size = pair(size)
    
    def __call__(self, img):
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - (left + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - (up + (H - OH) % 2)
        return img.crop((left, up, right, bottom))
        

class ToArray:
    '''
    转换PIL图片转换成numpy array
    '''
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
    
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL:
    def __call__(self, array):
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass


class Normalize:
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
    
    def __call__(self, array):
        mean, std = self.mean, self.std
        
        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    '''
    展平数组
    '''
    def __call__(self, array):
        return array.flatten()


class AsType:
    def __init__(self, dtype=np.float32):
        self.dtype = dtype
    
    def __call__(self, array):
        return array.astype(self.dtype)

ToFloat = AsType


class ToInt(AsType):
    def __init__(self, dtype=np.int_):
        self.dtype = dtype
    