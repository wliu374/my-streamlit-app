import numpy as np
import PIL.Image as Image
import torch
import cv2

class Resize(object):
    def __init__(self, ratio):
        super(Resize, self).__init__()
        self.ratio = ratio

    def __call__(self, img):
        h, w = img.shape[:2]
        nh = int(np.ceil(h * self.ratio))
        nw = int(np.ceil(w * self.ratio))
        img = cv2.resize(img, (nw, nh),interpolation=cv2.INTER_CUBIC)
        return img

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.scale = np.float32(1 / 255)
        self.mean = np.float32(np.array(mean)).reshape((1,1,3)) if mean is not None else None
        self.std = np.float32(np.array(std)).reshape((1,1,3)) if std is not None else None

    def __call__(self, img):
        if self.mean is None or self.std is None:
            self.calculate_mean_std(img)

        # pixel normalization
        img = (self.scale * img - self.mean) / self.std

        return img

    def calculate_mean_std(self, img):
        self.mean = np.float32(np.array([0, 0, 0]).reshape((1, 1, 3)))
        self.std = np.float32(np.array([0, 0, 0]).reshape((1, 1, 3)))

        if img.shape[2] == 3:
            x = img.transpose((2, 0, 1))
        x = x.reshape(3,-1)
        self.mean += x.mean(1).reshape(1,1,3)
        self.std += x.std(1).reshape(1,1,3)

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        pass

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        return img

if __name__ == '__main__':
    image = np.array(Image.open("fragm.JPG"))
