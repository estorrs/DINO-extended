import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop, RandomRotation, CenterCrop, Compose
from torch.utils.data import DataLoader, Dataset

class TileTransform(object):
    def __init__(self, size=(128, 128), deg=360):
        self.size = size
        self.deg = deg

    def __call__(self, x):
        """
        he - (n, H, W)
        """
        h = torch.randint(0, x.shape[-2] - self.size[-2] * 2, (1,)).item()
        w = torch.randint(0, x.shape[-1] - self.size[-1] * 2, (1,)).item()
        deg = torch.randint(0, self.deg, (1,)).item()

        crop = TF.crop(x, h, w, self.size[-2] * 2, self.size[-1] * 2)
        crop = TF.rotate(crop, deg)
        crop = TF.center_crop(crop, self.size)

        return crop


class TileDataset(Dataset):
    """Registration Dataset"""
    def __init__(self, imgs, size=(256, 256), scale=None, transform=None, length=2**13):
        self.length = length

        if scale is not None:
            imgs = [TF.resize(img, (int(img.shape[-2] * scale), int(img.shape[-1] * scale)), antialias=True)
                    for img in imgs]
        self.imgs = imgs

        self.transform = transform if transform is not None else TileTransform(size=size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = (idx + 1) % len(self.imgs)
        return self.transform(self.imgs[i])
    

class MultichannelAug(object):
    def __init__(self, means, stds,
                brightness=.8, contrast=.8, saturation=.8, hue=.2):
        self.means = means
        self.stds = stds
        
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, x):
        if np.random.rand() < .3:
            brightness = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            contrast = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            saturation = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            hue = np.random.uniform(-self.hue, self.hue)
            for i in range(x.shape[-3]):
                x[..., i:i+1, :, :] = TF.adjust_brightness(x[..., i:i+1, :, :], brightness)
                x[..., i:i+1, :, :] = TF.adjust_contrast(x[..., i:i+1, :, :], contrast)
                x[..., i:i+1, :, :] = TF.adjust_saturation(x[..., i:i+1, :, :], saturation)
                x[..., i:i+1, :, :] = TF.adjust_hue(x[..., i:i+1, :, :], hue)

        if np.random.rand() < .5:
            x = TF.vflip(x)
        if np.random.rand() < .5:
            x = TF.hflip(x)

        if np.random.rand() < .2:
            TF.gaussian_blur(x, (3, 3), (1., 2.))

        x = TF.normalize(x, self.means, self.stds)

        return x