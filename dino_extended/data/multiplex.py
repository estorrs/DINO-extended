import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop, RandomRotation, CenterCrop, Compose

class TileTransform(object):
    def __init__(self, size=(128, 128), degrees=180):
        self.size = size

        self.spatial_transform = Compose([
            RandomCrop((size[0] * 2, size[1] * 2), padding=size, padding_mode='reflect'),
            RandomRotation(degrees),
            CenterCrop(size)
        ])

    def __call__(self, he):
        """
        he - (n, H, W)
        """
        h = torch.randint(0, he.shape[-2] - self.size[-2] * 2, (1,)).item()
        w = torch.randint(0, he.shape[-1] - self.size[-1] * 2, (1,)).item()
        deg = torch.randint(0, 360, (1,)).item()

        crop = TF.crop(he, h, w, self.size[-2] * 2, self.size[-1] * 2)
        crop = TF.rotate(crop, deg)
        crop = TF.center_crop(crop, self.size)
        
        return crop