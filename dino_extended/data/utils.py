import os
import re

import numpy as np
import seaborn as sns
import scanpy as sc
import tifffile
import torch
import torchvision.transforms.functional as TF
import numpy as np
from einops import rearrange
from skimage.exposure import rescale_intensity
from tifffile import TiffFile
from ome_types import from_tiff, from_xml, to_xml, model


def listfiles(folder, regex=None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


def extract_ome_tiff(fp, channels=None):   
    tif = TiffFile(fp)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    d = {}
    img_channels = []
    for c, p in zip(im.pixels.channels, tif.pages):
        img_channels.append(c.name)

        if channels is None:
            img = p.asarray()
            d[c.name] = img
        elif c.name in channels:
            img = p.asarray()
            d[c.name] = img

    if channels is not None and len(set(channels).intersection(set(img_channels))) != len(channels):
        raise RuntimeError(f'Not all channels were found in ome tiff: {channels} | {img_channels}')

    return d


def get_ome_tiff_channels(fp):   
    tif = TiffFile(fp)
    ome = from_xml(tif.ome_metadata)
    im = ome.images[0]
    return [c.name for c in im.pixels.channels]


def make_pseudo(channel_to_img, cmap=None, contrast_pct=20.):
    cmap = sns.color_palette('tab10') if cmap is None else cmap

    new = np.zeros_like(next(iter(channel_to_img.values())))
    img_stack = []
    for i, (channel, img) in enumerate(channel_to_img.items()):
        color = cmap[i] if not isinstance(cmap, dict) else cmap[channel]
        new = img.copy().astype(np.float32)
        new -= new.min()
        new /= new.max()

        try:
            vmax = np.percentile(new[new>0], (contrast_pct)) if np.count_nonzero(new) else 1.
            new = rescale_intensity(new, in_range=(0., vmax))
        except IndexError:
            pass

        new = np.repeat(np.expand_dims(new, -1), 3, axis=-1)
        new *= color
        img_stack.append(new)
    stack = np.mean(np.asarray(img_stack), axis=0)
    stack -= stack.min()
    stack /= stack.max()
    return stack


def save_ome_tiff(channel_to_img, filepath):
    """
    Generate an ome tiff from channel to image map
    """
    n_channels = len(channel_to_img)
    logging.info(f'image has {n_channels} total biomarkers')

    with tifffile.TiffWriter(filepath, ome=True, bigtiff=True) as out_tif:
        biomarkers = []
        for i, (biomarker, img) in enumerate(channel_to_img.items()):
            x, y = img.shape[1], img.shape[0]
            biomarkers.append(biomarker)
            logging.info(f'writing {biomarker}')

            out_tif.write(img)
        o = model.OME()
        o.images.append(
            model.Image(
                id='Image:0',
                pixels=model.Pixels(
                    dimension_order='XYCZT',
                    size_c=n_channels,
                    size_t=1,
                    size_x=x,
                    size_y=y,
                    size_z=1,
                    type='float',
                    big_endian=False,
                    channels=[model.Channel(id=f'Channel:{i}', name=c) for i, c in enumerate(biomarkers)],
                )
            )
        )

        im = o.images[0]
        for i in range(len(im.pixels.channels)):
            im.pixels.planes.append(model.Plane(the_c=i, the_t=0, the_z=0))
        im.pixels.tiff_data_blocks.append(model.TiffData(plane_count=len(im.pixels.channels)))
        xml_str = to_xml(o)
        out_tif.overwrite_description(xml_str.encode())
