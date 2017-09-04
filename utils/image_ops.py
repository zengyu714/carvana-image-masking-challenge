import os

import cv2
import numpy as np
from scipy import misc
from skimage import measure
from skimage.exposure import adjust_gamma
from skimage.transform import warp, AffineTransform


def random_gamma(*inputs, low=0.9, high=1.1):
    """Adjust image intensity"""

    # `xs` has shape [height, width] with value in [0, 1].
    gamma = np.random.uniform(low=low, high=high)
    outputs = [adjust_gamma(item, gamma) for item in inputs]

    return outputs if len(inputs) > 1 else outputs[0]


def random_hsv(image,
               hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def random_affine(image, label,
                  scale_limit=(0.9, 1.1), trans_limit=(-0.0625, 0.0625), shear=None, rotation=None):
    scale = np.random.uniform(*scale_limit, size=2)
    translation = np.random.uniform(*np.multiply(image.shape[:-1], trans_limit), size=2)

    if shear is not None:
        shear = np.random.uniform(*np.deg2rad(shear))
    if rotation is not None:
        rotation = np.random.uniform(*np.deg2rad(rotation))

    tform = AffineTransform(scale=scale, rotation=rotation, shear=shear, translation=translation)
    return [warp(item, tform, mode='reflect', preserve_range=True) for item in [image, label]]


def random_hflip(image, label, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)

    return image, label


def shrink_to_bbox(*inputs):
    """
    Reduce volume/area to the size of bounding box

    Argument:
        Volumes:
        + modality_1, modality_2(optional), label

    Return:
        Reduced volumes
        + modality_1, modality_2(optional), label
        + bbox
    """
    ndim = inputs[0].ndim
    labeled = measure.label(inputs[0] > 0, connectivity=ndim)

    # bbox: (min_x, min_y, max_z, max_x, max_y, max_z)
    bb = measure.regionprops(labeled)[0].bbox

    # 3-d slice: (bb[0]: bb[3], bb[1]: bb[4], bb[2]: bb[5])
    # 2-d slice: (bb[0]: bb[2], bb[1]: bb[3])
    sl = [slice(bb[i], bb[i + ndim]) for i in range(ndim)]
    outputs = [item[sl] for item in inputs]

    return outputs + [bb]


def normalize(T):
    """
    Normalize volume's intensity to range [0, 1], for suing image processing
    Compute global maximum and minimum cause cerebrum is relatively homogeneous
    """
    _max = np.max(T)
    _min = np.min(T)
    # `T` with dtype float64
    return (T - _min) / (_max - _min)


def rename(root_dir='./data/base/'):
    """Strip names from 'BMPX.bmp' to 'X.bmp"""

    for dirname in os.listdir(root_dir):
        for filename in os.listdir(os.path.join(root_dir, dirname)):
            strip_name = filename.lstrip('BMP')
            oldname = os.path.join(root_dir, dirname, filename)
            newname = os.path.join(root_dir, dirname, strip_name)
            os.rename(oldname, newname)


def blend(image, label, alpha=0.5):
    """"Simulate colormap `jet`."""

    r = (label + 0.1) * 255 * alpha
    b = (image + image.mean()) * (1 - alpha)
    g = np.minimum(r, b)
    rgb = np.dstack([r, g, b] + image * 0.3)
    return rgb.astype(np.uint8)


def label_areas(data_dir, save_dir='./submit/ALL/', alpha=0.5):
    """Compute area of labels in `jet` colormap blend image and save.

    Arguments:
        data_dir: contains blend images
        save_dir: indicate directory to save results of computed areas
        alpha: parameter used to overlap labels and images
        
    Assume:
        label is in `r` color channel, i.e. label = rgb[0]
        
    Return:
        array of pairs of (image_idx, label_area) with shape [2, image_nums] 
    """
    get_idx = lambda name: int(name.split('_')[-1][: -4])
    images_path = sorted([os.path.join(data_dir, p) for p in os.listdir(data_dir)],
                         key=get_idx)
    # Filter the empty string in a list
    prefix = list(filter(None, data_dir.split('/')))[-1]
    idx, areas = [], []
    for p in images_path:
        # Append index
        idx.append(get_idx(p))
        # Append areas
        label = misc.imread(p)[..., 0]
        bw = label > (255 * alpha)
        areas.append(np.sum(bw))

    pairs = np.array([idx, areas])
    np.save(save_dir + prefix, pairs)
    return pairs
