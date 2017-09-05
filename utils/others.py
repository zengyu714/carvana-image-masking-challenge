import os
import numpy as np
from scipy.ndimage import imread
from skimage.transform import rescale


def png_to_png(mask_dir='./data/train_mask'):
    old_dir = os.getcwd()
    os.chdir(mask_dir)

    for i in os.listdir('.'):
        os.rename(i, i.replace('gif', 'png'))

    os.chdir(old_dir)


def swap_format(name):
    """ Swap path of images and masks.
    I.e., `02159e548029_06.jpg` <==> `02159e548029_06_mask.png`

    Arguments
        - fullname: either `path/to/XX_mask.png` or `path/to/XX.jpg`
    """
    if name.endswith('.jpg'):
        return name[:-4] + '_mask.png'
    elif name.endswith('.png'):
        return name[:-9] + '.jpg'
    raise Exception('Check format ends with \'.jpg\' or \'.png\', received incorrect \'{}\'.'.format(name))


def to_identity(names):
    if not isinstance(names, (list, tuple)):
        names = [names]
    mix = '_'.join(names).split('_')  # E.g., '00087a6bd4dc', '01.jpg', '00087a6bd4dc', '02.jpg'
    ids = np.unique([i for i in mix if len(i) == 12])
    if len(ids) == 1: return ids[0]
    return ids


def read_identidy_car(images_id, data_dir='./data/train/', scale=0.5):
    if not isinstance(images_id, (list, tuple)):
        images_id = [images_id]

    images = []
    if 'mask' in data_dir:
        ext = '{}_{:02d}_mask.png'
    else:
        ext = '{}_{:02d}.jpg'

    for i in images_id:
        for j in range(16):
            im = imread(os.path.join(data_dir, ext.format(i, j + 1)), mode='L')
            im = rescale(im, scale, mode='wrap', preserve_range=True)
            images += [im]

    return np.array(images)
