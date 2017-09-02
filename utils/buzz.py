import glob
import numpy as np

from tqdm import tqdm
from scipy.ndimage import zoom, imread
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image, remove_small_objects


def swap_format(fullpath):
    """ Swap path of images and masks.
    I.e., `./data/train/02159e548029_06.jpg` <==> `./data/train_mask/02159e548029_06_mask.gif`

    Arguments
        - fullname: either `path/to/XX.gif` or `path/to/XX.jpg`
    """
    name = fullpath[:-4]
    if fullpath.endswith('.jpg'):
        return name.replace('train', 'train_mask') + '_mask.gif'
    elif fullpath.endswith('.gif'):
        return name.replace('_mask', '') + '.jpg'
    raise Exception('Check format ends with \'.jpg\' or \'.gif\', received incorrect \'{}\'.'.format(fullpath))


def to_identity(fullpath):
    if not isinstance(fullpath, (list, tuple)):
        fullpath = [fullpath]
    base = '/'.join(fullpath[0].split('/')[:-1])
    name = np.unique([i.split('/')[-1].split('_')[0] for i in fullpath])
    return [base + '/' + i for i in name]


def read_identidy_bulk(images_id, zoom_factor=0.25):
    if not isinstance(images_id, (list, tuple)):
        images_id = [images_id]

    if 'train_mask' == images_id[0].split('/')[2]:
        ext = '{}_{:02d}_mask.gif'
    else:
        ext = '{}_{:02d}.jpg'

    for i in images_id:
        images = np.array([imread(ext.format(i, j + 1), mode='L') for j in range(16)])
    return zoom(images, (1, zoom_factor, zoom_factor))


def cars_shrink_to_bbox(*inputs):
    """Find the bounding box of a set of images.

    Arguments
        - inputs contains:
            images [16, height, width] (ndarray)
            labels [16, height, width] (ndarray)

    Return
        - images: [16, reduced_height, reduced_width]
        - bounding box: (min_x, min_y, max_x, max_y)
    """
    images = inputs[0]
    car_head = images[0]
    car_average = np.sum(images, axis=0) / 16
    im_roi = np.abs(car_head - car_average)
    bw_roi = im_roi > threshold_otsu(im_roi)
    remove_small_objects(bw_roi, min_size=3, in_place=True)
    chull = convex_hull_image(bw_roi)

    # 2-d slice: ()
    bb = regionprops(label(chull))[0].bbox
    # slice each input

    outputs = [item[:, bb[0]: bb[2], bb[1]: bb[3]] for item in inputs]
    return outputs + [bb]


def save_boundingbox(downscale=2):
    """Compute and save the bounding box dictionary
        {image_path: (min_x, min_y, max_x, max_y)}
    """
    boxes = {}
    for i in tqdm(to_identity(IMAGES_PATH)):
        images = read_identidy_bulk(i, zoom_factor=1 / downscale)
        _, bbox = cars_shrink_to_bbox(images)
        boxes[i] = bbox
    np.save('./data/bbox_downscale_x' + str(downscale), boxes)


IM_SIZE = np.array([1280, 1918])
IMAGES_PATH = sorted(glob.glob('./data/train/*.jpg'))
LABELS_PATH = [swap_format(i) for i in IMAGES_PATH]

if __name__ == '__main__':
    save_boundingbox()
