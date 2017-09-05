import os
import numpy as np

from tqdm import tqdm
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image, remove_small_objects

from others import read_identidy_car, to_identity


# TODO: could be improved by processing 2 images a time
def car_boxes(images):
    """Find the bounding box of a set of images.
    Difference between opposite direction --> threshold --> convex hull --> boundingbox.

    Arguments
        images:  [16, height, width] (ndarray)

    Returns
        bounding_box: np.ndarray with shape [16, 4]
                      each row contains [min_x, min_y, max_x, max_y]
    """
    num = images.shape[0]
    assert num == 16, 'Should have 16 images, while received {} images'.format(num)

    boxes = np.zeros((8, 4))  # save results

    for i in range(8):  # E.g., images[0] <==> images[8]
        pair_1, pair_2 = images[i], images[i + 8]
        diff = np.abs(pair_1 - pair_2)
        bw_diff = diff > threshold_otsu(diff)
        remove_small_objects(bw_diff, min_size=3, in_place=True)
        chull = convex_hull_image(bw_diff)

        boxes[i] = regionprops(label(chull))[0].bbox

    return np.vstack([boxes, boxes]).astype(int)  # [16, 4]


def save_boundingbox(mode='train'):
    """Compute and save the bounding box dictionary
        {image_path: (min_x, min_y, max_x, max_y)}
    """
    saved_boxes = {}
    scale = 0.2
    ids = to_identity(sorted(os.listdir('../data/' + mode)))
    for i in tqdm(ids):
        images = read_identidy_car(i, data_dir='../data/' + mode + '/', scale=scale)
        # 2-d slice: (bb[0]: bb[2], bb[1]: bb[3])
        boxes = car_boxes(images)
        for j in range(16):
            name = i + '_' + str(j).zfill(2) + '.jpg'
            saved_boxes[name] = boxes[j] / scale
    np.save('../data/morph_' + mode + '_boundingbox', saved_boxes)


if __name__ == '__main__':
    # save_boundingbox(mode='train')
    save_boundingbox(mode='test')
