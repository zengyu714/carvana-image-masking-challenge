import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.misc import imread

from utils.buzz import LABELS_PATH

# file
TRAIN_MASKS_CSV = './data/train_masks.csv'
TRAIN_MASKS_DEMO = './data/train_masks/0cdf5b5d0ce1_01_mask.gif'


def rle_encode(mask):
    pos = np.where(mask == 1)[0]
    sep = np.split(pos, np.where(np.diff(pos) != 1)[0] + 1)
    res = [f(s) for s in sep for f in (lambda s: s[0] + 1, lambda s: len(s))]
    return ' '.join(str(r) for r in res)


def validate_rle(mask_csv, mask_path=TRAIN_MASKS_DEMO):
    """ Validate run-length encoding.
    Arguments
        - maskfile: names like '0ce66b539f52', '0cdf5b5d0ce1'...
    """
    index = mask_path.split('/')[-1].replace('_mask.gif', '.jpg')

    true = mask_csv[mask_csv['img'] == index]['rle_mask']
    mask = imread(mask_path, mode='L').flatten() > 127
    pred = rle_encode(mask)
    assert (true == pred).bool(), 'Run-length encoding fails...'


def validate_rle_timeit():
    mask_csv = pd.read_csv(TRAIN_MASKS_CSV)

    for f in tqdm(LABELS_PATH):
        validate_rle(mask_csv, f)


if __name__ == '__main__':
    validate_rle_timeit()
