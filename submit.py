import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.misc import imread
from skimage.transform import resize

import visdom
import torch
from torch.autograd import Variable

from utils.others import read_infer_car
from utils.checkpoints import best_checkpoints
from nets.u_net import UNet_1024, UNet_512, UNet_256, UNet_128

# file
TRAIN_MASKS_CSV = './data/train_masks.csv'
TRAIN_MASKS_DIR = './data/train_masks/'
TRAIN_MASKS_PATH = [os.path.join(TRAIN_MASKS_DIR, p) for p in os.listdir(TRAIN_MASKS_DIR)]
TRAIN_MASKS_DEMO = TRAIN_MASKS_PATH[0]

MODEL = 'UNet_128'
INPUT_SIZE = int(MODEL.split('_')[-1])
IMAGE_SIZE = (1280, 1918)

model = eval(MODEL)()

torch.cuda.set_device(0)
model = model.cuda().eval()


def rle_encode(mask):
    pos = np.where(mask == 1)[0]
    sep = np.split(pos, np.where(np.diff(pos) != 1)[0] + 1)
    res = [f(s) for s in sep for f in (lambda s: s[0] + 1, lambda s: len(s))]
    return ' '.join(str(r) for r in res)


def submit():
    subdir = './{}/' + MODEL + '/'
    results_dir, checkpoint_dir = [subdir.format(p) for p in ['results', 'checkpoints']]
    best_path = best_checkpoints(results_dir, checkpoint_dir, keys=['val_dice_overlap'])
    print('===> Loading model from {}...'.format(best_path))

    best_model = torch.load(best_path)
    model.load_state_dict(best_model)

    ims_path = sorted(os.listdir('./data/test/'))
    batch_size = 2
    div_path = [ims_path[x: x + batch_size] for x in range(0, len(ims_path), batch_size)]

    rles = []
    names = []
    for i in tqdm(div_path):
        # Read data
        images = read_infer_car(list(i), size=INPUT_SIZE)  # convert ndarray to list
        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        images = Variable(images, volatile=True).float().cuda()

        # Infer
        output = model(images)

        pred = output.data.max(1)[1]
        pred = pred.cpu().numpy()

        for j, mask in enumerate(pred):
            mask = (resize(mask, IMAGE_SIZE, preserve_range=True) > 0.5).astype(int)
            rles += [rle_encode(mask)]
            names += [i[j]]

    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submit/submission.csv.gz', index=False, compression='gzip')


def validate_rle(mask_csv, mask_path=TRAIN_MASKS_DEMO):
    """ Validate run-length encoding.
    Arguments
        - maskfile: names like '0ce66b539f52', '0cdf5b5d0ce1'...
    """
    index = mask_path.split('/')[-1].replace('_mask.png', '.jpg')

    true = mask_csv[mask_csv['img'] == index]['rle_mask']
    mask = imread(mask_path, mode='L').flatten() > 127
    pred = rle_encode(mask)
    assert (true == pred).bool(), 'Run-length encoding fails...'


def validate_rle_timeit():
    mask_csv = pd.read_csv(TRAIN_MASKS_CSV)

    for f in tqdm(TRAIN_MASKS_PATH):
        validate_rle(mask_csv, f)


if __name__ == '__main__':
    submit()
    # validate_rle_timeit()
