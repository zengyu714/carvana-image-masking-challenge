import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from scipy.misc import imread
from skimage.transform import resize, rescale

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from inputs import SubmitCarDataset
from utils.image_ops import dense_crf
from nets.u_net import UNet_1024
from nets.atrous import Atrous_1024

INPUT_SIZE = (1024, 1024)  # [int(MODEL.split('_')[-1])] * 2
IMAGE_SIZE = (1280, 1918)
BATCH_SIZE = 16

import visdom

vis = visdom.Visdom()


def rle_encode(mask):
    pos = np.where(mask == 1)[0]
    sep = np.split(pos, np.where(np.diff(pos) != 1)[0] + 1)
    res = [f(s) for s in sep for f in (lambda s: s[0] + 1, lambda s: len(s))]
    return ' '.join(str(r) for r in res)


def rle_encode_faster(mask):
    """Reference: https://www.kaggle.com/stainsby/fast-tested-rle"""
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def submit(model_name='UNet_1024', device_id=1, use_crf=False):
    torch.cuda.set_device(device_id)

    model = eval(model_name)()  # say, 'UNet_1024'
    model = model.cuda().eval()

    best_path = 'checkpoints/{}/{}_best.pth'.format(model_name, model_name)
    print('===> Loading model from {}...'.format(best_path))

    best_model = torch.load(best_path)
    model.load_state_dict(best_model)

    submit_dataset = SubmitCarDataset()
    dataloader = DataLoader(submit_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    rles = []
    names = []
    for i_batch, (im_batch, x_batch, name_batch) in tqdm(enumerate(dataloader),
                                                         total=len(submit_dataset) // BATCH_SIZE, unit='batch'):
        x_batch = Variable(x_batch, volatile=True).float().cuda()
        output = model(x_batch)

        pred = output.data.max(1)[1]
        pred = pred.cpu().numpy()

        for j, mask in enumerate(pred):
            im = im_batch[j].numpy()
            mask = (resize(mask, IMAGE_SIZE, preserve_range=True) > 0.5).astype(np.uint8)

            if use_crf:
                mask = dense_crf(im, mask)
                # vis.image(rescale(mask, 0.25, preserve_range=True), opts=dict(title='CRF'))
            # C x H x W
            # vis.image(rescale(im, 0.25, preserve_range=True).transpose(2, 0, 1), opts=dict(title='IM'))
            # vis.image(rescale(mask, 0.25, preserve_range=True), opts=dict(title='ORI'))

            rles.append(rle_encode_faster(mask))
            names.append(name_batch[j])

        if i_batch % 512 == 0:
            # Save in case
            df = pd.DataFrame({'img': names, 'rle_mask': rles})
            df.to_csv('submit/submission_{}_{}.csv.gz'.format(model_name, i_batch), index=False, compression='gzip')


# file
TRAIN_MASKS_CSV = 'data/train_masks.csv'
TRAIN_MASKS_DIR = 'data/train_masks/'
TRAIN_MASKS_PATH = [os.path.join(TRAIN_MASKS_DIR, p) for p in os.listdir(TRAIN_MASKS_DIR)]
TRAIN_MASKS_DEMO = TRAIN_MASKS_PATH[0]


def validate_rle(mask_csv, mask_path=TRAIN_MASKS_DEMO):
    """ Validate run-length encoding.
    Arguments
        - maskfile: names like '0ce66b539f52', '0cdf5b5d0ce1'...
    """
    index = mask_path.split('/')[-1].replace('_mask.png', '.jpg')

    true = mask_csv[mask_csv['img'] == index]['rle_mask']
    mask = imread(mask_path, mode='L').flatten() > 127
    pred = rle_encode_faster(mask)
    assert (true == pred).bool(), 'Run-length encoding fails...'


def validate_rle_timeit():
    mask_csv = pd.read_csv(TRAIN_MASKS_CSV)

    for f in tqdm(TRAIN_MASKS_PATH):
        validate_rle(mask_csv, f)


if __name__ == '__main__':
    # submit(model_name='UNet_1024', device_id=2)
    submit(model_name='Atrous_1024', device_id=3)
    # validate_rle_timeit()
