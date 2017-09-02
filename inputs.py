import time

import numpy as np
import torch
import torch.utils.data as data

from scipy.misc import imread
from scipy.ndimage import zoom
from skimage.transform import resize

from utils.buzz import IMAGES_PATH, swap_format
from utils.image_ops import normalize, adjust_intensity

RESIZE = (1024, 1024)


def read_downscale(im, factor=4):
    seq = [1 / factor, 1 / factor]

    if im.ndim == 3:
        seq.append(1)
    im = zoom(im, seq, mode='nearest', prefilter=False)

    # SLOW random zoom
    # -----------------------------------------------------------------
    # rand_seq = np.random.uniform(low=0.8, high=1.2, size=2)
    # if im.ndim == 3:
    #     rand_seq = np.append(rand_seq, 1)
    # im = zoom(im, rand_seq, mode='nearest', order=5, prefilter=False)

    return im


class DatasetFromFolder(data.Dataset):
    def __init__(self, mode='train', train_size=4600, downscale=4, compress=20):
        super(DatasetFromFolder, self).__init__()
        self.downscale = downscale

        # Compress Dataloader size, for faster epoch
        self.compress = compress

        self.data_path = IMAGES_PATH
        np.random.shuffle(self.data_path)

        self.train_path, self.test_path = np.split(self.data_path, [train_size])
        self.image_path = eval('self.{}_path'.format(mode))
        self.label_path = [swap_format(p) for p in self.image_path]

    def __getitem__(self, index):
        np.random.seed(int(time.time()))

        # Read images, with size [height/4, width/4]
        index += np.random.choice(self.compress, 1) * (len(self.image_path) // self.compress)
        im_path, lb_path = self.image_path[index[0]], self.label_path[index[0]]
        images = read_downscale(imread(im_path))
        labels = np.expand_dims(read_downscale(imread(lb_path, mode='L')), -1)

        # Crop to smaller size
        # bb = self.bbox[im_path[:-7]]
        # images, labels = [item[bb[0]: bb[2], bb[1]: bb[3]] for item in [images, labels]]

        images = adjust_intensity(normalize(images))
        images, labels = [resize(item, RESIZE, preserve_range=True) for item in [images, labels]]
        labels = (labels > 127).astype(np.uint8)

        # Convert to tensor with shape [1/3, height, width]
        images, labels = [torch.from_numpy(item.transpose(2, 0, 1))
                          for item in [images, labels]]
        return images, labels

    def __len__(self):
        return len(self.image_path) // self.compress  # test


# ---------------------------------------------------------------------------------------------------------------------
class InputsTest():
    def test_read_identity(self):
        images_0 = read_identidy_bulk(to_identity(IMAGES_PATH[0]))
        labels_0 = read_identidy_bulk(to_identity(LABELS_PATH[0]))

        images, labels, bbox = cars_shrink_to_bbox(images_0, labels_0)
        print('Image Original shape:', images_0.shape)
        print('Label Reduced shape:', labels.shape)
        # 1 / 4
        [vis.images(np.expand_dims(zoom(item, [1, 0.5, 0.5], prefilter=True), 1)) for item in [images_0, labels_0]]

    def test_data_loader(self):
        training_data_loader = data.DataLoader(dataset=DatasetFromFolder(), num_workers=4, batch_size=16, shuffle=True)
        im, lb = next(iter(training_data_loader))
        print(im.size(), 'image')
        print(lb.size(), 'label')
        # If one-hot encode
        # ------------------------------------------------------------
        # lb = torch.index_select(lb, 1, torch.LongTensor([0]))
        # ------------------------------------------------------------
        vis.images(im.numpy(), opts=dict(title='Random selected images', caption='Shape: {}'.format(im.size())))
        vis.images(lb.numpy(), opts=dict(title='Random selected labels', caption='Shape: {}'.format(lb.size())))


if __name__ == '__main__':
    import visdom
    from utils.buzz import read_identidy_bulk, to_identity, cars_shrink_to_bbox, IMAGES_PATH, LABELS_PATH

    vis = visdom.Visdom()
    t = InputsTest()
    # t.test_read_identity()
    t.test_data_loader()
