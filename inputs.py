import glob
import time

import numpy as np
from scipy.misc import imread
from skimage.transform import resize, rescale

import torch
import torch.utils.data as data

from utils.image_ops import random_hsv, random_affine, random_hflip
from utils.morph_bbox import swap_format

RESIZE = (1024, 1024)
IM_SIZE = np.array([1280, 1918])
IMAGES_PATH = sorted(glob.glob('./data/train/*.jpg'))
LABELS_PATH = [swap_format(i) for i in IMAGES_PATH]


class DatasetFromFolder(data.Dataset):
    def __init__(self, mode='train', train_size=4600, length=1024, compress=20):
        super(DatasetFromFolder, self).__init__()
        self.compress = compress  # compress Dataloader size, for faster epoch
        self.length = [length] * 2
        self.data_path = sorted(glob.glob('./data/train/*.jpg'))
        np.random.shuffle(self.data_path)

        self.train_path, self.test_path = np.split(self.data_path, [train_size])
        self.image_path = eval('self.{}_path'.format(mode))
        self.label_path = [swap_format(p) for p in self.image_path]

        # self.bbox = np.load('./data/train_boundingbox.npy').item()

    def __getitem__(self, index):
        np.random.seed(int(time.time()))

        # Read image, with size [height, width]
        index += np.random.choice(self.compress, 1) * (len(self.image_path) // self.compress)
        im_path, lb_path = self.image_path[index[0]], self.label_path[index[0]]
        image, label = imread(im_path), imread(lb_path, mode='L')

        # # Crop to smaller size to increase foreground
        # bb = self.bbox[im_path[-19:]].astype(int)
        # image, label = [item[bb[1]: bb[3], bb[0]: bb[2]] for item in [image, label]]

        image = random_hsv(image)
        image, label = random_affine(image, label)
        image, label = random_hflip(image, label)
        image, label = [resize(item, self.length, preserve_range=True) for item in [image, label]]
        label = np.expand_dims((label > 127).astype(np.uint8), -1)

        # Convert to tensor with shape [1/3, height, width]
        image, label = [torch.from_numpy(item.transpose(2, 0, 1)) for item in [image, label]]
        return image.div(255), label

    def __len__(self):
        return len(self.image_path) // self.compress  # test


# ---------------------------------------------------------------------------------------------------------------------
class InputsTest():
    def test_read_identity(self):
        image_0 = read_identidy_bulk(to_identity(IMAGES_PATH[0]))
        label_0 = read_identidy_bulk(to_identity(LABELS_PATH[0]))

        image, label, bbox = cars_shrink_to_bbox(image_0, label_0)
        print('Image Original shape:', image_0.shape)
        print('Label Reduced shape:', label.shape)
        # 1 / 4
        [vis.image(np.expand_dims(rescale(item, [1, 0.5, 0.5], preserve_range=True), 1)) for item in [image_0, label_0]]

    def test_data_loader(self):
        training_data_loader = data.DataLoader(dataset=DatasetFromFolder(), num_workers=4, batch_size=16, shuffle=True)
        im, lb = next(iter(training_data_loader))
        print(im.size(), 'image')
        print(lb.size(), 'label')
        # If one-hot encode
        # ------------------------------------------------------------
        # lb = torch.index_select(lb, 1, torch.LongTensor([0]))
        # ------------------------------------------------------------
        vis.image(im.numpy(), opts=dict(title='Random selected image', caption='Shape: {}'.format(im.size())))
        vis.image(lb.numpy(), opts=dict(title='Random selected label', caption='Shape: {}'.format(lb.size())))


if __name__ == '__main__':
    import visdom
    from utils.morph_bbox import read_identidy_bulk, to_identity, cars_shrink_to_bbox

    vis = visdom.Visdom()
    t = InputsTest()
    # t.test_read_identity()
    t.test_data_loader()
