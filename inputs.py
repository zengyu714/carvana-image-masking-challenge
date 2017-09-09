import glob
import time

import numpy as np
from scipy.ndimage import imread
from skimage.transform import resize, rescale

import torch
import torch.utils.data as data

from utils.image_ops import random_hsv, random_affine, random_hflip
from utils.others import swap_format


class CarDataset(data.Dataset):
    def __init__(self, mode='train', use_bbox=False, train_size=4600, length=1024, compress=20):
        super(CarDataset, self).__init__()
        self.compress = compress  # compress Dataloader size, for faster epoch
        self.length = [length] * 2
        self.data_path = sorted(glob.glob('./data/train/*.jpg'))
        np.random.shuffle(self.data_path)

        self.train_path, self.test_path = np.split(self.data_path, [train_size])
        self.image_path = eval('self.{}_path'.format(mode))
        self.label_path = [swap_format(p).replace('train', 'train_masks') for p in self.image_path]

        self.use_bbox = use_bbox
        if use_bbox:
            self.bbox = np.load('./data/morph_train_boundingbox.npy').item()

    def __getitem__(self, index):
        np.random.seed(int(time.time()))

        # Read image, with size [height, width]
        index += np.random.choice(self.compress, 1) * (len(self.image_path) // self.compress)
        im_path, lb_path = self.image_path[index[0]], self.label_path[index[0]]
        image, label = imread(im_path), imread(lb_path, mode='L')

        # Crop to smaller size to increase foreground
        if self.use_bbox:
            bb = self.bbox[im_path[-19:]]
            image, label = [item[bb[0]: bb[2], bb[1]: bb[3]] for item in [image, label]]

        image = random_hsv(image)
        image, label = [resize(item, self.length, preserve_range=True) for item in [image, label]]

        if self.use_bbox:
            translation = None  # No need for translation
        else:
            translation = [-0.0625, 0.0625]
        image, label = random_affine(image, label, translation=translation)
        image, label = random_hflip(image, label)

        label = np.expand_dims((label > 127).astype(np.uint8), -1)

        # Convert to tensor with shape [1/3, height, width]
        image, label = [torch.from_numpy(item.transpose(2, 0, 1)) for item in [image, label]]
        return image.div(255), label

    def __len__(self):
        return len(self.image_path) // self.compress  # test


# test
# ---------------------------------------------------------------------------------------------------------------------
class InputsTest():
    def test_read_identity(self):
        images = read_identidy_car(to_identity(IMAGES_PATH[0]))
        labels = read_identidy_car(to_identity(LABELS_PATH[0]), data_dir='./data/train_mask/')
        boxes = car_boxes(images)
        bb = boxes[0]
        print(bb)
        image, label = [it[bb[0]: bb[2], bb[1]: bb[3]] for it in [images[0], labels[0]]]
        print('Image Original shape:', images[0].shape)
        print('Label Reduced shape:', label.shape)
        # 1 / 4
        [vis.image(rescale(item, 0.25, preserve_range=True)) for item in [image, label]]

    def test_data_loader(self):
        training_data_loader = data.DataLoader(dataset=CarDataset(), num_workers=4, batch_size=16, shuffle=True)
        im, lb = next(iter(training_data_loader))
        print(im.size(), 'image')
        print(lb.size(), 'label')
        im, lb = next(iter(training_data_loader))
        print(im.size(), 'image')
        print(lb.size(), 'label')
        # If one-hot encode
        # ------------------------------------------------------------
        # lb = torch.index_select(lb, 1, torch.LongTensor([0]))
        # ------------------------------------------------------------
        vis.images(im.numpy(), opts=dict(title='Random selected image', caption='Shape: {}'.format(im.size())))
        vis.images(lb.numpy(), opts=dict(title='Random selected label', caption='Shape: {}'.format(lb.size())))


if __name__ == '__main__':
    import visdom
    from configuration import IMAGES_PATH, LABELS_PATH
    from utils.others import read_identidy_car, to_identity
    from utils.morph_bbox import car_boxes

    vis = visdom.Visdom()
    t = InputsTest()
    # t.test_read_identity()
    t.test_data_loader()
