import os
import sys

import cv2
import tqdm
import visdom
import numpy as np

import torch
from torch.autograd import Variable

# Set SSD Module Root
# ------------------------------------------------------------------------
ssd_root = os.path.join(os.path.abspath('..'), 'lib/ssd')  # './lib/ssd'
if ssd_root not in sys.path:
    sys.path.append(ssd_root)

from ssd import build_ssd
from data import VOCDetection, VOCroot, AnnotationTransform

# Init
# -------------------------
vis = visdom.Visdom()
torch.cuda.set_device(3)

# Build SSD300 in test phase
net = build_ssd('test', 300, 21).cuda()
net.load_weights(os.path.join(ssd_root, 'weights/ssd300_mAP_77.43_v2.pth'))


def load_image(img_path=None, is_default=False):
    if is_default:
        # Load a sample image from the VOC07 dataset.
        testset = VOCDetection(VOCroot, [('2007', 'val')], None, AnnotationTransform)
        img_id = 30
        image = testset.pull_image(img_id)
    else:
        image = cv2.imread(img_path)

    # convert to rgb image for later annotation stack
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Pre-process the input
    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)
    # vis.image(rgb_image.transpose(2, 0, 1))
    # vis.image(x)
    return x, rgb_image


def get_positions(img_path='../data/train/0cdf5b5d0ce1_01.jpg'):
    x, rgb_image = load_image(img_path)
    var_x = Variable(x.unsqueeze(0)).cuda()  # wrap tensor in Variable

    y = net(var_x)
    car_detections = y.data[0, 7, 0]  # max score in car(No.7) class
    coords = car_detections[1:]

    w, h = rgb_image.shape[1::-1]
    scale = list([w, h]) * 2  # [w, h, w, h]
    pt = np.floor(coords.numpy() * scale)  # [left-top-x, left-top-y, right-bottom-x, right-bottom-y]
    robust_pt_1 = np.maximum([0, 0], pt[:1] - 30)
    robust_pt_2 = np.minimum([w, h], pt[2:] + 30)
    return np.hstack([robust_pt_1, robust_pt_2]).astype(int)


def save_boundingbox(subdir='test/'):
    data_dir = '../data/' + subdir
    test_images_path = sorted(os.listdir(data_dir))
    boxes = {}
    for i in tqdm.tqdm(test_images_path):
        bbox = get_positions(data_dir + i)
        boxes[i] = bbox
    # '../data/test_boundingbox'
    np.save('../data/ssd_' + subdir.rstrip('/') + '_boundingbox', boxes)


if __name__ == '__main__':
    save_boundingbox(subdir='train/')
# ------------------------------------------------------------------------------------
# Parse the Detections and View Results
#  20 + 1 classes
# ------------------------------------------------------------------------------------

# detections = y.data

# # scale each detection back up to the image
# scale = torch.Tensor([rgb_image.shape[1::-1], rgb_image.shape[1::-1]])

# top_k = 10
#
# plt.figure(figsize=(10, 10))
# colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()  # display different annotation
# plt.imshow(rgb_image)  # plot the image for stack
# currentAxis = plt.gca()

# for i in range(detections.size(1)):
#     j = 0
#     while detections[0, i, j, 0] >= 0.6:
#         score = detections[0, i, j, 0]
#         label_name = labels[i - 1]
#         display_txt = '%s: %.2f' % (label_name, score)
#         pt = (detections[0, i, j, 1:] * scale).numpy()
#         coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
#         color = colors[i]
#         currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
#         currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
#         j += 1

# plt.savefig('final_detect', bbox_inches='tight')
# ------------------------------------------------------------------------------------
