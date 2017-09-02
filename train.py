import os
import argparse
import numpy as np

import torch
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader

from inputs import DatasetFromFolder
from nets.u_net import UNet
from nets.layers import weights_init, softmax_flat, get_class_weight, get_statictis, pre_visdom, DiceLoss, NLLLoss
from utils.easy_visdom import EasyVisdom
from utils.checkpoints import load_checkpoints, save_checkpoints

parser = argparse.ArgumentParser(description='Carvana Image Masking Challenge')

parser.add_argument('-n', '--from-scratch', action='store_true',
                    help='train model from scratch')
parser.add_argument('-g', '--cuda-device', type=int, default=0,
                    help='choose which gpu to use')
parser.add_argument('-a', '--architecture', type=int, default=0,
                    help='0 ==> AtrousNext, 1 ==> DualPath')

args = parser.parse_args()


class Configuration:
    def __init__(self, prefix='UNet', cuda_device=args.cuda_device, from_scratch=args.from_scratch):
        self.cuda = torch.cuda.is_available()
        self.cuda_device = cuda_device

        self.batch_size = 4
        self.epochs = 20
        self.grad_acc_num = 8
        self.augment_size = 100
        self.train_size = 4000
        self.test_size = 1088
        self.learning_rate = 1e-6
        self.seed = 714
        self.threads = 4
        self.resume_step = -1
        self.from_scratch = from_scratch
        self.prefix = prefix

    def generate_dirs(self):
        self.result_dir = os.path.join('./results', self.prefix)
        self.checkpoint_dir = os.path.join('./checkpoints', self.prefix)
        # for d in [self.subdir, self.result_dir, self.checkpoint_dir]:
        #     if not os.path.exists(d):
        #         os.makedirs(d)
        [os.makedirs(d) for d in [self.result_dir, self.checkpoint_dir] if not os.path.exists(d)]


conf = Configuration()

# Dataset loader
# --------------------------------------------------------------------------------------------------------

train_data_loader = DataLoader(dataset=DatasetFromFolder('train', compress=conf.augment_size // 4),
                               num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)
test_data_loader = DataLoader(dataset=DatasetFromFolder('test', compress=conf.augment_size // 4),
                              num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)


def main():
    if args.architecture == 0:
        conf.prefix = 'UNet'
        # conf.learning_rate = 1e-7  # Step 210
        conf.learning_rate = 1e-8  # Step 305

        model = UNet()

    conf.generate_dirs()

    if conf.cuda:
        model = model.cuda()
    print('===> Building {}...'.format(conf.prefix))

    start_i = 1
    total_i = conf.epochs * conf.augment_size

    # Weights
    # ----------------------------------------------------------------------------------------------------
    if conf.from_scratch:
        model.apply(weights_init)
    else:
        start_i = load_checkpoints(model, conf.checkpoint_dir, conf.resume_step, conf.prefix)
    print('===> Begin training at epoch {}'.format(start_i))

    # Optimizer
    # ----------------------------------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)
    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('---> Learning rate: {} '.format(conf.learning_rate))

    # Loss
    # ----------------------------------------------------------------------------------------------------
    class_weight = get_class_weight(test_data_loader)
    print('---> Rescaled class weights: {}'.format(class_weight.numpy().T))
    if conf.cuda:
        class_weight = class_weight.cuda()

    # Visdom
    # ----------------------------------------------------------------------------------------------------
    ev = EasyVisdom(conf.from_scratch,
                    total_i,
                    start_i=start_i,
                    mode=['train', 'val'],
                    stats=['loss', 'acc', 'dice_overlap'],
                    results_dir=conf.result_dir,
                    env=conf.prefix)

    def train():
        epoch_loss, epoch_acc, epoch_overlap = np.zeros(3)

        # Sets the module in training mode, only on modules such as Dropout or BatchNorm.
        model.train()

        for partial_epoch, (image, label) in enumerate(train_data_loader, 1):
            image, label = Variable(image).float(), Variable(label).float()
            if conf.cuda:
                image, label = image.cuda(), label.cuda()

            output, target = model(image), label
            pred, true = softmax_flat(output, target)

            loss = NLLLoss(class_weight)(output, target) + DiceLoss()(pred, true)

            # Accumulate gradients
            if partial_epoch == 0:
                optimizer.zero_grad()

            loss.backward()
            if partial_epoch % conf.grad_acc_num == 0:
                optimizer.step()
                optimizer.zero_grad()

            accuracy, overlap = get_statictis(pred, true)

            epoch_acc += accuracy
            epoch_loss += loss.data[0]
            epoch_overlap += overlap

        avg_loss, avg_acc, avg_dice = np.array(
                [epoch_loss, epoch_acc, epoch_overlap]) / partial_epoch
        print_format = [avg_loss, avg_acc, avg_dice]
        print('===> Training step {} ({}/{})\t'.format(i, i // conf.augment_size + 1, conf.epochs),
              'Loss: {:.5f}   Accuracy: {:.5f}  |   Dice Overlap: {:.5f}'.format(*print_format))

        return (avg_loss, avg_acc, avg_dice, *pre_visdom(image, label, pred))

    def validate():
        epoch_loss, epoch_acc, epoch_overlap = np.zeros(3)

        # Sets the module in training mode, only on modules such as Dropout or BatchNorm.
        model.eval()

        for partial_epoch, (image, label) in enumerate(test_data_loader, 1):
            image, label = Variable(image, volatile=True).float(), Variable(label, volatile=True).float()
            if conf.cuda:
                image, label = image.cuda(), label.cuda()

            output, target = model(image), label
            pred, true = softmax_flat(output, target)

            loss = NLLLoss(class_weight)(output, target) + DiceLoss()(pred, true)

            accuracy, overlap = get_statictis(pred, true)

            epoch_acc += accuracy
            epoch_loss += loss.data[0]
            epoch_overlap += overlap

        avg_loss, avg_acc, avg_dice = np.array(
                [epoch_loss, epoch_acc, epoch_overlap]) / partial_epoch
        print_format = [avg_loss, avg_acc, avg_dice]
        print('===> ===> Validation Performance', '-' * 60,
              'Loss: {:.5f}   Accuracy: {:.5f}  |  Dice Overlap: {:.5f}'.format(*print_format))

        return (avg_loss, avg_acc, avg_dice, *pre_visdom(image, label, pred))

    for i in range(start_i, total_i + 1):
        # `results` contains [loss, acc, dice]

        *train_results, train_image, train_label, train_pred = train()
        *val_results, val_image, val_label, val_pred = validate()

        # Visualize - scalar
        ev.vis_scalar(i, train_results, val_results)

        # Visualize - images
        ev.vis_images(i,
                      show_interval=1,
                      im_titles=['input', 'label', 'prediction'],
                      train_images=[train_image, train_label, train_pred],
                      val_images=[val_image, val_label, val_pred])

        # Save checkpoints
        if i % 5 == 0:
            save_checkpoints(model, conf.checkpoint_dir, i, conf.prefix)
            # np.load('path/to/').item()
            np.save(os.path.join(conf.result_dir, 'results_dict.npy'), ev.results_dict)


if __name__ == '__main__':
    main()
