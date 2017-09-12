import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from inputs import CarDataset
from configuration import Configuration
from nets.u_net import UNet_1024, UNet_512, UNet_256, UNet_128, UNetReLU_128, UNetShallow_128, UNetMultiLoss_128
from nets.atrous import Atrous_256, Atrous_1024
from nets.layers import weights_init, flatten, avg_class_weight, get_statictis, pre_visdom, DiceLoss, NLLLoss
from utils.easy_visdom import EasyVisdom
from utils.lr_scheduler import ReduceLROnPlateau
from utils.checkpoints import load_checkpoints, save_checkpoints

parser = argparse.ArgumentParser(description='Carvana Image Masking Challenge')
parser.add_argument('-n', '--from-scratch', action='store_true',
                    help='train model from scratch')
parser.add_argument('-g', '--cuda-device', type=int, default=0,
                    help='choose which gpu to use')
parser.add_argument('-a', '--architecture', type=int, default=0,
                    help='see details in code')

args = parser.parse_args()
conf = Configuration(prefix='UNet_1024', cuda_device=args.cuda_device, from_scratch=args.from_scratch)


def main():
    # GPU configuration
    # --------------------------------------------------------------------------------------------------------
    if conf.cuda:
        torch.cuda.set_device(conf.cuda_device)
        print('===> Current GPU device is', torch.cuda.current_device())

    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)

    # Set models
    # --------------------------------------------------------------------------------------------------------
    if args.architecture == 0:
        conf.prefix = 'UNet_1024'
        conf.batch_size = 8
        model = UNet_1024()
    elif args.architecture == 1:
        conf.prefix = 'UNet_512'
        model = UNet_512()
    elif args.architecture == 2:
        conf.prefix = 'UNet_256'
        model = UNet_256()
    elif args.architecture == 3:
        conf.prefix = 'UNet_128'
        model = UNet_128()
    elif args.architecture == 4:
        conf.prefix = 'Atrous_1024'
        conf.batch_size = 8
        model = Atrous_1024()
    elif args.architecture == 5:
        conf.prefix = 'Atrous_256'
        model = Atrous_256()
    elif args.architecture == 6:
        conf.prefix = 'UNetShallow_128'
        model = UNetShallow_128()
    elif args.architecture == 7:
        conf.prefix = 'UNetMultiLoss_128'
        model = UNetMultiLoss_128()
    elif args.architecture == 8:
        conf.prefix = 'UNetLeaky_128'
        model = UNetReLU_128(act=nn.LeakyReLU())

    conf.generate_dirs()

    if conf.cuda:
        model = model.cuda()
    print('===> Building {}...'.format(conf.prefix))

    start_i = 1
    total_i = conf.epochs * conf.augment_size

    # Dataset loader
    # --------------------------------------------------------------------------------------------------------
    conf.length = int(conf.prefix.split('_')[-1])
    # actual epochs == conf.augment_size * conf.epochs / compress = 100 * 10 / (100 / 5) =  50
    train_data_loader = DataLoader(
            dataset=CarDataset('train', use_bbox=conf.use_bbox, length=conf.length,
                               compress=conf.augment_size // 5),
            num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)
    test_data_loader = DataLoader(
            dataset=CarDataset('test', use_bbox=conf.use_bbox, length=conf.length,
                               compress=conf.augment_size // 5),
            num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)

    # Weights
    # ----------------------------------------------------------------------------------------------------
    if conf.from_scratch:
        model.apply(weights_init)
    else:
        start_i = load_checkpoints(model, conf.checkpoint_dir, conf.resume_step, conf.prefix)
    print('===> Begin training at epoch {}'.format(start_i))

    # Optimizer
    # ----------------------------------------------------------------------------------------------------
    optimizer = optim.RMSprop(model.parameters(), lr=conf.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=100, cooldown=50, verbose=1, min_lr=1e-6)

    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('---> Learning rate: {} '.format(conf.learning_rate))

    # Loss
    # ----------------------------------------------------------------------------------------------------
    class_weight = avg_class_weight(test_data_loader)
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
            pred, true = flatten(output, target)

            loss = NLLLoss(class_weight)(output, target) + DiceLoss()(pred, true)
            accuracy, overlap = get_statictis(pred, true)

            # Accumulate gradients
            # >>> if partial_epoch == 0:
            # >>>     optimizer.zero_grad()
            # >>>
            # >>> loss.backward()
            # >>> if partial_epoch % conf.grad_acc_num == 0:
            # >>>     optimizer.step()
            # >>>     optimizer.zero_grad()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            epoch_acc += accuracy
            epoch_loss += loss.data[0]
            epoch_overlap += overlap

        avg_loss, avg_acc, avg_dice = np.array(
                [epoch_loss, epoch_acc, epoch_overlap]) / partial_epoch
        print_format = [avg_loss, avg_acc, avg_dice]
        print('===> Training step {} ({}/{})\t'.format(i, i // conf.augment_size + 1, conf.epochs),
              'Loss: {:.5f}   Accuracy: {:.5f}  |   Dice Overlap: {:.5f}'.format(*print_format))

        return (avg_loss, avg_acc, avg_dice,
                *pre_visdom(image, label, pred))

    def validate():
        epoch_loss, epoch_acc, epoch_overlap = np.zeros(3)

        # Sets the module in training mode, only on modules such as Dropout or BatchNorm.
        model.eval()

        for partial_epoch, (image, label) in enumerate(test_data_loader, 1):
            image, label = Variable(image, volatile=True).float(), Variable(label, volatile=True).float()
            if conf.cuda:
                image, label = image.cuda(), label.cuda()

            output, target = model(image), label
            pred, true = flatten(output, target)

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

        return (avg_loss, avg_acc, avg_dice,
                *pre_visdom(image, label, pred))

    best_result = 0
    for i in range(start_i, total_i + 1):
        # optimizer = exp_lr_scheduler(optimizer, i, init_lr=conf.learning_rate, lr_decay_epoch=conf.lr_decay_epoch)

        # `results` contains [loss, acc, dice]
        *train_results, train_image, train_label, train_pred = train()
        *val_results, val_image, val_label, val_pred = validate()

        scheduler.step(val_results[0], i)  # monitor validate loss

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
            is_best = val_results[-1] > best_result
            best_result = max(val_results[-1], best_result)
            save_checkpoints(model, conf.checkpoint_dir, i, conf.prefix, is_best=is_best)
            # np.load('path/to/').item()
            np.save(os.path.join(conf.result_dir, 'results_dict.npy'), ev.results_dict)


if __name__ == '__main__':
    main()
