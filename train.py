import os
import argparse
import numpy as np

import torch
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader

from inputs import CarDataset
from configuration import Configuration
from nets.u_net import DeepUNet_128, DeepUNet_HD
from nets.atrous import SCutAtrous_128, UNetAtrous_128, UNetAtrous_HD, SCutAtrous_HD, DeepAtrous_HD
from bluntools.layers import weights_init, active_flatten, multi_size, avg_class_weight, get_statistic, pre_visdom, \
    DiceLoss, NLLLoss
from bluntools.easy_visdom import EasyVisdom
from bluntools.lr_scheduler import auto_lr_scheduler, step_lr_scheduler
from bluntools.checkpoints import load_checkpoints, save_checkpoints

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
    # GPU (Default) configuration
    # --------------------------------------------------------------------------------------------------------
    assert conf.cuda, 'Use GPUs default, check cuda available'

    torch.cuda.manual_seed(conf.seed)
    torch.cuda.set_device(conf.cuda_device)
    print('===> Current GPU device is', torch.cuda.current_device())

    # Set models
    # --------------------------------------------------------------------------------------------------------
    if args.architecture == 0:
        conf.prefix = 'UNetAtrous_HD'
        conf.input_size = (1280, 1918)
        model = UNetAtrous_HD()
    elif args.architecture == 1:
        conf.prefix = 'SCutAtrous_HD'
        conf.input_size = (1280, 1918)
        model = SCutAtrous_HD()

    elif args.architecture == 2:
        conf.prefix = 'DeepAtrous_HD'
        conf.input_size = (1280, 1918)
        conf.batch_size = 2
        model = DeepAtrous_HD()
    elif args.architecture == 3:
        conf.prefix = 'UNetAtrous_128'
        model = UNetAtrous_128()

    conf.generate_dirs()

    model = model.cuda()
    print('===> Building {}...'.format(conf.prefix))
    print('---> Batch size: {}'.format(conf.batch_size))

    start_i = 1
    total_i = conf.epochs * conf.augment_size

    # Dataset loader
    # --------------------------------------------------------------------------------------------------------
    if conf.input_size is None:
        conf.input_size = tuple([int(conf.prefix.split('_')[-1])] * 2)
    print('---> Input image size: {}'.format(conf.input_size))

    # actual epochs == conf.augment_size * conf.epochs / compress = 100 * 10 / (100 / 5) =  50
    train_data_loader = DataLoader(
            dataset=CarDataset('train', input_size=conf.input_size, compress=conf.augment_size // 5,
                               use_bbox=conf.use_bbox, use_gta=True),
            num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)
    test_data_loader = DataLoader(
            dataset=CarDataset('test', use_bbox=conf.use_bbox, compress=conf.augment_size // 5,
                               input_size=conf.input_size, use_gta=True),
            num_workers=conf.threads, batch_size=conf.batch_size, shuffle=True)

    # Weights
    # ----------------------------------------------------------------------------------------------------
    if conf.from_scratch:
        model.apply(weights_init)
    else:
        start_i = load_checkpoints(model, conf.checkpoint_dir, conf.resume_step, conf.prefix)
    print('===> Begin training at epoch: {}'.format(start_i))

    # Optimizer
    # ----------------------------------------------------------------------------------------------------
    optimizer = optim.RMSprop(model.parameters(), lr=conf.learning_rate)
    scheduler = auto_lr_scheduler(optimizer, patience=150, cooldown=70, verbose=1, min_lr=1e-10)

    print('===> Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    print('---> Initial learning rate: {:.0e} '.format(conf.learning_rate))

    # Loss
    # ----------------------------------------------------------------------------------------------------
    class_weight = avg_class_weight(test_data_loader).cuda()
    print('---> Rescaled class weights: {}'.format(class_weight.cpu().numpy().T))

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
            image = Variable(image).float().cuda()

            outputs, targets = model(image), multi_size(label, size=conf.loss_size)  # 2D cuda Variable
            preds, trues, probs = active_flatten(outputs, targets)  # 1D

            loss = NLLLoss(class_weight)(outputs, targets) + DiceLoss()(probs, trues)

            pred, true = preds[0], trues[0]  # original size prediction
            accuracy, overlap = get_statistic(pred, true)

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

        avg_loss, avg_acc, avg_dice = np.array([epoch_loss, epoch_acc, epoch_overlap]) / partial_epoch
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
            image = Variable(image, volatile=True).float().cuda()

            outputs, targets = model(image), multi_size(label, size=conf.loss_size)  # 2D cuda Variable
            preds, trues, probs = active_flatten(outputs, targets)  # 1D
            loss = NLLLoss(class_weight)(outputs, targets) + DiceLoss()(probs, trues)

            pred, true = preds[0], trues[0]  # original size prediction
            accuracy, overlap = get_statistic(pred, true)

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
        # MultiStepLR monitors: epoch
        # optimizer = step_lr_scheduler(optimizer, i, milestones=[800, 1000, 1500],
        #                               init_lr=conf.learning_rate, instant=(start_i == i))

        # `results` contains [loss, acc, dice]
        *train_results, train_image, train_label, train_pred = train()
        *val_results, val_image, val_label, val_pred = validate()

        # ReduceLROnPlateau monitors: validate loss
        scheduler.step(val_results[0])

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
            temp_result = val_results[-1] + train_results[-1]
            is_best = temp_result > best_result
            best_result = max(temp_result, best_result)

            save_checkpoints(model, conf.checkpoint_dir, i, conf.prefix, is_best=is_best)
            # np.load('path/to/').item()
            np.save(os.path.join(conf.result_dir, 'results_dict.npy'), ev.results_dict)


if __name__ == '__main__':
    main()
