import os

import numpy as np
import torch


def get_resume_path(checkpoint_dir, resume_step=-1, prefix='Untitled_'):
    """Return latest checkpoints by default otherwise return the specified one.
    
    Notice:
        Filenames in `checkpoint_dir` should be 'XXX_100.path', 'XXX_300.path'...

    Arguments:
    + checkpoint_dir:
    + prefix: str('XXX_'), name of checkpoint
    + resume_step: uint, indicates specific step to resume training.
    + path of the resumed checkpoint.
    """
    names = [os.path.join(checkpoint_dir, p) for p in os.listdir(checkpoint_dir)]
    require = os.path.join(checkpoint_dir, prefix + '_' + str(resume_step) + '.pth')

    if resume_step == -1:
        return sorted(names, key=os.path.getmtime)[-1]
    elif os.path.isfile(require):
        return require
    raise Exception('\'%s\' dose not exist!' % require)


def load_checkpoints(model, checkpoint_dir, resume_step=-1, prefix='Untitled_'):
    """Load previous checkpoints"""

    cp = get_resume_path(checkpoint_dir, resume_step, prefix)
    pretrained_dict = torch.load(cp)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    cp_name = os.path.basename(cp)
    print('---> Loading checkpoint {}...'.format(cp_name))
    start_i = int(cp_name.split('_')[-1].split('.')[0]) + 1
    return start_i


def save_checkpoints(model, checkpoint_dir, step, prefix='Untitled_'):
    """Save 20 checkpoints at most"""

    names = os.listdir(checkpoint_dir)
    if len(names) >= 20:
        os.remove(os.path.join(checkpoint_dir, names[0]))
    # Recommend: save and load only the model parameters
    filename = prefix + '_' + str(step) + '.pth'
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, filename))
    print("===> ===> ===> Save checkpoint {} to {}".format(step, filename))


def best_checkpoints(results_dir, checkpoint_dir, keys=['val_dice_overlap']):
    """Return the path of the best checkpoint.
    
    Assume:
        checkpoint's format is 'Not_Important_123.pth'
    """
    checkpoints = os.listdir(checkpoint_dir)

    # '123' == '123.pth'[:-4] == ['Not', 'Important', '123.pth'][-1][:-4] == cp.split('_')[-1][:-4]
    resume_steps = [int(cp.split('_')[-1][:-4]) for cp in checkpoints]

    # prefix == 'Not_Important_{}.pth' == 'Not_Important' + '_{}.pth'
    prefix = '_'.join(checkpoints[0].split('_')[: -1]) + '_{}.pth'

    results_dict = np.load(os.path.join(results_dir, 'results_dict.npy')).item()
    selected_results = np.array([results_dict[key] for key in keys])
    sum_results = {step: selected_results[:, step - 1].sum() for step in resume_steps
                   if step > 0.7 * max(resume_steps)}

    best_checkpoints_step = sorted(sum_results, key=sum_results.get)[-1]
    return os.path.join(checkpoint_dir, prefix.format(best_checkpoints_step))

