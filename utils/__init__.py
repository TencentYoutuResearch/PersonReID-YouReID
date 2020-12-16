import torch
import shutil
from .logger import Logger, setup_logger
from .epoch_lr import EpochBaseLR, WarmupMultiStepLR, CosineAnnealingWarmRestarts, CosineAnnealingWarmUp
import os
import scipy.io as sio
import re
import errno
from PIL import Image
# from .default_parser import init_parser
from .sampler import RandomIdentitySampler, RandomCameraSampler, DistributeRandomIdentitySampler
import numpy
from .re_ranking import re_ranking

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def save_checkpoint(state, root='../../snapshot/'):

    # path = os.path.join(root, str(code))
    if not os.path.exists(root):
        os.makedirs(root)

    # if (state['epoch'] % 10) == 0:
    #     torch.save(state, path + '/' + str(state['epoch']) + '.pth')
    filename = os.path.join(root, 'checkpoint.pth')
    torch.save(state, filename)
    print('Save checkpoint at %s' % filename)
    # if is_best:
    #     shutil.copyfile(filename, filename.replace('checkpoint.pth', 'best.pth'))




def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_feature(part, code, data, features, labels, paths, cams, root='../../feature'):
    path = os.path.join(root, str(code))
    if not os.path.exists(path):
        os.makedirs(path)
    sio.savemat(os.path.join(path, data +'_'+ part+'.mat'),
                {'feature':features, 'label':labels, 'path':paths, 'cam':cams})

def parse_parameters(model, keywords):
    parameters = []
    for name, param in model.named_parameters():
        if keywords in name:
            parameters.append(param)
    return parameters

def filter_parameters(model, keywords):
    parameters = []
    for name, param in model.named_parameters():
        if keywords not in name:
            parameters.append(param)
    return parameters







def extract(test_data, args, net):

    model = net(pretrained=None, is_for_test=True)
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(os.path.join('../../snapshot', args.code, 'checkpoint.pth'))
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print(args.start_epoch, best_prec1)

    # switch to evaluate mode
    model.eval()


    for p, val_loader in test_data.items():


        with torch.no_grad():
            paths = []
            for i, (input, target, path, cam, _) in enumerate(val_loader):

                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                # compute output
                feat = model(input)
                if type(feat) is list:
                    feat = torch.cat(feat, dim=1)

                feature = feat.cpu()
                target = target.cpu()

                nd_label = target.numpy()
                nd_feature = feature.numpy()
                if i == 0:
                    all_feature = nd_feature
                    all_label = nd_label
                    all_cam = cam.numpy()
                else:
                    all_feature = numpy.vstack((all_feature, nd_feature))
                    all_label = numpy.concatenate((all_label, nd_label))
                    all_cam = numpy.concatenate((all_cam, cam.numpy()))
                paths.extend(path)
            all_label.shape = (all_label.size, 1)
            all_cam.shape = (all_cam.size, 1)
            print(all_feature.shape, all_label.shape, all_cam.shape)
            save_feature(p, args.code, args.data, all_feature, all_label, paths, all_cam)

def remote_debug():
    import sys
    sys.path.append("../../config/pycharm-debug-py3k.egg")
    import pydevd
    pydevd.settrace('172.26.97.100', port=60000, stdoutToServer=True, stderrToServer=True)
