import argparse
import dataset
import models

def init_parser():
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--step', default=40, type=int, metavar='N',
                        help=' period of learning rate decay.')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    ## for dataset
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--load_img_to_cash', default=1, type=int,
                        help='load img to cash')
    parser.add_argument('--least_image_per_class', default=4, type=int,
                        help='at least N image per class')
    parser.add_argument('--use_tf_sample', default=1, type=int,
                        help='use tf sample')

    parser.add_argument('--lr', '--learning-rate', default=0.003, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate models on validation set')
    parser.add_argument('--seed', default=0, type=int, help='the random seed')
    parser.add_argument('-g', '--gpu', type=str, default='2,3', metavar='G',
                        help='set the ID of GPU')
    parser.add_argument('-c', '--code', type=str, default='0000', help='the number of model')
    parser.add_argument('-sig', '--signature', type=str, default='', help='other setting')

    parser.add_argument('--net', type=str, default='Baseline', choices=models.__all__, help='nets: ' +' | '.join(models.__all__) +
                            ' (default: resnet50)')
    parser.add_argument('--pretrained', action='store_true',
                        help='use imagenet pretrained model')

    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'], help ='the optimizer of network')
    parser.add_argument('--other-pretrained', default='', type=str, metavar='PATH',
                        help='path to pretrained model checkpoint (default: none)')
    parser.add_argument('--rerank', action='store_true', help='use rerank', default=False)

    ## dataset setting
    parser.add_argument('--data', choices=dataset.__all__, help='dataset: ' +' | '.join(dataset.__all__) +
                            ' (default: KESCI)', default='Market1501')
    parser.add_argument('--part', type=str, default='train')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--use_random_pad', type=int, default=1)

    ## loss setting
    parser.add_argument('--margin', type=float, default=0.3, help='the margin of the triplet loss')
    parser.add_argument('--weight', type=float, default=1, help='the weight of the triplet loss')


    # debug
    parser.add_argument('--debug', action='store_true', help='use remote debug', default=False)


    return parser


