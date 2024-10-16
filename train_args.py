
# coding=utf-8

import argparse
import random
import os
import numpy as np
import torch

def get_args():
    dataset_names = ['CIFAR10', 'CIFAR100', 'RCV1']
    delay_types = ['fixed', 'random']
    parser = argparse.ArgumentParser(description="meta config of experiment")
    parser.add_argument('--dataset', default='CIFAR100', type=str, metavar='data', choices=dataset_names)
    parser.add_argument('--model', default='resnet', type=str, metavar='model')
    parser.add_argument('--num-epochs', default=200, type=int, metavar='N', help='number of epochs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--delay', default=16, type=int)
    parser.add_argument('--delay-type', default='random', type=str, choices=delay_types)
    parser.add_argument('--num-workers', default=16, type=int, metavar='W')
    parser.add_argument('--batch-size', default=16, type=int, metavar='b', help='batch size per worker')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--logdir', default='./log/', type=str)
    parser.add_argument('--lr-schedule', default='const', type=str, choices=['const', 'decay', 't'])
    parser.add_argument('--lr-decay', default=0.1, type=float)
    parser.add_argument('--cuda-device', default=0, type=int, metavar='c')
    parser.add_argument('--print-freq', default=50, type=int, metavar='p')
    parser.add_argument('--eval-freq', default=0, type=int)
    parser.add_argument("--cuda-ps", action='store_true')
    args = parser.parse_args()

    return args

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
