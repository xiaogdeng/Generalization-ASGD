import pickle
import os, datetime, sys
import time
import copy
import random
import shutil
from random import Random, shuffle
import numpy as np
import types

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

# import custom_models
from train_args import get_args, seed_torch
from models import *
from libsvm_data import *


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))
        # return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '}) (({sum' + self.fmt + '}))'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, *meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def print(self, epoch, batch):
        entries = [self.prefix + "[" + str(epoch) + str(", ") + str(batch) + "]"]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))


class Logger:
    def __init__(self, folder):
        self.logs = []
        self.folder = folder

    def update(self, vals):
        self.logs.append(vals)
        # self.logs.append([epoch, train_loss, train_accuracy, train_time, test_loss, test_accuracy, test_time])

    def save_log(self, filename):
        temp = np.asarray(self.logs)
        np.savetxt(filename, temp, delimiter=",")


def load_dataset(name, location, train=True):
    # Todo: custom for name in {'MNIST', 'CIFAR10', 'CIFAR100', ...}
    name = name.lower()

    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(root=location, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=location, train=False, download=True, transform=transform_test)

    elif name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                 std=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
        ])
        train_dataset = datasets.CIFAR100(root=location, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR100(root=location, train=False, download=True, transform=transform_test)

    elif name == 'rcv1':
        # of classes: 2
        # of data: 20,242 / 677,399 (testing)
        # of features: 47,236
        train_dataset = RCV1Binary("train", download=True, data_root=location)
        test_dataset = RCV1Binary("test", download=True, data_root=location)

    else:
        raise ValueError(name + ' is not known.')

    return train_dataset, test_dataset


def load_model(name=None, dataset=None):
    name = name.lower()
    dataset = dataset.lower()
    if name == 'resnet' and dataset == 'cifar10':
        net = resnet18(10)
    elif name == 'resnet' and dataset == 'cifar100':
        net = resnet18(100)
    elif name == 'linearnet_rcv1':
        net = LinearNet_rcv1()
    else:
        raise ValueError(name + ' is not known.')
    init_params(net)

    return net


def load_model_copy(name=None, dataset=None):
    name = name.lower()
    dataset = dataset.lower()
    if name == 'resnet' and dataset == 'cifar10':
        net = resnet18_copy(10)
    elif name == 'resnet' and dataset == 'cifar100':
        net = resnet18_copy(100)
    elif name == 'linearnet_rcv1':
        net = LinearNet_rcv1_copy()
    else:
        raise ValueError(name + ' is not known.')
    init_params(net)

    return net


def data_partition(num_workers, data_set):
    """
    generates a random shuffle of the size of the dataset, and returns the indices partitioned among the workers

    :param num_workers:
    :param data_set: type torch.data
    :param separate:
    :return:
    """
    # size = data_set.data.shape[0]
    size = len(data_set)
    ind = list(range(size))
    shuffle(ind)
    # worker_size is the number of samples per worker. The last worker either receives the additional samples
    # or the last samples are dropped.
    worker_size = size // num_workers
    data = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        if w is not num_workers - 1:
            data[w] = ind[w * worker_size: (w + 1) * worker_size]
        else:
            # drop last
            data[w] = ind[w * worker_size:(w + 1) * worker_size]

    return data


def data_partition_perturbation(num_workers, data_set):
    """
    generates a random shuffle of the size of the dataset, and returns the indices partitioned among the workers
    Randomly replacing a data

    :param num_workers:
    :param data_set: type torch.data
    :param separate:
    :return:
    """
    # size = data_set.data.shape[0]
    size = len(data_set)
    ind = list(range(size))
    shuffle(ind)

    # perturbation
    i = random.randint(1, size - 1)
    ind[i] = ind[i - 1]

    # worker_size is the number of samples per worker. The last worker either receives the additional samples
    # or the last samples are dropped.
    worker_size = size // num_workers
    data = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        if w is not num_workers - 1:
            data[w] = ind[w * worker_size: (w + 1) * worker_size]
        else:
            # drop last
            data[w] = ind[w * worker_size:(w + 1) * worker_size]

    return data


def load_data_from_inds(data_set, inds):
    """
    size= len(inds)
    returns data of dim size x dim of one data point, labels of dim size x 1
    :param data_set: type torch.utils.data
    :param inds: list of indices
    :return:
    """
    data = torch.cat([data_set[ind_][0].unsqueeze_(0) for ind_ in inds], 0)
    labels = torch.cat([torch.from_numpy(np.array(data_set[ind_][1])).unsqueeze_(0) for ind_ in inds], 0)

    return data, labels


class ParameterServer:
    def __init__(self, **kwargs):
        # server worker related parameters
        # location, foldername added new as compared to ParameterServer
        self.delay = kwargs['delay']
        self.num_workers = kwargs['num_workers']
        self.workers = []
        self.workers_copy = []
        self.batch_size = kwargs['batch_size']
        self.inf_batch_size = 10000
        self.cuda_ps = kwargs['cuda_ps']
        self.worker_delays = kwargs['worker_delays']

        # data loading
        self.dataset = kwargs['dataset']
        location = './dataset'

        self.train_data, self.test_data = load_dataset(self.dataset, location)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=2)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, num_workers=2)
        # self.partitions = {}
        self.partitions = data_partition(self.num_workers, self.train_data)
        self.partitions_copy = data_partition_perturbation(self.num_workers, self.train_data)

        # choosing model and loss function
        if kwargs['model']:
            self.model = load_model(name=kwargs['model'], dataset=self.dataset)
            self.model_copy = load_model_copy(name=kwargs['model'], dataset=self.dataset)

        # device to use
        if torch.cuda.is_available():
            if kwargs['device']:
                self.device = torch.device(kwargs['device'])
            else:
                self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        if self.cuda_ps:
            self.model.to(self.device)
            self.model_copy.to(self.device)
        # Training related
        self.num_epochs = kwargs['num_epochs']
        self.epoch = 0
        self.lr = kwargs['lr']
        self.lr_schedule = kwargs['lr_schedule']
        self.decay = kwargs['lr_decay']
        # Print related
        self.print_freq = kwargs['print_freq']  # iterations (not epochs)
        self.loss_meter = AverageMeter("loss:", ":.4e")
        self.time_meter = AverageMeter('time:', ":6.3f")
        self.compute_gradients_time_meter = AverageMeter('grads time:', "6.3f")
        self.aggregate_gradients_time_meter = AverageMeter('aggr time:', "6.3f")
        self.progress_meter = ProgressMeter(self.loss_meter, self.time_meter)
        # Logging related
        self.folder_name = kwargs['log_folder']
        self.model_checkpoints_folder = self.folder_name + "model_checkpoints/"
        self.model_copy_checkpoints_folder = self.folder_name + "model_copy_checkpoints/"
        self.logger = Logger(self.folder_name)
        self.delaylogger = Logger(self.folder_name)  # each column corresponds to each worker
        self.grad_norm_logger = Logger(self.folder_name)  # each column corresponds to each worker
        self.loss_logger = Logger(self.folder_name)  # each column corresponds to each worker
        self.lr_logger = Logger(self.folder_name)  # logs lr, norm of weights
        self.tb_writer = kwargs['tb_writer']
        self.eval_freq = kwargs['eval_freq']
        # Functions to be called at init
        self.initiate_workers(kwargs['model'])
        self.initiate_workers_copy(kwargs['model'])
        self.global_step = 0
        self.step_grads = {}
        self.step_grads_copy = {}

    def initiate_workers(self, model):
        for id_ in range(self.num_workers):
            loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                sampler=SubsetSequentialSampler(self.partitions[id_]),
                                num_workers=1)
            self.workers.append(Worker(self, id_, delay=self.worker_delays[id_],
                                       device=self.device, loader=loader,
                                       batch_size=self.batch_size, model=model,
                                       ))

    def initiate_workers_copy(self, model):
        for id_ in range(self.num_workers):
            loader_copy = DataLoader(self.train_data, batch_size=self.batch_size,
                                     sampler=SubsetSequentialSampler(self.partitions_copy[id_]),
                                     num_workers=1)
            self.workers_copy.append(Worker_Copy(self, id_, delay=self.worker_delays[id_],
                                                 device=self.device, loader=loader_copy,
                                                 batch_size=self.batch_size, model=model,
                                                 ))

    def lr_update(self, itr, epoch):
        if self.lr_schedule == 'const':
            self.lr = self.lr
        elif self.lr_schedule == 'decay':
            self.lr = self.lr / (1 + self.decay * self.epoch)
        elif self.lr_schedule == 't':
            # lr = c/t
            self.lr = self.lr / (1 + self.epoch)

    def compute_norm(self, parameters):
        total_norm = 0
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        for p in parameters:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm

    def train(self):
        num_iter_per_epoch = len(self.partitions[0]) // self.batch_size + 1
        self.running_itr = 0
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.loss_meter.reset()
            self.time_meter.reset()
            for itr in range(num_iter_per_epoch):
                start_time = time.time()
                try:
                    self.step()
                except Exception as e:
                    # propagating exception
                    print('exception in step ', e)
                    raise e
                step_time = time.time() - start_time
                self.running_itr += 1
                self.lr_update(self.running_itr, epoch)
                self.time_meter.update(step_time, 1)

                if itr % self.print_freq == self.print_freq - 1:
                    self.progress_meter.print(epoch, itr)

                if self.eval_freq > 0 and self.running_itr % self.eval_freq == 0:
                    self.evaluation(self.running_itr)

            self.progress_meter.print(epoch, itr)
            # Todo: Write a function that tracks progress, tensorboard integration (SummaryWriter)
            if self.eval_freq == 0:
                self.evaluation(self.running_itr)

        # self.save_logs()
        print("training completed")

    def evaluation(self, epoch):

        print("eval train loss")
        start_time = time.time()
        train_loss, train_acc = self.inference(test=False)
        train_time = time.time() - start_time
        print("train time", train_time)

        print("eval test loss")
        start_time = time.time()
        test_loss, test_acc = self.inference(test=True)
        test_time = time.time() - start_time
        print("test time", test_time)
        print("eval gen error", (test_loss - train_loss).item())
        print("eval gen error-abs", np.abs(test_loss - train_loss).item())

        self.logger.update([epoch, train_loss, train_acc, train_time, test_loss, test_acc, test_time])
        self.tb_writer.add_scalar(f'Eval/TrainLoss', train_loss, epoch)
        self.tb_writer.add_scalar(f'Eval/TestLoss', test_loss, epoch)
        self.tb_writer.add_scalar(f'Eval/TrainAcc', train_acc, epoch)
        self.tb_writer.add_scalar(f'Eval/TestAcc', test_acc, epoch)
        self.tb_writer.add_scalar(f'Eval/GenErr', test_loss - train_loss, epoch)
        self.tb_writer.add_scalar(f'Eval/GenErr-ABS', np.abs(test_loss - train_loss), epoch)

        # stability
        w_1 = [param.data for param in self.model.parameters()]
        w_2 = [param.data for param in self.model_copy.parameters()]
        diff = [0 for i in range(len(w_1))]
        for i in range(len(w_1)):
            diff[i] = (w_1[i] - w_2[i])

        stability = self.compute_norm(diff)
        stability_normal = stability / (self.compute_norm(w_1) + self.compute_norm(w_2))
        print("eval stability", stability)
        self.tb_writer.add_scalar(f'Eval/Stability', stability, epoch)
        self.tb_writer.add_scalar(f'Eval/Stability_normalization', stability_normal, epoch)

    def step(self):
        loss = 0
        batch_size = 0
        delays = []
        losses = []
        start_time = time.time()

        for id_ in range(self.num_workers):
            start_time = time.time()
            worker_loss, batch_size_ = self.workers[id_].compute_gradients(self.running_itr)
            self.compute_gradients_time_meter.update(time.time() - start_time, 1)
            start_time = time.time()

            worker_loss_copy, batch_size_copy = self.workers_copy[id_].compute_gradients_copy(self.running_itr)

            if torch.isnan(worker_loss) or torch.isinf(worker_loss).any():
                raise Exception('found Nan/Inf values')
            if torch.isnan(worker_loss_copy) or torch.isinf(worker_loss_copy).any():
                raise Exception('found Nan/Inf values')

            batch_size += batch_size_
            loss += worker_loss * batch_size_
            self.global_step += 1
            losses.append(worker_loss * batch_size)
            self.tb_writer.add_scalar(f'TrainLoss/worker_{[id_]}', worker_loss, self.running_itr)
            self.tb_writer.add_scalar(f'TrainLoss_perturbation/worker_{[id_]}', worker_loss_copy, self.running_itr)
            self.tb_writer.add_scalar(f'Delay/worker_{[id_]}', self.workers[id_].delay, self.running_itr)
            delays.append(self.workers[id_].delay)
            # Todo: Log worker's statistics (run time, loss, accuracy, model parameters at the end of epoch

            if self.running_itr % 50 == 0:
                # grad_diff ERROR server does not compute gradient
                if (self.running_itr - id_) % self.workers[id_].delay == 0:
                    g_1 = [param.grad.data for param in self.workers[id_].model.parameters()]
                    g_2 = [param.grad.data for param in self.workers_copy[id_].model_copy.parameters()]
                    diff_grad = [0 for i in range(len(g_1))]
                    for i in range(len(g_1)):
                        diff_grad[i] = (g_1[i] - g_2[i])
                    grad_diff = self.compute_norm(diff_grad)
                    # print("gradiedt difference", grad_diff)
                    self.tb_writer.add_scalar(f'Grad/grad_diff_norm_{[id_]}', grad_diff, self.running_itr)

                    grad_norm = self.compute_norm(g_1)
                    # print("gradiedt norm", grad_norm)
                    self.tb_writer.add_scalar(f'Grad/grad_norm_{[id_]}', grad_norm, self.running_itr)
                    if self.running_itr % 100 == 0:
                        print("gradiedt difference", grad_diff, "gradiedt norm", grad_norm)
        if self.step_grads != {}:
            ids = self.step_grads.keys()
            self.aggregate_gradients(list(self.step_grads.values()))
            self.step_grads = {}
            for i in ids:
                self.workers[i].get_server_weights()

            self.loss_logger.update(losses)
            self.delaylogger.update(delays)
            self.aggregate_gradients_time_meter.update(time.time() - start_time, 1)
            loss /= batch_size
            self.loss_meter.update(loss.data, batch_size)
            self.tb_writer.add_scalar(f'TrainLoss/avg', loss, self.running_itr)
            self.lr_logger.update([self.lr, self.compute_norm(self.model.parameters())])


        if self.step_grads_copy != {}:
            ids = self.step_grads_copy.keys()
            self.aggregate_gradients_copy(list(self.step_grads_copy.values()))
            self.step_grads_copy = {}

            for i in ids:
                self.workers_copy[i].get_server_weights_copy()

    def get_grad(self, id, grads):
        self.step_grads[id] = grads


    def get_grad_copy(self, id, grads):
        self.step_grads_copy[id] = grads

    def aggregate_gradients(self, grads):
        for id_ in range(1, len(grads)):
            for param1, param2 in zip(grads[0], grads[id_]):
                param1.data += param2.data

        # Assign grad data to model grad data. Update parameters of the model
        for param1, param2 in zip(self.model.parameters(), grads[0]):
            param1.data -= self.lr * param2.data
        self.lr_logger.update([self.lr, self.compute_norm(self.model.parameters())])
        self.tb_writer.add_scalar(f'Lr/', self.lr, self.running_itr)

    def aggregate_gradients_copy(self, grads):
        for id_ in range(1, len(grads)):
            for param1, param2 in zip(grads[0], grads[id_]):
                param1.data += param2.data

        # Assign grad data to model grad data. Update parameters of the model
        for param1, param2 in zip(self.model_copy.parameters(), grads[0]):
            param1.data -= self.lr * param2.data


    def inference(self, test=True):
        self.model.to(self.device)
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            if test:
                for data, labels in self.test_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs, batch_loss = self.model(data, labels)
                    loss += batch_loss * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            else:
                for data, labels in self.train_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs, batch_loss = self.model(data, labels)
                    loss += batch_loss * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        total_loss = loss / total
        total_correct = correct / total
        print('%d images- Accuracy: %6.3f %%, Loss: %6.6f ' % (
            total, 100 * total_correct, total_loss))
        if not self.cuda_ps:
            self.model.cpu()

        return total_loss.cpu(), total_correct

    def save_logs(self, folder_name=None):
        if not folder_name:
            folder_name = self.folder_name
        self.logger.save_log(folder_name + "stats.csv")
        self.delaylogger.save_log(folder_name + "delays.csv")
        self.grad_norm_logger.save_log(folder_name + "norms.csv")
        self.loss_logger.save_log(folder_name + 'losses.csv')
        self.lr_logger.save_log(folder_name + 'lr.csv')


class Worker:
    def __init__(self, *args, **kwargs):
        self.parent = args[0]
        self.id = args[1]
        self.loader = kwargs['loader']
        self.generator = enumerate(self.loader)

        if kwargs['model']:
            self.model = load_model(name=kwargs['model'], dataset=self.parent.dataset)
        else:
            raise NotImplementedError

        self.batch_size = kwargs['batch_size']
        self.delay = kwargs['delay']
        self.epoch = 0
        self.device = kwargs['device']
        self.model.to(self.device)
        # self.model_copy.to(self.device)
        self.data_loading_time_meter = AverageMeter('data time:', ":6.3f")
        self.model_loading_time_meter = AverageMeter('model time:', ":6.3f")
        self.nn_time_meter = AverageMeter('nn time', ":6.3f")
        self.progress_meter = ProgressMeter(self.model_loading_time_meter, self.data_loading_time_meter,
                                            self.nn_time_meter,
                                            prefix='worker:')
        self.get_server_weights()
        self.worker_loss = (torch.tensor(0), 0)


    def get_next_mini_batch(self):
        try:
            _, (data, labels) = next(self.generator)
        except StopIteration:
            self.generator = enumerate(self.loader)
            _, (data, labels) = next(self.generator)

        return data.to(self.device), labels.to(self.device)

    def get_server_weights(self):
        for param_1, param_2 in zip(self.model.parameters(), self.parent.model.parameters()):
            param_1.data = param_2.clone().detach().requires_grad_().data.to(self.device)

    def push_pull(self):
        if self.parent.cuda_ps:
            self.parent.get_grad(self.id, [param.grad.data for param in self.model.parameters()])
        else:
            self.parent.get_grad(self.id, [param.grad.data.cpu() for param in self.model.parameters()])

    def compute_gradients(self, global_step):
        if (global_step - self.id) % self.delay == 0:
            start_time = time.time()
            self.model_loading_time_meter.update(time.time() - start_time, 1)
            start_time = time.time()
            batchdata, batchlabels = self.get_next_mini_batch()  # passes to device already
            self.data_loading_time_meter.update(time.time() - start_time, 1)
            start_time = time.time()

            # origion model
            self.model.zero_grad()
            output, loss = self.model.forward(batchdata, batchlabels)
            loss.backward()
            self.push_pull()

            self.nn_time_meter.update(time.time() - start_time, 1)
            self.worker_loss = (loss.data, batchlabels.size(0))
            return self.worker_loss
        else:
            return self.worker_loss


class Worker_Copy:
    def __init__(self, *args, **kwargs):
        self.parent = args[0]
        self.id = args[1]
        self.loader = kwargs['loader']
        self.generator = enumerate(self.loader)

        if kwargs['model']:
            self.model_copy = load_model_copy(name=kwargs['model'], dataset=self.parent.dataset)
        else:
            raise NotImplementedError

        self.batch_size = kwargs['batch_size']
        self.delay = kwargs['delay']
        self.epoch = 0
        self.device = kwargs['device']
        self.model_copy.to(self.device)
        self.data_loading_time_meter = AverageMeter('data time:', ":6.3f")
        self.model_loading_time_meter = AverageMeter('model time:', ":6.3f")
        self.nn_time_meter = AverageMeter('nn time', ":6.3f")
        self.progress_meter = ProgressMeter(self.model_loading_time_meter, self.data_loading_time_meter,
                                            self.nn_time_meter,
                                            prefix='worker:')
        self.get_server_weights_copy()
        self.worker_loss_copy = (torch.tensor(0), 0)

    def get_next_mini_batch(self):
        try:
            _, (data, labels) = next(self.generator)
        except StopIteration:
            self.generator = enumerate(self.loader)
            _, (data, labels) = next(self.generator)

        return data.to(self.device), labels.to(self.device)

    def get_server_weights_copy(self):
        for param_1, param_2 in zip(self.model_copy.parameters(), self.parent.model_copy.parameters()):
            param_1.data = param_2.clone().detach().requires_grad_().data.to(self.device)

    def push_pull_copy(self):
        if self.parent.cuda_ps:
            self.parent.get_grad_copy(self.id, [param.grad.data for param in self.model_copy.parameters()])
        else:
            self.parent.get_grad_copy(self.id, [param.grad.data.cpu() for param in self.model_copy.parameters()])

    def compute_gradients_copy(self, global_step):
        if (global_step - self.id) % self.delay == 0:
            batchdata, batchlabels = self.get_next_mini_batch()  # passes to device already

            # perturbation model
            self.model_copy.zero_grad()
            output_copy, loss_copy = self.model_copy.forward(batchdata, batchlabels)
            loss_copy.backward()
            self.push_pull_copy()

            self.worker_loss_copy = (loss_copy.data, batchlabels.size(0))
            return self.worker_loss_copy
        else:
            return self.worker_loss_copy


if __name__ == '__main__':
    args = get_args()
    seed_torch(args.seed)
    delay = args.delay
    worker_delays = []
    if args.delay_type == 'random':
        for id_ in range(args.num_workers):
            if id_ == 0:
                worker_delays.append(delay)
            elif id_ == 1:
                worker_delays.append(1)
            else:
                worker_delays.append(np.random.randint(1, 1 + delay))
    elif args.delay_type == 'fixed':
        worker_delays = [delay for _ in range(args.num_workers)]
    else:
        assert False, 'not implemented delay type'

    task_name = f'{args.model}_{args.dataset}/runs_stability_worker-{args.num_workers}_{args.delay_type}-{args.delay}delay_bs-' \
                f'{args.batch_size}_lr-{args.lr}_sch-{args.lr_schedule}_epoch-{args.num_epochs}_seed-{args.seed}/'
    args.tb_writer = SummaryWriter(
        log_dir=args.logdir + task_name)
    args.log_folder = args.logdir + task_name + 'csv/'
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)

    args.device = 'cuda:' + str(args.cuda_device)
    args.worker_delays = worker_delays
    args.avg_delay = sum(worker_delays) / len(worker_delays)
    kwargs = vars(args)

    print('------------------------ arguments ------------------------', flush=True)
    str_list = []
    hparams = {}
    for arg in kwargs:
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        hparams[str(arg)] = str(getattr(args, arg))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print('-------------------- end of arguments ---------------------', flush=True)
    args.tb_writer.add_hparams(hparams, {'job': 1}, run_name=os.path.dirname(os.path.realpath(__file__))
                                                             + os.sep + args.logdir + task_name)
    p = ParameterServer(**kwargs)
    p.train()
