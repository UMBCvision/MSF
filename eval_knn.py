import builtins
from collections import Counter, OrderedDict
from random import shuffle
import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F
import numpy as np
import faiss

from tools import *
from models.resnet import resnet18, resnet50
from models.alexnet import AlexNet as alexnet
from models.mobilenet import MobileNetV2 as mobilenet
# from models.resnet_swav import resnet50w5, resnet50 as swav_resnet50
# from models.resnet_byol import resnet50 as byol_resnet50
# from models.resnet_gn_ws import l_resnet18, l_resnet50
from eval_linear import load_weights
# from file_dataset import FileDataset


parser = argparse.ArgumentParser(description='NN evaluation')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet',
                    choices=['imagenet', 'imagenet100', 'imagenet-lt'],
                    help='use full or subset of the dataset')
parser.add_argument('-j', '--workers', default=8, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('-a', '--arch', type=str, default='alexnet',
                        choices=['alexnet' , 'resnet18' , 'resnet50', 'mobilenet' ,
                                 'l_resnet18', 'l_resnet50', 
                                 'two_resnet50', 'one_resnet50', 
                                 'moco_alexnet' , 'moco_resnet18' , 'moco_resnet50', 'moco_mobilenet', 'resnet50w5', 'teacher_resnet18',  'teacher_resnet50',
                                 'sup_alexnet' , 'sup_resnet18' , 'sup_resnet50', 'sup_mobilenet', 'pt_alexnet', 'swav_resnet50', 'byol_resnet50'])
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=90, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--save', default='./output/cluster_alignment_1', type=str,
                    help='experiment output directory')
parser.add_argument('--weights', dest='weights', type=str,
                    help='pre-trained model weights')
parser.add_argument('--load_cache', action='store_true',
                    help='should the features be recomputed or loaded from the cache')
parser.add_argument('-k', default=1, type=int, help='k in kNN')
parser.add_argument('--debug', action='store_true', help='whether in debug mode or not')

TEMP = 0.04


def main():
    global logger

    args = parser.parse_args()
    makedirs(args.save)

    if not args.debug:
        logger = get_logger(
            logpath=os.path.join(args.save, 'logs'),
            # logpath=os.path.join(args.save, 'knn.logs'),
            filepath=os.path.abspath(__file__)
        )
        def print_pass(*args):
            logger.info(*args)
        builtins.print = print_pass

    print(args)

    main_worker(args)


def get_model(args):

    model = None
    if args.arch == 'alexnet' :
        model = alexnet()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {('module.'+k): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(model)
        print(msg)

    elif args.arch == 'pt_alexnet' :
        model = models.alexnet(num_classes=16000)
        checkpoint = torch.load(args.weights)
        sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=True)
        classif = list(model.classifier.children())[:5]
        model.classifier = nn.Sequential(*classif)
        model = torch.nn.DataParallel(model).cuda()
        print(model)
        print(msg)

    elif args.arch == 'resnet18' :
        model = resnet18()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {('module.'+k): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(model)
        print(msg)

    elif args.arch == 'one_resnet50' :
        model = resnet50()
        model.fc = nn.Sequential()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'projection' not in k}
        sd = {k: v for k, v in sd.items() if 'prediction' not in k}
        sd = {k: v for k, v in sd.items() if 'pred_' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_two' not in k}
        sd = {k.replace('encoder_one.', ''): v for k, v in sd.items()}
        sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'two_resnet50' :
        model = resnet50()
        model.fc = nn.Sequential()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'projection' not in k}
        sd = {k: v for k, v in sd.items() if 'prediction' not in k}
        sd = {k: v for k, v in sd.items() if 'pred_' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_one' not in k}
        sd = {k.replace('encoder_two.', ''): v for k, v in sd.items()}
        sd = {k.replace('backbone.', ''): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'l_resnet18' :
        model = l_resnet18()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {('module.'+k): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(model)
        print(msg)

    elif 'teacher_' in args.arch:
        if 'resnet18' in args.arch:
            model = resnet18()
        elif 'resnet50' in args.arch:
            model = resnet50()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']

        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'predict_q' not in k}
        sd = {k: v for k, v in sd.items() if 'queue' not in k}
        new_sd = {}
        for key in sd.keys():
            if 'encoder_k' in key and 'running_' not in key:
                new_sd['module.' + key.replace('encoder_k.', '')] = sd[key]
            if 'encoder_q' in key and 'running_' in key:
                new_sd['module.' + key.replace('encoder_q.', '')] = sd[key]
        msg = model.load_state_dict(new_sd, strict=True)
        print(model)
        print(msg)

    elif args.arch == 'mobilenet' :
        model = mobilenet()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        msg = model.load_state_dict(checkpoint['model'] , strict=False)
        print(model)
        print(msg)

    elif args.arch == 'resnet50' :
        model = resnet50()
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        sd = {('module.'+k): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=False)
        print(model)
        print(msg)

    elif args.arch == 'byol_resnet50' :
        model = byol_resnet50()
        model.fc = nn.Sequential()
        checkpoint = torch.load(args.weights)
        if 'model' in checkpoint:
            sd = checkpoint['model']
        else:
            sd = checkpoint['state_dict']
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        sd = {k: v for k, v in sd.items() if 'fc' not in k}
        sd = {k: v for k, v in sd.items() if 'encoder_k' not in k}
        sd = {k: v for k, v in sd.items() if 'predict_q' not in k}
        sd = {k: v for k, v in sd.items() if 'queue' not in k}
        sd = {k.replace('encoder_q.', ''): v for k, v in sd.items()}
        msg = model.load_state_dict(sd, strict=True)
        print(model)
        print(msg)
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'moco_alexnet' :
        model = alexnet()
        model.fc = nn.Sequential()
        model = nn.Sequential(OrderedDict([('encoder_q', model)]))
        model = model.cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)

    elif args.arch == 'moco_resnet18' :
        model = resnet18().cuda()
        model = nn.Sequential(OrderedDict([('encoder_q' , model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        msg = model.load_state_dict(checkpoint['state_dict'] , strict=False)
        print(msg)
        # model.module.encoder_q.fc = nn.Sequential()

    elif args.arch == 'moco_mobilenet' :
        model = mobilenet()
        model.fc = nn.Sequential()
        model = nn.Sequential(OrderedDict([('encoder_q', model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    elif args.arch == 'moco_resnet50' :
        model = resnet50().cuda()
        model = nn.Sequential(OrderedDict([('encoder_q' , model)]))
        model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['state_dict'] , strict=False)
        model.module.encoder_q.fc = nn.Sequential()

    elif args.arch == 'resnet50w5':
        model = resnet50w5()
        model.l2norm = None
        load_weights(model, args.weights)
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'swav_resnet50':
        model = swav_resnet50()
        model.l2norm = None
        load_weights(model, args.weights)
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'sup_alexnet' :
        # model = models.alexnet(pretrained=True)
        # modules = list(model.children())[:-1]
        # classifier_modules = list(model.classifier.children())[:-1]
        # modules.append(Flatten())
        # modules.append(nn.Sequential(*classifier_modules))
        # model = nn.Sequential(*modules)
        # model = model.cuda()
        ####### modified #######
        model = models.alexnet(pretrained=False)
        model.classifier = nn.Sequential()
        modules = list(model.children())
        modules.append(nn.Flatten())
        model = nn.Sequential(*modules)
        model = model.cuda()

    elif args.arch == 'sup_resnet18' :
        model = models.resnet18(pretrained=True)
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'sup_mobilenet' :
        model = models.mobilenet_v2(pretrained=True)
        model.classifier = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    elif args.arch == 'sup_resnet50' :
        model = models.resnet50(pretrained=True)
        model.fc = nn.Sequential()
        model = torch.nn.DataParallel(model).cuda()

    for param in model.parameters():
        param.requires_grad = False

    return model


class ImageFolderEx(datasets.ImageFolder) :
    def __getitem__(self, index):
        sample, target = super(ImageFolderEx, self).__getitem__(index)
        return index, sample, target


# class FileDatasetEx(FileDataset) :
#     def __getitem__(self, index):
#         sample, target = super(FileDatasetEx, self).__getitem__(index)
#         return index, sample, target


def get_loaders(dataset_dir, bs, workers, dataset='imagenet'):
    traindir = os.path.join(dataset_dir, 'train')
    valdir = os.path.join(dataset_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolderEx(traindir, augmentation)
    val_dataset = ImageFolderEx(valdir, augmentation)

    if dataset == 'imagenet100':
        subset_classes(train_dataset, num_classes=100)
        subset_classes(val_dataset, num_classes=100)


    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    return train_loader, val_loader


def main_worker(args):

    start = time.time()
    # Get train/val loader 
    # ---------------------------------------------------------------
    train_loader, val_loader = get_loaders(args.data, args.batch_size, args.workers, args.dataset)

    # Create and load the model
    # If you want to evaluate your model, modify this part and load your model
    # ------------------------------------------------------------------------
    # MODIFY 'get_model' TO EVALUATE YOUR MODEL
    model = get_model(args)

    # ------------------------------------------------------------------------
    # Forward training samples throw the model and cache feats
    # ------------------------------------------------------------------------
    cudnn.benchmark = True

    cached_feats = '%s/train_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        print('load train feats from cache =>')
        train_feats, train_labels, train_inds = torch.load(cached_feats)
    else:
        print('get train feats =>')
        train_feats, train_labels, train_inds = get_feats(train_loader, model, args.print_freq)
        torch.save((train_feats, train_labels, train_inds), cached_feats)

    cached_feats = '%s/val_feats.pth.tar' % args.save
    if args.load_cache and os.path.exists(cached_feats):
        print('load val feats from cache =>')
        val_feats, val_labels, val_inds = torch.load(cached_feats)
    else:
        print('get val feats =>')
        val_feats, val_labels, val_inds = get_feats(val_loader, model, args.print_freq)
        torch.save((val_feats, val_labels, val_inds), cached_feats)

    # ------------------------------------------------------------------------
    # Calculate NN accuracy on validation set
    # ------------------------------------------------------------------------

    # train_feats = l2_normalize(train_feats)
    # val_feats = l2_normalize(val_feats)

    # mean = torch.mean(train_feats, dim=0)
    # std = torch.std(train_feats, dim=0)

    # stdmean = std.mean()
    # train_feats = train_feats / stdmean
    # val_feats = val_feats / stdmean

    # train_feats = train_feats / std
    # val_feats = val_feats / std

    # train_feats = (train_feats - mean) / std
    # val_feats = (val_feats - mean) / std

    # train_feats = train_feats - mean
    # val_feats = val_feats - mean

    # train_feats = train_feats / TEMP
    # val_feats = val_feats / TEMP

    train_feats = l2_normalize(train_feats)
    val_feats = l2_normalize(val_feats)

    for k in [1,20]:
        print(k)
        acc = faiss_knn(train_feats, train_labels, val_feats, val_labels, k)
        nn_time = time.time() - start
        print('=> time : {:.2f}s'.format(nn_time))
        print(' * Acc {:.2f}'.format(acc))


def l2_normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)


def faiss_knn(feats_train, targets_train, feats_val, targets_val, k):
    feats_train = feats_train.numpy()
    targets_train = targets_train.numpy()
    feats_val = feats_val.numpy()
    targets_val = targets_val.numpy()

    d = feats_train.shape[-1]

    index = faiss.IndexFlatL2(d)  # build the index
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(feats_train)

    D, I = gpu_index.search(feats_val, k)

    pred = np.zeros(I.shape[0], dtype=np.int)
    conf_mat = np.zeros((1000, 1000), dtype=np.int)
    for i in range(I.shape[0]):
        votes = list(Counter(targets_train[I[i]]).items())
        shuffle(votes)
        pred[i] = max(votes, key=lambda x: x[1])[0]
        conf_mat[targets_val[i], pred[i]] += 1

    acc = 100.0 * (pred == targets_val).mean()
    assert acc == (100.0 * (np.trace(conf_mat) / np.sum(conf_mat)))

    # per_cat_acc = 100.0 * (np.diag(conf_mat) / np.sum(conf_mat, axis=1))
    # sparse_cats = [58, 155, 356, 747, 865, 234, 268, 384, 385, 491, 498, 538, 646, 650, 726, 860, 887, 15, 170, 231]
    # s = ' '.join('{}'.format(c) for c in sparse_cats)
    # print('==> cats: {}'.format(s))
    # s = ' '.join('{:.1f}'.format(a) for a in per_cat_acc[sparse_cats])
    # print('==> acc/cat: {}'.format(s))
    # print('==> mean acc: {}'.format(per_cat_acc[sparse_cats].mean()))

    return acc


def get_feats(loader, model, print_freq):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    feats, labels, indices, ptr = None, None, None, 0

    with torch.no_grad():
        end = time.time()
        for i, (index, images, target) in enumerate(loader):
            images = images.cuda(non_blocking=True)
            cur_targets = target.cpu()
            cur_feats = model(images).cpu()
            cur_indices = index.cpu()

            B, D = cur_feats.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                feats = torch.zeros((len(loader.dataset), D)).float()
                labels = torch.zeros(len(loader.dataset)).long()
                indices = torch.zeros(len(loader.dataset)).long()

            feats.index_copy_(0, inds, cur_feats)
            labels.index_copy_(0, inds, cur_targets)
            indices.index_copy_(0, inds, cur_indices)
            ptr += B

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print(progress.display(i))

    return feats, labels, indices


def subset_classes(dataset, num_classes=10):
    np.random.seed(1234)
    all_classes = sorted(dataset.class_to_idx.items(), key=lambda x: x[1])
    subset_classes = [all_classes[i] for i in np.random.permutation(len(all_classes))[:num_classes]]
    subset_classes = sorted(subset_classes, key=lambda x: x[1])
    dataset.classes_to_idx = {c: i for i, (c, _) in enumerate(subset_classes)}
    dataset.classes = [c for c, _ in subset_classes]
    orig_to_new_inds = {orig_ind: new_ind for new_ind, (_, orig_ind) in enumerate(subset_classes)}
    dataset.samples = [(p, orig_to_new_inds[i]) for p, i in dataset.samples if i in orig_to_new_inds]



if __name__ == '__main__':
    main()

