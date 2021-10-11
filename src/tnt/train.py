# -*- coding: utf-8 -*-
'''

Adapted from : https://github.com/kentaroy47/vision-transformers-cifar10

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import timm
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from models.vit import ViT
from models.tnt import TNT
from utils import progress_bar
from randomaug import RandAugment

# parsers
parser = argparse.ArgumentParser(description='PyTorch Vision Transformer Training')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='use randomaug')
parser.add_argument('--amp', action='store_true', help='enable AMP training')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='tnt')
parser.add_argument('--bs', default='64')
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch', default='4', type=int)
parser.add_argument('--pixel', default='2', type=int)
parser.add_argument('--convkernel', default='8', type=int)
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')

args = parser.parse_args()

watermark = f"{args.dataset}_{args.net}_lr{args.lr}_bs{args.bs}_nepochs{args.n_epochs}"
if args.net == 'tnt':
    watermark += f"_patch{args.patch}_pixel{args.pixel}"
if args.amp:
    watermark += "_useamp"
print(f'> experiment: {watermark}')

import wandb
wandb.init(project=f"{args.dataset}", name='_'.join(watermark.split('_')[1:]))
wandb.config.update(args)

if args.aug:
    import albumentations
bs = int(args.bs)
use_amp = args.amp

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
if args.net == "vit_timm":
    size = 384
elif args.net == "tnt_timm":
    size = 224
elif args.dataset == 'cifar10':
    size = 32
elif args.dataset == 'tiny-imagenet':
    size = 64
elif args.dataset == 'stl10':
    size = 96
    
print(f'> using image resize: ({size})')
transform_train = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomCrop(size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if args.aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))
    
def fix_tin_val_folder(root):
    val_img_dir = f'{root}/images'
    if os.path.exists(val_img_dir):
        data = open(f'{root}/val_annotations.txt', 'r').readlines()
        val_img_dict = {}
        for line in data:
            words = line.split('\t')
            val_img_dict[words[0]] = words[1]

        for img, folder in val_img_dict.items():
            new_dir = f'{root}/{folder}'
            os.makedirs(new_dir, exist_ok=True)
            old_file = f'{val_img_dir}/{img}'
            if os.path.exists(olf_file):
                os.rename(old_file, f'{new_dir}/{img}')

        import shutil
        shutil.rmtree(val_img_dir)

print('> Preparing data..')
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    n_classes = 10
elif args.dataset == 'tiny-imagenet':
    fix_tin_val_folder('./data/tiny-imagenet-200/val')
    trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_test)
    n_classes = 200
elif args.dataset == 'stl10':
    trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)
    n_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)


# Model
print('> Building model..')
if args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = n_classes,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    net = timm.create_model("vit_large_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, n_classes)
elif args.net=="tnt":
    net = TNT(
        image_size = size,
        patch_size = args.patch,
        pixel_size = args.pixel,
        num_classes = n_classes,
        patch_dim = 512,        # dimension of patch token
        pixel_dim = 24,
        depth = 6,
        heads = 8,
        attn_dropout = 0.1,
        ff_dropout = 0.1
    )
elif args.net=="tnt_timm":
    net = timm.create_model("tnt_s_patch16_224", pretrained=True)
    net.head = nn.Linear(net.head.in_features, n_classes)
    
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net) # make parallel
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine or reduce LR on Plateau scheduling
if not args.cos:
    from torch.optim import lr_scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

if args.cos:
    wandb.config.scheduler = "cosine"
else:
    wandb.config.scheduler = "ReduceLROnPlateau"

##### Training
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    acc = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), acc, correct, total))
    return train_loss/(batch_idx+1), acc

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Update scheduler
    if not args.cos:
        scheduler.step(test_loss)
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
              "scaler": scaler.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss/(batch_idx+1), acc

wandb.watch(net)
log = []
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = test(epoch)
    
    if args.cos:
        scheduler.step(epoch-1)
    
    # Log training..
    log.append({'epoch': epoch, 'trn_loss': train_loss, 'trn_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
    wandb.log({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"], "epoch_time": time.time()-start})

# writeout
import json
json.dump(log, open(f'logs/{watermark}.json', 'w'))
wandb.save("wandb_{}.h5".format(args.net))
    
