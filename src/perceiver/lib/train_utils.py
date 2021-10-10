import os
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torchvision import datasets
from torchvision import transforms as T
from perceiver_pytorch import Perceiver, PerceiverIO


def pp(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def get_data_transforms(dataset, random_aug):
    transforms = []
    if dataset == 'cifar10':
        if random_aug:
            random_transforms = [
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.RandomAffine(degrees=15, translate=(0.2, 0.2),
                                scale=(0.8, 1.2), shear=15,
                                resample=Image.BILINEAR),
                T.RandomEqualize(),
                T.RandomAutocontrast(),
                T.RandomAdjustSharpness(sharpness_factor=2),
            ]
            transforms.extend([
                T.RandomHorizontalFlip(),
                T.RandomChoice(random_transforms)])
        transforms.extend([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset == 'stl10':
        if random_aug:
            random_transforms = [
                T.ColorJitter(0.2, 0.2, 0.2, 0.2),
                T.transforms.RandomResizedCrop(96, scale=(0.5, 1)),
                T.RandomAffine(degrees=15, translate=(0.2, 0.2),
                               scale=(0.8, 1.2), shear=15,
                               resample=Image.BILINEAR),
                T.RandomEqualize(),
                T.RandomAutocontrast(),
                T.RandomAdjustSharpness(sharpness_factor=2),
            ]
            transforms.extend([
                T.RandomHorizontalFlip(),
                T.RandomChoice(random_transforms)])
        transforms.extend([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    elif dataset == 'tinyimagenet':
        transforms.extend([
            T.ToTensor(),
        ])

    return T.Compose(transforms)

def get_datasets(config, data_root):
    assert config.dataset in ('stl10', 'cifar10', 'tinyimagenet')
    data_root = os.path.abspath(os.path.expanduser(data_root))
    root_dir = os.path.join(data_root, config.dataset)
    train_transforms = get_data_transforms(config.dataset, config.train.random_aug)
    eval_transforms = get_data_transforms(config.dataset, False)

    if config.dataset == 'stl10':
        train_dataset = datasets.STL10(root=root_dir, split='train', transform=train_transforms, download=True)
        test_dataset = datasets.STL10(root=root_dir, split='test', transform=eval_transforms, download=True)
        train_dataset_eval = datasets.STL10(root=root_dir, split='train', transform=eval_transforms, download=True)
    elif config.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=root_dir, train=True, transform=train_transforms, download=True)
        test_dataset = datasets.CIFAR10(root=root_dir, train=False, transform=eval_transforms, download=True)
        train_dataset_eval = datasets.CIFAR10(root=root_dir, train=True, transform=eval_transforms, download=True)
    elif config.dataset == 'tinyimagenet':
        train_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=train_transforms)
        test_dataset = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=eval_transforms)
        train_dataset_eval = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=eval_transforms)

    return train_dataset, test_dataset, train_dataset_eval

def get_model(config):
    model_cls = {'perceiver': Perceiver,
                 'perceiver_io': PerceiverIO
                 }[config.model_type]
    model = model_cls(**config.model_args)
    if config.cuda:
        model.cuda()
    return model

def get_optimizer(model, config):
    optimizer_cls = {'adamw': optim.AdamW,
                     'rmsprop': optim.RMSprop
                     }[config.optimizer_type]
    optimizer = optimizer_cls(model.parameters(), **config.optimizer_args)
    return optimizer

def train(config):
    train_dataset, test_dataset, train_dataset_eval = get_datasets(config, 'data')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
                                               shuffle=True, pin_memory=config.cuda, drop_last=False)
    model = get_model(config)
    opt = get_optimizer(model, config)

    total_steps = 0
    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'minibatch_losses': [],
    }

    num_params = sum(np.prod(p.shape).item() for p in model.parameters())
    pp(f'Created {config.model_type} model with {num_params} parameters.')
    results['num_params'] = num_params

    for epoch in range(1, config.train.num_epochs+1):
        pp(f'Starting epoch {epoch}')

        for _, (x, y) in enumerate(train_loader):
            total_steps += 1 

            if config.cuda:
                x, y = x.cuda(), y.cuda()
            x = x.permute(0, 2, 3, 1)
            y_hat = model(x)

            opt.zero_grad()
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            opt.step()

            if total_steps % 10 == 0:
                pp(f'\repoch {epoch} step {total_steps} loss {loss.item():.4f}', end='')
                results['minibatch_losses'].append(loss.item())
        pp(f'\nepoch {epoch} step {total_steps} loss {loss.item():.4f}')

        ep_train_loss, ep_train_acc = evaluate(model, train_dataset_eval, config)
        pp(f'epoch {epoch} Train accuracy: {ep_train_acc:.4f} loss: {ep_train_loss:.4f}')
        ep_test_loss, ep_test_acc = evaluate(model, test_dataset, config)
        pp(f'epoch {epoch} Test accuracy: {ep_test_acc:.4f} loss: {ep_test_loss:.4f}')
        pp()

        results['train_loss'].append(ep_train_loss)
        results['train_acc'].append(ep_train_acc)
        results['test_loss'].append(ep_test_loss)
        results['test_acc'].append(ep_test_acc)

        with open(config.result_file, 'w') as f:
            yaml.dump(results, f, indent=4)


@torch.no_grad()
def evaluate(model, dataset, config):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.train.batch_size,
                                              shuffle=False, pin_memory=config.cuda,
                                              drop_last=False, num_workers=4)
    model.eval()
    losses, preds = [], []
    for x, y in data_loader:
        if config.cuda:
            x, y = x.cuda(), y.cuda()
        x = x.permute(0, 2, 3, 1)

        y_hat = model(x)
        y_pred = y_hat.argmax(axis=-1)
        loss = F.cross_entropy(y_hat, y, reduction='none')
        pred = (y_pred == y).long()
        losses.append(loss)
        preds.append(pred)

    avg_loss = torch.cat(losses).cpu().mean().item()
    accuracy = torch.cat(preds).cpu().float().mean().item()

    return avg_loss, accuracy
