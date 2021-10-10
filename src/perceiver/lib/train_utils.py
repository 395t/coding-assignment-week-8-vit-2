import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from perceiver_pytorch import Perceiver, PerceiverIO

def pp(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def get_datasets(dataset, data_root):
    assert dataset in ('stl10', 'cifar10', 'tinyimagenet')
    data_root = os.path.abspath(os.path.expanduser(data_root))
    root_dir = os.path.join(data_root, dataset)
    if dataset == 'stl10':
        train_dataset = datasets.STL10(root=root_dir, split='train', transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.STL10(root=root_dir, split='test', transform=transforms.ToTensor(), download=True)
    elif dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transforms.ToTensor())
    elif dataset == 'tinyimagenet':
        train_dataset = datasets.ImageFolder(os.path.join(root_dir, 'train'), transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.ImageFolder(os.path.join(root_dir, 'test'), transform=transforms.Compose([transforms.ToTensor()]))

    return train_dataset, test_dataset

def get_model(config):
    model_cls = {'perceiver': Perceiver,
                 'perceiver_io': PerceiverIO
                 }[config.model_type]
    model = model_cls(**config.model_args)
    param_count = sum(np.prod(p.shape).item() for p in model.parameters())
    pp(f'Created {config.model_type} model with {param_count} parameters.')
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
    train_dataset, test_dataset = get_datasets(config.dataset, 'data')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size,
                                               shuffle=True, pin_memory=config.cuda, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.train.batch_size,
                                              shuffle=False, pin_memory=config.cuda, drop_last=False)
    model = get_model(config)
    opt = get_optimizer(model, config)

    total_steps = 0
    for epoch in range(1, config.train.num_epochs+1):
        pp(f'Starting epoch {epoch}')

        for _, (x, y) in enumerate(train_loader):
            total_steps += 1 

            if config.cuda:
                x, y = x.cuda(), y.cuda()
            x = x.permute(0, 2, 3, 1) * 2 - 1
            y_hat = model(x)

            opt.zero_grad()
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            opt.step()

            if total_steps % 100 == 0:
                pp(f'epoch {epoch} step {total_steps} loss {loss.item():.4f}')
        pp(f'epoch {epoch} step {total_steps} loss {loss.item():.4f}')

        ep_train_loss, ep_train_acc = evaluate(model, train_dataset, config)
        pp(f'epoch {epoch} Train accuracy: {ep_train_acc:.4f} loss: {ep_train_loss:.4f}')
        ep_test_loss, ep_test_acc = evaluate(model, test_dataset, config)
        pp(f'epoch {epoch} Test accuracy: {ep_test_acc:.4f} loss: {ep_test_loss:.4f}')
        pp()


@torch.no_grad()
def evaluate(model, dataset, config):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=config.train.batch_size,
                                              shuffle=False, pin_memory=config.cuda,
                                              drop_last=False)
    model.eval()
    losses, preds = [], []
    for x, y in data_loader:
        if config.cuda:
            x, y = x.cuda(), y.cuda()
        x = x.permute(0, 2, 3, 1) * 2 - 1

        y_hat = model(x)
        y_pred = y_hat.argmax(axis=-1)
        loss = F.cross_entropy(y_hat, y, reduction='none')
        pred = (y_pred == y).long()
        losses.append(loss)
        preds.append(pred)

    avg_loss = torch.cat(losses).cpu().mean().item()
    accuracy = torch.cat(preds).cpu().float().mean().item()

    return avg_loss, accuracy
