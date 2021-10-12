import functools
import glob
import os
import yaml

# For plots
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt; plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{bm}']
import seaborn as sns; sns.set(context='paper', style='whitegrid', font_scale=1.5, font='Times New Roman')

import fire
import numpy as np
import torch


_MARKERS = ['o', '^', 's', 'x', '+']


def plot(fn_prefix, results, smooth=5, skip_first=None, set_kwargs=None, legend_loc='best'):
    fig = plt.figure(1, figsize=(6, 4))
    ax = fig.add_subplot()
    max_len = min(len(v) for v in results.values())
    for i, (label, values) in enumerate(results.items()):
        x = np.arange(1, len(values)+1)[:max_len]
        y = np.array(values)[:max_len]
        assert x.shape == y.shape
        if smooth is not None:
            y[smooth-1:] = np.convolve(y, np.ones(smooth), 'valid') / smooth
        if skip_first is not None:
            x = x[skip_first:]
            y = y[skip_first:]
        ax.plot(x, y, label=label)
    ax.grid(True, axis='y', linestyle='--', linewidth=1)
    ax.grid(True, axis='x', linestyle='--', linewidth=1)
    if set_kwargs is not None:
        ax.set(**set_kwargs)
    ax.legend(loc=legend_loc)
    fig.savefig(fn_prefix+'.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(fn_prefix+'.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_curves(*, out, files, labels, split, dataset):
    assert split in ('train', 'test')
    split_up = split[0].upper() + split[1:]
    files, labels = files.split(','), labels.split(',')
    assert len(files) == len(labels)
    loss_results = {}
    acc_results = {}
    for fn, label in zip(files, labels):
        with open(fn, 'r') as fp:
            dd = yaml.load(fp)
        loss_results[label] = dd[f'{split}_loss']
        acc_results[label] = dd[f'{split}_acc']

    plot(out + '_loss', loss_results, legend_loc='upper right', set_kwargs={
        'title': f'{dataset}: {split_up} Loss',
        'xlabel': 'Epoch',
        'ylabel': f'{split_up} Loss',
    })

    plot(out + '_acc', acc_results, legend_loc='lower right', set_kwargs={
        'title': f'{dataset}: {split_up} Accuracy',
        'xlabel': 'Epoch',
        'ylabel': f'{split_up} Accuracy',
    })


def best_vals(*files):
    print()
    for fn in files:
        with open(fn, 'r') as fp:
            dd = yaml.load(fp)
        best_train_loss = min(dd['train_loss'])
        best_train_acc = max(dd['train_acc']) * 100.
        best_test_loss = min(dd['test_loss'])
        best_test_acc = max(dd['test_acc']) * 100.
        num_params = dd['num_params']
        print(f'Results for: {fn}')
        print(f'  - best train acc: {best_train_acc:2f}')
        print(f'  - best test acc: {best_test_acc:2f}')
        print(f'  - best train loss: {best_train_loss:.4f}')
        print(f'  - best test loss: {best_test_loss:.4f}')
        print(f'  - parameter count: {num_params}')
        print()


if __name__ == '__main__':
    fire.Fire({
        'plot_train': functools.partial(plot_curves, split='train'),
        'plot_test': functools.partial(plot_curves, split='test'),
        'best_vals':  best_vals,
    })