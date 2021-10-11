# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 1e-2
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 1e-3
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 1e-4
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 1e-5

# CUDA_VISIBLE_DEVICES=1,2,3 python tiny-imagenet.py --lr 1e-3 --n_epochs 10
# CUDA_VISIBLE_DEVICES=1,2,3 python tiny-imagenet.py --lr 3e-4 --n_epochs 10
# CUDA_VISIBLE_DEVICES=1,2,3 python tiny-imagenet.py --lr 1e-4 --n_epochs 10
# CUDA_VISIBLE_DEVICES=1,2,3 python tiny-imagenet.py --lr 1e-5 --n_epochs 10

# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4 --patch 2
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4 --patch 8
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4 --patch 16
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4 --patch 8 --pixel 1
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4 --patch 8 --pixel 4
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4 --patch 8 --pixel 8
# CUDA_VISIBLE_DEVICES=1,2,3 python train_cifar10.py --lr 3e-4 --patch 16 --pixel 8

# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset cifar10 --lr 1e-3 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset cifar10 --lr 3e-4 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset cifar10 --lr 1e-4 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset cifar10 --lr 1e-5 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset cifar10 --lr 3e-6 --bs 64 --n_epochs 5

# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset stl10 --lr 1e-3 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset stl10 --lr 3e-4 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset stl10 --lr 1e-4 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset stl10 --lr 1e-5 --bs 64 --n_epochs 5
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --net tnt_timm --dataset stl10 --lr 3e-6 --bs 64 --n_epochs 5

python train.py --net tnt --dataset tiny-imagenet --lr 1e-3 --bs 128 --n_epochs 10
python train.py --net tnt --dataset tiny-imagenet --lr 1e-4 --bs 128 --n_epochs 10
python train.py --net tnt --dataset tiny-imagenet --lr 1e-5 --bs 128 --n_epochs 10
python train.py --net tnt --dataset tiny-imagenet --lr 1e-6 --bs 128 --n_epochs 10