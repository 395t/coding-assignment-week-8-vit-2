from configs import load_config
from lib.train_utils import train


def main():
    config = load_config('perceiver_cifar10')
    train(config)


if __name__ == '__main__':
    main()