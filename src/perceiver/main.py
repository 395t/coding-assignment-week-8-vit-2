import sys

from configs import load_config
from lib.train_utils import train


def main():
    config = load_config(sys.argv[1])
    train(config)


if __name__ == '__main__':
    main()