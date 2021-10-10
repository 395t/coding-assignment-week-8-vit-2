from . import perceiver_cifar10
from . import perceiver_stl10

def load_config(config_name):
    return {
        'perceiver_cifar10': perceiver_cifar10.get_config(),
        'perceiver_stl10': perceiver_stl10.get_config(),
    }[config_name]