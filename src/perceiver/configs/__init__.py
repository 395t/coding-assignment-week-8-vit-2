from . import perceiver_cifar10

def load_config(config_name):
    return {
        'perceiver_cifar10': perceiver_cifar10.get_config(),
    }[config_name]