from . import perceiver_cifar10
from . import perceiver_cifar10_large
from . import perceiver_cifar10_xl
from . import perceiver_stl10
from . import perceiver_stl10_large
from . import perceiver_stl10_xl
from . import perceiver_tinyimagenet_large
from . import perceiver_tinyimagenet_large_noaug
from . import perceiver_tinyimagenet_xl

def load_config(config_name):
    return {
        'perceiver_cifar10': perceiver_cifar10.get_config(),
        'perceiver_cifar10_large': perceiver_cifar10_large.get_config(),
        'perceiver_cifar10_xl': perceiver_cifar10_xl.get_config(),
        'perceiver_stl10': perceiver_stl10.get_config(),
        'perceiver_stl10_large': perceiver_stl10_large.get_config(),
        'perceiver_stl10_xl': perceiver_stl10_xl.get_config(),
        'perceiver_tinyimagenet_large': perceiver_tinyimagenet_large.get_config(),
        'perceiver_tinyimagenet_large_noaug': perceiver_tinyimagenet_large_noaug.get_config(),
        'perceiver_tinyimagenet_xl': perceiver_tinyimagenet_xl.get_config(),
    }[config_name]