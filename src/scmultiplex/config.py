import configparser
import os

from collections import OrderedDict

config = None
spacing = None

def init_config(config_file_path):
    global config
    if config is None:
        config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation() )
        with open(config_file_path, 'r') as config_file_fd:
            config.read_file(config_file_fd)
    return config


def get_round_names(config_file_path):
    config = init_config(config_file_path)
    config_params = {
        'round_names':      ('00BuildExperiment', 'round_names'),
        }
    params = get_workflow_params(config_file_path, config_params)
    round_names = params['round_names'].split(',')
    return round_names


def get_workflow_params(config_file_path, config_params):
    config = init_config(config_file_path)
    ret = OrderedDict()
    for param_key, param_loc in config_params.items():
        param_sec, param_opt = param_loc
        ret[param_key] = config.get(param_sec, param_opt)
    return ret


def compute_workflow_params(config_file_path, compute_param):
    config = init_config(config_file_path)
    ret = OrderedDict()
    for param_key, fp in compute_param.items():
        func, p = fp
        args = []
        for param_sec, param_opt in p:
            args.append(config.get(param_sec, param_opt))
        ret[param_key] = func(*args)
    return ret


def summary_csv_path(save_path, round_folder):
    return os.path.join(save_path, round_folder, 'summary.csv')


def commasplit(cstring):
    return cstring.split(',')


def str2bool(bstring):
    return bstring.lower() in ('true',)


def parse_spacing(spacing):
    return tuple(float(v) for v in commasplit(spacing))


def spacing_anisotropy_tuple(spacing):
    """Return the tuple of pixel spacings normalized to x-pixel spacing
    """
    return tuple(i / spacing[-1] for i in spacing)


def spacing_anisotropy_scalar(spacing):
    """Return the z-anisotropy scalar (z norm to x-pixel spacing)
    """
    if len(spacing) == 3:
        return list(spacing_anisotropy_tuple(spacing))[0]
    else:
        raise ValueError('expect 3-dimensional pixel spacing for z-anisotropy calculation')


def spacing_to2d(spacing):
    if len(spacing) == 3:
        return spacing[1:]
    return spacing
