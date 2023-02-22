import configparser

from collections import OrderedDict

config = None

def init_config(config_file_path):
    global config
    if config is None:
        config = configparser.ConfigParser(interpolation = configparser.ExtendedInterpolation() )
        config.read(config_file_path)
    return config

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

def summary_csv_path(folder_path):
    return os.path.join(folder_path, 'summary.csv')
