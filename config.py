import yaml
import os
from attack_method import *
import torch
import os


def get_bound(model_name):

    if model_name not in ['inception', 'inceptionv4', 'inceptionresv2', 'inceptionv3']:
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                input_size = [3, 224, 224]
    else:
                mean = (0.5, 0.5, 0.5)
                std = (0.5, 0.5, 0.5)
                input_size = [3, 299, 299]


    channel1_low_bound = (0 - mean[0]) / std[0]
    channel1_hign_bound = (1 - mean[0]) / std[0]
    channel2_low_bound = (0 - mean[1]) / std[1]
    channel2_hign_bound = (1 - mean[1]) / std[1]
    channel3_low_bound = (0 - mean[2]) / std[2]
    channel3_hign_bound = (1 - mean[2]) / std[2]
    bound = [channel1_low_bound, channel1_hign_bound, channel2_low_bound, channel2_hign_bound, channel3_low_bound, channel3_hign_bound]
    return bound






def get_config_info(attack_name):
    path = 'config.yaml'
    with open(path, 'r') as y:
        cfg = yaml.load(y, Loader = yaml.FullLoader)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(attack_name, cfg[attack_name])
    target_attack = cfg[attack_name]['target_attack']
    model_name = cfg[attack_name]['model_name']
    save_dir = cfg[attack_name]['save_dir']
    batch_size = cfg[attack_name]['batch_size']
    epsilon = cfg[attack_name]['epsilon'] / ( cfg[attack_name]['std_norm'] * 255.0 )
    bound = get_bound(cfg[attack_name]['model_name'])
    niters = cfg[attack_name]['niters']
    if attack_name == 'VIM':
        beta = cfg[attack_name]['beta']
        N = cfg[attack_name]['N']
        mu = cfg[attack_name]['mu']
        attack_type = VMI(epsilon, beta, N, niters, mu, device, bound)
    elif attack_name == 'SIM':
        m = cfg[attack_name]['m']
        mu = cfg[attack_name]['mu']
        attack_type = SIM(epsilon, niters, mu, m, device, bound)
    elif attack_name == 'AOA':
        eta = cfg[attack_name]['eta']
        lambda_ = cfg[attack_name]['lambda_']
        attack_type = AoA(epsilon, eta, lambda_, niters, device, bound)
    elif attack_name == 'NAAO':
        gamma = cfg[attack_name]['gamma']
        mu = cfg[attack_name]['mu']
        target_layer = cfg[attack_name]['target_layer']
        steps = cfg[attack_name]['steps']
        path_type = cfg[attack_name]['path_type']
        attack_type = NAAO(epsilon, steps, niters, mu, gamma, target_layer, path_type, device,  bound)
    elif attack_name == 'BLURIGO':
        gamma = cfg[attack_name]['gamma']
        mu = cfg[attack_name]['mu']
        steps = cfg[attack_name]['steps']
        grad_step = cfg[attack_name]['grad_step']
        sigma_max = cfg[attack_name]['sigma_max']
        attack_type = BLURIGO(epsilon, steps, sigma_max, grad_step, niters, gamma, mu, device,  bound)
    elif attack_name == 'NRDM':
        target_layer = cfg[attack_name]['target_layer']
        attack_type = NRDM(epsilon, niters, target_layer, device, bound)
    elif attack_name == 'FDA':
        target_layer = cfg[attack_name]['target_layer']
        mu = cfg[attack_name]['mu']
        attack_type = FDA(epsilon, niters, mu, target_layer, device, bound)
    elif attack_name == 'FIA':
        target_layer = cfg[attack_name]['target_layer']
        pro = cfg[attack_name]['pro']
        mu = cfg[attack_name]['mu']
        steps = cfg[attack_name]['steps']
        attack_type = FIA(epsilon, steps, niters, mu, pro, target_layer, device, bound)
    elif attack_name == 'DIM':
        p_value = cfg[attack_name]['p_value']
        mu = cfg[attack_name]['mu']
        attack_type = DIM(epsilon, niters, mu, p_value, device, bound)
    return attack_type, model_name, target_attack, batch_size, save_dir


