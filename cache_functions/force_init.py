import torch
from .force_scheduler import force_scheduler
def force_init(cache_dic, current, shape,device='cuda'):
    '''
    Initialization for Force Activation step.
    shape: [batch_size, seq_len]
        [1,4090]
        [1,512]
        [1,4608]
    '''
    cache_dic['cache_index'][-1][current['layer']][current['module']] = torch.zeros(shape[0], shape[1], dtype=torch.uint8, device=device)
    if cache_dic['fresh_ratio'] == 0:
        # FORA
        linear_step_weight = 0.0
    else: 
        # TokenCache
        linear_step_weight = 0.2
    step_factor = torch.tensor(1 - linear_step_weight + 2 * linear_step_weight * current['step'] / current['num_steps'])
    threshold = torch.round(cache_dic['fresh_threshold'] / step_factor)

    cache_dic['cal_threshold'] = threshold