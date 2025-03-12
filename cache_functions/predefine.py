from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
from .cache_cutfresh import *
from .cal_type import cal_type
from .force_init import force_init
from einops import repeat,rearrange
import torch
def predefine_cache_fresh_indices(cache_dic, current):
    device='cuda'
    hw = cache_dic.get('hw', None)
    assert hw is not None, "hw is None"
    for step in range(current['num_steps']):
        current['step'] = step
        cal_type(cache_dic, current)
        if current['type'] == 'full':
            for layer in range(19):
                current['layer'] = layer
                for module in ['attn', 'mlp']:
                    current['module'] = module
                    force_init(cache_dic, current, (1,hw[0]*hw[1]))
                for module in ['cross-attn', 'cross-mlp']:
                    current['module'] = module
                    force_init(cache_dic, current, (1,512))
            for layer in range(19,19+38):
                current['layer'] = layer
                for module in ['norm', 'attn', 'mlp']:
                    current['module'] = module
                    force_init(cache_dic, current, (1,hw[0]*hw[1]))
        elif current['type'] == 'ToCa':
            for layer in range(19+38):
                current['layer'] = layer
                for module in ['norm', 'attn', 'mlp','cross-attn','cross-mlp']:
                    current['module'] = module
                    
                    step = current['step']
                    layer = current['layer']
                    module = current['module']
                    
                    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)
                    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio), min=0, max=1)
                    if layer<19:
                        if current['module'] in ['mlp']:
                            score = torch.rand(1,hw[0]*hw[1],device=device)
                        elif current['module'] in ['cross-mlp']:
                            score = torch.rand(1,512,device=device)
                        else:
                            continue
                    else:
                        if current['module'] in ['norm','mlp']:
                            score = torch.rand(1,hw[0]*hw[1],device=device)
                        else:
                            continue
                    if cache_dic['cache_type'] !='random':
                        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])
                        score = score + cache_dic['soft_fresh_weight'] * soft_step_score #+ 0.1 *soft_layer_score
                            
                        if current['module'] in ['mlp','norm']:
                            score = edit_score(score, 0.8, current)
                        
                    indices = score.argsort(dim=-1, descending=True)
                    topk = int(fresh_ratio * score.shape[1])
                    fresh_indices = indices[:, :topk]
                    stale_indices = indices[:, topk:]
                    cache_dic['predefine_fresh_indices'][f"step{step},layer:{layer},module:{module}"] = fresh_indices
                    cache_dic['cache_index'][-1][layer][module] += 1
                    cache_dic['cache_index'][-1][layer][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.uint8, device=fresh_indices.device))