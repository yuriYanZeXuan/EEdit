import torch
import torch.nn as nn
from .scores import attn_score, similarity_score, norm_score, kv_norm_score
def score_evaluate(cache_dic, tokens, current) -> torch.Tensor:
    '''
    Return the score tensor (B, N) for the given tokens.
    '''

    # Just see more explanation in the version of DiT-ToCa if needed.

    if cache_dic['cache_type'] == 'random':
        score = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device)
        # score = torch.cat([score, score], dim=0).to(tokens.device)

    elif cache_dic['cache_type'] == 'straight':
        score = torch.ones(tokens.shape[0], tokens.shape[1]).to(tokens.device)
    
    elif cache_dic['cache_type'] == 'attention':
        score = attn_score(cache_dic, current)
        if score.shape[1] != tokens.shape[1]:
            score = score[:,:tokens.shape[1]]
    
    elif cache_dic['cache_type'] == 'similarity':
        score = similarity_score(cache_dic, current, tokens)

    elif cache_dic['cache_type'] == 'norm':
        score = norm_score(cache_dic, current, tokens)
        
    elif cache_dic['cache_type'] == 'kv-norm':
        score = kv_norm_score(cache_dic, current)
    elif cache_dic['cache_type'] == 'compress':
        score1 = torch.rand(int(tokens.shape[0]*0.5), tokens.shape[1])
        score1 = torch.cat([score1, score1], dim=0).to(tokens.device)
        score2 = cache_dic['attn_map'][-1][current['layer']].sum(dim=1)#.mean(dim=0) # (B, N)
        # normalize
        score2 = score2 / score2.max(dim=1, keepdim=True)[0]
        score = 0.5 * score1 + 0.5 * score2
    else:
        score = torch.rand(tokens.shape[0], tokens.shape[1], device=tokens.device)
    
    
    if (cache_dic['cache_type']=='random' and (cache_dic['force_fresh'] == 'global')):
        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])
    elif (cache_dic['cache_type']=='attention' and (cache_dic['force_fresh'] == 'global')):
        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])
        score = score + cache_dic['soft_fresh_weight'] * soft_step_score# + 0.1 *soft_layer_score    
    elif (cache_dic['cache_type']=='kv-norm' and (cache_dic['force_fresh'] == 'global')):
        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])
        score = score + cache_dic['soft_fresh_weight'] * soft_step_score #+ 0.1 *soft_layer_score 
    else :
        soft_step_score = cache_dic['cache_index'][-1][current['layer']][current['module']].float() / (cache_dic['fresh_threshold'])
        score = score + cache_dic['soft_fresh_weight'] * soft_step_score #+ 0.1 *soft_layer_score 
    return score.to(tokens.device)