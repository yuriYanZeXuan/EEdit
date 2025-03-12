import torch
from torch.quantization.observer import MinMaxObserver
# LPT=torch.float8_e4m3fn
LPT=torch.quint4x2
obs = MinMaxObserver(dtype=LPT)
def to_LPT(x):
    obs(x)
    scale, zero_point = obs.calculate_qparams()
    scale=scale.to(x.device)
    zero_point=zero_point.to(x.device)
    x = x.to(torch.float32)
    quantized_data = torch.quantize_per_tensor(x, scale, zero_point, dtype=LPT).to(x.device)
    return quantized_data
def update_cache(fresh_indices, fresh_tokens, cache_dic, current, fresh_attn_map=None):
    '''
    Update the cache with the fresh tokens.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    mix_precision=cache_dic['low_precision_region'] is not None
    if mix_precision:
        low_precision_region=cache_dic['low_precision_region']
        full_precision_region=cache_dic['full_precision_region']
    # Update the cached tokens at the positions
    if module == 'attn':
        # this branch is not used in the final version, but if you explore the partial fresh strategy of attention, it works (probably a few bugs).
        indices = fresh_indices#.sort(dim=1, descending=False)[0]
        # cache_dic['attn_map'][-1][layer].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_attn_map.shape[-1]), src=fresh_attn_map)
    elif module == 'cross-attn':
        indices = fresh_indices#.sort(dim=1, descending=False)[0]
        # cache_dic['cross_attn_map'][-1][layer].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_attn_map.shape[-1]), src=fresh_attn_map)
    elif module == 'mlp' or module == 'cross-mlp':
        indices = fresh_indices
    elif module == 'norm':
        indices = fresh_indices
    else:
        raise ValueError(f"module {module} is not supported when updating cache")
    if mix_precision:
        if module+'lpx' in cache_dic['cache'][-1][layer].keys():
            if current['layer']>18:
                lpx=cache_dic['cache'][-1][layer][module+'lpx'].dequantize().to(torch.bfloat16)
                hpx=cache_dic['cache'][-1][layer][module+'hpx']
                tmp=torch.zeros((lpx.shape[0],lpx.shape[1]+hpx.shape[1],lpx.shape[2]),dtype=torch.bfloat16,device=lpx.device)
                tmp[:,:512]=hpx[:,:512]
                tmp[:,512:][:,full_precision_region]=hpx[:,512:]
                tmp[:,512:][:,low_precision_region]=lpx.dequantize().to(torch.bfloat16)
                tmp.scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
                lpx=to_LPT(tmp[:,512:][:,low_precision_region])
                hpx=torch.cat([tmp[:,:512],tmp[:,512:][:,full_precision_region]],dim=1)
                cache_dic['cache'][-1][layer][module+'lpx']=lpx
                cache_dic['cache'][-1][layer][module+'hpx']=hpx
            else:
                lpx=cache_dic['cache'][-1][layer][module+'lpx'].dequantize().to(torch.bfloat16)
                hpx=cache_dic['cache'][-1][layer][module+'hpx']
                tmp=torch.zeros((lpx.shape[0],lpx.shape[1]+hpx.shape[1],lpx.shape[2]),dtype=torch.bfloat16,device=lpx.device)
                tmp.scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
                lpx=to_LPT(tmp[:,low_precision_region])
                hpx=tmp[:,full_precision_region]
                cache_dic['cache'][-1][layer][module+'lpx']=lpx
                cache_dic['cache'][-1][layer][module+'hpx']=hpx
        else:
            tmp=cache_dic['cache'][-1][layer][module].dequantize().to(torch.bfloat16)
            tmp.scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
            cache_dic['cache'][-1][layer][module]=to_LPT(tmp)
    else:
        cache_dic['cache'][-1][layer][module].scatter_(dim=1, index=indices.unsqueeze(-1).expand(-1, -1, fresh_tokens.shape[-1]), src=fresh_tokens)
    
    

        
        