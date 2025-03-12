from .predefine import predefine_cache_fresh_indices
def cache_init(model_kwargs, num_steps, edit_idx=None):   
    '''
    Initialization for cache.
    '''
    cache_dic = {}
    cache = {}
    cache_index = {}
    cache[-1]={}
    cache_index[-1]={}
    cache_index['layer_index']={}
    cache_dic['attn_map'] = {}
    cache_dic['attn_map'][-1] = {}
    cache_dic['cross_attn_map'] = {}
    cache_dic['cross_attn_map'][-1] = {}
    cache_dic['norm'] = {}
    cache_dic['norm'][-1] = {}
    cache_dic['hw'] = model_kwargs.get('hw', None)
    for j in range(19+38):
        cache[-1][j] = {}
        cache_index[-1][j] = {}
        cache_dic['attn_map'][-1][j] = {}
        cache_dic['cross_attn_map'][-1][j] = {}

    cache_dic['cache_type'] = model_kwargs['cache_type']
    cache_dic['cache_index'] = cache_index
    cache_dic['cache'] = cache
    cache_dic['fresh_ratio_schedule'] = model_kwargs['ratio_scheduler']
    cache_dic['fresh_ratio'] = model_kwargs['fresh_ratio']
    cache_dic['fresh_threshold'] = model_kwargs['fresh_threshold']
    cache_dic['force_fresh'] = model_kwargs['force_fresh']
    cache_dic['soft_fresh_weight'] = model_kwargs['soft_fresh_weight']
    cache_dic['tailing_step'] = model_kwargs['tailing_step']
    cache_dic['full_precision_region'] = model_kwargs.get('full_precision_region', None)
    cache_dic['low_precision_region'] = model_kwargs.get('low_precision_region', None)
    cache_dic['process_indices'] = {}
    current = {}
    current['num_steps'] = num_steps
    current['edit_base'] = model_kwargs.get('edit_base', 1)
    current['edit_idx'] = edit_idx
    if cache_dic['cache_type'] == 'ours_predefine':
        cache_dic['predefine_fresh_indices'] = {}
        predefine_cache_fresh_indices(cache_dic, current)
        
    return cache_dic, current
    