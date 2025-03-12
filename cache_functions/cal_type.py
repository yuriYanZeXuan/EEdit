def cal_type(cache_dic, current):
    '''
    Determine calculation type for this step
    '''

    first_step = (current['step'] == 0)
    force_fresh = cache_dic['force_fresh']
    tailing_step = (current['step'] >= current['num_steps']-cache_dic['tailing_step'])
    fresh_interval = cache_dic['fresh_threshold']
    if (current['step'] % fresh_interval == 0) or first_step or tailing_step:
        current['type'] = 'full'
    else:
        current['type'] = 'ToCa'