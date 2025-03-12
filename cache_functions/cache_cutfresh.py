from .fresh_ratio_scheduler import fresh_ratio_scheduler
from .score_evaluate import score_evaluate
import torch
from einops import repeat,rearrange

def cache_cutfresh(cache_dic, tokens, current):
    '''
    Cut fresh tokens from the input tokens and update the cache counter.
    
    cache_dic: dict, the cache dictionary containing cache(main extra memory cost), indices and some other information.
    tokens: torch.Tensor, the input tokens to be cut.
    current: dict, the current step, layer, and module information. Particularly convenient for debugging.
    '''
    step = current['step']
    layer = current['layer']
    module = current['module']
    hw = cache_dic.get('hw', None)
    fresh_ratio = fresh_ratio_scheduler(cache_dic, current)
    fresh_ratio = torch.clamp(torch.tensor(fresh_ratio, device = tokens.device), min=0, max=1)
    score = score_evaluate(cache_dic, tokens, current) # s1, s2, s3 mentioned in the paper
    if hw is None:
        if score.shape[1] == 4608:
            # for resolution 1024x1024
            score = score[:,:4096]
        elif score.shape[1] == 1536:
            # for resolution 512x512
            score = score[:,:1024]
        elif score.shape[1] == 256+512:
            # for resolution 256x256
            score = score[:,:256]
    else:
        if score.shape[1] != hw[0]*hw[1]:
            score = score[:,:hw[0]*hw[1]] 
    if cache_dic['cache_type'] !='random':
        if current['module'] in ['mlp', 'attn','cross-attn','norm']:
            if cache_dic['cache_type'] in ['toca','duca']:
                score = local_selection_with_bonus(score, 0.4, 4, hw) # Uniform Spatial Distribution s4 mentioned in the paper
            elif cache_dic['cache_type'] in ['random','ours_predefine','ours_cache']:
                score = edit_score(score, 0.8, current)
        
    indices = score.argsort(dim=-1, descending=True)
    topk = int(fresh_ratio * score.shape[1])
    fresh_indices = indices[:, :topk]
    cache_dic["process_indices"][f"step{step},layer:{layer},module:{module}"] = fresh_indices
    stale_indices = indices[:, topk:]

    # Updating the Cache Frequency Score s3 mentioned in the paper
    # stale tokens index + 1 in each ***module***, fresh tokens index = 0
    cache_dic['cache_index'][-1][layer][module] += 1
    cache_dic['cache_index'][-1][layer][module].scatter_(dim=1, index=fresh_indices, 
                                                                    src = torch.zeros_like(fresh_indices, dtype=torch.uint8, device=fresh_indices.device))
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    if module in ['mlp', 'attn', 'cross-mlp', 'cross-attn','norm']:

        fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)

        return fresh_indices, fresh_tokens
    else:
        raise ValueError("Unrecognized module?", module)

def predefine_cache_cutfresh(cache_dic, tokens, current):
    step = current['step']
    layer = current['layer']
    module = current['module']
    fresh_indices = cache_dic['predefine_fresh_indices'][f"step{step},layer:{layer},module:{module}"]
    fresh_indices_expand = fresh_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    fresh_tokens = torch.gather(input = tokens, dim = 1, index = fresh_indices_expand)
    return fresh_indices, fresh_tokens


def local_selection_with_bonus(score, bonus_ratio, grid_size=4,hw=None):
    """
    hw: (h,w) latent patch size
    """
    batch_size, num_tokens = score.shape
    block_size = grid_size * grid_size
    if hw is None:
        image_size = int(num_tokens ** 0.5)
        hw=(image_size,image_size)
    
    # Step 1: Reshape score to group it by blocks score size: [batch_size, num_blocks, block_size]
    if hw[0] % grid_size != 0 or hw[1] % grid_size != 0:
        score2 = rearrange(score,'b (c1 c2) -> b c1 c2',c1=hw[0],c2=hw[1])
        score_rec = score2[:,:hw[0]//grid_size*grid_size,:hw[1]//grid_size*grid_size]
    else:
        score_rec = score
    score_reshaped = score_rec.view(batch_size, hw[0] // grid_size, grid_size, hw[1] // grid_size, grid_size)
    score_reshaped = score_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    score_reshaped = score_reshaped.view(batch_size, -1, block_size)  # [batch_size, num_blocks, block_size]
    # Step 2: Find the max token in each block
    max_scores, max_indices = score_reshaped.max(dim=-1, keepdim=True)  # [batch_size, num_blocks, 1]
    
    # Step 3: Create a mask to identify max score tokens
    mask = torch.zeros_like(score_reshaped)
    mask.scatter_(-1, max_indices, 1)  # Set mask to 1 at the max indices
    
    # Step 4: Apply the bonus only to the max score tokens
    score_reshaped = score_reshaped + (mask * max_scores * bonus_ratio)  # Apply bonus only to max tokens
    
    if hw[0] % grid_size != 0 or hw[1] % grid_size != 0:
        score_reshaped=score_reshaped.view(batch_size, hw[0] // grid_size, hw[1] // grid_size, grid_size,grid_size)
        score_reshaped=score_reshaped.permute(0,1,3,2,4).contiguous()
        score2[:,:score_rec.shape[1],:score_rec.shape[2]]=score_reshaped.view(batch_size,score_rec.shape[1],score_rec.shape[2])
        score_modified = score2.view(batch_size, num_tokens)
        return score_modified
    else:
        # Step 5: Reshape the score back to its original shape
        score_modified = score_reshaped.view(batch_size, hw[0] // grid_size, grid_size, hw[1] // grid_size, grid_size)
        score_modified = score_modified.permute(0, 1, 3, 2, 4).contiguous()
        score_modified = score_modified.view(batch_size, num_tokens)
        return score_modified

def edit_score(score, bonus_ratio, current):
    batch_size, num_tokens = score.shape
    if 'edit_idx_merged' in current :
        score=score*current['edit_idx_merged']
        return score
    elif current['edit_idx'] is not None:
        edit_base=current.get('edit_base',1)
        for i in range(len(current['edit_idx'])):
            edit_idx = current['edit_idx'][i]
            score[:, edit_idx] = score[:, edit_idx] *(1+edit_base*bonus_ratio**(1+i))
        return score
    else:
        return score
    