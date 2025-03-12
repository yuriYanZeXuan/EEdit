import torch
import torch.nn as nn
import torch.nn.functional as F

def attn_score(cache_dic, current):
    attention_score = F.normalize(cache_dic['attn_map'][-1][current['layer']].sum(dim=1), dim=1, p=2)
    cmap=cache_dic['cross_attn_map'][-1][current['layer']]
    if cmap!={}:
        cross_attention_entropy = -torch.sum(cmap * torch.log(cmap + 1e-7), dim=-2)
        cross_attention_score   = F.normalize(1 + cross_attention_entropy, dim=1, p=2) # Note here "1" does not influence the sorted sequence, but provie stability.

        # In PixArt, the cross_attention_score (s2) is used as the score, for a better text-image alignment.

        # You can try conbining the self_attention_score (s1) and cross_attention_score (s2) as the final score, there exists a balance.
        cross_weight = 0.5
        score =  (1-cross_weight) * attention_score + cross_weight * cross_attention_score
        return score
    else:
        return attention_score

def similarity_score(cache_dic, current, tokens):
    cosine_sim = F.cosine_similarity(tokens, cache_dic['cache'][-1][current['layer']][current['module']], dim=-1)

    return F.normalize(1- cosine_sim, dim=-1, p=2)

def norm_score(cache_dic, current, tokens):
    norm = tokens.norm(dim=-1, p=2)
    return F.normalize(norm, dim=-1, p=2)


def kv_norm_score(cache_dic, current):
    if current['module'].startswith('cross'):
        v_norm=cache_dic['cache'][-1][current['layer']]['cross_attn_v_norm']
    else:
        v_norm=cache_dic['cache'][-1][current['layer']]['attn_v_norm']
    kv_norm = 1 -v_norm
    
    return F.normalize(kv_norm.sum(dim=1), p=2)

