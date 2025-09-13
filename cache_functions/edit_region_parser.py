from typing import Optional
import torch
import numpy as np


def edit_region_parser(
    x1,y1,x2,y2,
    resolution: Optional[int] = 512,
    vae_scale_factor=16,
    cascade_num=3,
    height=None,
    width=None,
):   
    '''
    calculate and return the index of the edit region by input [x1,y1,x2,y2]
    '''
    if cascade_num ==0:
        return None
    if height is None:
        height=resolution
    if width is None:
        width=resolution
    edit_idx=[[]for _ in range(cascade_num)]
    assert resolution in [512,1024,2048,None], "resolution must be 512 or 1024 or 2048 or None"
    if resolution is None:
        resolution=max(height,width)
        assert 0<=x1<=x2<=width and 0<=y1<=y2<=height, f"{x1},{y1},{x2},{y2} must be in the range of resolution"
    else:
        width=resolution
        height=resolution
        assert 0<=x1<=x2<=width and 0<=y1<=y2<=height, f"{x1},{y1},{x2},{y2} must be in the range of resolution"
    scaled_resolution = resolution // vae_scale_factor
    scaled_width=width//vae_scale_factor
    scaled_height=height//vae_scale_factor
    scaled_x1 = int(x1 / vae_scale_factor)
    scaled_y1 = int(y1 / vae_scale_factor) 
    scaled_x2 = int(x2 / vae_scale_factor+0.5)
    scaled_y2 = int(y2 / vae_scale_factor+0.5)
    
    scaled_x1 = max(0, min(scaled_x1, scaled_width))
    scaled_y1 = max(0, min(scaled_y1, scaled_height))
    scaled_x2 = max(0, min(scaled_x2, scaled_width))
    scaled_y2 = max(0, min(scaled_y2, scaled_height))
    
    indices = []
    for y in range(scaled_y1, scaled_y2):
        for x in range(scaled_x1, scaled_x2):
            edit_idx[0].append(x+y*scaled_width)
            indices.append((x,y))
    edit_idx[0]=torch.tensor(list(set(edit_idx[0])))
    region_set=set(indices)
    tmp_set=set()
    for k in range(1,cascade_num):
        for i in range(0,scaled_width):
            for j in range(0,scaled_height):
                if (i,j) not in region_set:
                    if (i+1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j-1) in region_set:
                        tmp_set.add((i,j))
        region_set=region_set.union(tmp_set)
        edit_idx[k]=torch.tensor([i+j*scaled_width for i,j in tmp_set])
        tmp_set.clear()
    for k in range(cascade_num-1, -1, -1):
        if len(edit_idx[k]) == 0:
            del edit_idx[k]
    return edit_idx

def edit_pts_parser(
    src_pts,tgt_pts,
    resolution: Optional[int] = 512,
    vae_scale_factor=16,
    cascade_num=3,
    height=None,
    width=None,
):   
    if cascade_num ==0:
        return None
    if height is None:
        height=resolution
    if width is None:
        width=resolution
    edit_idx=[[]for _ in range(cascade_num)]
    assert src_pts.shape==tgt_pts.shape, "src_pts and tgt_pts must have the same shape"
    assert src_pts.shape[1]==2, "src_pts and tgt_pts must have 2 columns"
    vae_src_pts=src_pts//vae_scale_factor
    vae_tgt_pts=tgt_pts//vae_scale_factor
    scaled_resolution = resolution // vae_scale_factor
    scaled_width=width//vae_scale_factor
    scaled_height=height//vae_scale_factor
    vae_src_pts = [tuple(pt) for pt in vae_src_pts.tolist()]
    vae_tgt_pts = [tuple(pt) for pt in vae_tgt_pts.tolist()]
    indices = vae_src_pts + vae_tgt_pts
    edit_idx[0] = [i+j*scaled_width  for i,j in indices if 0<=i<scaled_width and 0<=j<scaled_height]
    edit_idx[0]=torch.tensor(list(set(edit_idx[0])))
    region_set = set(indices)
    # return region_set
    tmp_set=set()
    for k in range(1,cascade_num):
        for i in range(0,scaled_width):
            for j in range(0,scaled_height):
                if (i,j) not in region_set:
                    if (i+1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j-1) in region_set:
                        tmp_set.add((i,j))
        region_set=region_set.union(tmp_set)
        for i,j in tmp_set:
            edit_idx[k].append(i+j*scaled_width)
        edit_idx[k]=torch.tensor(list(set(edit_idx[k])))
        tmp_set.clear()
    for k in range(cascade_num-1, -1, -1):
        if len(edit_idx[k]) == 0:
            del edit_idx[k]
    return edit_idx
    
def edit_mask_parser(
    mask,
    resolution: Optional[int] = 512,
    vae_scale_factor=16,
    cascade_num=3,
    height=512,
    width=512,
):
    if cascade_num == 0:
        return None
    if height is None:
        height=resolution
    if width is None:
        width=resolution
    edit_idx = [[] for _ in range(cascade_num)]
    scaled_resolution = resolution // vae_scale_factor
    scaled_width=width//vae_scale_factor
    scaled_height=height//vae_scale_factor
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    
    if len(mask.shape) > 2:
        mask = mask.mean(axis=2)  
        
    scaled_mask = mask[::vae_scale_factor, ::vae_scale_factor]
    
    indices = set()
    h, w = scaled_mask.shape
    for i in range(h):
        for j in range(w):
            if scaled_mask[i, j] > 0:
                indices.add((j, i)) # NOTE: The original code might have had a bug here. indices are usually (x, y) which corresponds to (w, h) or (j, i)
                
    edit_idx[0] = [x + y * scaled_width for x, y in indices if 0 <= x < scaled_width and 0 <= y < scaled_height]
    edit_idx[0]=torch.tensor(list(set(edit_idx[0])))
    
    region_set = indices.copy()
    tmp_set = set()
    
    for k in range(1, cascade_num):
        for i in range(0, scaled_width):
            for j in range(0, scaled_height):
                if (i,j) not in region_set:
                    if (i+1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j-1) in region_set:
                        tmp_set.add((i,j))
                        
        region_set = region_set.union(tmp_set)
        edit_idx[k] = torch.tensor([i+j*scaled_width for i,j in tmp_set])
        tmp_set.clear()
    for k in range(cascade_num-1, -1, -1):
        if len(edit_idx[k]) == 0:
            del edit_idx[k]
    return edit_idx

def edit_resize_parser(
    size_region,
    seg_region,
    cascade_num=3,
    resolution: Optional[int] = 512,
    vae_scale_factor=16,
    height=None,
    width=None,
):
    if cascade_num == 0:
        return None
    
    if size_region is None and seg_region is None:
        return None
    if height is None:
        height=resolution
    if width is None:
        width=resolution
    bbox_idx = None
    if size_region is not None:
        x1, y1, x2, y2 = size_region
        bbox_idx = edit_region_parser(
            x1, y1, x2, y2,
            resolution=resolution,
            vae_scale_factor=vae_scale_factor,
            cascade_num=cascade_num,
            height=height,
            width=width
        )
    
    mask_idx = None
    if seg_region is not None:
        mask_idx = edit_mask_parser(
            seg_region,
            resolution=resolution,
            vae_scale_factor=vae_scale_factor,
            cascade_num=cascade_num,
            height=height,
            width=width
        )
    
    if bbox_idx is None:
        return mask_idx
    if mask_idx is None:
        return bbox_idx
        
    combined_idx = [[] for _ in range(cascade_num)]
    for i in range(cascade_num):
        combined_set = set(bbox_idx[i].tolist()).union(set(mask_idx[i].tolist()))
        combined_idx[i] = torch.tensor(list(combined_set))
    for k in range(cascade_num-1, -1, -1):
        if len(combined_idx[k]) == 0:
            del combined_idx[k]
    return combined_idx
    
    

def edit_ratio_parser(
    ratio=0.3,
    resolution: Optional[int] = 512,
    vae_scale_factor=16,
    cascade_num=10,
    height=None,
    width=None,
):   
    '''
    calculate and return the index of the edit region by input [x1,y1,x2,y2]
    '''
    if cascade_num ==0:
        return None
    if height is None:
        height=resolution
    if width is None:
        width=resolution
    edit_idx=[[]for _ in range(cascade_num)]
    assert resolution in [512,1024,2048,None], "resolution must be 512 or 1024 or 2048 or None"
    assert 0<=ratio<=1, "ratio must be in the range of 0 to 1"
    if resolution is None:
        resolution=max(height,width)
    else:
        width=resolution
        height=resolution
    scaled_resolution = resolution // vae_scale_factor
    scaled_width=width//vae_scale_factor
    scaled_height=height//vae_scale_factor
    x_c=scaled_width//2
    y_c=scaled_height//2
    scaled_x1 = int(x_c-ratio*scaled_width)
    scaled_y1 = int(y_c-ratio*scaled_height) 
    scaled_x2 = int(x_c+ratio*scaled_width)
    scaled_y2 = int(y_c+ratio*scaled_height)
    
    scaled_x1 = max(0, min(scaled_x1, scaled_width))
    scaled_y1 = max(0, min(scaled_y1, scaled_height))
    scaled_x2 = max(0, min(scaled_x2, scaled_width))
    scaled_y2 = max(0, min(scaled_y2, scaled_height))
    
    indices = []
    for y in range(scaled_y1, scaled_y2):
        for x in range(scaled_x1, scaled_x2):
            edit_idx[0].append(x+y*scaled_width)
            indices.append((x,y))
    edit_idx[0]=torch.tensor(list(set(edit_idx[0])))
    region_set=set(indices)
    tmp_set=set()
    for k in range(1,cascade_num):
        for i in range(0,scaled_width):
            for j in range(0,scaled_height):
                if (i,j) not in region_set:
                    if (i+1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j) in region_set:
                        tmp_set.add((i,j))
                    elif (i,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j-1) in region_set:
                        tmp_set.add((i,j))
                    elif (i-1,j+1) in region_set:
                        tmp_set.add((i,j))
                    elif (i+1,j-1) in region_set:
                        tmp_set.add((i,j))
        region_set=region_set.union(tmp_set)
        edit_idx[k]=torch.tensor([i+j*scaled_width for i,j in tmp_set])
        tmp_set.clear()
    for k in range(cascade_num-1, -1, -1):
        if len(edit_idx[k]) == 0:
            del edit_idx[k]
    return edit_idx

def convert_to_cache_index(edit_idx,edit_base=2,bonus_ratio=0.8,resolution=512,vae_scale_factor=16,height=None,width=None):
    assert resolution in [512,1024,2048,None], "resolution must be 512 or 1024 or 2048 or None"
    if resolution is None:
        resolution=max(height,width)
    if width is None:
        width=resolution
    if height is None:
        height=resolution
    scaled_width=width//vae_scale_factor
    scaled_height=height//vae_scale_factor
    score=torch.ones(scaled_width*scaled_height)
    if edit_idx is None:
        return score
    for i in range(len(edit_idx)):
        src = edit_idx[i]
        score[src] = score[src] *(1+edit_base*bonus_ratio**(1+i))
    return score
