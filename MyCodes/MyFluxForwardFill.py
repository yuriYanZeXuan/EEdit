from typing import Any, Dict, Optional
import torch

def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
):
    """
    This forward pass is adapted for the flux-fill model, while preserving the custom caching logic.
    The main difference is the initial embedding of hidden_states to match the transformer's inner_dim.
    """
    joint_attention_kwargs = joint_attention_kwargs if joint_attention_kwargs is not None else {}
    
    # 1. Input projections
    # This is the key fix: The VAE output (packed into hidden_states) needs to be projected to the transformer's inner dimension.
    # The original MyFluxForward was missing this step, causing a shape mismatch.
    hidden_states = self.x_embedder(hidden_states)
    
    timestep_embed, guidance_embed, pooled_projections = self.time_text_embed(
        timestep, guidance, pooled_projections
    )
    temb = (timestep_embed + guidance_embed).unsqueeze(1)
    
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 2. Rotary Positional Embeddings
    image_rotary_emb = self.pos_embed(img_ids)
    
    # 3. Cache Initialization and Loop
    if joint_attention_kwargs.get('use_cache', False):
        current = joint_attention_kwargs['current']
        current['num_blocks'] = len(self.transformer_blocks) + len(self.single_transformer_blocks)

    # 4. Main Transformer Blocks
    for i, block in enumerate(self.transformer_blocks):
        if joint_attention_kwargs.get('use_cache', False):
            current['layer'] = i
        
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # 5. Single Transformer Blocks
    for i, block in enumerate(self.single_transformer_blocks):
        if joint_attention_kwargs.get('use_cache', False):
            current['layer'] = i + len(self.transformer_blocks)
        
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # 6. Output
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    if not return_dict:
        return (hidden_states,)

    return hidden_states
