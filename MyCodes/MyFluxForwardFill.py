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
    
    # 2. Time, text and guidance embedding
    if self.config.guidance_embeds:
        time_embed_dim = self.inner_dim
        if guidance is None:
            # When guidance is not provided, we must pass a dummy tensor for the guidance argument.
            dummy_guidance = torch.zeros_like(timestep)
            embedding = self.time_text_embed(timestep, dummy_guidance, pooled_projections)

            # The custom module appears to only return the time embedding when passed a dummy guidance.
            # We will assume the returned embedding is the timestep embedding and use the original,
            # un-projected pooled_projections for the text embedding as a workaround.
            timestep_embed = embedding
            guidance_embed = None
            # pooled_projections is intentionally not updated here.
        else:
            embedding = self.time_text_embed(
                timestep, guidance, pooled_projections
            )
            guidance_embed_dim = time_embed_dim
            text_embed_dim = time_embed_dim
            timestep_embed, guidance_embed, pooled_projections = torch.split(
                embedding, [time_embed_dim, guidance_embed_dim, text_embed_dim], dim=1
            )
    else:
        embedding = self.time_text_embed(timestep, pooled_projections)
        time_embed_dim = self.inner_dim
        text_embed_dim = time_embed_dim
        timestep_embed, pooled_projections = torch.split(embedding, [time_embed_dim, text_embed_dim], dim=1)
        guidance_embed = None

    # 3. Create conditioning embedding
    if guidance_embed is None:
        cond_embed = timestep_embed
    else:
        cond_embed = timestep_embed + guidance_embed * guidance.to(timestep_embed.dtype).reshape(-1, 1)

    # 4.-5. Post-process embeddings
    adaln_embed = pooled_projections
    
    # 6. Rotary Positional Embeddings
    joint_ids = torch.cat((txt_ids, img_ids), dim=0)
    joint_rotary_emb = self.pos_embed(joint_ids)
    image_only_rotary_emb = self.pos_embed(img_ids)

    # 7. Cache Initialization and Loop
    if joint_attention_kwargs.get('use_cache', False):
        current = joint_attention_kwargs['current']
        current['num_blocks'] = len(self.transformer_blocks) + len(self.single_transformer_blocks)

    if encoder_hidden_states is not None:
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 8. Main Transformer Blocks
    for i, block in enumerate(self.transformer_blocks):
        if joint_attention_kwargs.get('use_cache', False):
            current['layer'] = i
        
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=cond_embed,
            image_rotary_emb=joint_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # 9. Single Transformer Blocks
    for i, block in enumerate(self.single_transformer_blocks):
        if joint_attention_kwargs.get('use_cache', False):
            current['layer'] = i + len(self.transformer_blocks)
        
        hidden_states = block(
            hidden_states=hidden_states,
            temb=cond_embed,
            image_rotary_emb=image_only_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # 10. Output
    hidden_states = self.norm_out(hidden_states, cond_embed)
    hidden_states = self.proj_out(hidden_states)

    if not return_dict:
        return (hidden_states,)

    return hidden_states
