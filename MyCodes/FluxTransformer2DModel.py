# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    # FluxAttnProcessor2_0,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from .MyFromOriginalModelMixin import MyFromOriginalModelMixin
from MyCodes.MyAttnProcessor import MyFluxAttnProcessor2_0
from cache_functions import global_force_fresh, cache_cutfresh, update_cache, force_init,cal_type


# logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
@maybe_allow_in_graph
class FluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, mlp_ratio=4.0):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = AdaLayerNormZeroSingle(dim)
        self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)

        processor = MyFluxAttnProcessor2_0()
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        # norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        # mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        if joint_attention_kwargs['use_cache']:
            # print(f"use cache")
            cache_dic = joint_attention_kwargs['cache_dic']
            current = joint_attention_kwargs['current']
            
            cache_type = cache_dic['cache_type']
            layer = current['layer']
            
            cal_type(cache_dic=cache_dic, current=current)

            if current['type'] == 'full': # Force Activation: Compute all tokens and save them in cache
                current['module'] = 'norm'
                norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
                # print(f"1 norm_hidden_states: {norm_hidden_states.shape}, gate: {gate.shape}")
                # norm_hidden_states: torch.Size([1, 1536, 3072]), gate: torch.Size([1, 3072])
                mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
                # print(f"1 mlp_hidden_states: {mlp_hidden_states.shape}")
                # mlp_hidden_states: torch.Size([1, 1536, 12288])
                cache_dic['cache'][-1][layer]['norm'] = mlp_hidden_states
                force_init(cache_dic, current, mlp_hidden_states.shape)
                
                current['module'] = 'attn'
                cache_dic['cache'][-1][layer]['attn'], cache_dic['attn_map'][-1][layer] = self.attn(
                    hidden_states=norm_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )
                # print(f"single attn map:MyCodes/FluxTransformer2DModel.py:120: {cache_dic['attn_map'][-1][layer].shape}，layer：{layer},hidden_states: {hidden_states.shape}")
                force_init(cache_dic, current, hidden_states.shape)
                attn_output = cache_dic['cache'][-1][layer]['attn']
                
                current['module'] = 'mlp'
                hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
                cache_dic['cache'][-1][layer]['mlp']  =  gate * self.proj_out(hidden_states)
                # gate = gate.unsqueeze(1)
                force_init(cache_dic, current, hidden_states.shape)
                hidden_states =cache_dic['cache'][-1][layer]['mlp']


            elif current['type'] == 'ToCa': # Partial Computation: Compute only fresh tokens and save them in cache, no attention token computation in the final version

                # no extra computation for final version, you can add the following line to do some partial computation, but always worse results caused by error propagation mentioned in the paper.
                current['module'] = 'norm'
                fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, hidden_states, current)
                norm_hidden_states, gate = self.norm(fresh_tokens, emb=temb)
                mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
                update_cache(fresh_indices, fresh_tokens=mlp_hidden_states, cache_dic=cache_dic, current=current)

                current['module'] = 'attn' 
                attn_output = cache_dic['cache'][-1][layer]['attn']

                current['module'] = 'mlp'
                hidden_states = torch.cat([attn_output, cache_dic['cache'][-1][layer]['norm']], dim=2)
                fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, hidden_states, current)
                fresh_tokens = gate*self.proj_out(fresh_tokens)
                update_cache(fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic, current=current)
                hidden_states = cache_dic['cache'][-1][layer]['mlp']
                
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states
        else:
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            
            attn_output, attn_map = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            hidden_states = gate * self.proj_out(hidden_states)
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states


@maybe_allow_in_graph
class FluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, qk_norm="rms_norm", eps=1e-6):
        super().__init__()

        self.norm1 = AdaLayerNormZero(dim)

        self.norm1_context = AdaLayerNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = MyFluxAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        # norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        # norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
        #     encoder_hidden_states, emb=temb
        # )
        
        joint_attention_kwargs = joint_attention_kwargs or {}
        use_attn_map = joint_attention_kwargs.get('use_attn_map', False)
        if joint_attention_kwargs['use_cache']:
            # use cache
            # print(f"use cache")
            cache_dic = joint_attention_kwargs['cache_dic']
            current = joint_attention_kwargs['current']
        
            cache_type = cache_dic['cache_type']
            layer = current['layer']
            cal_type(cache_dic=cache_dic, current=current)

            if current['type'] == 'full':
                current['module'] = 'shift'
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )
                cache_dic['cache'][-1][current['layer']][current['module']] = {
                    'gate_msa': gate_msa,
                    'shift_mlp': shift_mlp,
                    'scale_mlp': scale_mlp,
                    'gate_mlp': gate_mlp,
                    'c_gate_msa': c_gate_msa,
                    'c_shift_mlp': c_shift_mlp,
                    'c_scale_mlp': c_scale_mlp,
                    'c_gate_mlp': c_gate_mlp,
                }
                # force_init(cache_dic, current, torch.cat([norm_hidden_states, norm_encoder_hidden_states], dim=1))
                
                current['module'] = 'attn'
                attn_output, context_attn_output, attn_map = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    **joint_attention_kwargs,
                )
                B1,H1,L1=norm_hidden_states.shape
                B2,H2,L2=norm_encoder_hidden_states.shape
                self_attn_map=None
                cross_attn_map=None
                if attn_map is not None:
                    B,H,W = attn_map.shape
                    cross_attn_map = attn_map[:,:H2,H2:]
                    self_attn_map = attn_map[:,H2:,H2:]
                    assert H1+H2==H,f"H1+H2!=H,H1:{H1},H2:{H2},H:{H}"
                
                cache_dic['cache'][-1][current['layer']][current['module']], cache_dic['attn_map'][-1][current['layer']] = attn_output, self_attn_map
                attn_output = gate_msa.unsqueeze(1) * cache_dic['cache'][-1][current['layer']][current['module']]
                force_init(cache_dic, current, attn_output.shape)
                hidden_states = hidden_states + attn_output

                current['module'] = 'mlp'
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                cache_dic['cache'][-1][current['layer']][current['module']] = self.ff(norm_hidden_states)
                force_init(cache_dic, current, norm_hidden_states.shape)
                ff_output = gate_mlp.unsqueeze(1) * cache_dic['cache'][-1][current['layer']][current['module']]
                hidden_states = hidden_states + ff_output

                # Process attention outputs for the `encoder_hidden_states`.
                current['module'] = 'cross-attn'
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                cache_dic['cache'][-1][current['layer']][current['module']], cache_dic['cross_attn_map'][-1][current['layer']] = context_attn_output, cross_attn_map
                force_init(cache_dic, current, context_attn_output.shape)
                encoder_hidden_states = encoder_hidden_states + context_attn_output

                current['module'] = 'cross-mlp'
                norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
                norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
                cache_dic['cache'][-1][current['layer']][current['module']] = context_ff_output
                force_init(cache_dic, current, encoder_hidden_states.shape)
                encoder_hidden_states = encoder_hidden_states+c_gate_mlp.unsqueeze(1) * context_ff_output
                
                
                if encoder_hidden_states.dtype == torch.float16:
                    encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

                return encoder_hidden_states, hidden_states
                
            else:
                current['module'] = 'shift'
                cache_dict = cache_dic['cache'][-1][current['layer']][current['module']]
                gate_msa = cache_dict['gate_msa']
                shift_mlp = cache_dict['shift_mlp'] 
                scale_mlp = cache_dict['scale_mlp']
                gate_mlp = cache_dict['gate_mlp']
                c_gate_msa = cache_dict['c_gate_msa']
                c_shift_mlp = cache_dict['c_shift_mlp']
                c_scale_mlp = cache_dict['c_scale_mlp'] 
                c_gate_mlp = cache_dict['c_gate_mlp']
                
                current['module'] = 'attn'
                attn_output = gate_msa.unsqueeze(1)*cache_dic['cache'][-1][current['layer']][current['module']]
                hidden_states = hidden_states + attn_output

                current['module'] = 'mlp'
                fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, hidden_states, current)
                fresh_tokens = self.norm2(fresh_tokens)
                fresh_tokens = fresh_tokens * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                fresh_tokens = self.ff(fresh_tokens)
                update_cache(fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic, current=current)
                hidden_states = hidden_states + gate_mlp.unsqueeze(1) * cache_dic['cache'][-1][current['layer']][current['module']]

                current['module'] = 'cross-attn'
                context_attn_output = cache_dic['cache'][-1][current['layer']][current['module']]
                context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
                encoder_hidden_states = encoder_hidden_states + context_attn_output
                
                current['module'] = 'cross-mlp'
                fresh_indices, fresh_tokens = cache_cutfresh(cache_dic, encoder_hidden_states, current)
                fresh_tokens = self.norm2_context(fresh_tokens)
                fresh_tokens = fresh_tokens * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
                fresh_tokens = self.ff_context(fresh_tokens)
                update_cache(fresh_indices, fresh_tokens=fresh_tokens, cache_dic=cache_dic, current=current)
                encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * cache_dic['cache'][-1][current['layer']][current['module']]

                if encoder_hidden_states.dtype == torch.float16:
                    encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

                return encoder_hidden_states, hidden_states
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            attn_output, context_attn_output, attn_map = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

            return encoder_hidden_states, hidden_states


class FluxTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, MyFromOriginalModelMixin):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["FluxTransformerBlock", "FluxSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        out_channels: Optional[int] = None, # Make compatible with different configs
    ):
        super().__init__()
        print("init with MyFluxAttnProcessor2_0")
        # Use out_channels if provided, otherwise fall back to in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )
        
        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        
    
    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)


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
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        implement refer to MyFluxForward.py
        """
        print("FluxTransformer2DModel empty forward")
        pass