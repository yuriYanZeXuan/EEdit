# EEdit
EEdit⚡: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing

# install
```
conda create -n eedit python=3.12.3
conda activate eedit
pip install -r EEdit/requirements.txt
```

# checkpoints & datasets
For masks generated from mapping_file.json, we provide scripts that follows:
`python MyCodes/myutils.py`

<details>
<summary>checkpoints structure</summary>
weights
├── flux1-dev.safetensors
├── model_index.json
├── scheduler
│   └── scheduler_config.json
├── sd_vae_ft_mse
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── text_encoder
│   ├── config.json
│   └── model.safetensors
├── text_encoder_2
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   ├── model.safetensors.index-1.json
│   └── model.safetensors.index.json
├── tokenizer
│   ├── merges-1.txt
│   ├── merges.txt
│   ├── special_tokens_map-1.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config-1.json
│   ├── tokenizer_config.json
│   ├── vocab-1.json
│   └── vocab.json
├── tokenizer_2
│   ├── special_tokens_map-1.json
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── transformer_config.json
└── vae
    ├── config.json
    └── diffusion_pytorch_model.safetensors
</details>


<details>
<summary>datasets structure</summary>
input
├── composition
│   ├── Real-Cartoon
│   │   ├── 0000 a cartoon animation of a sheep in the forest
│   │   │   ├── bg58.png
│   │   │   ├── cp_bg_fg.jpg
│   │   │   ├── dccf_image.jpg
│   │   │   ├── fg35_63d22cda1f5b66e8e5aca776.jpg
│   │   │   ├── fg35_mask.png
│   │   │   └── mask_bg_fg.jpg
│   ├── Real-Painting
│   ├── Real-Sketch
│   ├── Real-Real
    ...

├── drag_data
│   ├── dragbench-dr
│   │   ├── animals
│   │   │   ├── JH_2023-09-14-1820-16
│   │   │   │   ├── meta_data.pkl
│   │   │   │   ├── meta_data_region.pkl
│   │   │   │   ├── original_image.png
│   │   │   │   └── user_drag.png
│   └── dragbench-sr
│       ├── art_0
│       │   ├── meta_data.pkl
│       │   ├── meta_data_region.pkl
│       │   ├── original_image.png
│       │   └── user_drag.png
    ...


├── inpaint
│   ├── annotation_images 
│   │   ├── 0_random_140
│   │   │   ├── 000000000000.jpg
│   │   │   
│   │   ├── 1_change_object_80
│   │   │   ├── 1_artificial
│   │   │   │   ├── 1_animal
│   │   │   │   │   ├── 111000000000.jpg
│   │   │   │   ├── 2_human
│   │   │   │   │   ├── 112000000000.jpg
│   │   ├── 2_add_object_80
│   │   │   ├── 1_artificial
│   │   │   │   ├── 1_animal
│   │   │   │   │   ├── 211000000000.jpg
        ...
│   ├── mapping_file.json
│   └── masks
│       ├── mask-000.png
|       ├── mask-001.png
|       ├── ...

</details>

# generation
`source run_gen.sh`