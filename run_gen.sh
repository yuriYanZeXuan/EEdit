#!/bin/bash
echo "Current PID: $$"
# log start time
start_time=$(date +%s)

output_dir=./EEdit_outputs
weights_dir=./weights

# -----------------------------------------------------------------------------------
# composition Generation
img_config=composition_images
cfgs_comp=(
    "cache_configs"
)

for cfg in "${cfgs_comp[@]}"; do
    python composition_gen.py --weights_dir $weights_dir --config_path ./configs/composition/${cfg}.json --img_config ./configs/composition/$img_config.json --output_dir ${output_dir}/comp/${cfg} --use_predefine 1
done


# -----------------------------------------------------------------------------------
# Prompt-guided edit
cfgs_edit=(
    cache_configs
)

for cfg in "${cfgs_edit[@]}"; do
    output_dir_tag="inpaint/${cfg}"
    config_path="./configs/inpaint/$cfg.json"
    img_config_path="./configs/inpaint/prompt_images.json"
    python inpaint_gen.py --weights_dir $weights_dir --config_path $config_path --img_config $img_config_path --output_dir ${output_dir}/${output_dir_tag} --use_predefine 1
done

# -----------------------------------------------------------------------------------
# Dragging
cfgs_drag=(
    cache_configs
)

for cfg in "${cfgs_drag[@]}"; do
    config_path=./configs/drag/$cfg.json
    img_config_path=./configs/drag/dragging_images.json
    
    python drag_gen.py --weights_dir $weights_dir --config_path $config_path --img_config $img_config_path --output_dir ${output_dir}/drag/${cfg} --use_predefine 1
done

echo "All completed!"
# calculate total time
echo "Calculating total time..."
end_time=$(date +%s)
total_time=$((end_time - start_time))
hours=$((total_time / 3600))
minutes=$(( (total_time % 3600) / 60 ))
seconds=$((total_time % 60))

echo "Total time: ${hours} hours ${minutes} minutes ${seconds} seconds"
