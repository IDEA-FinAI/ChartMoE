CUDA_VISIBLE_DEVICES=0 python chartmoe_construction.py \
    --moe_aligned_pth_path output/moe_aligned/mlp_moe.pth \
    --chartmoe_hf_dir ckpt/chartmoe \
    --adapter_model_name output/sft \
    --output_path output/sft/chartmoe_reproduced