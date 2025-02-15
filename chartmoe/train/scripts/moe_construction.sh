CUDA_VISIBLE_DEVICES=0 python moe_construction.py \
    --base_model ckpt/InternLM-XComposer2_Enhanced \
    --root_dir ./output \
    --save_name moe_aligned