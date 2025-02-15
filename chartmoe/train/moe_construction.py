"""
    FEATURE: Construct MLP-MoE Connector with three aligned MLP and the general MLP Connector
    AUTHOR: Brian Qu
    URL: https://arxiv.org/abs/2409.03277
"""
from mlp_moe import MLPMoE
import argparse
from glob import glob
import os
from copy import deepcopy

import torch
import torch.nn as nn
import transformers

def main(args):
    root = args.root_dir
    table_ckpt = glob(f"{root}/table_proj/checkpoint-*/mm_mlp.bin")[0]
    json_ckpt = glob(f"{root}/json_proj/checkpoint-*/mm_mlp.bin")[0]
    code_ckpt = glob(f"{root}/code_proj/checkpoint-*/mm_mlp.bin")[0]
    
    base_model = transformers.AutoModel.from_pretrained(
                    args.base_model,
                    trust_remote_code=True,
                    device_map="cuda",
                    attn_implementation="eager",
                )
    base_proj = deepcopy(base_model.vision_proj)
    del base_model
    
    table_proj = torch.load(table_ckpt)
    json_proj = torch.load(json_ckpt)
    code_proj = torch.load(code_ckpt)
    
    mlp_moe = MLPMoE(args.mlp_smoe_experts, args.mlp_smoe_topk, 1024, 4096)
    for idx, expert in enumerate(mlp_moe.experts):
        print(idx % args.mlp_smoe_experts)
        if idx % args.mlp_smoe_experts == 0:
            for target_layer, source_layer in zip(expert, base_proj):
                if isinstance(target_layer, nn.Linear) and isinstance(source_layer, nn.Linear):
                    target_layer.weight = deepcopy(source_layer.weight)
                    target_layer.bias = deepcopy(source_layer.bias)
            print(f"{idx} expert: load base_proj")
        if idx % args.mlp_smoe_experts == 1:
            for ii in [0,2]:
                expert[ii].weight.data = table_proj[f'vision_proj.{ii}.weight']
                expert[ii].bias.data = table_proj[f'vision_proj.{ii}.bias']
            print(f"{idx} expert: load table_proj")
        if idx % args.mlp_smoe_experts == 2:
            for ii in [0,2]:
                expert[ii].weight.data = json_proj[f'vision_proj.{ii}.weight']
                expert[ii].bias.data = json_proj[f'vision_proj.{ii}.bias']
            print(f"{idx} expert: load json_proj")
        if idx % args.mlp_smoe_experts == 3:
            for ii in [0,2]:
                expert[ii].weight.data = code_proj[f'vision_proj.{ii}.weight']
                expert[ii].bias.data = code_proj[f'vision_proj.{ii}.bias']
            print(f"{idx} expert: load code_proj")
        
    os.makedirs(f"{root}/{args.save_name}", exist_ok=True)
    mlp_moe_state_dict = mlp_moe.state_dict()
    mlp_moe_state_dict['num_selected'] = args.mlp_smoe_topk
    torch.save(mlp_moe_state_dict, f"{root}/{args.save_name}/mlp_moe.pth")
    
                        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--mlp_smoe_experts", type=int, default=4)
    parser.add_argument("--mlp_smoe_topk", type=int, default=2)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--save_name", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
    