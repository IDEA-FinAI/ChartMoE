"""
    FEATURE: Construct ChartMoE after Post-Training, including: 1. Merge LoRA to LLM 2. Adapt to ChartMoE HF Implementation
    AUTHOR: Brian Qu
    URL: https://arxiv.org/abs/2409.03277
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from mlp_moe import MLPMoE
import os
import shutil


@dataclass
class ScriptArguments:
    """The input names representing the Adapter and Base model fine-tuned with
    PEFT, and the output name representing the merged model."""

    moe_aligned_pth_path: Optional[str] = field(
        default=None, metadata={'help': 'the path of aligned moe .pth file'}
    )
    chartmoe_hf_dir: Optional[str] = field(
        default=None, metadata={'help': 'the path of downloaded chartmoe hf dir'}
    )
    adapter_model_name: Optional[str] = field(
        default=None, metadata={'help': 'the adapter name'}
    )
    output_path: Optional[str] = field(
        default=None, metadata={'help': 'the merged model saved path'}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.moe_aligned_pth_path is not None, 'please provide the path of aligned moe .pth file'
assert script_args.adapter_model_name is not None, 'please provide the name of the Adapter you would like to merge'
assert script_args.output_path is not None, 'please provide the the merged model saved path'
assert script_args.chartmoe_hf_dir is not None, 'please provide the path of downloaded chartmoe hf dir'

# get base model path from adapter_config.json
peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
base_model_path = peft_config.base_model_name_or_path
print(f"\033[31mLoad base model from {base_model_path}\033[0m")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda",
    attn_implementation="eager",
)

mlp_moe_state_dict = torch.load(script_args.moe_aligned_pth_path, map_location="cpu")

num_experts = mlp_moe_state_dict['gate.weight'].size(0)
num_selected = mlp_moe_state_dict.pop('num_selected')
    
mlp_moe = MLPMoE(num_experts, num_selected, 1024, 4096).to(model.device)
mlp_moe.load_state_dict(mlp_moe_state_dict)

print("\033[32mload aligned moe...\033[0m")
model.vision_proj = mlp_moe

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path, trust_remote_code=True
)

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

print("\033[33mmerge the lora weights and `modules_to_save` weights\033[0m")
model = model.merge_and_unload()

model.save_pretrained(f'{script_args.output_path}')
tokenizer.save_pretrained(f'{script_args.output_path}')

print("\033[34msave and adapt to `ChartMoE` format...\033[0m")
# adjust the contained files
chartmoe_files = [
    'special_tokens_map.json', 
    'configuration_chartmoe.py', 
    'modeling_internlm2.py', 
    'README.md', 
    'config.json', 
    'generation_config.json', 
    '.gitattributes', 
    'teaser.png', 
    'zero_to_fp32.py', 
    'pytorch_model.bin.index.json', 
    'tokenization_internlm_xcomposer2.py', 
    'build_mlp.py', 
    'tokenizer_config.json', 
    'build_moe_connector.py', 
    'tokenizer.model', 
    'modeling_chartmoe.py',
]
keep_files = [
    'pytorch_model-00001-of-00002.bin', 
    'pytorch_model-00002-of-00002.bin'
]

for fn in os.listdir(script_args.output_path):
    if fn not in keep_files:
        os.remove(os.path.join(script_args.output_path, fn))

for fn in chartmoe_files:
    if fn != 'config.json':
        shutil.copy(os.path.join(script_args.chartmoe_hf_dir, fn), os.path.join(script_args.output_path, fn))
    else:
        import json
        config = json.load(open(os.path.join(script_args.chartmoe_hf_dir, fn), encoding='utf-8'))
        config["num_experts"] = num_experts
        config["num_selected"] = num_selected
        with open(os.path.join(script_args.output_path, fn), 'w', encoding='utf-8') as fo:
            json.dump(config, fo, indent=4, ensure_ascii=False)

