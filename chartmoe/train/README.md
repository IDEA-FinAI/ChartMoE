<div align="center">
<h1>Training Recipes of ChartMoE</h1>
</div>

<div align="center">
<h3>Datasets are released at 🤗https://huggingface.co/datasets/Coobiw/ChartMoE-Data!</h3>
</div>

In this part, I'll introduct the training recipes for reproducing ChartMoE. Except for the training recipes, I also provided a checkpoint that can be reproduced according to following instructions. You can find it at [🤗](https://huggingface.co/Coobiw/ChartMoE_Reproduced). **This version has better performance on ChartQA(both with & without PoT).**


## Download and Organize the ChartMoE-Data
[🤗ChartMoE Data](https://huggingface.co/datasets/Coobiw/ChartMoE-Data) has been released! You can download it by running:

```bash
cd chartmoe/train
python scripts/chartmoe_data_download.py
```
Datasets will appear at `chartmoe/train/data`.

Then, please unzip these two files.
```bash
unzip ChartMoE-Align.zip
unzip SFT.zip
```

Additionally, I want to announce that the `ChartY_replot` in `ChartMoE-Align` contains data with higher quality and bilingual texts! It may be a good choice to sample more from `ChartY_replot`.

### Data Format
```python
  [
    {
      "id": "0",
      "image": ['path/to/image_0.jpg']
      "conversations": [
        {
          "from": "user",
          "value": "<ImageHere> Please describe these two images in detail."
        },
        {
          "from": "assistant",
          "value": "......"
        }
      ]
    },
    {
      "id": "1",
      "image": ['path/to/image_1.jpg']
      "conversations": [
        {
          "from": "user",
          "value": "<ImageHere> what is the color of the dog"
        },
        {
          "from": "assistant",
          "value": "it is ...."
        }
      ]
    }
  ]
```

## Download InternLM_XComposer2_Enhanced

**Note: I've supported `flash-attn` and `batchified training` for InternLM-XComposer2 on [Coobiw/InternLM-XComposer2_Enhanced](https://huggingface.co/Coobiw/InternLM-XComposer2_Enhanced). This will indeed acclerate training.**

Run:

```bash
cd chartmoe/train
python scripts/internlm_xc2_download.py
```

Then, ChartMoE will appear at `chartmoe/train/ckpt/InternLM-XComposer2_Enhanced`.

## Diversely-Aligned MoE-MLP Training

### Training Pipeline of ChartMoE

![Overview](../../asset/train_pipeline.png)

Run:

```bash
cd chartmoe/train
bash scripts/multi_align.sh
```

Then, the table/json/code MLP connector will appear at `chartmoe/train/output/{}_proj`.format(table/json/code)!

After diversely alignment, we can construct the MoE-MLP connector by running:

```bash
cd chartmoe/train
bash scripts/moe_construction.sh
```

The MoE-MLP connnector will appear at `chartmoe/train/output/moe_aligned/mlp_moe.pth`.

## SFT

*Note: In this Repo, we don't add "High-Quality Knowledge Learning" mid-training.*

Please notice [the path of MoE-MLP connector](./scripts/sft.sh#L24).

Run:

```bash
cd chartmoe/train
mkdir -p logs/sft
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/sft.sh 2>&1 | tee logs/sft/tee_logs.txt
```

## Merge MLP-MoE Connector and LoRA Weights for ChartMoE Construction
Run:

```bash
cd chartmoe/train
bash scripts/chartmoe_construction.sh
```

## Evaluation on ChartQA
w/o PoT:

```bash
CUDA_VISIBLE_DEVICES=0 python chartmoe/eval_ChartQA.py --ckpt_path chartmoe/train/output/sft/chartmoe_reproduced --save_path chartmoe/train/output/sft/chartmoe_reproduced/ChartQA_wo-PoT
```

Result:
```
+-----------+--------+--------+--------+
|    @AP    |  0.05  |  0.1   |  0.2   |
+-----------+--------+--------+--------+
|   Human   | 0.704  | 0.7376 | 0.772  |
| Augmented | 0.9056 | 0.9192 | 0.9352 |
|  Averaged | 0.8048 | 0.8284 | 0.8536 |
+-----------+--------+--------+--------+
```

PoT:
```bash
CUDA_VISIBLE_DEVICES=0 python chartmoe/eval_ChartQA.py --ckpt_path chartmoe/train/output/sft/chartmoe_reproduced --save_path chartmoe/train/output/sft/chartmoe_reproduced/ChartQA_PoT --pot --pot_idx 1
```

Result:
```
+-----------+--------+--------+-------+
|    @AP    |  0.05  |  0.1   |  0.2  |
+-----------+--------+--------+-------+
|   Human   | 0.7952 | 0.8128 | 0.828 |
| Augmented | 0.904  | 0.9176 | 0.932 |
|  Averaged | 0.8496 | 0.8652 |  0.88 |
+-----------+--------+--------+-------+
```
