# Neuro-VL
> University of Michigan


## Overview
![Neuro-VL overview](docs/figure_1.png)

## Updates
**(2026-11)** Release Neuro-VL website https://neurovl.mlins.org/, model weight [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/mlinslab/neuro-vl-8b-thinking), evaluation benchmark [![huggingface Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-yellow)](https://huggingface.co/datasets/mlinslab/neurobench), and this official implementation [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/MLNeurosurg/neuro-vl).

## Getting Started

### Install 
```bash
git@github.com:zch0414/neuro-vl.git
cd neuro-vl
conda env create -f ./environment.yaml
conda activate neuro_vl
```

### Model
Download the model from [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/mlinslab/neuro-vl-8b-thinking):
```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download(
    repo_id="mlinslab/neuro-vl-8b-thinking",
    local_dir="/path/to/neuro-vl-8b-thinking",
    token=True
)

print(f"Checkpoint folder: {model_dir}")
```

### Demo
Join the waitlist for the online demo at TODO, or download the model and run a demo locally:
```bash
python demo.py --model /path/to/neuro-vl-8b-thinking/ --input CQ500CT107 --user_prompt "What do you see?"
```

The <code>--input</code> parameter controls which data is used. <code>CQ500CT107</code>, <code>ReMIND-071/study_1</code>, and <code>ReMIND-071/study_2</code> are three single-study examples, while <code>ReMIND-071</code> is a longitudinal example.

Use <code>--provide_scan_type</code> to explicitly provide the scan type for each input scan to the model. Otherwise, the model will predict the scan type based on the image feature.

### Evaluation
Download the processed tensor dataset from [![huggingface Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-yellow)](https://huggingface.co/datasets/mlinslab/neurobench):
```python
from pathlib import Path
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mlinslab/neurobench",
    repo_type="dataset",
    allow_patterns="data/tensor/**",
    local_dir="./neurobench",
)

root = Path("./neurobench/data/tensor/rsna")
for batch in root.glob("batch_[0-4]"):
    for patient in batch.iterdir():
        patient.rename(root / patient.name)
    batch.rmdir()
```

Run the scripts in `/src/neuro_vl_test/scripts/` to generate model responses for each question. To score the generated responses, follow the instructions in the NeuroBench repository on [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/MLNeurosurg/neurobench).


### Training
Pre-train your visual encoder using the HLIP repository on [![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white)](https://github.com/zch0414/hlip) or directly download from [![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-yellow)](https://huggingface.co/zch0414/clip-vit_large-slice_scan_study-dualdinotxt1568). Then, run the commands below from `src/neuro_vl_sft/`:



**Stage One**

```bash
export WANDB_PROJECT=neuro-vl

accelerate launch --config_file ./fsdp_config.yaml main.py \
--stage stage-one \
--train_data /path/to/mri/='"/path/to/hlip/mri/train.txt"' /path/to/ct/='"/path/to/hlip/ct/train.txt"' \
--valid_data /path/to/mri/='"/path/to/hlip/mri/valid.txt"' /path/to/ct/='"/path/to/hlip/ct/valid.txt"' \
--max_num_scans 8 \
--visual_resolution 8 7 7 \
--context_length 4096 \
--visual_encoder_name vit_large_slice_scan_study_dualdinotxt1568 \
--visual_encoder_path /path/to/hlip.pt \
--language_model_name_or_path /path/to/Qwen3-8B \
--system_message "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." \
--output_dir /path/to/neuro_vl/ \
--freeze_visual_encoder \
--freeze_language_model \
--gradient_checkpointing \
--num_train_epochs 1 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 1 \
--dataloader_num_workers 16 \
--gradient_accumulation_steps 8 \
--learning_rate 1e-3 \
--lr_scheduler_type cosine \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--bf16 \
--tf32
```

**Stage Two**

Use the connector checkpoint saved by Stage One.

```bash
export WANDB_PROJECT=neuro-vl

accelerate launch --config_file ./fsdp_config.yaml main.py \
--stage stage-two \
--train_data /path/to/mri/='"/path/to/neuro_vl/mri/train.txt"' /path/to/ct/='"/path/to/neuro_vl/ct/train.txt"' \
--valid_data /path/to/mri/='"/path/to/neuro_vl/mri/valid.txt"' /path/to/ct/='"/path/to/neuro_vl/ct/valid.txt"' \
--max_num_scans 8 \
--visual_resolution 8 7 7 \
--context_length 5632 \
--visual_encoder_name vit_large_slice_scan_study_dualdinotxt1568 \
--visual_encoder_path /path/to/hlip.pt \
--connector_path /path/to/neuro_vl/run_stage_one/connector_checkpoint.pt \
--language_model_name_or_path /path/to/Qwen3-8B \
--use_confession_statement \
--enable_thinking \
--system_message "You are Neuro-VL, created by MLiNS @ University of Michigan. You are a helpful assistant." \
--output_dir /path/to/neuro_vl/ \
--gradient_checkpointing \
--num_train_epochs 1 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 1 \
--dataloader_num_workers 16 \
--gradient_accumulation_steps 8 \
--learning_rate 2e-5 \
--visual_encoder_learning_rate 2e-6 \
--lr_scheduler_type cosine \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--bf16 \
--tf32
```

**Stage Three**

Use the visual encoder, connector, and language model checkpoints saved by Stage Two.

```bash
export WANDB_PROJECT=neuro-vl

accelerate launch --config_file ./fsdp_config.yaml main.py \
--stage stage-three \
--train_data /path/to/mri/='"/path/to/neuro_vl/mri/train_longitudinal.csv"' /path/to/ct/='"/path/to/neuro_vl/ct/train_longitudinal.csv"' \
--valid_data /path/to/mri/='"/path/to/neuro_vl/mri/valid_longitudinal.csv"' /path/to/ct/='"/path/to/neuro_vl/ct/valid_longitudinal.csv"' \
--max_num_scans 16 \
--visual_resolution 8 7 7 \
--encode_studies_independently \
--context_length 8192 \
--visual_encoder_name vit_large_slice_scan_study_dualdinotxt1568 \
--visual_encoder_path /path/to/neuro_vl/run_stage_two/vit_large_slice_scan_study_dualdinotxt1568.pt \
--connector_path /path/to/neuro_vl/run_stage_two/connector_checkpoint.pt \
--language_model_name_or_path /path/to/neuro_vl/run_stage_two/Qwen3-8B \
--enable_thinking \
--system_message "You are Neuro-VL, created by MLiNS @ University of Michigan. You are a helpful assistant." \
--output_dir /path/to/neuro_vl/ \
--gradient_checkpointing \
--num_train_epochs 1 \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 1 \
--dataloader_num_workers 16 \
--gradient_accumulation_steps 16 \
--learning_rate 1e-5 \
--visual_encoder_learning_rate 2e-6 \
--lr_scheduler_type cosine \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--bf16 \
--tf32
```

## Citation
If you find this repository helpful, please consider citing:
```bib
@article{zhao2026towards,
  title={Towards Scalable Language-Image Pre-training for 3D Medical Imaging},
  author={Chenhui Zhao and Yiwei Lyu and Asadur Zaman Chowdury and Edward S Harake and Akhil Kondepudi and Akshay T Rao and Xinhai Hou and Honglak Lee and Todd C Hollon},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=WxHf4EcBWA}
}
```
