# MQA-RefAVS

Official implementation of **Audit After Segmentation: Reference-Free Mask Quality Assessment for Language-Referred Audio-Visual Segmentation**.

We introduce Mask Quality Assessment under the Ref-AVS context (**MQA-RefAVS**), a new task that aims to automatically infer the quality of candidate segmentation masks without access to ground-truth annotations. Given a video, audio, referring expression, frame image, and candidate segmentation mask, it aims to estimate the IoU,predict the mask type and an audit action. 

This project is released with MQ-RAVSBench and can be viewed as a sister project to [TGS-Agent](https://github.com/jasongief/TGS-Agent), focusing on reference-free mask quality assessment after language-referred audio-visual segmentation.

- Paper: https://arxiv.org/pdf/2602.03892
- Dataset: https://huggingface.co/datasets/Jinxing1/MQ-RAVSBench
- Pretrained weights: https://huggingface.co/Jinxing1/MQ-Auditor

## Repository Structure

```text
MQA-RefAVS/
  configs/                 # Dataclass argument definitions
  dataset/                 # Dataset loaders and collators; MQ-RAVSBench is downloaded separately
  deepspeed/               # DeepSpeed configs
  models/                  # Llama-based multimodal auditor modules
  pretrained_weights/      # Local upstream model/checkpoint placeholders
  scripts/finetune/        # MQ-Auditor training and evaluation entry points
  scripts/pretrain/        # Optional audio/visual pretraining scripts
  utils/                   # Training, evaluation, and checkpoint utilities
```

## Paths

Run commands from the repository root. The default layout is:

```text
parent_dir/
  MQA-RefAVS/
    pretrained_weights/
    checkpoints/
      MQ-Auditor/
  MQ-RAVSBench/
```

Default paths:

```text
PRETRAINED_WEIGHTS_DIR=pretrained_weights
MQ_RAVSBENCH_DIR=../MQ-RAVSBench
MQ_AUDITOR_CKPT_DIR=checkpoints/MQ-Auditor
```

If your files are stored elsewhere, edit these variables at the top of `scripts/finetune/finetune_hyperlora.sh` and `scripts/finetune/inference_hyper_lora.sh`.

## Installation

```bash
conda create -n mqa python=3.10 -y
conda activate mqa
pip install -r requirements.txt
```


## Data

Download MQ-RAVSBench from Hugging Face and place it next to this repository:

```bash
cd MQA-RefAVS
huggingface-cli download Jinxing1/MQ-RAVSBench \
  --repo-type dataset \
  --local-dir ../MQ-RAVSBench
```

The default scripts expect:

```text
../MQ-RAVSBench/train_test_meta_files/metadata.csv
../MQ-RAVSBench/train_test_meta_files/train_audit_only_filtered.json
../MQ-RAVSBench/train_test_meta_files/test_s_image_filtered.json
../MQ-RAVSBench/train_test_meta_files/test_u_image_filtered.json
../MQ-RAVSBench/train_test_meta_files/test_s_video_filtered.json
../MQ-RAVSBench/train_test_meta_files/test_u_video_filtered.json
```


## Pretrained Weights

This repository keeps only empty placeholder directories under `pretrained_weights/`. Download the required upstream files locally before running training or evaluation.

By default, place upstream weights under `MQA-RefAVS/pretrained_weights/`:

```text
pretrained_weights/
  Llama-2-7b-chat-hf/
  clip-vit-large-patch14/
  google-bert-base-uncased/
  BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
  audio_pretrain.bin
  visual_pretrain.bin
```

The first three entries are directories containing the local Hugging Face model files for Llama-2-7B-Chat, CLIP ViT-L/14, and BERT-base. `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` is the BEATs checkpoint. `audio_pretrain.bin` and `visual_pretrain.bin` are used by the MQ-Auditor training script to initialize the audio and visual projectors.

For upstream checkpoint preparation, also refer to the setup used by [TGS-Agent](https://github.com/jasongief/TGS-Agent).

Download the released MQ-Auditor checkpoint separately:

```bash
huggingface-cli download Jinxing1/MQ-Auditor \
  --local-dir checkpoints/MQ-Auditor
```

The evaluation script expects:

```text
checkpoints/MQ-Auditor/
  non_lora_trainables.bin
  checkpoint-960/
    finetune_weights.bin
```

## Evaluation

Run the released checkpoint on MQ-RAVSBench:

```bash
cd MQA-RefAVS
bash scripts/finetune/inference_hyper_lora.sh
```

Configure evaluation options at the top of `scripts/finetune/inference_hyper_lora.sh`:

```text
TEST_NAME=test_s                 # test_s or test_u
REFAVS_EVAL_MODE=image           # image or video
REFAVS_MASK_TYPE_FILTER=perfect  # perfect, merge, full_neg, cutout, erode, or dilate
REFAVS_MASK_RANK_FILTER=-1       # -1 for all masks; 1 for Hard; 2 for Medium hard
DEVICE=cuda:0
```

Use the following fixed mask-type/rank settings for evaluation:

```text
perfect: -1
merge: -1
full_neg: -1
cutout: 1 or 2
erode: 1 or 2
dilate: 1 or 2
```

For `cutout`, `erode`, and `dilate`, rank `1` denotes Hard (H) samples and rank `2` denotes Medium hard (M) samples. For `perfect`, `merge`, and `full_neg`, rank `-1` evaluates all masks of that type.


## Training

The reference fine-tuning entry point is:

```bash
cd MQA-RefAVS
bash scripts/finetune/finetune_hyperlora.sh
```

The released model was trained with the following main setting:

```text
epochs96_lr1e-4_bs4_gradacc8_lora_r32alpha64_pos0.5_ioulosswei0
```

Default MQ-RAVSBench training inputs:

```text
../MQ-RAVSBench/train_test_meta_files/metadata.csv
../MQ-RAVSBench/train_test_meta_files/train_audit_only_filtered.json
```

Default mask input mode:

```text
mask_and_masked_frame
```

## License

The MQ-Auditor source code is released under the MIT License.
MQ-RAVSBench and the released MQ-Auditor weights are provided for non-commercial research purposes only under CC BY-NC-SA 4.0-style terms. Since MQ-RAVSBench incorporates videos and annotations from previous datasets, including Ref-AVSBench and AVSBench, users must also comply with the licenses and terms of the original datasets. 

## Citation

```bibtex
@article{zhou2026audit,
  title={Audit After Segmentation: Reference-Free Mask Quality Assessment for Language-Referred Audio-Visual Segmentation},
  author={Zhou, Jinxing and Zhou, Yanghao and Wang, Yaoting and Han, Zongyan and Ma, Jiaqi and Ding, Henghui and Anwer, Rao Muhammad and Cholakkal, Hisham},
  journal={arXiv preprint arXiv:2602.03892},
  year={2026}
}
```
