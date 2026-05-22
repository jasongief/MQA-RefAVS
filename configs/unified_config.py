
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers

@dataclass
class ModelArguments:  
    # llm
    model_name_or_path: Optional[str] = field(default="")
    freeze_backbone: bool = field(default=True, metadata={"help": "Whether to freeze the LLM backbone."})
    llm_name: str = field(default='llama')
    ## visual module
    vit_ckpt_path: str = field(default='')
    select_layer_list = [14,22,23]  # [-11,-2,-1]
    select_feature: str = field(default='patch')
    image_size: int = field(default=224)
    patch_size: int = field(default=14)
    visual_query_token_nums: int = field(default=32)
    ## audio module
    BEATs_ckpt_path: str = field(default='') 
    audio_query_token_nums: int = field(default=32)
    ## seg module
    prompt_embed_dim: int = field(default=256)
    mask_decoder_transformer_depth: int = field(default=2)
    low_res_mask_size: int = field(default=112)
    image_scale_nums: int = field(default=2)
    token_nums_per_scale: int = field(default=3)
    avs_query_num: int = field(default=300)
    num_classes: int = field(default=1)
    query_generator_num_layers: int = field(default=2)


@dataclass
class InferenceArguments:
    # used for inference
    ckpt_dir: str = field(default='')
    avs_ckpt_dir: str = field(default='')
    nonlora_ckpt_dir: str = field(default='')
    
    # for infer avs
    # avs_ckpt_dir: str = field(default='')
    avss_ckpt_dir: str = field(default='')
    adapter_ckpt_path: str = field(default=None)
    test_name: str = field(default='test') # for ref-avs: test_u,test_s,test_n

    device: str = field(default='cuda:0')
    

@dataclass
class DataArguments:
    # pretrain
    video_frame_nums: int = field(default=8)
    image_size = ModelArguments.image_size
    image_caption_task: bool = field(default=False)
    video_caption_task: bool = field(default=False)
    audio_caption_task: bool = field(default=False)
    segmentation_task: bool = field(default=False)
    # fine-tune
    avqa_task: bool = field(default=False)
    ave_task: bool = field(default=False)
    avvp_task: bool = field(default=False)
    arig_task: bool = field(default=False)
    ms3_task: bool = field(default=False)
    s4_task : bool = field(default=False)
    avss_task: bool = field(default=False)
    avcap_task: bool = field(default=False)
    ref_avs_task: bool = field(default=False)
    #  training
    refavs_meta_csv_path: str = field(default="../MQ-RAVSBench/train_test_meta_files/metadata.csv")
    
    refavs_test_image_json_path: str = field(default="../MQ-RAVSBench/train_test_meta_files/test_s_image_filtered.json")
    refavs_test_u_image_json_path: str = field(default="../MQ-RAVSBench/train_test_meta_files/test_u_image_filtered.json")
    refavs_test_video_json_path: str = field(default="../MQ-RAVSBench/train_test_meta_files/test_s_video_filtered.json")
    refavs_test_u_video_json_path: str = field(default="../MQ-RAVSBench/train_test_meta_files/test_u_video_filtered.json")

    refavs_cot_json_path: str = field(default="../MQ-RAVSBench/train_test_meta_files/train_audit_only_filtered.json")

    refavs_data_root: str = field(default="../MQ-RAVSBench")
    # prior method whose predicted masks are audited: eemc / tgsagent
    refavs_prior_method: str = field(default="tgsagent")

    # image / video
    refavs_eval_mode: str = field(default="image")
    # 评测时可选固定某一类 mask：perfect/cutout/erode/dilate/merge/full_neg/null/all
    refavs_mask_type_filter: str = field(default="all")
    # 对 cutout/erode/dilate 支持 rank 过滤（1/2），-1 表示不过滤
    refavs_mask_rank_filter: int = field(default=-1)
    # mask 编码方式：
    # - mask: 仅输入二值 mask；
    # - masked_image: 将 mask 作用到原帧后送入视觉编码；
    # - both: 在 mask / masked_image 间随机二选一（向后兼容）；
    # - mask_and_masked_frame: 同时提供二者用于消融。
    refavs_mask_encode_mode: str = field(default="masked_image")
    # 训练时希望在一个 batch 内平衡正负样本比例，默认一半 gt、一半各种负样本
    refavs_pos_ratio: float = field(default=0.5)
    multi_frames: bool = field(default=False) # avs task input single frame
    
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the data directory"}
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr: Optional[float] = None
    freeze_mm_mlp_adapter: bool = field(default=False)
    remove_unused_columns: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)
    # Training Data Arguments 
    group_by_modality_length: bool = field(default=False)
    # model_max_length: int = field(
    #     default=512,
    #     metadata={
    #         "help":
    #         "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    #     },
    # )
    # Lora or Quant Arguments
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=32,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    # use_hyper_lora

    ce_loss_weight: float = field(default=1.0)
    dice_loss_weight: float = field(default=0.5)
    bce_loss_weight: float = field(default=2.0)
    # output_loss_weight: float = field(default=1.0)
    iou_loss_weight: float = field(default=1.0)

    audio_branch: bool = field(default=False)
    visual_branch: bool = field(default=False)
    seg_branch: bool = field(default=False)

    pretrain_ckpt_dir: str = field(default='')
    finetune_ckpt_dir: str = field(default='')

    save_modules: str = field(default='vl_projector,al_projector,lora')

    exp_desc: str = field(default='exp')

    use_process: bool = field(default=True)

    use_hyper_lora: bool = field(default=True)

    
    # data_path: str = field(
    #     default=None,
    #     metadata={"help": "Path to the data directory"}
    # )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "Evaluation strategy to use"}
    )
    do_train: bool = field(default=False)
