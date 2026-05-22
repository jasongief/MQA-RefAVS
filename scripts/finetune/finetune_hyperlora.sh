# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=4
MASTER_PORT=6666
RANK=0

PRETRAINED_WEIGHTS_DIR="${PRETRAINED_WEIGHTS_DIR:-pretrained_weights}"
MQ_RAVSBENCH_DIR="${MQ_RAVSBENCH_DIR:-../MQ-RAVSBench}"

llama2_ckpt_path="${LLAMA2_CKPT_PATH:-${PRETRAINED_WEIGHTS_DIR}/Llama-2-7b-chat-hf}"
clip_ckpt_path="${CLIP_CKPT_PATH:-${PRETRAINED_WEIGHTS_DIR}/clip-vit-large-patch14}"
beats_ckpt_path="${BEATS_CKPT_PATH:-${PRETRAINED_WEIGHTS_DIR}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt}"

# Training Arguments
NUM_TRAIN_EPOCHS=96
LEARNING_RATE=1e-4
LOCAL_BATCH_SIZE=4 
GRADIENT_ACCUMULATION_STEPS=8
GLOBAL_BATCH_SIZE=$WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE*$GRADIENT_ACCUMULATION_STEPS

IOU_LOSS_WEIGHT=0 # IoU regression head disabled; keep flag for compatibility
REFAVS_POS_RATIO=0.5 


LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05

BASE_META_DIR="${MQ_RAVSBENCH_DIR}/train_test_meta_files"

#! audit only for SFT
REFAVS_COT_JSON_PATH="$BASE_META_DIR/train_audit_only_filtered.json"

#! Train data csv path
REFAVS_META_CSV_PATH="$BASE_META_DIR/metadata.csv"


#! mask / masked_image  / mask_and_masked_frame 
REFAVS_MASK_ENCODE_MODE="mask_and_masked_frame"  
WANDB_PROJECT_NAME=Auditonly_${REFAVS_MASK_ENCODE_MODE}


export RUN_NAME="epochs${NUM_TRAIN_EPOCHS}_lr${LEARNING_RATE}_bs${LOCAL_BATCH_SIZE}_gradacc${GRADIENT_ACCUMULATION_STEPS}_lora_r${LORA_R}alpha${LORA_ALPHA}_pos${REFAVS_POS_RATIO}_ioulosswei${IOU_LOSS_WEIGHT}"
OUTP_DIR=results_epoch${NUM_TRAIN_EPOCHS}
# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT=${WANDB_PROJECT_NAME}
export TOKENIZERS_PARALLELISM='true'
export ASCEND_LAUNCH_BLOCKING='1'



torchrun --nproc_per_node $NPROC_PER_NODE \
    --master_port $MASTER_PORT \
    scripts/finetune/finetune_hyperlora.py \
    --deepspeed deepspeed/stage2-offload.json \
    --llm_name llama \
    --model_name_or_path "$llama2_ckpt_path" \
    --exp_desc "exp" \
    --freeze_backbone True \
    --lora_enable True \
    --bits 32 \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --pretrain_ckpt_dir "$PRETRAINED_WEIGHTS_DIR" \
    --ref_avs_task True \
    --refavs_mask_encode_mode ${REFAVS_MASK_ENCODE_MODE} \
    --refavs_pos_ratio ${REFAVS_POS_RATIO} \
    --refavs_cot_json_path $REFAVS_COT_JSON_PATH \
    --refavs_meta_csv_path $REFAVS_META_CSV_PATH \
    --refavs_data_root "$MQ_RAVSBENCH_DIR" \
    --save_modules vl_projector,al_projector,lora \
    --multi_frames False \
    --visual_branch True \
    --video_frame_nums 10 \
    --vit_ckpt_path "$clip_ckpt_path" \
    --select_feature patch \
    --image_size 224 \
    --patch_size 14 \
    --visual_query_token_nums 32 \
    --audio_branch True \
    --BEATs_ckpt_path "$beats_ckpt_path" \
    --audio_query_token_nums 32 \
    --seg_branch False \
    --prompt_embed_dim 256 \
    --mask_decoder_transformer_depth 2 \
    --low_res_mask_size 112 \
    --image_scale_nums 2 \
    --token_nums_per_scale 3 \
    --avs_query_num 300 \
    --num_classes 1 \
    --query_generator_num_layers 2 \
    --iou_loss_weight $IOU_LOSS_WEIGHT \
    --ce_loss_weight 1.0 \
    --dice_loss_weight 0.5 \
    --bce_loss_weight 1.0 \
    --output_dir $OUTP_DIR/$WANDB_PROJECT/$RUN_NAME \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --ddp_find_unused_parameters True \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps -1 \
    --save_total_limit $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --half_precision_backend "auto" \
    --dataloader_num_workers 4 \
    --report_to all \
