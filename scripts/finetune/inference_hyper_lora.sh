#!/bin/bash

WORLD_SIZE=1
NPROC_PER_NODE=1
MASTER_PORT=6666
RANK=0

PRETRAINED_WEIGHTS_DIR="${PRETRAINED_WEIGHTS_DIR:-pretrained_weights}"
MQ_RAVSBENCH_DIR="${MQ_RAVSBENCH_DIR:-../MQ-RAVSBench}"
MQ_AUDITOR_CKPT_DIR="${MQ_AUDITOR_CKPT_DIR:-checkpoints/MQ-Auditor}"

llama2_ckpt_path="${LLAMA2_CKPT_PATH:-${PRETRAINED_WEIGHTS_DIR}/Llama-2-7b-chat-hf}"
clip_ckpt_path="${CLIP_CKPT_PATH:-${PRETRAINED_WEIGHTS_DIR}/clip-vit-large-patch14}"
beats_ckpt_path="${BEATS_CKPT_PATH:-${PRETRAINED_WEIGHTS_DIR}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt}"

BASE_META_DIR="${MQ_RAVSBENCH_DIR}/train_test_meta_files"

REFAVS_TEST_IMAGE_JSON_PATH="${BASE_META_DIR}/test_s_image_filtered.json"
REFAVS_TEST_U_IMAGE_JSON_PATH="${BASE_META_DIR}/test_u_image_filtered.json"
REFAVS_TEST_VIDEO_JSON_PATH="${BASE_META_DIR}/test_s_video_filtered.json"
REFAVS_TEST_U_VIDEO_JSON_PATH="${BASE_META_DIR}/test_u_video_filtered.json"

REFAVS_MASK_ENCODE_MODE="${REFAVS_MASK_ENCODE_MODE:-mask_and_masked_frame}"
TEST_NAME="${TEST_NAME:-test_s}"
REFAVS_EVAL_MODE="${REFAVS_EVAL_MODE:-image}"
REFAVS_MASK_TYPE_FILTER="${REFAVS_MASK_TYPE_FILTER:-all}"
REFAVS_MASK_RANK_FILTER="${REFAVS_MASK_RANK_FILTER:--1}"
DEVICE="${DEVICE:-cuda:0}"

python scripts/finetune/inference_hyper_lora.py \
    --llm_name llama \
    --model_name_or_path "$llama2_ckpt_path" \
    --freeze_backbone True \
    --lora_enable True \
    --use_hyper_lora False \
    --use_process True \
    --bits 32 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --bf16 True \
    --tf32 False \
    --fp16 False \
    --device "$DEVICE" \
    --ckpt_dir "${MQ_AUDITOR_CKPT_DIR}/checkpoint-960" \
    --refavs_mask_encode_mode "$REFAVS_MASK_ENCODE_MODE" \
    --refavs_eval_mode "$REFAVS_EVAL_MODE" \
    --refavs_mask_type_filter "$REFAVS_MASK_TYPE_FILTER" \
    --refavs_mask_rank_filter "$REFAVS_MASK_RANK_FILTER" \
    --test_name "$TEST_NAME" \
    --refavs_test_image_json_path "$REFAVS_TEST_IMAGE_JSON_PATH" \
    --refavs_test_u_image_json_path "$REFAVS_TEST_U_IMAGE_JSON_PATH" \
    --refavs_test_video_json_path "$REFAVS_TEST_VIDEO_JSON_PATH" \
    --refavs_test_u_video_json_path "$REFAVS_TEST_U_VIDEO_JSON_PATH" \
    --refavs_data_root "$MQ_RAVSBENCH_DIR" \
    --ref_avs_task True \
    --nonlora_ckpt_dir "$MQ_AUDITOR_CKPT_DIR" \
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
    --output_dir test
