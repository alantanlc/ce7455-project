#!/bin/sh
MODEL=$1
DATA_SIZE=$2
# OUTPUT_DIR="./models/${MODEL}_${DATA_SIZE}"
OUTPUT_DIR="./models/t5-base_xl_mtp4"

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment_t5.py \
--model_type t5 \
--model_name_or_path "./models/t5-base_l_mtp5/" \
--output_dir "./models/t5-base_l_mtp5/" \
--task_name winogrande_ps \
--tokenizer_name 't5-base' \
--do_lower_case \
--data_dir ./data \
--max_seq_length 100 \
--seed 42 \
--data_cache_dir ./data/cache/ \
--do_prediction 


# 
# python scripts/run_experiment.py \
# --model_type roberta_mc \
# --model_name_or_path .output/models \
# --task_name winogrande \
# --do_predict \
# --do_lower_case \
# --data_dir ./data \
# --max_seq_length 80 \
# --per_gpu_eval_batch_size 4 \
# --output_dir ./output/models/ \
# --data_cache_dir ./output/cache/ \