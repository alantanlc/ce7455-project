#!/bin/sh
DATA_SIZE=$1
# OUTPUT_DIR="./models/roberta-large_$1"
OUTPUT_DIR="./models/test"

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment.py \
--model_type roberta_mc \
--model_name_or_path roberta-large \
--data_size $DATA_SIZE \
--task_name winogrande \
--do_eval \
--do_lower_case \
--data_dir ./data \
--max_seq_length 80 \
--per_gpu_eval_batch_size 4 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--output_dir $OUTPUT_DIR \
--do_train \
--logging_steps 500 \
--save_steps 500 \
--seed 42 \
--data_cache_dir ./data/cache/ \
--warmup_pct 0.1 \
--overwrite_output_dir \
--evaluate_during_training \