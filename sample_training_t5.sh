#!/bin/sh
DATA_SIZE=$1
# OUTPUT_DIR="./models/$2"
OUTPUT_DIR="./models/test"

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment_t5.py \
--model_type t5 \
--model_name_or_path t5-base \
--data_size $DATA_SIZE \
--task_name winogrande_ps \
--do_eval \
--do_lower_case \
--data_dir ./data \
--max_seq_length 80 \
--per_gpu_eval_batch_size 2 \
--per_gpu_train_batch_size 8 \
--gradient_accumulation_steps 16 \
--learning_rate 1e-3 \
--num_train_epochs 8 \
--output_dir $OUTPUT_DIR \
--do_train \
--logging_steps 2 \
--save_steps 2 \
--seed 42 \
--data_cache_dir ./data/cache/ \
--warmup_pct 0.1 \
--overwrite_output_dir \
--evaluate_during_training \