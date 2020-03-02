#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python ./scripts/run_experiment.py \
--model_type bert_mc \
--model_name_or_path bert-base-uncased \
--data_size xs \
--task_name winogrande \
--do_eval \
--do_lower_case \
--data_dir ./data \
--max_seq_length 80 \
--per_gpu_eval_batch_size 4 \
--per_gpu_train_batch_size 4 \
--learning_rate 1e-5 \
--num_train_epochs 1 \
--output_dir ./output/test \
--do_train \
--logging_steps 4752 \
--save_steps 4750 \
--seed 42 \
--data_cache_dir ./output/cache/ \
--overwrite_output_dir \
--warmup_pct 0.1 \
--evaluate_during_training
