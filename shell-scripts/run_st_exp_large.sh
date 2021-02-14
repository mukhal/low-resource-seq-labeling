#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:p100l:4
#SBATCH --mem-per-cpu=24000M               # memory (per node)
#SBATCH --time=0-06:00            # time (DD-HH:MM)


k=$1
source ~/py36/bin/activate
python run_clf.py --data_dir=data/sentiment-analysis/sarcasm-norm/zero-shot-dia-large-dev-test/ --task_name=sa --output_dir saved_models/sentiment_analysis/self-training/xlmr-large-sarcasm-dia-zero-shot-st-K$k-balanced --max_seq_length=320 --num_train_epochs 5 --do_eval --warmup_proportion=0.1 --pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.large  --learning_rate 0.000005 --gradient_accumulation_steps 1 --eval_on test --dropout 0.1 --train_batch_size 8 --eval_batch_size 16  --patience 5 --do_train --self_training --K=$k --balanced --unlabeled_data_dir data/unlabeled_aoc_maghrebi/ --no_pbar --seed 12
