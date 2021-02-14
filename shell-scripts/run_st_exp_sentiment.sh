#!/bin/bash
#SBATCH --account=def-mageed
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=24000M               # memory (per node)
#SBATCH --time=0-05:00            # time (DD-HH:MM)


k=$1
source ~/py36/bin/activate

python run_clf.py --data_dir=data/sentiment-analysis/zero-shot-Arsenti/-dev-test/ --task_name=sa \
	--output_dir="saved_models/sentiment_analysis/arsenti-experiment"\
	--max_seq_length=320 \
	--num_train_epochs 5 \
	--do_eval \
	--warmup_proportion=0.1 \
	--pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.base  --learning_rate 0.00001 \
	--gradient_accumulation_steps 1 --eval_on test --dropout 0.1 --train_batch_size 16 --eval_batch_size 128  --patience 5 --do_train \
	 --unlabeled_data_dir data/unlabeled_aoc_maghrebi/ --no_pbar;


