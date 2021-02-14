#!/bin/bash
source ~/py36/bin/activate

python main.py --data_dir=data/zero-shot-dialect/ --task_name=ner --output_dir=adv2_zero_xlmr_large_shot_msa --max_seq_length=320 --num_train_epochs 10 --do_eval --warmup_proportion=0.2 --pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.large/ --learning_rate 0.000006 --gradient_accumulation_steps 1 --eval_on test --dropout 0.1 --train_batch_size 1 --do_train --eval_batch_size 1 --adversarial_objective

