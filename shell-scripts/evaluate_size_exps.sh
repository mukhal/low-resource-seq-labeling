#!/bin/bash

sizes=(10 20 40 80 100)
dia="lev"
#sizes=(80 100)

for i in "${sizes[@]}"
do
	echo "**********************Self Training $i percent**********************************"
	echo "pos_few_shot_saved_models/self_training_K100_for_real_few_shot_($dia)_$i-examples"
	python main.py --data_dir="data/pos/zero-shot-$dia/$i-examples" --task_name=pos \
	--output_dir="pos_few_shot_saved_models/self_training_K100_for_real_few_shot_($dia)_$i-examples" \
       	--max_seq_length=320 --num_train_epochs 5 \
	--do_eval --warmup_proportion=0.1 \
	--pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.base/ \
	--learning_rate 0.00001\
	--gradient_accumulation_steps 1 --eval_on dev --dropout 0.1\
	--train_batch_size 16 --eval_batch_size 128  --self_training 
	
	echo "**********************Finetuning $i percent**********************************"
	python main.py --data_dir="data/pos/zero-shot-$dia/$i-examples" --task_name=pos \
	--output_dir="pos_few_shot_saved_models/finetuning_few_shot_($dia)_$i-examples" \
       	--max_seq_length=320 --num_train_epochs 20 \
	--do_eval --warmup_proportion=0.1 \
	--pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.base/ \
	--learning_rate 0.00001\
	--gradient_accumulation_steps 1 --eval_on dev --dropout 0.1\
	--train_batch_size 16 --eval_batch_size 128  

done
