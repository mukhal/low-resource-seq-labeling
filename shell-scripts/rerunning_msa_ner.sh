#!/bin/bash

ss=(0.8 0.9 0.95)

for i in "${ss[@]}"
do
	echo "**********************Self Training S=$i **********************************"
	python main.py --data_dir="data/large-zero-shot-msa" --task_name=ner \
	--output_dir="/tmp/msa_tau=$i" \
       	--max_seq_length=320 --num_train_epochs 5 \
	--do_eval --warmup_proportion=0.1 \
	--pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.base/ \
	--learning_rate 0.00001\
	--gradient_accumulation_steps 1 --eval_on test --dropout 0.1\
	--train_batch_size 16 --eval_batch_size 128  --self_training --do_train --K=$i
	
	#echo "**********************Finetuning $i percent**********************************"
	#python main.py --data_dir="data/pos/zero-shot-$dia/$i-examples" --task_name=pos \
	#--output_dir="pos_few_shot_saved_models/finetuning_few_shot_($dia)_$i-examples" \
       	#--max_seq_length=320 --num_train_epochs 20 \
	#--do_eval --warmup_proportion=0.1 \
	#--pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.base/ \
	#--learning_rate 0.00001\
	#--gradient_accumulation_steps 1 --eval_on test --dropout 0.1\
	#--train_batch_size 16 --eval_batch_size 128   --do_train

done
