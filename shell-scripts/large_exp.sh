
#!/bin/bash


python main.py --data_dir=data/pos/zero-shot-egy/ --task_name=pos --output_dir=saved_models/large/zero-shot-egy --max_seq_length=320 --num_train_epochs 10 --do_eval --warmup_proportion=0.1 --pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.large/  --learning_rate 0.00001 --gradient_accumulation_steps 1 --eval_on test --dropout 0.1 --train_batch_size 2 --eval_batch_size 16 --self_training --K=0.90 --do_train


python main.py --data_dir=data/pos/zero-shot-glf/ --task_name=pos --output_dir=saved_models/large/zero-shot-glf --max_seq_length=320 --num_train_epochs 10 --do_eval --warmup_proportion=0.1 --pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.large/  --learning_rate 0.00001 --gradient_accumulation_steps 1 --eval_on test --dropout 0.1 --train_batch_size 2 --eval_batch_size 16 --self_training --K=0.90 --do_train

python main.py --data_dir=data/pos/zero-shot-lev/ --task_name=pos --output_dir=saved_models/large/zero-shot-lev --max_seq_length=320 --num_train_epochs 10 --do_eval --warmup_proportion=0.1 --pretrained_path ../../pretrained-sumarization/pretrained_models/xlmr.large/  --learning_rate 0.00001 --gradient_accumulation_steps 1 --eval_on test --dropout 0.1 --train_batch_size 2 --eval_batch_size 16 --self_training --K=0.90 --do_train

