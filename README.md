
This repository holds the Code for EACL 2021 paper "Self-Training Pre-Trained Language Models for Zero- and Few-Shot Multi-Dialectal Arabic Sequence Labeling"



## Requirements
Please make sure you have pytorch >=1.4 and fairseq >= 0.9 installed.

## Setting Up the Data
### NER
Format your data using IOB format with a token per line and an empty line separating sentences. For example: 
```
لنزار B-PERS
عدد O
كبير O
من O
الكتب O
النثرية O
أهمها O
```

### POS Tagging 
Similar to NER: 
```
و       CONJ
WB      WB
هو      PRON
WB      WB
في      PREP
WB      WB
محام    NOUN
TB      TB
ين      NSUFF
WB      WB
ه       FUT_PART
TB      TB
يعبر    V
TB      TB
```
Typically each data folder has 3 files: `train.txt`, `valid.txt`, and `test.txt`

## XLMR models

We first need XLM-R models. You can donwload XLM-R models using the following commands
```
mkdir pretrained_models

wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.base.tar.gz # base model
tar -xzvf xlmr.base.tar.gz # extract it

wget https://dl.fbaipublicfiles.com/fairseq/models/xlmr.large.tar.gz # large model
tar -xzvf xlmr.large.tar.gz
```


## Finetuning
To finetune XLM-RoBERTa without self-training use the following command: 

```
 python main.py --data_dir=data_path --task_name=ner \
        --output_dir="/tmp/msa_tau=$i" \
        --max_seq_length=320 --num_train_epochs 5 \
        --do_eval --warmup_proportion=0.1 \
        --pretrained_path pretrained_models/xlmr.base/ \
        --learning_rate 0.00001\
        --gradient_accumulation_steps 1 --eval_on test --dropout 0.1\
        --train_batch_size 16 --eval_batch_size 128 --do_train

```


