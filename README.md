## Zero-shot Multidialectal Arabic Sequence Labeling (NER and POS tagging)


#### Code for EACL 2021 paper [Self-Training Pre-Trained Language Models for Zero- and Few-Shot Multi-Dialectal Arabic Sequence Labeling](https://arxiv.org/abs/2101.04758)

## Requirements
Please make sure you have `pytorch >=1.4` and `fairseq >= 0.9` installed.

## Datasets
The `data/` folder includes some of the datasets used in the paper (Some of the datasets can only be accessed through the LDC). 

### NER
* `NER/twitter`: social media NER dataset from [(Darwish, 2013)](https://www.aclweb.org/anthology/P13-1153.pdf)
* `NER/twitter.norm`: same dataset but normalized.
* `NER/ANERCorp`: MSA dataset obtained from [(Benajiba et al., 2007)](https://link.springer.com/chapter/10.1007/978-3-540-70939-8_13)
* `NER/zero-shot-dialect`: this is the zero-shot dialectal setting. training data is from ANERCorp (Benajiba et al., 2007) while validation and test sets come from the dialectal portion of the Twitter data (Darwish, 2013).
* `NER/zero-shot-msa`:  same as above but validation and test sets come from the MSA portion of the Twitter data.
### POS tagging 
* `POS-tagging/egy`, `POS-tagging/glf`, `POS-tagging/lev`, and `POS-tagging/msa` dialectal POS tagging datasets obtained from [(Darwish et al., 2018)](https://www.aclweb.org/anthology/L18-1015.pdf)
* `POS-tagging/zero-shot-*`: training data is MSA, development, and test data from dialects.

### Unalebeled data
These are unlabeled examples used for self-training.
* `unlabeled_aoc`: Unlabeled AOC tweets taken from [(El Araby and Mageed, 2018)](https://www.aclweb.org/anthology/W18-3930.pdf).

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
هو      PRON
في      PREP
محام    NOUN
TB      TB
ين      NSUFF
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
        --output_dir=output_dir \
        --max_seq_length=320 --num_train_epochs 5 \
        --do_eval --warmup_proportion=0.1 \
        --pretrained_path pretrained_models/xlmr.base/ \
        --learning_rate 0.00001\
        --gradient_accumulation_steps 1 --eval_on test --dropout 0.1\
        --train_batch_size 16 --eval_batch_size 128 --do_train

```

## Self-training
To fine-tune your model with self-training, you need to add the flag `--self_training`. Also, you need to specify your selection mechanism:  
* Use integer values > 1 for fixed size selection. For example, 
```
python main.py --data_dir=data_path --task_name=ner \
        --output_dir=output_dir \
        --max_seq_length=320 --num_train_epochs=5 \
        --do_eval --warmup_proportion=0.1 \
        --pretrained_path=pretrained_models/xlmr.base/ \
        --learning_rate=0.00001\
        --gradient_accumulation_steps=1 --eval_on test --dropout=0.1\
        --train_batch_size=16 --eval_batch_size=128 --do_train --self_training --K=100
```

* Use float values <=1.0 for probability threshold. For example:
```
python main.py --data_dir=data_path --task_name=ner \
        --output_dir=output_dir \
        --max_seq_length=320 --num_train_epochs=5 \
        --do_eval --warmup_proportion=0.1 \
        --pretrained_path=pretrained_models/xlmr.base/ \
        --learning_rate 0.00001\
        --gradient_accumulation_steps=1 --eval_on test --dropout=0.1\
        --train_batch_size=16 --eval_batch_size=128 --do_train --self_training --K=0.90
```
To train for POS tagging, run with `--task_name=pos`. 

## Evaluation
To evaluate or predict labels using a finetuned model: 
```
python predict.py  --task_name=ner \
        --load_model=path/to/finetuned/model.pt \
        --pretrained_path pretrained_models/xlmr.base/ \
        --predict_file=path/to/IOB/file
```


## Citation 
If you use this code, please cite this paper
```
@article{khalifa2021self,
  title={Self-Training Pre-Trained Language Models for Zero-and Few-Shot Multi-Dialectal Arabic Sequence Labeling},
  author={Khalifa, Muhammad and Abdul-Mageed, Muhammad and Shaalan, Khaled},
  journal={arXiv preprint arXiv:2101.04758},
  year={2021}
}
```

