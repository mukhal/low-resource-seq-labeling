from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, ConcatDataset)

from seqeval.metrics import classification_report
from model.xlmr_for_token_classification import (XLMRForTokenClassification, 
                                                XLMRForTokenClassificationWithCRF, Discriminator)
from utils.train_utils import add_xlmr_args, evaluate_model, get_top_confidence_samples
from utils.data_utils import DataProcessor, create_ner_dataset, SentClfProcessor, create_clf_dataset

from tqdm import tqdm as tqdm
from tqdm import trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)
    parser.add_argument('--predict_file', type=str, default='')
    parser.add_argument('--out_file', type=str, default='')


    args = parser.parse_args()

    data_processor = DataProcessor(task=args.task_name)
    label_list = data_processor.get_labels()
    num_labels = len(label_list) + 1  # add one for IGNORE label


    model_cls = XLMRForTokenClassification
 
    hidden_size = 768 if 'base' in args.pretrained_path else 1024 # TODO: move this inside model.__init__
    device = 'cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'

   
    # creating model
    model = model_cls(pretrained_path=args.pretrained_path,
                                       n_labels=num_labels, hidden_size=hidden_size,
                                       dropout_p=args.dropout, device=device)


    # load best/ saved model
    state_dict = torch.load(open(args.load_model, 'rb'))
    model.load_state_dict(state_dict)
    logger.info("Loaded saved model")

    model.to(device)

    pred_examples = data_processor.get_pred_examples(args.predict_file)
    pred_features = data_processor.convert_examples_to_features(
        pred_examples, label_list, 320, model.encode_word)
    
    pred_data = create_ner_dataset(pred_features)
    f1_score, report, y_true, y_pred = evaluate_model(model, pred_data, label_list, args.eval_batch_size, args.use_crf, device, pred=True)

   
    logger.info("\n%s", report)
    output_pred_file = args.out_file
    with open(output_pred_file, "w") as writer:
        for ex, pred in zip(pred_examples, y_pred):
            writer.write("Ex text: {}\n".format(ex.text))
            writer.write("Ex labels: {}\n".format(ex.label))
            writer.write("Ex preds: {}\n".format(pred)) 

            writer.write("*******************************\n")


if __name__ == "__main__":
    main()

