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
from model.xlmr_for_sequence_classification import XLMRForSequenceClassification 
from utils.train_utils import add_xlmr_args, evaluate_model_seq_classification, get_top_confidence_samples_seq_classification
from utils.data_utils import SequenceClassificationProcessor, create_ner_dataset, create_clf_dataset

from tqdm import tqdm as tqdm
from tqdm import trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    parser = argparse.ArgumentParser()
    parser = add_xlmr_args(parser)
    parser.add_argument('--self_training', action='store_true', default=False)
    parser.add_argument('--unlabeled_data_dir', type=str, default='data/unlabeled_data')
    parser.add_argument('--self_training_confidence', type=float, default=0.9)
    parser.add_argument('--K', type=float, default=50)
    parser.add_argument('--patience', type=float, default=10)
    parser.add_argument('--balanced', action='store_true', default=False)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    data_processor = SequenceClassificationProcessor()
    label_list = data_processor.get_labels()
    num_labels = len(label_list)  # add one for IGNORE label

    train_examples = None
    num_train_optimization_steps = 0

    if args.do_train:

        train_examples = data_processor.get_examples(args.data_dir, split='train')
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        
            # preparing model configs
    hidden_size = 768 if 'base' in args.pretrained_path else 1024 # TODO: move this inside model.__init__

    device = args.device if (torch.cuda.is_available() and not args.no_cuda) else 'cpu'
    model_cls = XLMRForSequenceClassification
    
    # creating model
    model = model_cls(pretrained_path=args.pretrained_path,
                                       n_labels=num_labels, hidden_size=hidden_size,
                                       dropout_p=args.dropout, device=device)

    model.to(device)

    if args.load_model is not None:
        logging.info("Loading saved model {}".format(args.load_model))
        state_dict = torch.load(args.load_model)
        model.load_state_dict(state_dict, strict=True)

    no_decay = ['bias', 'final_layer_norm.weight']

    params = list(model.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    
    # freeze model if necessary
    if args.freeze_model:
        logger.info("Freezing XLM-R model...")
        for n, p in model.named_parameters():
            if 'xlmr' in n and p.requires_grad:
                p.requires_grad = False

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    
    label_map = {i: label for i, label in enumerate(label_list, 1)}

    if args.do_train:
        train_features = data_processor.convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, model.encode_sent)

        if args.self_training:
            self_training_examples = data_processor.get_unlabeled_examples(args.unlabeled_data_dir)
            self_training_features = data_processor.convert_examples_to_features(self_training_examples, label_list, args.max_seq_length, model.encode_sent) 
            
            logging.info("Loaded {} Unlabeled examples".format(len(self_training_examples)))


        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        train_data = create_clf_dataset(train_features)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)


        val_examples = data_processor.get_examples(args.data_dir, split="valid")
        val_features = data_processor.convert_examples_to_features(
            val_examples, label_list, args.max_seq_length, model.encode_sent)

        val_data = create_clf_dataset(val_features)
        best_val_f1 = 0.0

       ############################# Self Training Loop ######################
        n_iter=0
        optimizer = AdamW(model.parameters(),
                              lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(
                optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)

        patience = 0 
        best_val_report=None
        while 1:
            ############################ Inner Training Loop #####################
            n_iter+=1
            print(len(train_dataloader))
            
            for epoch_ in tqdm(range(
                args.num_train_epochs), desc="Epoch", disable=args.no_pbar):

                tr_loss = 0
                tbar = tqdm(train_dataloader, desc="Iteration", 
                    disable=args.no_pbar)
                
                model.train()
                for step, batch in enumerate(tbar):
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, label_ids = batch
                    loss, _ = model(input_ids, label_ids, get_sent_repr=True)

                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps
                    
                    tr_loss += loss.item()
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 10)
                    
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                       optimizer.step()
                       scheduler.step()  # Update learning rate schedule
                       model.zero_grad()
                
                    tbar.set_description('Loss = %.4f' %(tr_loss / (step+1)))
                logger.info("Evaluating on validation set...\n")
                #torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
                f1, report = evaluate_model_seq_classification(model, val_data, label_list, args.eval_batch_size, device)
                if f1 > best_val_f1:
                    best_val_f1 = f1
                    logger.info("\nFound better f1=%.4f on validation set. Saving model\n" %(f1))
                    logger.info("\n%s\n" %(report))
                    
                    torch.save(model.state_dict(), open(os.path.join(args.output_dir, 'model.pt'), 'wb'))
                    patience=0
                    best_val_report = report
                else :
                    logger.info("\nNo better F1 score: {}\n".format(f1))
                    patience+=1
            
            ######################################################################
            if not args.self_training:
                break
            if patience >= args.patience:
                logger.info("No more patience. Existing")
                break
            ## get confidence and update train_data, train_dataloader
            # convert unlabeled examples to features 

            if len(self_training_features) <= 0: # no more self-training data
                break

            confident_features, self_training_features = get_top_confidence_samples_seq_classification(model, self_training_features, batch_size=args.eval_batch_size, K=args.K, balanced=args.balanced, n_classes=len(label_list))
            

            for f in confident_features:
                l_id = f.label_id
            logging.info("Got %d confident samples"%(len(confident_features)))
            # append new features 
            #train_features = data_processor.convert_examples_to_features(
            #         train_examples, label_list, args.max_seq_length, model.encode_word)


            train_features.extend(confident_features)

            print("now we have %d total examples"% len(train_features))

            train_data = create_clf_dataset(train_features)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=args.train_batch_size)
                
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate

            scheduler.step(0)

            #print("Loading best last model...")
            #model.load_state_dict(torch.load(open(os.path.join(args.output_dir, 'model.pt'), 'rb')))


    # load best/ saved model
    state_dict = torch.load(open(os.path.join(args.output_dir, 'model.pt'), 'rb'))
    model.load_state_dict(state_dict)
    logger.info("Loaded saved model")

    model.to(device)

    if args.do_eval:
        if args.eval_on == "dev":
            eval_examples = data_processor.get_examples(args.data_dir, split='valid')
        elif args.eval_on == "test":
            eval_examples = data_processor.get_examples(args.data_dir, split='test')
        else:
            raise ValueError("eval on dev or test set only")
        eval_features = data_processor.convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, model.encode_sent)
        
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        
        eval_data = create_clf_dataset(eval_features)
        f1_score, report = evaluate_model_seq_classification(model, eval_data, label_list, args.eval_batch_size, device)

        logger.info("DEV RESULTS")
        logger.info("\n%s", best_val_report)
        logger.info("TEST RESULTS\n")
        logger.info("\n%s", report)
        output_eval_file = os.path.join(args.output_dir, "%s_results.txt"%(args.eval_on))
        logger.info("dataset = {}".format(args.data_dir))
        logger.info("model = {}".format(args.output_dir))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Writing results to file *****")
            writer.write("DEV RESULTS\n")
            writer.write(best_val_report)
            writer.write("TEST RESULTS\n")
            writer.write(report)
            logger.info("Done.")
        
        ## write arguments to file
        output_args_file = os.path.join(args.output_dir, "training_args.txt")
        with open(output_args_file, "w") as writer:
            writer.write(json.dumps(vars(args)))

if __name__ == "__main__":
    main()

