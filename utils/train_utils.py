from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score, classification_report, accuracy_score
import sklearn
import torch
import torch.nn.functional as F
from .data_utils import InputFeatures
import numpy as np

def add_xlmr_args(parser):
     """
     Adds training and validation arguments to the passed parser
     """

     parser.add_argument("--data_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
     parser.add_argument("--pretrained_path", default=None, type=str, required=True,
                         help="pretrained XLM-Roberta model path")
     parser.add_argument("--task_name",
                         default=None,
                         type=str,
                         required=True,
                         help="The name of the task to train.")
     parser.add_argument("--output_dir",
                         default=None,
                         type=str,
                         required=True,
                         help="The output directory where the model predictions and checkpoints will be written.")
     # Other parameters
     parser.add_argument("--cache_dir",
                         default="",
                         type=str,
                         help="Where do you want to store the pre-trained models downloaded from s3")
     parser.add_argument("--max_seq_length",
                         default=128,
                         type=int,
                         help="The maximum total input sequence length after WordPiece tokenization. \n"
                              "Sequences longer than this will be truncated, and sequences shorter \n"
                              "than this will be padded.")
     parser.add_argument("--do_train",
                         action='store_true',
                         help="Whether to run training.")
     parser.add_argument("--do_eval",
                         action='store_true',
                         help="Whether to run eval or not.")
     parser.add_argument("--eval_on",
                         default="dev",
                         help="Whether to run eval on the dev set or test set.")
     parser.add_argument("--do_lower_case",
                         action='store_true',
                         help="Set this flag if you are using an uncased model.")
     parser.add_argument("--train_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for training.")
     parser.add_argument("--eval_batch_size",
                         default=32,
                         type=int,
                         help="Total batch size for eval.")
     parser.add_argument("--learning_rate",
                         default=5e-5,
                         type=float,
                         help="The initial learning rate for Adam.")
     parser.add_argument("--num_train_epochs",
                         default=3,
                         type=int,
                         help="Total number of training epochs to perform.")
     parser.add_argument("--warmup_proportion",
                         default=0.1,
                         type=float,
                         help="Proportion of training to perform linear learning rate warmup for. "
                              "E.g., 0.1 = 10%% of training.")
     parser.add_argument("--weight_decay", default=0.01, type=float,
                         help="Weight deay if we apply some.")
     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                         help="Epsilon for Adam optimizer.")
     parser.add_argument("--max_grad_norm", default=1.0, type=float,
                         help="Max gradient norm.")
     parser.add_argument("--no_cuda",
                         action='store_true',
                         help="Whether not to use CUDA when available")
     parser.add_argument('--seed',
                         type=int,
                         default=42,
                         help="random seed for initialization")
     parser.add_argument('--gradient_accumulation_steps',
                         type=int,
                         default=1,
                         help="Number of updates steps to accumulate before performing a backward/update pass.")
     parser.add_argument('--fp16',
                         action='store_true',
                         help="Whether to use 16-bit float precision instead of 32-bit")
     parser.add_argument('--fp16_opt_level', type=str, default='O1',
                         help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                              "See details at https://nvidia.github.io/apex/amp.html")
     parser.add_argument('--loss_scale',
                         type=float, default=0,
                         help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                              "0 (default value): dynamic loss scaling.\n"
                              "Positive power of 2: static loss scaling value.\n")
     parser.add_argument('--dropout', 
                         type=float, default=0.3,
                         help = "training dropout probability")
     
     parser.add_argument('--freeze_model', 
                         action='store_true', default=False,
                         help = "whether to freeze the XLM-R base model and train only the classification heads")
     
     parser.add_argument('--use_crf', 
                         action='store_true', default=False,
                         help = "whether to add a CRF layer on top of the classification head")
     
     parser.add_argument('--no_pbar', 
                         action='store_true', default=False,
                         help = "disable tqdm progress bar")

     
     parser.add_argument('--load_model', 
                         type=str, default=None,
                         help = "saved model to load")



     return parser

def get_top_confidence_samples_seq_labeling(model, features, batch_size=16,  K=40, device='cuda', balanced=False, n_classes=7):

     """
     Runs model on data, return the set of examples whose prediction confidence is equal of above min_confidence_per_sample
     Args:

        model: the model
        data: set of unlabeled examples 
        min_confidence_per_sample: threshold by which we select examples

    Returns:
        A set of indices of the selected example

     """
     model.eval() # turn of dropout
     y_true = []
     y_pred = []

     filtered_examples = []

     all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
     all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
     all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
     all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.uint8)

     predictions = []
     confidences = []

     confident_features, rest_features=[], []
     for idx in range(0, all_input_ids.size(0), batch_size):

        input_ids = all_input_ids[idx:idx+ batch_size]
        label_ids = all_label_ids[idx:idx+batch_size]
        valid_ids = all_valid_ids[idx:idx+batch_size]
        l_mask = all_lmask_ids[idx:idx+batch_size]

        input_ids = input_ids.to(device)
        valid_ids = valid_ids.to(device)
        l_mask = l_mask.to(device)

        with torch.no_grad():
             logits = model(input_ids, labels=None, labels_mask=l_mask,
                              valid_mask=valid_ids)

        prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=2).max(dim=2) # B x L
        
        prediction_prob[~l_mask.bool()] = 1e7 # so they will be ignored by min
        #prediction_prob[label_ids==0] = 1e7 # ignore WB
        #prediction_prob[label_ids==1] = 1e7 # ignore TB

        min_confidence  , _= prediction_prob.min(dim=-1) # B
        # mean 
        #prediction_prob[~l_mask.bool()] = 0 # so they would be ignored by sum
        #min_confidence = prediction_prob.sum(dim=-1) / l_mask.sum(dim=-1)

        predictions.append(predicted_labels)
        confidences.append(min_confidence)
 
     confidences = torch.cat(confidences, dim=0)
     predictions = torch.cat(predictions, dim=0)
 
     idx_sorted = torch.argsort(confidences, descending=True)
     if K>=1.0:
         K=int(K)
         if balanced:
             pass
         else:
             top_k_idx = idx_sorted[:K]
             rest_idx = idx_sorted[K:]
     else:
        top_k_idx = (confidences >= K)
        rest_idx = (confidences < K)
     
     rest_idx = torch.tensor([i for i in range(len(confidences)) if i not in top_k_idx]).long()
        
     selected_ids = all_input_ids[top_k_idx].cpu().numpy().tolist()
     selected_lbls = predictions[top_k_idx].cpu().numpy().tolist()
     selected_masks = all_lmask_ids[top_k_idx].cpu().numpy().tolist()
     selected_valid = all_valid_ids[top_k_idx].cpu().numpy().tolist()

     # add them to examples
     for ids, lbls, msks, valids in zip(selected_ids, selected_lbls, selected_masks, selected_valid):
         #print(lbls)
         confident_features.append(InputFeatures(input_ids=ids, label_id=lbls, label_mask=msks, valid_ids=valids))
 
     # select those that don't satisfy the confidence
     non_selected_ids = all_input_ids[rest_idx].cpu().numpy().tolist()
     non_selected_lbls = all_label_ids[rest_idx].cpu().numpy().tolist()
     non_selected_masks = all_lmask_ids[rest_idx].cpu().numpy().tolist()
     non_selected_valid = all_valid_ids[rest_idx].cpu().numpy().tolist()
     
     for ids, lbls, msks, valids in zip(non_selected_ids, non_selected_lbls, non_selected_masks, non_selected_valid):
         #print(lbls)
         rest_features.append(InputFeatures(input_ids=ids, label_id=lbls, label_mask=msks, valid_ids=valids))
    
     print(len(rest_features))
     print(len(confident_features))
     print(len(features))
     #assert len(features) == len(rest_features) + len(confident_features) # sanity check
 
     return confident_features, rest_features

def get_top_confidence_samples_seq_classification(model, features, batch_size=16,  K=40, device='cuda', balanced=False, n_classes=3):

     """
     Runs model on data, return the set of examples whose prediction confidence is equal of above min_confidence_per_sample
     Args:

        model: the model
        data: set of unlabeled examples 
        min_confidence_per_sample: threshold by which we select examples

    Returns:
        A set of indices of the selected example

     """
     model.eval() # turn of dropout
     y_true = []
     y_pred = []

     filtered_examples = []

     all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    
     all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)


     predictions = []
     confidences = []

     confident_features, rest_features=[], []
     for idx in range(0, all_input_ids.size(0), batch_size):

        input_ids = all_input_ids[idx:idx+ batch_size]
        input_ids = input_ids.to(device)

        with torch.no_grad():
             logits = model(input_ids, labels=None)

        prediction_prob, predicted_labels = torch.nn.functional.softmax(logits, dim=1).max(dim=1) # B 
        
        predictions.append(predicted_labels)
        confidences.append(prediction_prob)
 
     confidences = torch.cat(confidences, dim=0)
     predictions = torch.cat(predictions, dim=0)
 
     idx_sorted = torch.argsort(confidences, descending=True)
     MAX_SAMPLES = 1000

     if K>1.0:
        
         K=int(K)
         if balanced:
             K = K // n_classes
             top_k_idx=[]
             # Taking K examples from each class
             for class_id in range(n_classes):
                 i=0; n_taken=0
                 while i < idx_sorted.size(0) and n_taken < K:
                     idx = idx_sorted[i]
                     if predictions[idx] == class_id:
                         n_taken+=1
                         top_k_idx.append(idx)
                     i += 1
                 if n_taken == K:
                     print("collected all for class %d"%(class_id))

             top_k_idx = torch.LongTensor(top_k_idx)
             rest_idx = torch.tensor([i for i in range(len(confidences)) if i not in top_k_idx]).long()

         else:
             top_k_idx = idx_sorted[:K]
             rest_idx = idx_sorted[K:]
     else:
        top_k_idx = (confidences >= K)
        rest_idx = (confidences < K)
      
 
     selected_ids = all_input_ids[top_k_idx].cpu().numpy().tolist()
     selected_lbls = predictions[top_k_idx].cpu().numpy().tolist()

     unique, counts = np.unique(selected_lbls, return_counts=True)
     frequencies = np.asarray((unique, counts)).T
     print(frequencies)
 
     # add them to examples
     for ids, lbl in zip(selected_ids, selected_lbls):
         confident_features.append(InputFeatures(input_ids=ids, label_id=lbl))
 
     # select those that don't satisfy the confidence
     non_selected_ids = all_input_ids[rest_idx].cpu().numpy().tolist()
     non_selected_lbls = all_label_ids[rest_idx].cpu().numpy().tolist()
     
     for ids, lbl in zip(non_selected_ids, non_selected_lbls):
         #print(lbls)
         rest_features.append(InputFeatures(input_ids=ids, label_id=lbl))

     assert len(features) == len(rest_features) + len(confident_features) # sanity check
 
     return confident_features, rest_features



def evaluate_model_seq_labeling(model, eval_dataset, label_list, batch_size, use_crf, device, pred=False):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     # Run prediction for full data
     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     label_map = {i: label for i, label in enumerate(label_list, 1)}
     label_map[0] = "IGNORE"


     for input_ids, label_ids, l_mask, valid_ids in eval_dataloader:

          input_ids = input_ids.to(device)
          label_ids = label_ids.to(device)

          valid_ids = valid_ids.to(device)
          l_mask = l_mask.to(device)

          with torch.no_grad():
               logits = model(input_ids, labels=None, labels_mask=None,
                              valid_mask=valid_ids)

          if use_crf:
               predicted_labels = model.decode_logits(logits, mask=l_mask, device=device)
          else :     
               predicted_labels = torch.argmax(logits, dim=2)

          predicted_labels = predicted_labels.detach().cpu().numpy()
          label_ids = label_ids.cpu().numpy()

          for i, cur_label in enumerate(label_ids):
               temp_1 = []
               temp_2 = []

               for j, m in enumerate(cur_label):
                   if valid_ids[i][j] and label_map[m] not in ['WB' , 'TB']: #'PROG_PART', 'NEG_PART']:  # if it's a valid label
                         temp_1.append(label_map[m])
                         temp_2.append(label_map[predicted_labels[i][j]])

               assert len(temp_1) == len(temp_2)
               y_true.append(temp_1)
               y_pred.append(temp_2)

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='Macro')
     acc = accuracy_score(y_true, y_pred)

     s = "Accuracy = {}".format(acc)
     print(s)
     report +='\n\n'+ s

     if 'NOUN' in label_map.values():
         print("Returning acc")
         f1=acc
    
     if pred:
         return f1, report, y_true, y_pred
     return f1, report

def evaluate_model_seq_classification(model, eval_dataset, label_list, batch_size, device, pred=False):
     """
     Evaluates an NER model on the eval_dataset provided.
     Returns:
          F1_score: Macro-average f1_score on the evaluation dataset.
          Report: detailed classification report 
     """

     from sklearn.metrics import f1_score, classification_report, accuracy_score

     # Run prediction for full data
     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     label_map = {i: label for i, label in enumerate(label_list)}
     for input_ids, label_ids in eval_dataloader:

          input_ids = input_ids.to(device)
          label_ids = label_ids.to(device)

          with torch.no_grad():
               logits = model(input_ids, labels=None)
               predicted_labels = torch.argmax(logits, dim=1)

          predicted_labels = predicted_labels.detach().cpu().numpy()
          label_ids = label_ids.cpu().numpy()
          y_true.extend(label_ids)
          y_pred.extend(predicted_labels)

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='macro')
     acc = accuracy_score(y_true, y_pred)
     
     model.train()

     if pred:
         return f1, report, y_true, y_pred
     return f1, report
