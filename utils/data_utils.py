import os
import logging

import torch 
from torch.utils.data import TensorDataset

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_id, input_mask=None, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

class SequenceClassificationProcessor:
   
    def get_examples(self, data_dir, split='train'):
        examples = []

        cnt=0
        f=open(os.path.join(data_dir, '%s.tsv'%(split)), encoding='utf-8')
        for i, sent in enumerate(f):
            cnt+=1
            #print("Line %d "%(cnt))

            cls, sent= sent.split('\t')
            cls= cls.lower()
            if cls not in self.get_labels():
                logging.info("Unknown label {}. Skipping line...".format(cls))
                continue
            #print(sent, " ---> " + cls)
            examples.append(
                    InputExample(guid=cnt, text_a=sent, label=cls))
        logging.info("loaded %d classification examples"%(len(examples)))
        return examples
    
    def get_pred_examples(self, file_dir):
        """See base class."""
        return self._create_examples(self._read_file(file_dir),"pred")


    def get_labels(self):
        return ['pos', 'neg'] #'neut']
        
    
    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        examples = []
        cnt=0
        f=open(os.path.join(data_dir, 'sents.txt'), encoding='utf-8')
        for sent in f:
            cnt+=1
            examples.append(
                    InputExample(guid=cnt, text_a=sent, label=None))
        logging.info("loaded %d unlabeled examples"%(len(examples)))
        return examples


    def convert_examples_to_features(self, examples, label_list, max_seq_length, encode_sent_method):

        labels = list(set([e.label for e in examples]))
        logging.info("labels = {}".format(labels))

        label_map= {label_list[i]:i for i in range(len(label_list))}
        
        features=[]
        for e in examples:
            input_ids = encode_sent_method(e.text_a)
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]


            while len(input_ids) < max_seq_length:
                input_ids.append(1)

            features.append(InputFeatures(input_ids=input_ids,
                label_id= -1 if e.label is None else label_map[e.label]))

        return features


class SequenceLabelingProcessor:
    """Processor for the CoNLL-2003 data set."""
    def __init__(self, task):
        assert task in ['ner', 'pos']
        if task == 'ner':
            self.labels = ["O", "B-PERS", "I-PERS", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "U"]

        elif task =='pos':
            self.labels = ['TB', 'WB', 'PART', 'V', 'ADJ', 'DET', 'HASH', 'NOUN', 'PUNC',
                           'CONJ', 'PREP', 'PRON', 'EOS', 'CASE', 'EMOT', 'NSUFF', 'NUM',
                                  'URL', 'ADV', 'MENTION', 'FUT_PART', 'ABBREV', 'FOREIGN', 'PROG_PART', 'NEG_PART','U']
            
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_pred_examples(self, file_dir):
        """See base class."""
        return self._create_examples(self._read_file(file_dir),"pred")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")
    
    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "sents.unlabeled")), "unlabeled")


    def get_labels(self):
        return self.labels

    def _read_file(self, filename):
        '''
        read file
        '''
        f = open(filename, encoding='utf-8', errors='ignore')
        data = []
        sentence = []
        label = []

        # get all labels in file

        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n" or line[0] == '.' or line.split()[0]=='EOS':
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue

            splits = line.split()
            if len(splits) <= 1:
                logging.info("skipping line")
                continue
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, tag = splits[0], splits[-1]
            
            if tag not in self.get_labels():
                logging.info("ignoring unknown tag {} in line {}".format(tag, i))
                continue
            #if tag in ['WB', "TB"]:
            #    tag = "IGNORE"

            sentence.append(word.strip())
            label.append(tag.strip())

        if len(sentence) > 0:
            data.append((sentence, label))
            print(label)
            sentence = []
            label = []
        return data

    def _create_examples(self, lines, set_type):
        examples = []

        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        
        logging.info("max sentence length = %d" %(max(len(ex.text_a.split()) for ex in examples)))
        return examples
    
    
    def convert_examples_to_features(self, examples, label_list, max_seq_length, encode_method):
        """Converts a set of examples into XLMR compatible format

        * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
        * Other positions are labeled with 0 ("IGNORE")
    
        """
        ignored_label = "IGNORE"
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        label_map[ignored_label] = 0  # 0 label is to be ignored
        
        features = []
        for (ex_index, example) in enumerate(examples):

            textlist = example.text_a.split(' ')
            labellist = example.label
            labels = []
            valid = []
            label_mask = []
            token_ids = []
           
            for i, word in enumerate(textlist):  
                tokens = encode_method(word.strip())  # word token ids   
                token_ids.extend(tokens)  # all sentence token ids
                label_1 = labellist[i]
                for m in range(len(tokens)):

                    if m == 0:  # only label the first BPE token of each work
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        labels.append(ignored_label)  # unlabeled BPE token
                        label_mask.append(0)
                        valid.append(0)

            if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
                token_ids = token_ids[0:(max_seq_length-2)]
                labels = labels[0:(max_seq_length-2)]
                valid = valid[0:(max_seq_length-2)]
                label_mask = label_mask[0:(max_seq_length-2)]

            # adding <s>
            token_ids.insert(0, 0)
            labels.insert(0, ignored_label)
            label_mask.insert(0, 0)
            valid.insert(0, 0)

            # adding </s>
            token_ids.append(2)
            labels.append(ignored_label)
            label_mask.append(0)
            valid.append(0)

            assert len(token_ids) == len(labels)
            assert len(valid) == len(labels)

            label_ids = []
            for i, _ in enumerate(token_ids):
                label_ids.append(label_map[labels[i]])

            assert len(token_ids) == len(label_ids)
            assert len(valid) == len(label_ids)

            input_mask = [1] * len(token_ids)

            while len(token_ids) < max_seq_length:
                token_ids.append(1)  # token padding idx
                input_mask.append(0)
                label_ids.append(label_map[ignored_label])  # label ignore idx
                valid.append(0)
                label_mask.append(0)

            while len(label_ids) < max_seq_length:
                label_ids.append(label_map[ignored_label])
                label_mask.append(0)

            assert len(token_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            features.append(
                InputFeatures(input_ids=token_ids,
                              input_mask=input_mask,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
        return features


def create_ner_dataset(features):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.uint8)

    return TensorDataset(
        all_input_ids, all_label_ids, all_lmask_ids, all_valid_ids)

def create_clf_dataset(features):
    
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids)
