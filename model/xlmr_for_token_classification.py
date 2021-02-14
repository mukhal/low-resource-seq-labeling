from fairseq.models.roberta import XLMRModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
#from TorchCRF import CRF

class Discriminator(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1=nn.Linear(hidden_size, hidden_size)
        self.fc2=nn.Linear(hidden_size, hidden_size)
        self.fc3=nn.Linear(hidden_size, hidden_size)
        self.fc4=nn.Linear(hidden_size, hidden_size)
        self.fc5=nn.Linear(hidden_size, 1)

        self.fc1.weight.data.normal_(
            mean=0.0, std=0.03)
        self.fc2.weight.data.normal_(
            mean=0.0, std=0.03)
        self.fc3.weight.data.normal_(
            mean=0.0, std=0.03)

        
    def forward(self, sent_repr):
        d_interm = nn.functional.relu(self.fc1(sent_repr))
        d_interm = nn.functional.relu(self.fc2(d_interm))
        d_interm = nn.functional.relu(self.fc3(d_interm))
        d_interm = nn.functional.relu(self.fc4(d_interm))
        d_logits = self.fc5(d_interm)
        return torch.sigmoid(d_logits)




class XLMRForTokenClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p, label_ignore_idx=0,
                 head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.classification_head = nn.Linear(hidden_size, n_labels)

        self.label_ignore_idx = label_ignore_idx

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)

        self.device = device

        # initializing classification head
        self.classification_head.weight.data.normal_(
            mean=0.0, std=head_init_range)
        
       
    def forward_generator(self, inputs_ids):
        '''
        Computes a forward pass to generate embeddings 

        Args:
            inputs_ids: tensor of shape (bsz, max_seq_len), pad_idx=1
            labels: temspr pf soze (bsz)

        '''
        transformer_out, _ = self.model(inputs_ids, features_only=True)
        generator_representation = transformer_out.mean(dim=1) # bsz x hidden
        return generator_representation


    def forward(self, inputs_ids, labels, labels_mask, valid_mask, get_sent_repr= False):
        '''
        Computes a forward pass through the sequence agging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask and valid_mask: indicate where loss gradients should be propagated and where 
            labels should be ignored

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out, _ = self.model(inputs_ids, features_only=True)
        sent_repr = transformer_out.mean(dim=1)

        out_1 = F.relu(self.linear_1(transformer_out))
        out_1 = self.dropout(out_1)
        logits = self.classification_head(out_1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            # Only keep active parts of the loss
            if labels_mask is not None:
                active_loss = valid_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                #print("Preds = ", active_logits.argmax(dim=-1))
                #print("Labels = ", active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))
            
            if get_sent_repr:
                return loss, sent_repr
            return loss
        else:
            return logits

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]


class XLMRForTokenClassificationWithCRF(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p, label_ignore_idx=0,
                 head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.classification_head = nn.Linear(hidden_size, n_labels)

        self.label_ignore_idx = label_ignore_idx

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.crf = CRF(n_labels, batch_first=True)

        self.dropout = nn.Dropout(dropout_p)
        self.device = device

        # initializing classification head
        self.classification_head.weight.data.normal_(
            mean=0.0, std=head_init_range)

    def forward(self, inputs_ids, labels, labels_mask, valid_mask):
        '''
        Computes a forward pass through the sequence tagging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask and valid_mask: indicate where loss gradients should be propagated and where 
            labels should be ignored

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out, _ = self.model(inputs_ids, features_only=True)
        bsz, seq_len, _ = transformer_out.size()

        transformer_out = self.dropout(transformer_out)
        #out_1 = F.relu(self.linear_1(transformer_out))
        #out_1 = self.dropout(out_1)
        logits = self.classification_head(transformer_out)

        if labels is not None:
            labels_mask = labels_mask.view(-1) == 1

            loss = torch.tensor(0.0)
            for i in range(logits.size(0)):
                
                sent_logits = logits[i]
                sent_labels = labels[i]

                active_logits = sent_logits.view(-1, self.n_labels)[labels_mask] # (seq_len, n_labels)
                active_labels = sent_labels.view(-1)[labels_mask] # (seq_len)

                sent = active_logits.unsqueeze(0)
                lbls = active_labels.unsqueeze(0)

                ls = -1 * self.crf(sent, lbls)
                loss += ls
            
            loss /= logits.size(0)
            return loss

        else:
            # return raw logits
            return logits

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]

    def decode_logits(self, logits, mask, device):
        """
        Takes as input logits of shape: (bsz, seq_len, n_labels)

        Returns:
            decoded_tags: most probable sequence of tags (bsz, seq_len) 
        """
        bsz, seq_len, _ = logits.size()

        mask = mask.view(-1) == 1
        decoded_tags = torch.zeros([bsz, seq_len]).long()

        for i in range(logits.size(0)):
            sent_logits = logits[i]
            active_logits = sent_logits.view(-1, self.n_labels)[mask] # (seq_len, n_labels)
            sent = active_logits.unsqueeze(0)

            tags = self.crf.decode(sent)[0]
            decoded_tags[i, :len(tags)] = torch.LongTensor(tags)
    
        return decoded_tags
