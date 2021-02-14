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


class XLMRForSequenceClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p, label_ignore_idx=-1,
                 head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.classification_head = nn.Linear(hidden_size, n_labels)

        self.label_ignore_idx = label_ignore_idx
        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.model.register_classification_head('clf', n_labels)

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


    def forward(self, inputs_ids, labels, get_sent_repr=False):
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
        self.eval()
        #transformer_out, _ = self.model(inputs_ids, features_only=True)
        #sent_repr = transformer_out[:, 0, :]# <s> token
        #sent_repr = transformer_out.mean(dim=1)
        #print(inputs_ids)
        #out_1 = F.relu(self.linear_1(sent_repr))
        #out_1 = self.dropout(out_1)
        #logits = self.classification_head(sent_repr)
        #print(logits)

        logits, states = self.model(inputs_ids, features_only=True, classification_head_name='clf')
        
        if labels is not None:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, labels, reduction='sum')
            if get_sent_repr:
                return loss, states
            return loss
        
        else:
            return logits

    def encode_sent(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        return tensor_ids.cpu().numpy().tolist()

