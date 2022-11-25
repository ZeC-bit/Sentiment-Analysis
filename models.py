import torch.nn as nn 
import torch
class LSTMRNN(nn.Module):
    def __init__(self,
                 vocab_size,
                 num_layers = 3,
                 hidden_dim = 64,
                 embedding_dim = 64,
                 dropout = 0.5):

        super(LSTMRNN,self).__init__()
        self.num_layers = num_layers
        self.output_dim = 3
        self.hidden_dim = hidden_dim
        # Dimensionality reduction from vocab_size to embedding_dim
        # Necessary if we use pre-trained tokenizer (which is the case)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer 
        self.lstm = nn.LSTM (
            input_size=embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers, 
            batch_first=True
        )
        
        # dropout layer
        self.dropout = nn.Dropout(dropout)
    
        # linear layer for final classification 
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x, hidden):
        embeds = self.embedding(x) 
        lstm_out, hidden = self.lstm(embeds, hidden)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = out[:, -1]
        return out, hidden
        
        
    def init_hidden(self, batch_size, device):
        # Initializes hidden state
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.num_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.num_layers,batch_size,self.hidden_dim)).to(device)
        return (h0, c0)

def get_lstm(vocab_size, num_layers):
    return LSTMRNN(
        vocab_size=vocab_size, num_layers=num_layers
    )

from transformers import BertForSequenceClassification
def get_bert():
    # Huggingface BERT model for sequence classification
    # There is zero chance we can train bert from scratch....
    # Loading pre-trained weights is absolutely necessary.
    return BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            # To use bert large, use 'bert-large-uncased' instead.
            num_labels = 3,
            output_attentions = False,
            output_hidden_states = False,
            return_dict = False
        )
