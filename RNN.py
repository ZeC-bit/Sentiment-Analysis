import preprocess, models
import seaborn as sns, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tqdm.auto import tqdm
from sklearn.metrics import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")
batch_size=64

df = preprocess.load_from('data.json', to_df=True)
dataset_train, dataset_val, vocab_size = preprocess.prepare_dataset(df, bert_label=False)
dataloader_train, dataloader_val = preprocess.prepare_dataloader(dataset_train, dataset_val, batch_size)
model = models.get_lstm(vocab_size=vocab_size, num_layers=3).to(device)
model.to(device)

print(model)
print(f"Model Params : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

from torch.optim import AdamW
# AdamW (ICLR 2019) is a variant of Adam that is designed to generalize better

criterion = nn.CrossEntropyLoss()
epochs = 50
optimizer = AdamW(
    model.parameters(),
    lr = 1e-3,
)

clip = 1

# This main training loop is essentially universal in pytorch usage. 

def lstm_iteration(model, loader, is_train, optimizer=None, criterion=None):
    model.train() if is_train else model.eval()
    labels, predictions = [], []
    # For RNNs, we first initialize the hidden state.
    h = model.init_hidden(batch_size, device)
    for (input, _, label) in loader:

        input, label = input.to(device), label.to(device)   
        # RNN hidden state
        h = tuple([each.data for each in h])
        if is_train: model.zero_grad()
        # "Recurrent" Neural Network... hidden state is updated. 
        output, h = model(input, h)
        if is_train: 
   
            loss = criterion(output, label)
            # For training iterations, perform optimization
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        # Get the index of the max probability (=index of the predicted class)
        _, pred = torch.max(output, 1)
        # For assessment of the model, get the true and predicted label
        labels.extend(label.to("cpu").tolist())
        predictions.extend(pred.to("cpu").tolist()) 
    return labels, predictions

def evaluate(label, pred):
    # accuracy and recall : data should be in type of [0, 1, 2], [1, 2, 1] (not one-hot)
    return accuracy_score(label, pred), recall_score(label, pred, average='macro'),

# Train for several epochs
for epoch in range(epochs):
  
    train_loader = tqdm(dataloader_train)
    train_labels, train_predictions = lstm_iteration(model, train_loader, is_train=True, optimizer=optimizer, criterion=criterion)
    train_evaluation = evaluate(train_labels, train_predictions)
        
    valid_loader = tqdm(dataloader_val)
    valid_labels, valid_predictions = lstm_iteration(model, valid_loader, is_train=False)
    valid_evaluation = evaluate(valid_labels, valid_predictions)
    
    print(f'Epoch {epoch+1}') 
    print(f'train acc  : {train_evaluation[0]:.3f}, recall : {train_evaluation[1]:.3f}')
    print(f'valid acc  : {valid_evaluation[0]:.3f}, recall : {valid_evaluation[1]:.3f}')

test_loader = tqdm(dataloader_val)
test_labels, test_predictions = lstm_iteration(model, test_loader, is_train=False)
test_evaluation = evaluate(test_labels, test_predictions)
print(f'TEST acc  : {test_evaluation[0]:.3f}, recall : {test_evaluation[1]:.3f}')

# Unsplitted dataset
full_dataset, vocab_size = preprocess.prepare_dataset(df, bert_label=False, split=False) 
full_loader = tqdm(DataLoader(
    full_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
))
# Final iteration to get all predictions
labels, predictions = lstm_iteration(model, full_loader, is_train=False)
# Last batch is dropped, so we drop them from dataframe
df = df[:len(predictions)]
df['pred'] = predictions
# Save predictions for plotting
plot_data = pd.DataFrame({
    "pos" : df[(df['pred']==2)].groupby([df['date']]).count()['pred'],
    "neu" : df[(df['pred']==1)].groupby([df['date']]).count()['pred'],
    "neg" : df[(df['pred']==0)].groupby([df['date']]).count()['pred'],
})
sns.set(rc = {'figure.figsize':(15,8)})
sns.lineplot(data=plot_data)
# Save figure
plt.savefig("rnn_fig.png")
