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

dataset_train, dataset_val, vocab_size = preprocess.prepare_dataset(df, bert_label=True)
dataloader_train, dataloader_val = preprocess.prepare_dataloader(dataset_train, dataset_val, batch_size)
model = models.get_bert().to(device)
model.to(device)

print(model)
print(f"Model Params : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

from torch.optim import AdamW

epochs = 50
optimizer = AdamW(
    model.parameters(),
    lr = 1e-5,
)

clip = 1

def bert_iteration(model, loader, is_train, optimizer=None):
    model.train() if is_train else model.eval()
    labels, predictions = [], []
    for batch in loader:
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[2]
        }
        if is_train: model.zero_grad()
        loss, output = model(**inputs)
        if is_train: 
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        _, pred = torch.max(output, 1)
        # Because BERT labels are one-hot encoded, we need argmax here
        labels.extend(torch.argmax(batch[2].to("cpu"), dim=1).tolist())
        predictions.extend(pred.to("cpu").tolist())
    return labels, predictions

def evaluate(label, pred):
    return accuracy_score(label, pred), recall_score(label, pred, average='macro'),

for epoch in range(epochs):
    train_loader = tqdm(dataloader_train)
    # BERT comes together with internally defined loss function
    # which is Cross Entropy (same as in RNN)
    train_labels, train_predictions = bert_iteration(model, train_loader, is_train=True, optimizer=optimizer)
    train_evaluation = evaluate(train_labels, train_predictions)
        
    valid_loader = tqdm(dataloader_val)
    valid_labels, valid_predictions = bert_iteration(model, valid_loader, is_train=False)
    valid_evaluation = evaluate(valid_labels, valid_predictions)
    
    print(f'Epoch {epoch+1}') 
    print(f'train acc  : {train_evaluation[0]:.3f}, recall : {train_evaluation[1]:.3f}')
    print(f'valid acc  : {valid_evaluation[0]:.3f}, recall : {valid_evaluation[1]:.3f}')

test_loader = tqdm(dataloader_val)
test_labels, test_predictions = bert_iteration(model, test_loader, is_train=False)
test_evaluation = evaluate(test_labels, test_predictions)
print(f'TEST acc  : {test_evaluation[0]:.3f}, recall : {test_evaluation[1]:.3f}')

full_dataset, vocab_size = preprocess.prepare_dataset(df, bert_label=True, split=False) # Unsplitted dataset
full_loader = tqdm(DataLoader(
    full_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
))
labels, predictions = bert_iteration(model, full_loader, is_train=False)
df = df[:len(predictions)]
df['pred'] = predictions
plot_data = pd.DataFrame({
    "pos" : df[(df['pred']==2)].groupby([df['date']]).count()['pred'],
    "neu" : df[(df['pred']==1)].groupby([df['date']]).count()['pred'],
    "neg" : df[(df['pred']==0)].groupby([df['date']]).count()['pred'],
})
sns.set(rc = {'figure.figsize':(15,8)})
sns.lineplot(data=plot_data)
plt.savefig("bert_fig.png")
