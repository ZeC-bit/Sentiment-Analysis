import pandas as pd, numpy as np 
from sklearn.metrics import *
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
import requests
import pickle, sys
import json
import re 
import torch
from tqdm.auto import tqdm
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
sys.setrecursionlimit(100000)

def crawl():
    pages = []
    # Crawl the news from aastocks.com
    pbar = tqdm(range(1208101, 1200000, -1))
    cnt = 0
    for i in pbar:
        url = f"http://www.aastocks.com/en/stocks/news/aafn-con/NOW.{i}/company-news/HK6"
        get_page = requests.get(url)
        soup = BeautifulSoup(get_page.text, "html.parser")
        headline = [div.text for div in soup.select('div.newshead5')]
        if len(headline) > 0:
            # if page is valid, then append it to the list of pages
            cnt += 1
            pages.append(soup)
            pbar.set_postfix_str(f"Found {cnt} news")
    return pages

def cleanup(pages):
    stock_regex = re.compile(r'[A-Z -]+.[(][0-9]+[.]HK[)]')
    numerics_regex = re.compile(r'[+-][0-9.]+ [(][+-][0-9.]+[%][)]')
    # stock regex : [UPPERCASES] ([NUMERIC][.][NUMERIC][HK])
    # numerics : erase stuff such as +0.something (+0.something%) since this might confuse NLP model.
    all_news = []
    for soup in tqdm(pages):
        try:
            news = {}
            # get the headline, abstract and pos/neg labels
            headline = [div.text for div in soup.select('div.newshead5')][0].strip()
            abstract = [div.text for div in soup.select('div.newscontent5')][0]
            positive = int([div.find('div', attrs={'class' : 'value'}).text for div in soup.findAll('div', attrs={'class': 'divBullish'})][0])
            negative = int([div.find('div', attrs={'class' : 'value'}).text for div in soup.findAll('div', attrs={'class': 'divBearish'})][0])
            date = [div.text for div in soup.select('div.newstime5')][0]

            # Data cleaning : remove some unicode characters and repetitive phrases
            abstract = abstract.replace(u'\xa0', u' ')
            abstract = abstract.partition("(HK")[0]
            delete_tokens = numerics_regex.findall(abstract)
            for delete_token in delete_tokens:
                abstract = abstract.replace(delete_token, "")

            m = [s.strip() for s in stock_regex.findall(abstract)]
            stocks = []
            for s in stock_regex.findall(abstract):
                _s = s
                s = s.strip().split(' ')
                name, id = (' '.join(s[:len(s)-1]), s[-1][1:-1])
                stocks.append((name, id))
                abstract = abstract.replace(_s, name)
            news["headline"] = headline
            news["abstract"] = abstract
            news["date"] = date
            news["positive"] = positive
            news["negative"] = negative
            news["stocks"] = stocks
            all_news.append(news)
        except Exception:
            continue
    return all_news

def label_mapping(pos, neg):
    if pos > neg * 2: return 2
    elif neg > pos: return 0
    else: return 1
    
def load_from(jsonfile, to_df = True):
    # This is not a good practice, but I don't have time to do it better...
    # In general, never use "eval"
    jsondata = eval(json.loads(json.load(open('data.json', 'r'))))
    if to_df:
        df = pd.DataFrame(jsondata)
        df['label'] = [
            label_mapping(pos, neg)
            for (pos, neg) in zip(df['positive'], df['negative'])
        ]
        text = []
        for (headline, abstract) in zip(df['headline'], df['abstract']):
            text.append(headline + abstract)
        df['text'] = text
        df['date'] = [
            datetime.strptime(s, "%Y/%m/%d %H:%M").date()
            for s in df['date']
        ]
        return df
    else: return jsondata

from sklearn.model_selection import train_test_split
def prepare_dataset(df : pd.DataFrame, bert_label = True, split = True):
    # Train-test split (80 : 20)
    X, y = df['text'].values, df['label'].values
    if split:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else: X_train, y_train = X, y

    # Tokenize using pretrained BERT tokenizer
    # Our dataset size is small, hence training tokenizer from scratch is not viable.
    from transformers import BertTokenizer
    from torch.utils.data import TensorDataset
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True,
        strip_accents=True
    )
    encoded_data_train = tokenizer.batch_encode_plus(
        X_train,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=128,
        return_tensors='pt'
    )
    input_ids_train = encoded_data_train['input_ids']
    attention_masks_train = encoded_data_train['attention_mask']

    # BERT takes one-hot label, in float type
    if bert_label:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
        y_train = onehot.fit_transform(y_train.reshape(-1, 1))
        labels_train = torch.tensor(y_train, dtype=torch.float32)
    else:
        labels_train = torch.tensor(y_train)
    dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
    
    if split:
        # Make validation dataset together
        encoded_data_val = tokenizer.batch_encode_plus(
            X_val,
            add_special_tokens=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        if bert_label:
            y_val = onehot.transform(y_val.reshape(-1, 1))
            labels_val = torch.tensor(y_val, dtype=torch.float32)
        else:
            labels_val = torch.tensor(y_val)
        dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

    if split:
        return dataset_train, dataset_val, tokenizer.vocab_size
    else:
        return dataset_train, tokenizer.vocab_size

from torch.utils.data import DataLoader
def prepare_dataloader(dataset_train, dataset_val, batch_size=32):
    # Dataset to dataloader
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    return dataloader_train, dataloader_val

    
if __name__ == "__main__":
    # If this file is called as main, then we will prepare the dataset by crawling
    pages = crawl()
    news = cleanup(pages)
    serialized = json.dumps(news)
    json.dump(serialized, open("data.json", 'w'))
