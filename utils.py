import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import jieba
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, pad_packed_sequence, pad_sequence
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

add_punc='‘ ？《》⑦，】℃“！—。￥”’：）；、（【'
filter_punc = string.punctuation + add_punc
filter_punc = re.sub("[!?？！]", "", filter_punc)
filter_punc = re.escape(filter_punc)
stop_words  = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# set random seeds to keep the results identical
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
GLOBAL_SEED = 2019
setup_seed(GLOBAL_SEED)

def transform_punc(text):
    text = re.sub("(ca|wo)n't", r"\1n not", text)
    text = re.sub("t's", "t is", text)
    text = re.sub("'re", " are", text)
    text = re.sub("n't", " not", text)
    text = re.sub("([!?])", r" \1 ", text)
    text = re.sub(r'\\n|<br /><br />', " ", text)
    return text

def sent_split(text):
    sentences = re.split(r'\\n|<br /><br />|\…{1,2}|\.{3,}|[\.\?\!;]', text)
    return sentences

def sent_split_zh(text):
    text = re.sub(r'(\s+|\…{1,2}|\.{3,}|[。，！｜；？\.\?\!,;])', r'\1 ', text, re.U)
    return text.split() 


def clean_text(text):
    text = re.sub("(ca|wo)n't", r"\1n not", text)
    text = re.sub("t's", "t is", text)
    text = re.sub("'re", " are", text)
    text = re.sub("n't", " not", text)
    text = re.sub(r'\\n|<br /><br />', " ", text)
    text = re.sub(r'[{}]'.format(filter_punc), ' ', text)
    text = re.sub("([!?])", r" \1 ", text)
    text = re.sub(r'\s+',' ',text, re.U)
    text = text.lower()
    text = [lemmatizer.lemmatize(token, "v") for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token) for token in text]
    # text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

def clean_text_zh(text):
    text = re.sub(r'[{}]'.format(filter_punc), ' ', text)
    text = re.sub("[！!]", " ！ ", text)
    text = re.sub("[？?]", " ？ ", text)
    text = re.sub(r'\s+',' ',text, re.U)
    return text


def process(text):
    sents = sent_split(text)
    res = []
    for sent in sents:
        sent = clean_text(sent)
        split_sen = sent.split()
        if len(split_sen) > 0:
            res.append(split_sen)
    return res

def process_zh(text):
    sents = sent_split_zh(text)
    res = []
    for sent in sents:
        sent = clean_text_zh(sent)
        split_sen = list(jieba.cut(sent))
        if len(split_sen) > 0:
            res.append(split_sen)
    return res

def pad_and_sort_batch(Batch):
    batch_split = list(zip(*Batch))
    seqs, labels, lengths = batch_split[0], batch_split[1], batch_split[2]
    seq_lengths, idxs = torch.tensor(lengths).sort(0, descending=True)
    seqs = pad_sequence(seqs, batch_first=True)
    seq_tensor = seqs[idxs]
    labels_tensor = torch.tensor(labels, dtype=torch.int64)[idxs]
    return seq_tensor.to(DEVICE), labels_tensor.to(DEVICE), seq_lengths.to(DEVICE)


def preprocess_for_batch(X_train, y_train, BATCH_SIZE):
    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1)
    sorted_index = np.argsort([len(sen) for sen in X_train])
    index_batchs = []
    n = sorted_index.shape[0]
    # pack every batch
    for i in range(0, int(np.ceil(n/BATCH_SIZE))):
        if (i+1)*BATCH_SIZE > n: 
            break # drop last batch if size < BATCH_SIZE
        else:
            index_batchs.append(sorted_index[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

    # shuffle the batchs
    idxs = list(range(len(index_batchs)))
    np.random.shuffle(idxs)
    shuffle_batchs = np.array(index_batchs)[idxs]

    # convert list item to tensor, avoid memory leak in dataset
    X_train_sorted_tensors = [torch.tensor(X_train[idx], dtype=torch.int64) for index_batch in shuffle_batchs for idx in index_batch]
    y_train_sorted_tensors = [torch.tensor(y_train[idx], dtype=torch.int64) for index_batch in shuffle_batchs for idx in index_batch]
    return X_train_sorted_tensors, y_train_sorted_tensors

def pad_or_truncate(seq, maxlen=512, pad=False):
    if seq.size(0) >= maxlen:
        return seq[:maxlen]
    if pad:
        return torch.cat((seq, torch.zeros(maxlen-seq.size(0), dtype=torch.int64)))
    else:
        return seq

def pad_and_truncate_hierarchical(Batch, MAX_SEN_LEN, MAX_SEN_NUM):
    batch_split = list(zip(*Batch))
    seq, labels = batch_split[0], batch_split[1]
    try:
        max_sen_num = min(max([len(sens) for sens in seq]), MAX_SEN_NUM)
        max_sen_len = min(max([len(sen) for sens in seq for sen in sens]), MAX_SEN_LEN)
    except:
        max_sen_num, max_sen_len = 1, 1
    X = torch.zeros(len(seq), max_sen_num, max_sen_len, dtype=torch.int64)
    for idx1, sens in enumerate(seq):
        for idx2, sen in enumerate(sens):
            if idx2 >= max_sen_num:
                break
            if len(sen) >= max_sen_len:
                sen = sen[:max_sen_len]
            X[idx1, idx2, :len(sen)] = torch.tensor(sen.astype(int), dtype=torch.int64)
    return X.to(DEVICE), torch.tensor(labels, dtype=torch.int64).to(DEVICE)


def preprocess_for_batch_hierarchical(X_train, y_train, BATCH_SIZE):
    X_train = np.array(X_train)
    y_train = np.array(y_train).reshape(-1, 1).astype(int)
    sorted_index = np.argsort([len(sen) for sen in X_train])
    index_batchs = []
    n = sorted_index.shape[0]
    # pack every batch
    for i in range(0, int(np.ceil(n/BATCH_SIZE))):
        if (i+1)*BATCH_SIZE > n: 
            break # drop last batch if size < BATCH_SIZE
        else:
            index_batchs.append(sorted_index[i*BATCH_SIZE:(i+1)*BATCH_SIZE])

    # shuffle the batchs
    idxs = list(range(len(index_batchs)))
    np.random.shuffle(idxs)
    shuffle_batchs = np.array(index_batchs)[idxs]

    X_train_sorted = np.array([X_train[idx] for index_batch in shuffle_batchs for idx in index_batch])
    y_train_sorted = np.array([y_train[idx] for index_batch in shuffle_batchs for idx in index_batch])
    return X_train_sorted, y_train_sorted
