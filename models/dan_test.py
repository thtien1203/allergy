import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score, confusion_matrix

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import argparse
import time
import nltk

kUNK = '<unk>'
kPAD = '<pad>'

def class_labels(data):
    class_to_i = {}
    i_to_class = {}
    i = 0
    for _, ans in data:
        if ans not in class_to_i.keys():
            class_to_i[ans] = i
            i_to_class[i] = ans
            i+=1
    return class_to_i, i_to_class

def load_data(df: pd.DataFrame, lim=None):
    """
    load the json file into data list
    """
    
    data = list()

    if lim:
        df = df.iloc[:lim]

    for _, row in df.iterrows():
        input_text = nltk.word_tokenize(row['input'])
        label = row['labels']
        if label:
            data.append((input_text, label))
    
    return data


def load_words(exs):
    """
    vocabuary building
    Keyword arguments:
    exs: list of input questions-type pairs
    """

    words = set()
    word2ind = {kPAD: 0, kUNK: 1}
    ind2word = {0: kPAD, 1: kUNK}
    for input_text, _ in exs:
        for w in input_text:
            words.add(w)
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [kPAD, kUNK] + words
    return words, word2ind, ind2word


class AllergyDataset(Dataset):
    """
    Pytorch data class for questions
    """

    def __init__(self, examples, word2ind, num_classes, class2ind=None):
        self.text = []
        self.labels = []

        for ii, ll in examples:
            self.text.append(ii)
            self.labels.append(ll)
        
        if type(self.labels[0]) == str:
            for i in range(len(self.labels)):
                try:
                    self.labels[i] = class2ind[self.labels[i]]
                except:
                    self.labels[i] = num_classes
        self.word2ind = word2ind
    
    def __getitem__(self, index):
        return self.vectorize(self.text[index], self.word2ind), \
          self.labels[index]
    
    def __len__(self):
        return len(self.text)

    @staticmethod
    def vectorize(ex, word2ind):
        """
        vectorize a single example based on the word2ind dict. 
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        vec_text = [0] * len(ex)

        for i, word in enumerate(ex):
            if word in word2ind:
                vec_text[i] = word2ind[word]
            else:
                vec_text[i] = word2ind[kUNK]

        return vec_text

def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 
    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    input_len = list()
    label_list = list()
    for ex in batch:
        input_len.append(len(ex[0]))
        label_list.append(ex[1])

    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(input_len), max(input_len)).zero_()
    for i in range(len(input_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    batch = {'text': x1, 'len': torch.FloatTensor(input_len), 'labels': target_labels}
    return batch


def evaluate(data_loader, model, device):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    num_samples = 0
    all_top_i = []
    all_labels = []

    for idx, batch in enumerate(data_loader):
        input_text = batch['text'].to(device)
        input_len = batch['len'].to(device)
        labels = batch['labels']

        ####Your code here
        logits = model(input_text, input_len)
        top_n, top_i = logits.topk(1)
        num_examples += input_text.size(0)
        num_samples += len(batch['text'])

        top_i = top_i.data.cpu()
        labels = labels.data.cpu()

        all_top_i.append(top_i.squeeze().T)
        all_labels.append(labels.squeeze().T)

    all_labels = torch.cat(all_labels)
    all_top_i = torch.cat(all_top_i)

    recall = recall_score(all_labels, all_top_i, average='macro')
    precision = precision_score(all_labels, all_top_i, average='macro')
    f1 = f1_score(all_labels, all_top_i, average='macro')
    balanced_acc = balanced_accuracy_score(all_labels, all_top_i)

    print(f"recall: {recall}, precision: {precision}, f1: {f1}, balanced accuracy: {balanced_acc}")

    


def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model
    Keyword arguments:
    args: arguments 
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### modify the following code to complete the training funtion

    for idx, batch in enumerate(train_data_loader):
        input_text = batch['text'].to(device)
        input_len = batch['len'].to(device)
        labels = batch['labels']

        #### Your code here

        logits = model(input_text, input_len)
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # clip_grad_norm_(model.parameters(), args.grad_clipping) 
        print_loss_total += loss.data.cpu().numpy()
        epoch_loss_total += loss.data.cpu().numpy()

        # if idx % args.checkpoint == 0 and idx > 0:
        #     print_loss_avg = print_loss_total / args.checkpoint

        #     print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
        #     print_loss_total = 0
        #     evaluate(dev_data_loader, model, device)
        
    return accuracy




class DanModel(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_dim=1000,
                 n_hidden_units=512, nn_dropout=0.2):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(self.nn_dropout)
        self.bn1 = nn.BatchNorm1d(emb_dim)
        self.fc1 = nn.Linear(emb_dim, n_hidden_units)
        self.dropout2 = nn.Dropout(self.nn_dropout)
        self.bn2 = nn.BatchNorm1d(n_hidden_units)
        self.fc2 = nn.Linear(n_hidden_units, n_classes)
        self._softmax = nn.Softmax(dim=1)


        self.net = nn.Sequential(
            self.dropout1,
            # self.bn1,
            self.fc1,
            self.dropout2,
            # self.bn2,
            self.fc2
        )
        
       
    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass, returns the logits of the predictions.
        
        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """
        # Complete the forward funtion.  First look up the word embeddings.

        embedded = self.embeddings(input_text)

        # Then average them

        average_embeddings = (embedded.sum(dim=1) / text_len.view(-1, 1))

        # Before feeding them through the network

        logits = self.net(average_embeddings)


        if is_prob:
            logits = self._softmax(logits)

        return logits



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--checkpoint', type=int, default=5)
    parser.add_argument("--limit", help="Number of training documents", type=int, default=-1, required=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = "data"
    label_file = "labels.csv"
    data_file = "dataset_with_labeled.csv"

    labels = pd.read_csv(os.path.join("..", data_dir, label_file))
    data = pd.read_csv(os.path.join("..", data_dir, data_file))

    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    TEST_RATIO = 0.2

    assert (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) == 1, "train, val, test ratio must sum to 1.0" 

    train_df, test_df = train_test_split(data, test_size=TEST_RATIO)
    train_df, val_df = train_test_split(train_df, test_size=(VAL_RATIO / (1 - TEST_RATIO)))

    ### Load data
    train_exs = load_data(train_df)
    dev_exs = load_data(val_df)
    test_exs = load_data(test_df)

    ### Create vocab
    voc, word2ind, ind2word = load_words(train_exs)

    #get num_classes from training + dev examples - this can then also be used as int value for those test class labels not seen in training+dev.
    num_classes = len(list(set([ex[1] for ex in train_exs+dev_exs])))

    print(num_classes)


    #get class to int mapping
    class2ind, ind2class = class_labels(train_exs + dev_exs)

    model = DanModel(num_classes, len(voc))
    model.to(device)
    print(model)
    #### Load batchifed dataset
    train_dataset = AllergyDataset(train_exs, word2ind, num_classes, class2ind)
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

    dev_dataset = AllergyDataset(dev_exs, word2ind, num_classes, class2ind)
    dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                            sampler=dev_sampler, num_workers=0,
                                            collate_fn=batchify)
    
    accuracy = 0
    for epoch in range(args.num_epochs):
        print('start epoch %d' % epoch)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            sampler=train_sampler, num_workers=0,
                                            collate_fn=batchify)
        accuracy = train(args, model, train_loader, dev_loader, accuracy, device)
    print('start testing:\n')

    test_dataset = AllergyDataset(test_exs, word2ind, num_classes, class2ind)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            sampler=test_sampler, num_workers=0,
                                            collate_fn=batchify)
    evaluate(test_loader, model, device)