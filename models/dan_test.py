import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import argparse
import time
import nltk
from collections import Counter

nltk.download('punkt_tab')

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
        vec_text = [0] * len(ex)

        for i, word in enumerate(ex):
            if word in word2ind:
                vec_text[i] = word2ind[word]
            else:
                vec_text[i] = word2ind[kUNK]

        return vec_text

def batchify(batch):

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

        all_top_i.append(top_i.squeeze())
        all_labels.append(labels.squeeze())

    all_labels = torch.cat(all_labels)
    all_top_i = torch.cat(all_top_i)

    recall = recall_score(all_labels, all_top_i, average='macro', zero_division=0)
    precision = precision_score(all_labels, all_top_i, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_top_i, average='macro', zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_top_i)

    print(f"recall: {recall}, precision: {precision}, f1: {f1}, balanced accuracy: {balanced_acc}")
    label_indices = list(range(len(ind2class)))
    print(classification_report(all_labels, all_top_i, labels=label_indices, target_names=[ind2class[i] for i in range(len(ind2class))], zero_division=0))




def train(args, model, train_data_loader, dev_data_loader, accuracy, device, class_weights=None):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=1e-2)
    # criterion = nn.CrossEntropyLoss()
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()


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

def load_glove_embeddings(glove_path, word2ind, embedding_dim):
    print("Loading GloVe embeddings...")

    embedding_matrix = np.random.normal(0, 0.6, (len(word2ind), embedding_dim))

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')

            if word in word2ind:  
                idx = word2ind[word]
                embedding_matrix[idx] = vector

    return torch.tensor(embedding_matrix, dtype=torch.float)


class DanModel(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=0):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.dropout1 = nn.Dropout(self.nn_dropout)
        self.fc1 = nn.Linear(emb_dim, n_hidden_units)
        self.dropout2 = nn.Dropout(self.nn_dropout)
        self.fc2 = nn.Linear(n_hidden_units, n_classes)
        self._softmax = nn.Softmax(dim=1)


        self.net = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.dropout1,
            self.fc2,
            self.dropout2
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
        mask = (input_text != 0).float().unsqueeze(-1)
        # Then average them
        masked_embedded = embedded * mask
        average_embeddings = (masked_embedded.sum(dim=1) / text_len.view(-1, 1))

        # Before feeding them through the network

        logits = self.net(average_embeddings)


        if is_prob:
            logits = self._softmax(logits)

        return logits

def save_predictions(data_loader, model, device, original_df, ind2class, output_file):
    model.eval()
    all_predictions = []
    all_labels = []
    for idx, batch in enumerate(data_loader):
        input_text = batch['text'].to(device)
        input_len = batch['len'].to(device)
        labels = batch['labels']
        
        logits = model(input_text, input_len)
        top_n, top_i = logits.topk(1)
        top_i = top_i.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        all_predictions.extend(top_i.flatten())
        all_labels.extend(labels)
    pred_labels = [ind2class[pred] for pred in all_predictions]
    true_labels = [ind2class[true] for true in all_labels]
    results_df = original_df.copy()
    results_df['predicted_label'] = pred_labels
    results_df['correct'] = [pred == true for pred, true in zip(pred_labels, true_labels)]
    results_df.to_csv(output_file, index=False)

def train_with_early_stopping(args, model, train_dataset, dev_loader, device, class_weights, patience=5):
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)
    
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch + 1}/{args.num_epochs}')
        
        # Training phase
        model.train()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler, 
            num_workers=0, 
            collate_fn=batchify
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
        
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        epoch_loss = 0
        for idx, batch in enumerate(train_loader):
            input_text = batch['text'].to(device)
            input_len = batch['len'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_text, input_len)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Training Loss: {avg_loss:.4f}')
        
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dev_loader:
                input_text = batch['text'].to(device)
                input_len = batch['len'].to(device)
                labels = batch['labels']
                
                logits = model(input_text, input_len)
                _, preds = logits.topk(1)
                
                all_preds.extend(preds.squeeze().cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        val_acc = balanced_accuracy_score(all_labels, all_preds)
        
        print(f'Validation F1: {val_f1:.4f}, Balanced Acc: {val_acc:.4f}')
        
        # early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f'NEW BEST! Validation F1: {best_val_f1:.4f}')
        else:
            patience_counter += 1
            print(f'No improvement (patience: {patience_counter}/{patience})')
        
        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break  
    
    # Load best model after loop
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f'\nLoaded best model (Validation F1: {best_val_f1:.4f})')
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--num-epochs', type=int, default=400)
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

    TRAIN_RATIO = 0.70 #0.6
    VAL_RATIO = 0.15 # 0.20
    TEST_RATIO = 0.15 #0.20

    assert (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) == 1, "train, val, test ratio must sum to 1.0" 

    # train_df, test_df = train_test_split(data, test_size=TEST_RATIO)
    # train_df, val_df = train_test_split(train_df, test_size=(VAL_RATIO / (1 - TEST_RATIO)))
    train_df, test_df = train_test_split(data, test_size=TEST_RATIO, 
                                      stratify=data['labels'])
    train_df, val_df = train_test_split(train_df, 
                                     test_size=(VAL_RATIO / (1 - TEST_RATIO)),
                                     stratify=train_df['labels'])
    
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

    train_labels = [label for _, label in train_exs]
    label_counts = Counter(train_labels)
    
    total_samples = len(train_labels)
    class_weights = []
    for i in range(num_classes):
        label_name = ind2class[i]
        count = label_counts.get(label_name, 1)
        weight = total_samples / (num_classes * count)
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights)
    print("\nClass Weights")
    for i, weight in enumerate(class_weights):
        print(f"{ind2class[i]}: {weight:.4f}")

    # model = DanModel(num_classes, len(voc), emb_dim=100)

    model = DanModel(num_classes, len(voc), emb_dim=100, n_hidden_units=512, nn_dropout=0.2)
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
    
    # accuracy = 0
    # for epoch in range(args.num_epochs):
    #     print('start epoch %d' % epoch)
    #     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                         sampler=train_sampler, num_workers=0,
    #                                         collate_fn=batchify)
    #     accuracy = train(args, model, train_loader, dev_loader, accuracy, device, class_weights)
    # early stopping
    model = train_with_early_stopping(args, model, train_dataset, dev_loader, 
                                       device, class_weights, patience=100)
    print('start testing:\n')

    print('\n=== Evaluating on Training Set ===')
    train_loader_eval = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                     sampler=torch.utils.data.sampler.SequentialSampler(train_dataset),
                                                     num_workers=0,
                                                     collate_fn=batchify)
    evaluate(train_loader_eval, model, device)

    print('\n=== Evaluating on Validation Set ===')
    evaluate(dev_loader, model, device)
    
    print('\n=== Evaluating on Test Set ===')

    test_dataset = AllergyDataset(test_exs, word2ind, num_classes, class2ind)
    test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                            sampler=test_sampler, num_workers=0,
                                            collate_fn=batchify)
    evaluate(test_loader, model, device)
    # save_predictions(test_loader, model, device, test_df, ind2class, output_file='./data/dan_predictions.csv')