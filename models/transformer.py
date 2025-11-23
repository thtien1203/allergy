from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import os
import numpy as np

def tokenize_function(examples, tokenizer):
    # tokenize the description
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

def evaluate_model(trainer, dataset):
    ###################### UPDATE FOR OUR METRICS ###################################
    if MODEL_NUM == 1:
        predictions = trainer.predict(dataset).predictions
    else:
        predictions = trainer.predict(dataset).predictions
    labels = np.array(dataset['label'])

    predictions = np.argmax(predictions, axis=1)

    recall = recall_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    balanced_accuracy = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')

    print(f"recall: {recall}, precision: {precision}, balanced accuracy: {balanced_accuracy}, f1: {f1}")

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = "data"
label_file = "labels.csv"
data_file = "dataset_with_labeled.csv"

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

assert (TRAIN_RATIO + VAL_RATIO + TEST_RATIO) == 1, "train, val, test ratio must sum to 1.0"

MODEL_NUM = 1

if __name__ == '__main__':
    
    if MODEL_NUM == 1:
        model_id = 'bert-base-cased'
    else:
        print(f"Model number {MODEL_NUM} is invalid - please change it to an allowed range of [1-3] and try again.")
        exit(-1)

    # use the model's recommended tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    df = pd.read_csv(os.path.join("..", data_dir, data_file))

    df = df.rename(columns={'input': 'text', 'labels': 'label'})

    df = df[['text', 'label']]

    labels = pd.read_csv(os.path.join("..", data_dir, label_file))
    label_to_val = dict(zip([label for label in labels['labels']], labels.index.to_list()))

    df['label'] = df['label'].map(label_to_val)

    target = "label"

    dataset = Dataset.from_pandas(df)

    dataset = dataset.train_test_split(test_size=TEST_RATIO)

    train_dataset = dataset['train']
    # also split for val set
    train_dataset = train_dataset.train_test_split(test_size=(VAL_RATIO / (1 - TEST_RATIO)))
    
    val_dataset = train_dataset['test']
    train_dataset = train_dataset['train']
    test_dataset = dataset['test']

    # tokenize
    train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)


    config = AutoConfig.from_pretrained(model_id, num_labels=len(labels), ignore_mismatched_sizes=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)


    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=1e-4,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        eval_strategy="epoch",
        eval_accumulation_steps=5,
        # use_cpu=True
    )

    # fine-tune the models on training data and evaluate it on validation data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    print("Test performance:")
    evaluate_model(trainer, test_dataset)