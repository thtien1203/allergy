from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import pandas as pd
import os

def tokenize_function(examples, tokenizer):
    # tokenize the description
    return tokenizer(examples['input'], padding="max_length", truncation=True, max_length=512)

def evaluate_model(trainer, dataset):
    ###################### UPDATE FOR OUR METRICS ###################################
    if MODEL_NUM == 1:
        predictions = trainer.predict(dataset).predictions[0]
    else:
        predictions = trainer.predict(dataset).predictions
    labels = dataset['label']
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    print(f"MSE: {mse}\nR2: {r2}")

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

    target = "labels"

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


    config = AutoConfig.from_pretrained(model_id, num_labels=1, ignore_mismatched_sizes=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, config=config)


    training_args = TrainingArguments(
        output_dir='./results',
        learning_rate=1e-4,
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
        eval_accumulation_steps=5 # this transfers memory from GPU to CPU, comment out if your hardware is less prohibitive
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