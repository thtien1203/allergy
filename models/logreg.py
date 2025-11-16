import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score, f1_score, confusion_matrix
from scipy.sparse import hstack

# 0 for bag-of-words; 1 for tf-idf
FEATURE_TYPE = 1

data_dir = "data"
label_file = "labels.csv"
data_file = "dataset_with_labeled.csv"

labels = pd.read_csv(os.path.join("..", data_dir, label_file))
data = pd.read_csv(os.path.join("..", data_dir, data_file))

label_to_val = dict(zip([label for label in labels['labels']], labels.index.to_list()))

data['labels'] = data['labels'].map(label_to_val)

data['allergens'] = data['allergens'].str.replace(',','')

if FEATURE_TYPE == 0:
    vectorizer_text = CountVectorizer()
elif FEATURE_TYPE == 1:
    vectorizer_text = TfidfVectorizer()

vectorized_text = vectorizer_text.fit_transform(data['input'])

if FEATURE_TYPE == 0:
    vectorizer_allergens = CountVectorizer()
elif FEATURE_TYPE == 1:
    vectorizer_allergens = TfidfVectorizer()

vectorized_allergens = vectorizer_allergens.fit_transform(data['allergens'])

X = hstack([vectorized_text, vectorized_allergens])
y = data['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("recall: ", recall_score(y_test, y_pred, average='macro'))
print("precision: ", precision_score(y_test, y_pred, average='macro'))
print("f1: ", f1_score(y_test, y_pred, average='macro'))
print("balanced accuracy: ", balanced_accuracy_score(y_test, y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))