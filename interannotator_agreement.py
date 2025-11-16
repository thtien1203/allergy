import pandas as pd
import numpy as np
import os
from sklearn.metrics import cohen_kappa_score

def print_label_idx(label_map):
    s = ""
    for k, v in label_map.items():
        s += f"({v}) {k}; "
    
    s += f"({len(label_map)}) unknown"

    print(s)

def get_category_input(label_map):
    while True:
        try:
            val = int(input("Please categorize the text above (by number): "))
            if val < 0 or val >= len(label_map):
                raise ValueError
            break
        except ValueError:
            print("Please enter a valid integer.")
    return val


NUM_TO_SAMPLE = 50

cwd = os.getcwd()
data_dir = "data"
label_file = "labels.csv"
data_file = "dataset_with_labeled.csv"

labels = pd.read_csv(os.path.join(cwd, data_dir, label_file))

label_to_val = dict(zip([label.lower() for label in labels['labels']], labels.index.to_list()))


dataset_df = pd.read_csv(os.path.join(cwd, data_dir, data_file))

sampled_df = dataset_df.sample(NUM_TO_SAMPLE, ignore_index=True, axis=0)

self_labels = []

print("ALLERGY ANNOTATION")
print(f"This script will sample {NUM_TO_SAMPLE} data points from the allergy dataset. You will have to classify them by their input text.")
print(f"These are the categories to choose from. Please input them by number")
print_label_idx(label_to_val)
input("Presse enter to start: ")

for index, row in sampled_df.iterrows():
    os.system('cls' if os.name == 'nt' else 'clear')

    print(f"{index + 1} / {NUM_TO_SAMPLE}")
    print("TEXT:")
    print(row['input'])
    print()
    print_label_idx(label_to_val)
    val = get_category_input(label_to_val)
    self_labels.append(val)

clustered_labels = np.array([label_to_val[label.lower()] for label in sampled_df['labels']])
self_labels = np.array(self_labels)

cohen_kappa = cohen_kappa_score(clustered_labels, self_labels)

# Cohen's Kappa
print("Cohen's Kappa score (interannotator agreement):")
print(cohen_kappa)