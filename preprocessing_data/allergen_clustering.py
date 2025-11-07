import pandas as pd
import numpy as np
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

df_embeddings = pd.read_csv("./data/word_embeddings.csv")

# convert json back to numpy arrays
embeddings = np.array([json.loads(vec) for vec in df_embeddings['word_embeddings']])
valid_allergens = df_embeddings['allergens'].tolist()
# count how often each allergen appears
df_filtered = pd.read_csv("./data/filtering_extracted_allergens.csv")
all_allergens = []
for allergen_str in df_filtered['allergens'].dropna():
    allergens = [a.strip().lower() for a in str(allergen_str).split(',')]
    all_allergens.extend(allergens)
allergen_counts = Counter(all_allergens)

# print(f"Loaded {len(embeddings)} allergen embeddings")
# print(f"Embedding dimension: {embeddings.shape[1]}")
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(embeddings)
manual_corrections = {
    'chicken':0, 'mushrooms': 0, 'tomatoes': 0,'alcohol': 0,   
    'latex': 4,
}
# manual corrections
for i, allergen in enumerate(valid_allergens):
    if allergen in manual_corrections:
        cluster_labels[i] = manual_corrections[allergen]

insect_allergens = {"bee", "wasp", "mosquito", "insect bite"}
new_cluster_id = cluster_labels.max() + 1   # this will be 6, new cluster = 6

for i, allergen in enumerate(valid_allergens):
    if allergen in insect_allergens:
        cluster_labels[i] = new_cluster_id
df_embeddings['cluster'] = cluster_labels
n_clusters = cluster_labels.max() + 1
# reduce dimensions for visualization
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(embeddings)
fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)

ax1 = fig.add_subplot(gs[0])  # Scatter plot
ax2 = fig.add_subplot(gs[1])
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#bcbd22']
cluster_to_category = {
    0: "Food",
    1: "Pet",
    2: "Environmental",
    3: "Seafood",
    4: "Contact",
    5: "Drug",
    6: "Insect"
}
cluster_name = [f'Cluster {i}' for i in range(n_clusters)]
# plot each cluster
for i in range(n_clusters):
    mask = cluster_labels == i
    ax1.scatter(
        embeddings_2d[mask, 0],
        embeddings_2d[mask, 1],
        c=colors[i%len(colors)],
        label=cluster_name[i],
        alpha=0.7,
        s=150,
        edgecolors='black',
        linewidth=0.8
    )
# place allergens on scatter plot
for allergen, (x, y), label in zip(valid_allergens, embeddings_2d, cluster_labels):
    if allergen in manual_corrections:
        ax1.text(
            x + 0.03, y + 0.03,
            allergen,
            fontsize=9,
            color='black',
        )
    else:
        ax1.text(
            x + 0.03, y + 0.03,
            allergen,
            fontsize=8,
            color='black'
        )
ax1.set_xlabel('PCA Component 1', fontsize=13, fontweight='bold')
ax1.set_ylabel('PCA Component 2', fontsize=13, fontweight='bold')
ax1.set_title(f'K-means Clustering (K={n_clusters})', 
              fontsize=15, fontweight='bold', pad=15)
ax1.legend(loc='best', fontsize=10, framealpha=0.95)
ax1.grid(True, alpha=0.3, linestyle='--')

ax2.set_title("Allergens per Cluster", fontsize=14, fontweight="bold", pad=15)
ax2.axis("off") 
y = 0.98  # start at top
for i in range(n_clusters):
    # get allergens in this cluster
    cluster_allergens = [
        allergen for allergen, label in zip(valid_allergens, cluster_labels)
        if label == i
    ]
    # sort by frequency in original data
    cluster_allergens_sorted = sorted(
        cluster_allergens,
        key=lambda x: allergen_counts.get(x, 0),
        reverse=True
    )
    allergens_text = ", ".join(cluster_allergens_sorted)

    ax2.text(
        0.02, y,
        f'Cluster {i} ({cluster_to_category.get(i)}):\n{allergens_text}',
        transform=ax2.transAxes,
        fontsize=9.5,
        va='top',
        wrap=True,
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor=colors[i % len(colors)],
            alpha=0.22,
            edgecolor=colors[i % len(colors)],
        )
    )
     # base gap between boxes
    base_gap = 0.07
    # extra gap depending on how long the list is
    extra_gap = 0.012 * (len(cluster_allergens_sorted) / 5)
    y -= (base_gap + extra_gap)
plt.tight_layout()
plt.show()

df_embeddings["labels"] = df_embeddings["cluster"].map(cluster_to_category)
df_embeddings.to_csv("./data/word_embeddings_labeled.csv", index=False)