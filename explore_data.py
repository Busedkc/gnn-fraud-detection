import pandas as pd

# Dosyaları yükle
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
classes = pd.read_csv('data/elliptic_txs_classes.csv')
edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

# Genel boyutlara bak
print("Features shape:", features.shape)
print("Classes shape:", classes.shape)
print("Edges shape:", edges.shape)

print("\n **Features ilk 5 satır** ")
print(features.head())

print("\n **Classes ilk 5 satır** ")
print(classes.head())

print("\n **Classes dağılımı** ")
print(classes['class'].value_counts())

print("\n **Edges ilk 5 satır** ")
print(edges.head())