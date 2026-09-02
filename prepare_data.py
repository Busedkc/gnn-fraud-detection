import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np

# 1. Veriyi yükle
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
classes = pd.read_csv('data/elliptic_txs_classes.csv')
edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

# İlk sütuna isim verelim (txId), okunabilirlik için
features = features.rename(columns={0: 'txId', 1: 'time_step'})

# 2. txId'leri 0'dan başlayan sıralı index'lere çevir
# PyG düğümleri 0, 1, 2, ... şeklinde ardışık index bekliyor, txId'ler rastgele büyük sayılar
all_tx_ids = features['txId'].values
txid_to_idx = {tx_id: idx for idx, tx_id in enumerate(all_tx_ids)}

# 3. Node özelliklerini (feature matrix) hazırla
# txId ve time_step sütunlarını çıkarıyoruz, geriye kalan 165 sütun gerçek özellikler
node_features = features.drop(columns=['txId', 'time_step']).values
x = torch.tensor(node_features, dtype=torch.float)

print("Node feature matrix (x) shape:", x.shape)

# 4. Class etiketlerini hazırla
# classes df'inde txId sırası features ile birebir aynı olmayabilir, o yüzden txId üzerinden eşleştiriyoruz
classes_map = dict(zip(classes['txId'], classes['class']))

# Her txId için etiketi, features sırasına göre diz
labels = []
labeled_mask = []  # True: etiketli (1 veya 2), False: unknown

for tx_id in all_tx_ids:
    cls = classes_map.get(tx_id, 'unknown')
    if cls == '1':          # illicit
        labels.append(1)
        labeled_mask.append(True)
    elif cls == '2':        # licit
        labels.append(0)
        labeled_mask.append(True)
    else:                   # unknown
        labels.append(-1)   # placeholder, eğitimde kullanılmayacak
        labeled_mask.append(False)

y = torch.tensor(labels, dtype=torch.long)
labeled_mask = torch.tensor(labeled_mask, dtype=torch.bool)

print("Label tensor (y) shape:", y.shape)
print("Etiketli node sayısı:", labeled_mask.sum().item())

# 5. Edge listesini txId yerine yeni index'lerle oluştur
# Bazı edge'lerde features'ta olmayan txId'ler olabilir, onları filtreliyoruz
valid_edges = edges[
    edges['txId1'].isin(txid_to_idx) & edges['txId2'].isin(txid_to_idx)
]

edge_index = torch.tensor([
    valid_edges['txId1'].map(txid_to_idx).values,
    valid_edges['txId2'].map(txid_to_idx).values
], dtype=torch.long)

print("Edge index shape:", edge_index.shape)

# 6. PyG Data objesini oluştur
data = Data(x=x, edge_index=edge_index, y=y)
data.labeled_mask = labeled_mask

print("\n--- PyG Data objesi ---")
print(data)

# 7. İleride tekrar kullanmak için kaydet
torch.save(data, 'data/graph_data.pt')
print("\nData objesi 'data/graph_data.pt' olarak kaydedildi.")