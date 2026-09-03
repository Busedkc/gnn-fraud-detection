import torch
import pandas as pd

data = torch.load('data/graph_data.pt', weights_only=False)
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
time_steps = features[1].values

# Her time step için illicit sayısını hesapla
results = []
for t in range(1, 50):
    mask = (time_steps == t)
    node_indices = mask.nonzero()[0] if hasattr(mask, 'nonzero') else [i for i, v in enumerate(mask) if v]
    labels_in_t = data.y[node_indices]
    illicit_count = (labels_in_t == 1).sum().item()
    total_count = len(node_indices)
    results.append((t, illicit_count, total_count))

# İllicit sayısına göre sırala, en yüksek 10'u göster
results.sort(key=lambda x: x[1], reverse=True)
print("Time Step | Illicit Sayısı | Toplam Düğüm")
for t, illicit, total in results[:10]:
    print(f"{t:9d} | {illicit:15d} | {total:12d}")