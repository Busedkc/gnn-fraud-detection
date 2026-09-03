import torch
import torch.nn.functional as F
import random
import numpy as np
import datetime
from model import GCN, GraphSAGE, GAT
from sklearn.metrics import precision_score, recall_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Veriyi yükle (bir kere, tüm denemeler için ortak)
data = torch.load('data/graph_data.pt', weights_only=False)

labeled_indices = data.labeled_mask.nonzero(as_tuple=True)[0]
num_labeled = labeled_indices.size(0)

perm = torch.randperm(num_labeled)
train_size = int(0.7 * num_labeled)
val_size = int(0.15 * num_labeled)

train_idx = labeled_indices[perm[:train_size]]
val_idx = labeled_indices[perm[train_size:train_size + val_size]]
test_idx = labeled_indices[perm[train_size + val_size:]]

class_counts = torch.bincount(data.y[train_idx])
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * 2


def evaluate(model, idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        y_true = data.y[idx].numpy()
        y_pred = pred[idx].numpy()
        acc = (pred[idx] == data.y[idx]).float().mean().item()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return acc, precision, recall, f1


def train_one_config(model, max_epochs=200, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val_f1 = 0
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_idx], data.y[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()

        _, _, _, val_f1 = evaluate(model, val_idx)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            best_state = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return best_val_f1


# 2. Denenecek kombinasyonlar
configs = []

for hidden in [16, 32, 64]:
    configs.append(('GCN', {'hidden_channels': hidden}))

for hidden in [16, 32, 64]:
    configs.append(('GraphSAGE', {'hidden_channels': hidden}))

for hidden, heads in [(8, 4), (16, 2), (16, 8)]:
    configs.append(('GAT', {'hidden_channels': hidden, 'heads': heads}))

# 3. Her kombinasyonu dene
results = []

for model_name, params in configs:
    if model_name == 'GCN':
        model = GCN(num_features=data.x.size(1), num_classes=2, **params)
    elif model_name == 'GraphSAGE':
        model = GraphSAGE(num_features=data.x.size(1), num_classes=2, **params)
    elif model_name == 'GAT':
        model = GAT(num_features=data.x.size(1), num_classes=2, **params)

    val_f1 = train_one_config(model)
    test_acc, test_precision, test_recall, test_f1 = evaluate(model, test_idx)

    print(f"{model_name} {params} -> Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
    results.append((model_name, params, val_f1, test_acc, test_precision, test_recall, test_f1))

# 4. Her model tipi için en iyi kombinasyonu bul
print("\n=== EN İYİ KONFİGÜRASYONLAR ===")
with open('results.md', 'a') as f:
    f.write(f"\n## Hyperparameter Tuning - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    for model_name in ['GCN', 'GraphSAGE', 'GAT']:
        model_results = [r for r in results if r[0] == model_name]
        best = max(model_results, key=lambda r: r[2])  # val_f1'e göre en iyisi
        name, params, val_f1, acc, prec, rec, f1 = best
        line = (f"- **{name}** best config: {params} | Val F1: {val_f1:.4f} | "
                f"Test Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
        print(line)
        f.write(line + "\n")