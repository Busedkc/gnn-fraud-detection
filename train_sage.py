import torch
import torch.nn.functional as F
from model import GraphSAGE
import random
import numpy as np
import datetime

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 1. Hazırladığımız veriyi yükle
data = torch.load('data/graph_data.pt', weights_only=False)
print("Data yüklendi:", data)

# 2. Etiketli düğümleri train/val/test olarak üçe ayır
# %70 train, %15 validation, %15 test
labeled_indices = data.labeled_mask.nonzero(as_tuple=True)[0]
num_labeled = labeled_indices.size(0)

perm = torch.randperm(num_labeled)  # rastgele karıştır

train_size = int(0.7 * num_labeled)
val_size = int(0.15 * num_labeled)
# test_size geri kalan her şey

train_idx = labeled_indices[perm[:train_size]]
val_idx = labeled_indices[perm[train_size:train_size + val_size]]
test_idx = labeled_indices[perm[train_size + val_size:]]

print(f"Train düğüm sayısı: {train_idx.size(0)}")
print(f"Val düğüm sayısı:   {val_idx.size(0)}")
print(f"Test düğüm sayısı:  {test_idx.size(0)}")

# 3. Modeli oluştur
model = GraphSAGE(num_features=data.x.size(1), hidden_channels=32, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 4. Eğitim döngüsü
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    # Class weight hesapla: az örnekli sınıfa (illicit) daha fazla ağırlık ver
    class_counts = torch.bincount(data.y[train_idx])
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2  # normalize et

    loss = F.cross_entropy(out[train_idx], data.y[train_idx], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()

# 5. Genel değerlendirme fonksiyonu (val ya da test için ortak kullanılabilir)
def evaluate(idx):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        correct = (pred[idx] == data.y[idx]).sum()
        acc = int(correct) / idx.size(0)

        from sklearn.metrics import precision_score, recall_score, f1_score
        y_true = data.y[idx].numpy()
        y_pred = pred[idx].numpy()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        return acc, precision, recall, f1

# 6. Eğitim döngüsü — early stopping ile
best_val_f1 = 0
best_epoch = 0
patience = 20          # bu kadar epoch boyunca iyileşme olmazsa dur
epochs_without_improvement = 0
best_model_state = None

max_epochs = 300        # üst sınır, buraya kadar gitmesin diye güvenlik

for epoch in range(1, max_epochs + 1):
    loss = train()
    acc, precision, recall, f1 = evaluate(val_idx)

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val Acc: {acc:.4f} | "
              f"Val Precision: {precision:.4f} | Val Recall: {recall:.4f} | Val F1: {f1:.4f}")

    if f1 > best_val_f1:
        best_val_f1 = f1
        best_epoch = epoch
        epochs_without_improvement = 0
        best_model_state = model.state_dict()  # en iyi modeli sakla
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"\nEarly stopping: {patience} epoch boyunca val F1 iyileşmedi. "
              f"Epoch {epoch}'te durduruldu.")
        break

print(f"\nEn iyi val F1: {best_val_f1:.4f} (epoch {best_epoch})")

# En iyi modeli geri yükle (döngü sonundaki model değil, en iyi olan)
model.load_state_dict(best_model_state)

# 7. Eğitim bitti, TEST seti ile SADECE BİR KERE nihai sonucu ölç
test_acc, test_precision, test_recall, test_f1 = evaluate(test_idx)
print("\n--- NİHAİ TEST SONUCU ---")
print(f"Test Acc:       {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall:    {test_recall:.4f}")
print(f"Test F1:        {test_f1:.4f}")


with open('results.md', 'a') as f:
    f.write(f"\n## {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    f.write(f"- Model: GraphSAGE (2 layer, hidden=32) + class weight + early stopping\n")
    f.write(f"- Best epoch: {best_epoch}, Best val F1: {best_val_f1:.4f}\n")
    f.write(f"- Test Acc: {test_acc:.4f} | Precision: {test_precision:.4f} | "
            f"Recall: {test_recall:.4f} | F1: {test_f1:.4f}\n")