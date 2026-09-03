
## 2026-09-01 20:00
- Model: GCN (2 layer, hidden=32) + class weight + early stopping
- Best epoch: 195, Best val F1: 0.6645
- Test Acc: 0.9062 | Precision: 0.5152 | Recall: 0.8813 | F1: 0.6503
## 2026-09-01 20:30
- Model: GraphSAGE (2 layer, hidden=32) + class weight + early stopping
- Best epoch: 155, Best val F1: 0.8145
- Test Acc: 0.9561 | Precision: 0.7182 | Recall: 0.9146 | F1: 0.8046

## 2026-09-01 20:58
- Model: GAT (2 layer, hidden=8, heads=4) + class weight + early stopping
- Best epoch: 198, Best val F1: 0.7080
- Test Acc: 0.9191 | Precision: 0.5553 | Recall: 0.9161 | F1: 0.6914

## Hyperparameter Tuning - 2026-09-02 11:55
- **GCN** best config: {'hidden_channels': 64} | Val F1: 0.6815 | Test Acc: 0.9133 | Precision: 0.5376 | Recall: 0.8799 | F1: 0.6674
- **GraphSAGE** best config: {'hidden_channels': 64} | Val F1: 0.8463 | Test Acc: 0.9661 | Precision: 0.7809 | Recall: 0.9132 | F1: 0.8419
- **GAT** best config: {'hidden_channels': 16, 'heads': 8} | Val F1: 0.7602 | Test Acc: 0.9455 | Precision: 0.6591 | Recall: 0.9291 | F1: 0.7712
