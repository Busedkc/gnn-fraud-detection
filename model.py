import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # 1. katman: komşulardan bilgi topla, non-linearity uygula
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # 2. katman: sınıflandırma için son çıktı
        x = self.conv2(x, edge_index)
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        return x
    
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=4):
        super().__init__()
        # İlk katman: birden fazla attention head kullanıyor, çıktılar birleştiriliyor (concat)
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads)
        # İkinci katman: tek head, doğrudan sınıf sayısına indiriyor
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # GAT literatüründe genelde ELU tercih edilir, ReLU yerine
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x