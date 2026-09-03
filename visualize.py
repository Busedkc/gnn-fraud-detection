import torch
import networkx as nx
import pandas as pd
import random
import plotly.graph_objects as go

random.seed(42)

# 1. Veriyi yükle
print("Veri yükleniyor...")
data = torch.load('data/graph_data.pt', weights_only=False)

print("CSV okunuyor...")
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
time_steps = features[1].values

print("Graf oluşturuluyor...")
edge_list = data.edge_index.t().tolist()
G_full = nx.Graph()
G_full.add_edges_from(edge_list)

# 2. Belirli bir time step seç
chosen_time_step = 32  # 1-49 arası

print("Time step filtreleniyor...")
selected_nodes = [i for i, t in enumerate(time_steps) if t == chosen_time_step]
print(f"Time step {chosen_time_step}: {len(selected_nodes)} düğüm")

G_sub = G_full.subgraph(selected_nodes)
print(f"Alt-graf (sadece bu time step'teki bağlantılar): "
      f"{G_sub.number_of_nodes()} düğüm, {G_sub.number_of_edges()} kenar")

# Bağlantısız (izole) düğümleri elemek için en büyük bağlantılı bileşeni bul
components = list(nx.connected_components(G_sub))
largest_component = max(components, key=len)
print(f"En büyük bağlantılı bileşen: {len(largest_component)} düğüm")

# Eğer hala çok büyükse, bu bileşen içinden örneklem al (rastgele değil,
# bir düğümden başlayıp komşuluk gezerek - bağlantı korunsun)
MAX_NODES = 250
if len(largest_component) > MAX_NODES:
    start = random.choice(list(largest_component))
    bfs_nodes = set([start])
    frontier = {start}
    while len(bfs_nodes) < MAX_NODES and frontier:
        new_frontier = set()
        for node in frontier:
            neighbors = set(G_sub.neighbors(node)) & largest_component
            new_frontier.update(neighbors - bfs_nodes)
        bfs_nodes.update(new_frontier)
        frontier = new_frontier
        if not frontier:
            break
    final_nodes = list(bfs_nodes)[:MAX_NODES]
else:
    final_nodes = list(largest_component)

G_sub = G_sub.subgraph(final_nodes)
print(f"Görselleştirilecek alt-graf: {G_sub.number_of_nodes()} düğüm, "
      f"{G_sub.number_of_edges()} kenar")

# 3. Yerleşim (layout) hesapla
pos = nx.spring_layout(G_sub, seed=42, k=1.5, iterations=50)

# 4. Kenarları çiz (çizgiler)
edge_x, edge_y = [], []
for edge in G_sub.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# 5. Düğümleri çiz (hover bilgisiyle)
node_x, node_y, node_color, node_text = [], [], [], []
label_map = {1: 'Illicit', 0: 'Licit', -1: 'Unknown'}
color_map = {1: '#e74c3c', 0: '#2ecc71', -1: '#bdc3c7'}

for node in G_sub.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)
    label = data.y[node].item()
    degree = G_sub.degree(node)
    node_color.append(color_map[label])
    node_text.append(
        f"Node ID: {node}<br>"
        f"Etiket: {label_map[label]}<br>"
        f"Bağlantı sayısı (derece): {degree}"
    )

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    text=node_text,
    marker=dict(
        color=node_color,
        size=14,
        line=dict(width=1, color='black')
    )
)

# 6. Figürü oluştur
fig = go.Figure(data=[edge_trace, node_trace])
fig.update_layout(
    title=f"Elliptic Bitcoin İşlem Ağı - Time Step {chosen_time_step} ({G_sub.number_of_nodes()} düğüm)<br>"
          f"<sub>Kırmızı: Illicit | Yeşil: Licit | Gri: Unknown — fareyi düğümlerin üzerine getir</sub>",
    showlegend=False,
    hovermode='closest',
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='white',
    width=1000,
    height=800
)

fig.write_html('subgraph_visualization.html')
fig.show()

print("İnteraktif görsel 'subgraph_visualization.html' olarak kaydedildi.")