import torch
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
import copy
import numpy as np

# Normalize features (x), edge attributes (edge_attr), and targets (y)
all_x = torch.cat([data.x for data in dataset], dim=0)
all_edge_attr = torch.cat([data.edge_attr for data in dataset], dim=0)
all_targets = torch.cat([data.y for data in dataset], dim=0)

# Normalize node features (x)
min_x = all_x.min(dim=0)[0]
max_x = all_x.max(dim=0)[0]
all_x = (all_x - min_x) / (max_x - min_x + 1e-6)

# Normalize edge attributes (edge_attr) if they exist
if all_edge_attr.nelement() > 0:  # Check if all_edge_attr is non-empty
    min_edge_attr = all_edge_attr.min(dim=0)[0]
    max_edge_attr = all_edge_attr.max(dim=0)[0]
    all_edge_attr = (all_edge_attr - min_edge_attr) / (max_edge_attr - min_edge_attr + 1e-6)

# Normalize targets (y)
min_targets = all_targets.min()
max_targets = all_targets.max()
all_targets = (all_targets - min_targets) / (max_targets - min_targets + 1e-6)

# Apply normalization to each graph in the dataset
for data in dataset:
    if max_x.nelement() > 0:  # If max_x is non-empty
        data.x = (data.x - min_x) / (max_x - min_x + 1e-6)
    if all_edge_attr.nelement() > 0:  # If all_edge_attr is non-empty
        data.edge_attr = (data.edge_attr - min_edge_attr) / (max_edge_attr - min_edge_attr + 1e-6)
    data.y = (data.y - min_targets) / (max_targets - min_targets + 1e-6)


# Simple Attention Layer to apply attention mechanism to node features
class SimpleAttentionLayer(torch.nn.Module):
    def __init__(self, feature_size):
        super(SimpleAttentionLayer, self).__init__()
        # Linear layer to compute attention scores
        self.att_weight = Linear(feature_size, 1)

    def forward(self, x):
        # Compute attention scores using sigmoid to keep them between 0 and 1
        att_scores = torch.sigmoid(self.att_weight(x))
        # Apply attention scores to the node features
        attended_x = x * att_scores
        return attended_x


# Graph-based neural network for PCE regression
class PCERegressor(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features):
        super(PCERegressor, self).__init__()
        nn1 = Sequential(Linear(num_edge_features, 128), ReLU(), Linear(128, num_node_features * 128))
        self.conv1 = NNConv(num_node_features, 128, nn1, aggr='mean')
        self.bn1 = BatchNorm1d(128)
        self.att1 = SimpleAttentionLayer(128)

        nn2 = Sequential(Linear(num_edge_features, 128), ReLU(), Linear(128, 128 * 64))
        self.conv2 = NNConv(128, 64, nn2, aggr='mean')
        self.bn2 = BatchNorm1d(64)
        self.att2 = SimpleAttentionLayer(64)

        nn3 = Sequential(Linear(num_edge_features, 64), ReLU(), Linear(64, 64 * 32))
        self.conv3 = NNConv(64, 32, nn3, aggr='mean')
        self.bn3 = BatchNorm1d(32)
        self.att3 = SimpleAttentionLayer(32)

        self.fc1 = Linear(32, 16)
        self.fc2 = Linear(16, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # First convolution + attention + dropout
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = self.att1(x)
        x = F.dropout(x, training=self.training)

        # Second convolution + attention + dropout
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = self.att2(x)
        x = F.dropout(x, training=self.training)

        # Third convolution + attention + global pooling
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        x = self.att3(x)
        x = global_mean_pool(x, batch)  # Pooling to get graph-level representation

        # Final fully connected layers
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# Dataset splitting into train and test sets
train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

# Dataloaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PCERegressor(num_node_features=10, num_edge_features=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()