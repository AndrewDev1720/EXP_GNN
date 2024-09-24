import torch
import torch.nn as nn
import dgl.function as fn
import dgl

# Define custom weighted GCN layer
class WeightedGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(WeightedGCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, h, eweight):
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['w'] = eweight

            # Apply weighted message passing
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))

            h_new = g.ndata['h_new']
            return self.linear(h_new)

# Define GCN model with edge weights
class GCNWithEdgeWeight(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCNWithEdgeWeight, self).__init__()
        self.layer1 = WeightedGCNLayer(in_feats, hidden_size)
        self.layer2 = WeightedGCNLayer(hidden_size, num_classes)

    def forward(self, g, features, eweight, target_node=None):
        h = self.layer1(g, features, eweight)
        h = torch.relu(h)
        h = self.layer2(g, h, eweight)
        return h
