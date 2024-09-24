import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import networkx as nx
import dgl
import matplotlib.pyplot as plt
from utils.argument import arg_parse_exp_node_ba_shapes  # You can reuse an argument parser or create a new one
from models.explainer_models import NodeExplainerEdgeMulti  # Use the explainer model
from models.gcn_custom import GCNWithEdgeWeight
from models.explainer_models import NodeExplainerEdgeMulti
from dgl.data import CoraGraphDataset
from utils.argument import arg_parse_exp_node_cora
from models.explainer_models import NodeExplainerEdgeMulti

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(1000)
    np.random.seed(0)
    np.set_printoptions(threshold=sys.maxsize)

    # Parse experiment arguments
    exp_args = arg_parse_exp_node_cora()
    print("Arguments:", exp_args)

    # Load the saved model and dataset path
    model_path = exp_args.model_path
    train_indices = np.load(os.path.join(model_path, 'train_mask.npy'), allow_pickle=True)
    test_indices = np.load(os.path.join(model_path, 'test_mask.npy'), allow_pickle=True)

    # Load the Cora dataset (single graph)
    dataset = CoraGraphDataset()
    G_dataset = dataset[0]  # Single DGLHeteroGraph

    # Move the graph to the device (GPU or CPU)
    device = torch.device('cuda:%s' % exp_args.cuda if exp_args.gpu else 'cpu')
    G_dataset = G_dataset.to(device)  # Move the graph to the correct device

    # Add edge weights (initialize to 1.0)
    G_dataset.edata['eweight'] = torch.ones(G_dataset.num_edges()).to(device)  # Assign edge weights to the device

    # Prepare node features and labels
    features = G_dataset.ndata['feat'].to(device)
    labels = G_dataset.ndata['label'].to(device)
    test_mask = G_dataset.ndata['test_mask'].to(device)

    # Load the pre-trained model using the correct architecture
    base_model = GCNWithEdgeWeight(features.shape[1], hidden_size=16, num_classes=len(torch.unique(labels))).to(device)
    base_model.load_state_dict(torch.load(os.path.join(model_path, 'model.model')))  # Load the saved model
    base_model.eval()

    # Freeze model parameters (no gradient required)
    for param in base_model.parameters():
        param.requires_grad = False

    # Create the explainer for the nodes
    explainer = NodeExplainerEdgeMulti(
        base_model=base_model,
        G_dataset=G_dataset,
        args=exp_args,
        test_indices=np.where(test_mask.cpu())[0]  # Test node indices for explanation
    )
    if 'eweight' in G_dataset.edata:
        G_dataset.edata['weight'] = G_dataset.edata['eweight']
    else:
        # Initialize weights as ones if not already present
        G_dataset.edata['weight'] = torch.ones(G_dataset.num_edges()).to(device)

    # Run the explanation process for nodes
    explainer.explain_nodes_gnn_stats()