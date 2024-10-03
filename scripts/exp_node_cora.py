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

import networkx as nx
import matplotlib.pyplot as plt
import dgl

def generate_and_visualize_subgraph(masked_adj, graph, node_id, threshold=0.5):
    """
    Generate and visualize the subgraph for a given node based on the masked adjacency matrix.
    Highlight the node currently being explained in red, with its ID also shown in red.
    """
    # Ensure masked_adj is 2D (source-destination pairs)
    num_edges = graph.num_edges()
    masked_adj = masked_adj.view(-1)  # Flatten the adjacency matrix if it's 1D
    edge_indices = graph.edges(order='eid')  # Get source and destination indices of edges

    # Apply threshold to get the important edges
    important_edges = (masked_adj > threshold).nonzero(as_tuple=False).view(-1)
    
    if important_edges.numel() == 0:
        print(f"No important edges found for node {node_id} with threshold {threshold}.")
        return
    
    # Get the source and destination nodes for the important edges
    src_nodes = edge_indices[0][important_edges].cpu().numpy()  # Source nodes
    dst_nodes = edge_indices[1][important_edges].cpu().numpy()  # Destination nodes
    
    # Create a list of edges
    edge_list = list(zip(src_nodes, dst_nodes))

    # Use DGL's edge subgraph function to generate the subgraph based on the selected edges
    subgraph = dgl.edge_subgraph(graph, important_edges, relabel_nodes=True).to('cpu')  # Move subgraph to CPU
    
    # Convert DGL graph to NetworkX for visualization
    nx_subgraph = dgl.to_networkx(subgraph)
    
    # Set node colors: Red for the node being explained, lightblue for others
    node_colors = []
    for node in nx_subgraph.nodes():
        if node == node_id:
            node_colors.append('red')  # Highlight the explained node in red
        else:
            node_colors.append('lightblue')  # Default color for other nodes
    
    # Create node labels and highlight the explained node label in red
    labels = {n: str(n) for n in nx_subgraph.nodes()}
    font_colors = ['Yellow' if n == node_id else 'black' for n in nx_subgraph.nodes()]  # Red font for the explained node
    
    # Plot the subgraph with the explained node highlighted in red
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(nx_subgraph)  # Layout for the graph
    nx.draw(nx_subgraph, pos, with_labels=False, node_color=node_colors, edge_color='gray', node_size=500, font_size=10)
    
    # Draw labels with font colors, making the explained node's label red
    for node, label in labels.items():
        nx.draw_networkx_labels(nx_subgraph, pos, labels={node: label}, font_color=font_colors[list(nx_subgraph.nodes()).index(node)], font_size=12)
    
    plt.title(f"Explanation Subgraph for Node {node_id}")
    plt.show()


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

    model_path = os.path.join(model_path, 'Cora_gcn_model.pth')

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
    exp_dict, num_dict = explainer.explain_nodes_gnn_stats()

    for node_id, masked_adj in exp_dict.items():
        print(f"Node {node_id}: Explanation Masked Adjacency")
        print(masked_adj)
        print(f"Number of explanations (edges): {num_dict[node_id]}")
    
    for node_id, masked_adj in exp_dict.items():
        print(f"Node {node_id}: Explanation Masked Adjacency")
        print(len(masked_adj))
        generate_and_visualize_subgraph(masked_adj, G_dataset, node_id, threshold=0.7)