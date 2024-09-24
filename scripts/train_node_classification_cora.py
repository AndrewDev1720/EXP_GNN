import numpy as np
import torch
import os
import time
from pathlib import Path
import sys
import dgl
import dgl.function as fn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gcn import GCNNodeBAShapes  # Can reuse the same GCN model or define a new one
from utils.argument import arg_parse_train_node_ba_shapes
from utils.graph_init import graph_init_real
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader
from dgl.data import CoraGraphDataset
from models.gcn_custom import GCNWithEdgeWeight



def train_node_classification(args):
    # Set device to GPU or CPU
    device = torch.device(f'cuda:{args.cuda}' if args.gpu else 'cpu')

    # Load the Cora dataset and move it to the correct device
    dataset = CoraGraphDataset()
    G_dataset = dataset[0].to(device)

    # Prepare node features and labels
    features = G_dataset.ndata['feat'].to(device)
    labels = G_dataset.ndata['label'].to(device)
    train_mask = G_dataset.ndata['train_mask'].to(device)
    test_mask = G_dataset.ndata['test_mask'].to(device)

    # Initialize edge weights as ones
    eweight = torch.ones(G_dataset.num_edges()).to(device)
    G_dataset.edata['weight'] = eweight  # Store edge weights in the graph
    # Create the GCN model
    model = GCNWithEdgeWeight(in_feats=features.shape[1], hidden_size=16, num_classes=len(torch.unique(labels))).to(device)

    # Create the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define the output path to save the model and training indices
    out_path = os.path.join(args.save_dir, args.dataset + "_logs")
    Path(out_path).mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        logits = model(G_dataset, features, eweight)  # Forward pass
        loss = loss_fn(logits[train_mask], labels[train_mask])  # Calculate loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Save model and training/test indices
    torch.save(model.state_dict(), os.path.join(out_path, 'model.model'))  # Save the model state dict
    np.save(os.path.join(out_path, 'train_mask.npy'), train_mask.cpu().numpy())  # Save the train mask
    np.save(os.path.join(out_path, 'test_mask.npy'), test_mask.cpu().numpy())  # Save the test mask
    print("Model and masks saved successfully to:", out_path)

    return model


def evaluate_model(model, graph, features, labels, mask, eweight):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features, eweight)
        pred = logits[mask].argmax(dim=1)
        correct = (pred == labels[mask]).sum().item()
        accuracy = correct / len(labels[mask])
    return accuracy


if __name__ == "__main__":
    import argparse

    # Argument parser function for the training script
    def arg_parse_train_node_cora():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='Cora')
        parser.add_argument('--gpu', action='store_true', help="Use GPU if available")
        parser.add_argument('--cuda', type=str, default='0', help="CUDA device ID")
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=5e-4)
        parser.add_argument('--num_epochs', type=int, default=3000)
        parser.add_argument('--save_dir', type=str, default="log", help="Directory to save the model and logs")
        return parser.parse_args()

    # Parse arguments
    args = arg_parse_train_node_cora()

    # Set CUDA device if using GPU
    if args.gpu:
        print(f"Using CUDA device {args.cuda}")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # Train the model
    trained_model = train_node_classification(args)

    # Evaluate the model
    dataset = CoraGraphDataset()
    G_dataset = dataset[0]
    device = torch.device(f'cuda:{args.cuda}' if args.gpu else 'cpu')
    G_dataset = G_dataset.to(device)
    features = G_dataset.ndata['feat'].to(device)
    labels = G_dataset.ndata['label'].to(device)
    test_mask = G_dataset.ndata['test_mask'].to(device)
    eweight = torch.ones(G_dataset.num_edges()).to(device)

    test_acc = evaluate_model(trained_model, G_dataset, features, labels, test_mask, eweight)
    print(f'Test Accuracy: {test_acc * 100:.2f}%')


