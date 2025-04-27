import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNConvNet(nn.Module):
    """Graph Convolutional Network to extract the feature of the network."""
    
    def __init__(self, input_dim, output_dim, embedding_dim, num_layers, batch_norm=False, dropout=0.0):
        super(GCNConvNet, self).__init__()
        self.num_layers = num_layers
        
        # Add the layers to the model
        for layer_id in range(self.num_layers):
            # Add GCNConv layer
            if self.num_layers == 1:
                self.add_module(f"conv_{layer_id}", GCNConv(input_dim, output_dim))
            elif layer_id == 0:
                self.add_module(f"conv_{layer_id}", GCNConv(input_dim, embedding_dim))
            elif layer_id == self.num_layers - 1:
                self.add_module(f"conv_{layer_id}", GCNConv(embedding_dim, output_dim))
            else:
                self.add_module(f"conv_{layer_id}", GCNConv(embedding_dim, embedding_dim))

            # Add BatchNorm layer (if required)
            if batch_norm:
                normalization_dim = output_dim if layer_id == self.num_layers - 1 else embedding_dim
                self.add_module(f"bn_{layer_id}", nn.BatchNorm1d(normalization_dim))
            else:
                self.add_module(f"bn_{layer_id}", nn.Identity())

            # Add Dropout layer (if required)
            if dropout > 0.0:
                self.add_module(f"dropout_{layer_id}", nn.Dropout(dropout))
            else:
                self.add_module(f"dropout_{layer_id}", nn.Identity())
            

    def forward(self, input):
        x, edge_index = input['x'], input['edge_index']

        for layer_id in range(self.num_layers):
            conv = getattr(self, f"conv_{layer_id}")
            bn = getattr(self, f"bn_{layer_id}")
            dropout = getattr(self, f"dropout_{layer_id}")

            x = conv(x, edge_index)
            
            if layer_id == self.num_layers - 1:
                x = dropout(bn(x))
            else:
                x = F.leaky_relu(dropout(bn(x)))
        
        return x
            