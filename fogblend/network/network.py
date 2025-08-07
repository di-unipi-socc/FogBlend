import torch
import torch.nn as nn
from fogblend.network.gnn import GCNConvNet
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch


class Encoder(nn.Module):
    """Encoder to extract the feature of the network."""

    def __init__(self, input_dim, embedding_dim, num_layers, batch_norm=False, dropout=0.0):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.gnn = GCNConvNet(input_dim=embedding_dim, output_dim=embedding_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout)
        self.mean_pooling = global_mean_pool


    def forward(self, data):
        """Forward pass to extract the node and graph-level embeddings.
        
        Args:
            data: The input data containing the node features and batch information.

        Returns:
            graph_embeddings: The graph-level embeddings after mean pooling. Shape: [batch_size, embedding_dim].
            node_embeddings_concatenated: The concatenated node embeddings with the graph-level representation. Shape: [batch_size, num_nodes, embedding_dim*3].
            mask: The mask indicating valid nodes in the batch. Shape: [batch_size, num_nodes].
        """
        # Unpack the data
        x, batch = data['x'], data['batch']

        # Apply linear transformation to the input features
        node_init_embeddings = self.linear(x)
        
        # Update the data object with the new node features
        data = data.clone()
        data['x'] = node_init_embeddings 
        
        # Apply GNN layers
        node_embeddings = self.gnn(data)

        # Apply mean pooling to get the graph-level representation (shape: [batch_size, embedding_dim])
        graph_embeddings = self.mean_pooling(node_embeddings, batch)

        # Convert node embeddings to dense format 
        node_embeddings_dense, _ = to_dense_batch(node_embeddings, batch)
        node_init_embeddings_dense, mask = to_dense_batch(node_init_embeddings, batch)

        # Repeat the graph-level representation for each node in the batch to allow concatenation
        graph_embeddings_repeated = graph_embeddings.unsqueeze(1).expand(-1, node_embeddings_dense.size(1), -1)

        # Concatenate the node embeddings with the graph-level representation  (shape: [batch_size, num_nodes, embedding_dim*3])
        node_embeddings_concatenated = torch.cat((node_embeddings_dense, node_init_embeddings_dense, graph_embeddings_repeated), dim=-1)

        return graph_embeddings, node_embeddings_concatenated, mask



class CriticNetwork(nn.Module):
    """Critic network to estimate the value of the state."""
    
    def __init__(self, p_net_feature_dim, v_net_feature_dim, embedding_dim=128, num_layers=2, dropout_prob=0.0, batch_norm=False, p_net_encoder=None, v_net_encoder=None):
        super(CriticNetwork, self).__init__()
        if p_net_encoder is not None and v_net_encoder is not None:
            self.p_net_encoder = p_net_encoder
            self.v_net_encoder = v_net_encoder
        else:
            self.p_net_encoder = Encoder(input_dim=p_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
            self.v_net_encoder = Encoder(input_dim=v_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
        
        # From [4*embedding_dim] to [embedding_dim]
        self.head = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim * 4),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

        # From [embedding_dim] to scalar 
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1)
        )

    
    def forward(self, data):
        """Forward pass to estimate the value of the state.
        
        Args:
            data: The input data containing the physical and virtual network information.
            
        Returns:
            value: The estimated value of the state. Shape: [batch_size].
        """
        # Unpack the data
        p_net_data, v_net_data = data['p_net'], data['v_net']

        # Extract features from the physical network and virtual network
        _, p_net_node_embeddings, p_mask = self.p_net_encoder(p_net_data)
        v_graph_embeddings, _, _ = self.v_net_encoder(v_net_data)

        # Concatenate the graph-level representation of the virtual network with the node embeddings of the physical network (shape: [batch_size, num_nodes, embedding_dim*4]) 
        node_embeddings_concat = torch.cat((p_net_node_embeddings, v_graph_embeddings.unsqueeze(1).expand(-1, p_net_node_embeddings.size(1), -1)), dim=-1)  

        # Apply transformation to each node independently (shape: [batch_size, num_nodes, embedding_dim])
        transformed = self.head(node_embeddings_concat)  

        # Multiply mask to zero out padded nodes
        transformed = transformed * p_mask.unsqueeze(-1)  # shape: [batch_size, num_nodes, embedding_dim]

        # Sum only over valid nodes
        sum_pooled = transformed.sum(dim=1)  # shape: [batch_size, embedding_dim]
        valid_node_counts = p_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # shape: [batch_size, 1]
        
        # Mean over valid nodes (shape: [batch_size, embedding_dim])
        pooled = sum_pooled / valid_node_counts
        
        # Final transformation to scalar value (shape: [batch_size, 1])
        value = self.value_head(pooled)

        return value.squeeze(-1)  # shape: [batch_size]
    


class ActorNetwork(nn.Module):
    """Actor network to select the action."""

    def __init__(self, p_net_feature_dim, v_net_feature_dim, embedding_dim=128, num_layers=2, dropout_prob=0.0, batch_norm=False, p_net_encoder=None, v_net_encoder=None):
        super(ActorNetwork, self).__init__()
        if p_net_encoder is not None and v_net_encoder is not None:
            self.p_net_encoder = p_net_encoder
            self.v_net_encoder = v_net_encoder
        else:
            self.p_net_encoder = Encoder(input_dim=p_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
            self.v_net_encoder = Encoder(input_dim=v_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
        
        self.high_policy = nn.Sequential(
            nn.Linear(4 * embedding_dim, embedding_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1),
        )

        self.low_policy = nn.Sequential(
            nn.Linear(6 * embedding_dim, embedding_dim * 3),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward_high(self, data):
        """Forward pass for high-level action (virtual node) selection.
        
        Args:
            data: The input data containing the physical and virtual network information.
            
        Returns:
            high_action_logits: The logits for the high-level action (virtual node). Shape: [batch_size, num_v_nodes].
        """
        # Unpack the data
        p_net_data, v_net_data = data['p_net'], data['v_net']

        # Extract features from the physical network and virtual network
        p_graph_embeddings, _, _ = self.p_net_encoder(p_net_data)
        _, v_node_embeddings, v_mask = self.v_net_encoder(v_net_data)

        # Repeat the physical network graph-level representation for each virtual node (shape: [batch_size, num_v_nodes, embedding_dim])
        p_graph_embeddings_repeated = p_graph_embeddings.unsqueeze(1).expand(-1, v_node_embeddings.size(1), -1)

        # Concatenate virtual node embeddings with repeated physical graph embeddings (shape: [batch_size, num_v_nodes, embedding_dim*4])
        v_node_embeddings_concat = torch.cat((v_node_embeddings, p_graph_embeddings_repeated), dim=-1)

        # Apply high-level policy network (shape: [batch_size, num_v_nodes, 1])
        high_action_logits = self.high_policy(v_node_embeddings_concat).squeeze(-1)
        
        # Apply mask to ignore padded virtual nodes
        high_action_logits = high_action_logits.masked_fill(v_mask == 0, -1e9)

        return high_action_logits

    
    def forward_low(self, data, high_level_action):
        """Forward pass for low-level action (physical node) selection.
        
        Args:
            data: The input data containing physical and virtual network information.
            high_level_action: The selected high-level action (virtual node). Shape: [batch_size].
        
        Returns:
            low_level_logits: The logits for the low-level action (physical node). Shape: [batch_size, num_p_nodes].
        """
        # Unpack the data
        p_net_data, v_net_data = data['p_net'], data['v_net']

        # Extract features from the physical network and virtual network
        _, p_node_embeddings, p_mask = self.p_net_encoder(p_net_data)
        _, v_node_embeddings, _ = self.v_net_encoder(v_net_data)

        # Get embedding for the selected virtual node (shape: [batch_size, embedding_dim * 3])
        curr_v_node_id = high_level_action.unsqueeze(1).unsqueeze(1).long()
        curr_v_node_embedding = v_node_embeddings.gather(1, curr_v_node_id.expand(v_node_embeddings.size(0), -1, v_node_embeddings.size(-1))).squeeze(1)

        # Repeat virtual node embedding across all physical nodes (shape: [batch_size, num_p_nodes, embedding_dim * 3])
        curr_v_node_embedding_repeated = curr_v_node_embedding.unsqueeze(1).expand(-1, p_node_embeddings.size(1), -1)

        # Concatenate with physical node embeddings (shape: [batch_size, num_p_nodes, embedding_dim * 6])
        state_embeddings = torch.cat([p_node_embeddings, curr_v_node_embedding_repeated], dim=-1)
        
        # Apply low-level policy network (shape: [batch_size, num_p_nodes])
        low_level_logits = self.low_policy(state_embeddings).squeeze(-1)

        # Apply mask to ignore padded physical nodes
        low_level_logits = low_level_logits.masked_fill(p_mask == 0, -1e9)

        return low_level_logits  # shape: [batch_size, num_p_nodes]


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network to select actions and estimate the value of the state."""

    def __init__(self, p_net_feature_dim, v_net_feature_dim, embedding_dim=128, num_layers=2, dropout_prob=0.0, batch_norm=False, shared_encoder=False):
        super(ActorCriticNetwork, self).__init__()
        if shared_encoder:
            # Use the same encoder for both actor and critic
            p_net_encoder = Encoder(input_dim=p_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
            v_net_encoder = Encoder(input_dim=v_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
        else:
            # Use separate encoders for actor and critic
            p_net_encoder = None
            v_net_encoder = None
        self.actor = ActorNetwork(p_net_feature_dim, v_net_feature_dim, embedding_dim, num_layers, dropout_prob, batch_norm, p_net_encoder, v_net_encoder)
        self.critic = CriticNetwork(p_net_feature_dim, v_net_feature_dim, embedding_dim, num_layers, dropout_prob, batch_norm, p_net_encoder, v_net_encoder)


    def forward_high(self, data):
        """Forward pass for high-level action (virtual node) selection.
        
        Args:
            data: The input data containing the physical and virtual network information.
            
        Returns:
            high_action_logits: The logits for the high-level action (virtual node). Shape: [batch_size, num_v_nodes].
        """
        return self.actor.forward_high(data)
    

    def forward_low(self, data, high_level_action):
        """Forward pass for low-level action (physical node) selection.
        
        Args:
            data: The input data containing physical and virtual network information.
            high_level_action: The selected high-level action (virtual node). Shape: [batch_size].
        
        Returns:
            low_level_logits: The logits for the low-level action (physical node). Shape: [batch_size, num_p_nodes].
        """
        return self.actor.forward_low(data, high_level_action)
    
    
    def forward_critic(self, data):
        """Forward pass to estimate the value of the state.
        
        Args:
            data: The input data containing the physical and virtual network information.
            
        Returns:
            value: The estimated value of the state. Shape: [batch_size].
        """
        return self.critic(data)