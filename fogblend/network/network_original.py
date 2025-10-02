import torch
import torch.nn as nn
from fogblend.network.gnn import GCNConvNet
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch


class EncoderOriginal(nn.Module):
    """Encoder to extract the feature of the network."""

    def __init__(self, input_dim, embedding_dim, num_layers, batch_norm=False, dropout=0.0):
        super(EncoderOriginal, self).__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.gnn = GCNConvNet(input_dim=embedding_dim, output_dim=embedding_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout)
        self.mean_pooling = global_mean_pool


    def forward(self, data):
        """Forward pass to extract the node and graph-level embeddings.
        
        Args:
            data: The input data containing the node features and batch information.

        Returns:
            graph_embeddings: The graph-level embeddings after mean pooling. Shape: [batch_size, embedding_dim].
            node_embeddings_summed: The node embeddings summed with the graph-level representation. Shape: [batch_size, num_nodes, embedding_dim*3].
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

        # Sum the node embeddings with the graph-level representation  (shape: [batch_size, num_nodes, embedding_dim])
        node_embeddings_summed = node_embeddings_dense + node_init_embeddings_dense + graph_embeddings.unsqueeze(1)

        return graph_embeddings, node_embeddings_summed, mask



class CriticNetworkOriginal(nn.Module):
    """Critic network to estimate the value of the state."""
    
    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=128, num_layers=2, dropout_prob=0.0, batch_norm=False, p_net_encoder=None, v_net_encoder=None):
        super(CriticNetworkOriginal, self).__init__()
        if p_net_encoder is not None and v_net_encoder is not None:
            self.p_net_encoder = p_net_encoder
            self.v_net_encoder = v_net_encoder
        else:
            self.p_net_encoder = EncoderOriginal(input_dim=p_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
            self.v_net_encoder = EncoderOriginal(input_dim=v_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
        
        # From [emedding_dim] to [1] (for each node)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2), 
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1)
        )

        # From [p_net_num_nodes] to scalar 
        self.value_head = nn.Sequential(
            nn.Linear(p_net_num_nodes, p_net_num_nodes * 2),
            nn.ReLU(),
            nn.Linear(p_net_num_nodes * 2, p_net_num_nodes),
            nn.ReLU(),
            nn.Linear(p_net_num_nodes, 1)
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
        _, p_net_node_embeddings, _ = self.p_net_encoder(p_net_data)
        v_graph_embeddings, _, _ = self.v_net_encoder(v_net_data)

        # Sum the graph-level representation of the virtual network with the node embeddings of the physical network (shape: [batch_size, num_nodes, embedding_dim]) 
        node_embeddings_summed = p_net_node_embeddings + v_graph_embeddings.unsqueeze(1)

        # Apply transformation to each node independently (shape: [batch_size, num_nodes, 1])
        transformed = self.head(node_embeddings_summed).squeeze(-1)
        
        # Final transformation to scalar value (shape: [batch_size, 1])
        value = self.value_head(transformed)

        return value.squeeze(-1)  # shape: [batch_size]
    


class ActorNetworkOriginal(nn.Module):
    """Actor network to select the action."""

    def __init__(self, p_net_feature_dim, v_net_feature_dim, embedding_dim=128, num_layers=2, dropout_prob=0.0, batch_norm=False, p_net_encoder=None, v_net_encoder=None):
        super(ActorNetworkOriginal, self).__init__()
        if p_net_encoder is not None and v_net_encoder is not None:
            self.p_net_encoder = p_net_encoder
            self.v_net_encoder = v_net_encoder
        else:
            self.p_net_encoder = EncoderOriginal(input_dim=p_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
            self.v_net_encoder = EncoderOriginal(input_dim=v_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
        
        self.high_policy = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

        self.low_policy = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
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

        # Sum virtual node embeddings with repeated physical graph embeddings (shape: [batch_size, num_v_nodes, embedding_dim])
        v_node_embeddings_summed = v_node_embeddings + p_graph_embeddings.unsqueeze(1)

        # Apply high-level policy network (shape: [batch_size, num_v_nodes])
        high_action_logits = self.high_policy(v_node_embeddings_summed).squeeze(-1)
        
        # Apply mask to ignore padded virtual nodes
        high_action_logits = high_action_logits.masked_fill(v_mask == 0, -1e9)

        return high_action_logits # shape: [batch_size, num_v_nodes]

    
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

        # Get embedding for the selected virtual node (shape: [batch_size, embedding_dim])
        curr_v_node_id = high_level_action.unsqueeze(1).unsqueeze(1).long()
        curr_v_node_embedding = v_node_embeddings.gather(1, curr_v_node_id.expand(v_node_embeddings.size(0), -1, v_node_embeddings.size(-1))).squeeze(1)

        # Sum with physical node embeddings (shape: [batch_size, num_p_nodes, embedding_dim])
        state_embeddings = p_node_embeddings + curr_v_node_embedding.unsqueeze(1)
        
        # Apply low-level policy network (shape: [batch_size, num_p_nodes])
        low_level_logits = self.low_policy(state_embeddings).squeeze(-1)

        # Apply mask to ignore padded physical nodes
        low_level_logits = low_level_logits.masked_fill(p_mask == 0, -1e9)

        return low_level_logits  # shape: [batch_size, num_p_nodes]


class ActorCriticNetworkOriginal(nn.Module):
    """Actor-Critic network to select actions and estimate the value of the state."""

    def __init__(self, p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim=128, num_layers=2, dropout_prob=0.0, batch_norm=False, shared_encoder=False):
        super(ActorCriticNetworkOriginal, self).__init__()
        if shared_encoder:
            # Use the same encoder for both actor and critic
            p_net_encoder = EncoderOriginal(input_dim=p_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
            v_net_encoder = EncoderOriginal(input_dim=v_net_feature_dim, embedding_dim=embedding_dim, num_layers=num_layers, batch_norm=batch_norm, dropout=dropout_prob)
        else:
            # Use separate encoders for actor and critic
            p_net_encoder = None
            v_net_encoder = None
        self.actor = ActorNetworkOriginal(p_net_feature_dim, v_net_feature_dim, embedding_dim, num_layers, dropout_prob, batch_norm, p_net_encoder, v_net_encoder)
        self.critic = CriticNetworkOriginal(p_net_num_nodes, p_net_feature_dim, v_net_feature_dim, embedding_dim, num_layers, dropout_prob, batch_norm, p_net_encoder, v_net_encoder)


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