import numpy as np
import torch
from fognetx.config import Config
from fognetx.environment.environment import Environment
from fognetx.network import ActorNetwork, CriticNetwork 
from torch.optim import Adam
from torch.distributions import Categorical
from fognetx.ppo.buffer import PPOBuffer


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(self, config: Config, env: Environment):
        self.config = config

        # Initialize the actor and critic networks 
        p_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 5 # (+5 for v_node_size, v_num_placed_nodes, p_node_status, p_net_node_degrees, average_distance)
        v_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 3 # (+3 for v_node_size, v_num_placed_nodes, v_node_status)

        self.actor = ActorNetwork(p_net_features_dim, v_net_features_dim, config.embedding_dim, config.gcn_num_layers, config.dropout_prob, config.batch_norm).to(config.device)
        self.critic = CriticNetwork(p_net_features_dim, v_net_features_dim, config.embedding_dim, config.gcn_num_layers, config.dropout_prob, config.batch_norm).to(config.device)

        # Initialize the optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Initialize the buffer
        self.buffer = PPOBuffer()  


    def act(self, state, mask=None, sample=True):
        """Return action and log probabilities
        
        Args:
            state: The current state of the environment.
            mask: The mask to apply to the action logits. If None, no mask is applied. (shape: [batch_size, num_v_nodes, num_p_nodes])
            sample: If True, sample an action from the distribution. If False, take the most probable action.
        
        Returns:
            high_action: The selected high-level action (virtual node).
            low_action: The selected low-level action (physical node).
            log_prob: The log probability of the selected actions.
        """
        # Get high-level action logits
        high_action_logits = self.actor.forward_high(state)

        # Apply mask to logits if provided
        if mask is not None:
            high_action_mask = (mask.sum(2) != 0).float()
            high_action_logits = high_action_logits.masked_fill(high_action_mask == 0, -1e9)

        # Transform logits to probabilities
        high_action_dist = Categorical(logits=high_action_logits) 

        # Sample action from the distribution or take the most probable action
        if sample:
            high_action = high_action_dist.sample()
        else:
            high_action = high_action_dist.probs.argmax(dim=-1)

        # Compute log prob of the action
        high_action_log_prob = high_action_dist.log_prob(high_action)

        # Get low-level action logits
        low_action_logits = self.actor.forward_low(state, high_action)
        
        # Apply mask to logits if provided
        if mask is not None:
            low_level_mask = mask[torch.arange(mask.shape[0]), :, high_action]  # extracts the feasibility mask only for the chosen virtual node in each batch
            low_action_logits = low_action_logits.masked_fill(low_level_mask == 0, -1e9)

        # Transform logits to probabilities
        low_action_dist = Categorical(logits=low_action_logits)

        # Sample action from the distribution or take the most probable action
        if sample:
            low_action = low_action_dist.sample()
        else:
            low_action = low_action_dist.probs.argmax(dim=-1)

        # Compute log prob of the action
        low_action_log_prob = low_action_dist.log_prob(low_action)

        # Combine log probs
        log_prob = high_action_log_prob + low_action_log_prob

        # Return actions and log probs
        return high_action, low_action, log_prob


    def evaluate(self, state):
        """Return value of the state"""
        # Get value from the critic network
        value = self.critic(state)
        return value


    def update(self):
        # PPO update logic
        pass


    def save(self, path): 
        pass


    def load(self, path): 
        pass