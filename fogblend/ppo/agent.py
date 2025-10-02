# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fogblend.utils.types import Config
# REGULAR IMPORTS
import os
import torch
import fogblend.utils as utils
from torch.optim import Adam
from fogblend.ppo.buffer import PPOBuffer
from torch.distributions import Categorical
from fogblend.network import ActorCriticNetwork, ActorCriticNetworkOriginal

# Suppress torch.load warnings
import warnings
warnings.filterwarnings("ignore", message=".*?.*?torch.load.*?", category=FutureWarning)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(self, config: Config):
        # Store the configuration
        self.config = config

        # Initialize the actor and critic networks 
        p_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 5 # (+5 for v_node_size, v_num_placed_nodes, p_node_status, p_net_node_degrees, average_distance)
        v_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 3 # (+3 for v_node_size, v_num_placed_nodes, v_node_status)

        # Initialize the policy network based on the selected architecture
        if config.architecture == "new":
            self.policy = ActorCriticNetwork(p_net_features_dim, v_net_features_dim, config.embedding_dim, config.gcn_num_layers, config.dropout_prob, config.batch_norm, config.shared_encoder).to(config.device)
        elif config.architecture == "original":
            if config.num_nodes is None: 
                raise ValueError("Value of -num_nodes must be specified for 'original' architecture if selected topology is not GEANT.")
            self.policy = ActorCriticNetworkOriginal(config.num_nodes, p_net_features_dim, v_net_features_dim, config.embedding_dim, config.gcn_num_layers, config.dropout_prob, config.batch_norm, config.shared_encoder).to(config.device)
        else:
            raise ValueError(f"Unknown architecture: {config.architecture}")

        # Initialize the optimizer
        self.optimizer = Adam(self.policy.parameters(), lr=config.lr, weight_decay=config.weight_decay)

        # Initialize the buffer
        self.buffer = PPOBuffer(config.seed)


    def act(self, state, mask, sample=True):
        """Return action and log probabilities
        
        Args:
            state: A dictionary containing the state of the environment (p_net and v_net).
            mask: The mask to apply to the action logits (shape: [batch_size, num_v_nodes, num_p_nodes]).
            sample: If True, sample an action from the distribution. If False, take the most probable action.
        
        Returns:
            high_action: The selected high-level action (virtual node).
            low_action: The selected low-level action (physical node).
            log_prob: The log probability of the selected actions.
        """
        # Get high-level action logits
        high_action_logits = self.policy.forward_high(state)

        # Apply mask to logits (virtual nodes are always masked)
        high_action_mask = (mask.sum(2) != 0).float() # Sum over the physical nodes to get a mask for the virtual nodes
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
        low_action_logits = self.policy.forward_low(state, high_action)
        
        # Apply mask to logits (physical nodes are masked if required by the config)
        if self.config.mask_actions:
            # Get the mask for the low-level actions
            low_level_mask = mask[torch.arange(mask.size(0)), high_action]
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
        """Return value of the state
        
        Args:
            state: A dictionary containing the state of the environment (p_net and v_net).
            
        Returns:
            value: The value of the state.
        """
        # Get value from the critic network
        value = self.policy.critic(state)
        return value


    def evaluate_actions(self, obs, masks, high_actions, low_actions) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log probabilities of the actions and values of the states
        
        Args:
            obs: A dictionary containing the state of the environment (p_net and v_net).
            masks: The mask to apply to the action logits (shape: [batch_size, num_v_nodes, num_p_nodes]).
            high_actions: The selected high-level actions (virtual nodes).
            low_actions: The selected low-level actions (physical nodes).
            
        Returns:
            log_prob: The log probabilities of the selected actions.
            entropy: The entropy of the action distributions.
            value: The value of the state.
        """
        # Get high-level action logits
        high_action_logits = self.policy.forward_high(obs)

        # Apply mask to logits (virtual nodes are always masked)
        high_action_masks = (masks.sum(2) != 0).float()
        high_action_logits = high_action_logits.masked_fill(high_action_masks == 0, -1e9)

        # Transform logits to probabilities
        high_action_dist = Categorical(logits=high_action_logits) 

        # Get log prob of the high-level action
        high_action_log_prob = high_action_dist.log_prob(high_actions)

        # Get low-level action logits
        low_action_logits = self.policy.forward_low(obs, high_actions)

        # Apply mask to logits (physical nodes are masked if required by the config)
        if self.config.mask_actions:
            # Get the mask for the low-level actions
            low_level_mask = masks[torch.arange(masks.size(0)), high_actions]
            low_action_logits = low_action_logits.masked_fill(low_level_mask == 0, -1e9)

        # Transform logits to probabilities
        low_action_dist = Categorical(logits=low_action_logits)

        # Get log prob of the low-level action
        low_action_log_prob = low_action_dist.log_prob(low_actions)

        # Combine log probs
        log_prob = high_action_log_prob + low_action_log_prob

        # Get value from the critic network
        value = self.policy.critic(obs)

        # Compute entropy of the distributions
        entropy = high_action_dist.entropy() + low_action_dist.entropy()

        return log_prob, entropy, value
    

    def evaluation_mode(self) -> None:
        """Set the agent in evaluation mode"""
        self.policy.eval()


    def save(self, epoch) -> None: 
        """Save the model in a pickle file
        
        Args:
            epoch: The current epoch of the training.
        """
        path = utils.get_model_save_path(epoch, self.config)
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)


    def load(self, path) -> None: 
        """Load the model from a pickle file
        
        Args:
            path: The path to the pickle file.
        """
        # Check that a path is provided
        if path is None:
            raise ValueError("Path to the model must be specified.")
        # Check if the file exists
        if os.path.exists(path) == False:
            raise FileNotFoundError(f"File {path} does not exist")
        # Load the model
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])