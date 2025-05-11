# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config, PPOAgent
# REGULAR IMPORTS
import os
import yaml
import torch
from dataclasses import asdict
from torch_geometric.data import Data, Batch


def path_to_links(path: list) -> list:
    """
    Converts a given path to a list of tuples containing two elements each.

    Args:
        path (list): A list of elements representing a path.

    Returns:
        list: A list of tuples, each tuple containing two elements from the given path.
    """
    if len(path) == 1:
        return [(path[0], path[0])]
    return [(path[i], path[i+1]) for i in range(len(path)-1)]


def warmup_agent(agent: PPOAgent, config: Config) -> None:
    """
    Warm up the agent by creating a dummy state and getting the action and value from the agent.
    This is useful in testing to avoid overhead in the first step of inference.
    
    Args:
        agent (PPOAgent): The agent to be warmed up.
        config (Config): The configuration object.
    """
    p_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 5 
    v_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 3

    # Create a dummy state with the correct dimensions
    obs = {
        'p_net': Batch.from_data_list([Data(x=torch.zeros((2, p_net_features_dim)).to(config.device), edge_index=torch.tensor([[0, 1], [1, 0]]).to(config.device))]),
        'v_net': Batch.from_data_list([Data(x=torch.zeros((2, v_net_features_dim)).to(config.device), edge_index=torch.tensor([[0, 1], [1, 0]]).to(config.device))]),
    }

    # Crate a dummy mask
    mask = torch.zeros((1, 2, 2)).to(config.device)

    with torch.no_grad():
        # Get the action from the agent
        agent.act(obs, mask)
        # Get the value from the agent
        agent.evaluate(obs)


def get_model_save_path(epoch, config: Config) -> str:
    """
    Get the model path for a given epoch. Creates the directory if it doesn't exist.

    Args:
        epoch (int): The epoch number.

    Returns:
        str: The model save path.
    """
    # Compose the save directory path
    save_dir = os.path.join(config.save_dir, config.unique_folder, 'model')

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Include filename in the path
    save_dir = os.path.join(save_dir, 'model_{}.pkl'.format(epoch))

    return save_dir


def batch_states(obs: list, masks: list, device):
    """
    Create a batch of observations and masks from a list of states.
    
    Args:
        obs (list): A list of state dicts to be batched. Each contains 'p_net' and 'v_net'.
        masks (list): A list of mask tensors of shape [1, num_v_nodes, num_p_nodes].
        device (str): The device to move the batch to.

    Returns:
        tuple: A tuple containing the batched observations and masks.
    """
    p_net_list = [item['p_net'].to_data_list()[0] for item in obs]
    v_net_list = [item['v_net'].to_data_list()[0] for item in obs]

    batched_p_net = Batch.from_data_list(p_net_list).to(device)
    batched_v_net = Batch.from_data_list(v_net_list).to(device)

    if masks[0] is not None:
        # Pad the masks to the same size (v_nodes can change in size)
        max_v_nodes = max(mask.size(1) for mask in masks)
        padded_masks = [torch.cat([mask, torch.zeros(1, max_v_nodes - mask.size(1), mask.size(2)).to(device)], dim=1) for mask in masks]
        # Batch masks: [batch_size, num_v_nodes, num_p_nodes]
        batched_masks = torch.cat(padded_masks, dim=0).to(device)
    else:
        batched_masks = None

    return {'p_net': batched_p_net, 'v_net': batched_v_net}, batched_masks


def compute_arrival_rate(p_num_nodes: int) -> float:
    """
    Compute the arrival rate based on the number of physical nodes.
    
    Args:
        p_num_nodes (int): The number of physical nodes.

    Returns:
        float: The computed arrival rate.
    """
    # Compute the arrival rate using a polynomial function
    # The coefficients are based on the polynomial regression of the data presented in the paper
    return -0.046 + 0.0012*p_num_nodes - 0.0000011*(p_num_nodes**2) + 1.777*(10**(-8))*(p_num_nodes**3)


def save_config(config: Config) -> None:
    """
    Save the configuration to a file.
    
    Args:
        config (Config): The configuration object.
    """
    # Compose the save directory path
    save_dir = os.path.join(config.save_dir, config.unique_folder)

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Convert dataclass to dictionary
    config_dict = asdict(config)

    # Save the configuration to a YAML file
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        

def log_update(loss, actor_loss, critic_loss, mean_entropy, mean_reward, mean_value, 
               mean_returns, mean_advantage, gradient_norm, config: Config) -> None:
    """
    Log the update information.
    
    Args:
        loss (float): The total loss.
        actor_loss (float): The actor loss.
        critic_loss (float): The critic loss.
        mean_entropy (float): The mean entropy.
        mean_reward (float): The mean reward.
        mean_value (float): The mean value.
        mean_returns (float): The mean returns.
        mean_advantage (float): The mean advantage.
        config (Config): The configuration object.
    """

    # Fields to log
    fields = [
        'loss', 'actor_loss', 'critic_loss',
        'mean_entropy', 'mean_reward', 'mean_value',
        'mean_returns', 'mean_advantage', 'gradient_norm'
    ]

    if config.verbose:
        print(f"Loss: {loss:.4f}, Actor Loss: {actor_loss:.4f}, Critic Loss: {critic_loss:.4f}, "
              f"Mean Entropy: {mean_entropy:.4f}, Mean Reward: {mean_reward:.4f}, "
              f"Mean Value: {mean_value:.4f}, Mean Returns: {mean_returns:.4f}, "
              f"Mean Advantage: {mean_advantage:.4f}, Gradient Norm: {gradient_norm:.4f}\n")
        
    if config.save:
        # Compose the log directory path
        log_dir = os.path.join(config.save_dir, config.unique_folder, 'logs')

        # Create the directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Log the update information to a CSV file
        with open(os.path.join(log_dir, 'updateLog.csv'), 'a') as f:
            # Write the header if the file is empty
            if os.stat(os.path.join(log_dir, 'updateLog.csv')).st_size == 0:
                f.write(','.join(fields) + '\n')
            # Write the update information
            f.write(f"{loss:.4f},{actor_loss:.4f},{critic_loss:.4f},"
                    f"{mean_entropy:.4f},{mean_reward:.4f},{mean_value:.4f},"
                    f"{mean_returns:.4f},{mean_advantage:.4f},{gradient_norm:.4f}\n"
                    )
            