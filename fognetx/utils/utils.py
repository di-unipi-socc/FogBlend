# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config, PPOAgent
# REGULAR IMPORTS
import os
import torch
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


def warmup_agent(agent: PPOAgent, config: Config):
    """
    Warm up the agent by creating a dummy state and getting the action and value from the agent.
    This is useful in testing to avoid overhead in the first step of inference.
    
    Args:
        agent: The agent to be warmed up.
        config (Config): The configuration object.
    """
    p_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 5 
    v_net_features_dim = len(config.node_resources) + len(config.link_resources)*3 + 3

    # Create a dummy state with the correct dimensions
    obs = {
        'p_net': Batch.from_data_list([Data(x=torch.zeros((2, p_net_features_dim)).to(config.device), edge_index=torch.tensor([[0, 1], [1, 0]]).to(config.device))]),
        'v_net': Batch.from_data_list([Data(x=torch.zeros((2, v_net_features_dim)).to(config.device), edge_index=torch.tensor([[0, 1], [1, 0]]).to(config.device))]),
    }

    with torch.no_grad():
        # Get the action from the agent
        agent.act(obs, None)
        # Get the value from the agent
        agent.evaluate(obs)


def get_model_save_path(epoch, config: Config) -> str:
    """
    Get the model path for a given epoch. Creates the directory if it doesn't exist.

    Args:
        epoch (int): The epoch number.

    Returns:
        str: The model path.
    """
    # Compose the save directory path
    save_dir = os.path.join(config.save_dir, config.unique_folder, 'model', 'model_{}'.format(epoch))

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def batch_states(obs: list, masks: list, device):
    """
    Create a batch of observations and masks from a list of states.
    
    Args:
        obs (list): A list of state dicts to be batched. Each contains 'p_net' and 'v_net'.
        masks (list): A list of mask tensors of shape [1, num_v_nodes, num_p_nodes].
        device (str or torch.device): The device to move the batch to.
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