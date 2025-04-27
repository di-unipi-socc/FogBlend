import argparse
from datetime import datetime
from dataclasses import dataclass
import os
from typing import Optional

import torch

parser = argparse.ArgumentParser(description="FogNetX Configuration")

# Constants
NODE_RESOURCES = ["cpu", "gpu", "ram"]
LINK_RESOURCES = ["bandwidth"]

SAVE_DIR = "save"
timestamp = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
UNIQUE_FOLDER = f"run_{timestamp}"
SUMMARY_FILENAME = "global_summary.csv"


# Data (physical network)
parser.add_argument("-p_net_topology", type=str, help="Physical network topology")
parser.add_argument("-num_nodes", type=int, help="Number of nodes in the physical network")
parser.add_argument("-p_net_min_size", type=int, help="Minimum size of the physical network")
parser.add_argument("-p_net_max_size", type=int, help="Maximum size of the physical network")
parser.add_argument("-p_net_min_node_resources", type=int, help="Lower bound of physical node resources")
parser.add_argument("-p_net_max_node_resources", type=int, help="Upper bound of physical node resources")
parser.add_argument("-p_net_min_link_resources", type=int, help="Lower bound of physical link resources")
parser.add_argument("-p_net_max_link_resources", type=int, help="Upper bound of physical link resources")


# Data (virtual network)
parser.add_argument("-num_v_net", type=int, help="Number of requests")
parser.add_argument("-v_net_min_size", type=int, help="Minimum size of the virtual network")
parser.add_argument("-v_net_max_size", type=int, help="Maximum size of the virtual network")
parser.add_argument("-v_net_min_node_resources", type=int, help="Lower bound of virtual node resources")
parser.add_argument("-v_net_max_node_resources", type=int, help="Upper bound of virtual node resources")
parser.add_argument("-v_net_min_link_resources", type=int, help="Lower bound of virtual link resources")
parser.add_argument("-v_net_max_link_resources", type=int, help="Upper bound of virtual link resources")
parser.add_argument("-arrival_rate", type=float, help="Arrival rate of requests")
parser.add_argument("-request_avg_lifetime", type=int, help="Lifetime of requests")


# Training
parser.add_argument("-num_workers", type=int, help="Number of workers to distributedly train the model")
parser.add_argument("-num_epochs", type=int, help="Number of training epochs")
parser.add_argument("-lr", type=float, help="Learning rate for the optimizer")
parser.add_argument("-clip_grad", type=float, help="Gradient clipping value")
parser.add_argument("-weight_decay", type=float, help="Weight decay for the optimizer")


# Reinforcement Learning
parser.add_argument("-rl_gamma", type=float, help="Discount factor for RL")
parser.add_argument("-gae_lambda", type=float, help="Generalized Advantage Estimation lambda")
parser.add_argument("-eps_clip", type=float, help="PPO epsilon clip")
parser.add_argument("-batch_size", type=int, help="Batch size for training")
parser.add_argument("-target_steps", type=int, help="Number of steps to collect before updating the model")
parser.add_argument("-update_times", type=int, help="Number of updates per iteration")
parser.add_argument("-entropy_coeff", type=float, help="Entropy coefficient for exploration")
parser.add_argument("-critic_coeff", type=float, help="Critic coefficient for value loss")
parser.add_argument("-norm_advantage", type=bool,help="Normalize advantage values")


# Model
parser.add_argument("-embedding_dim", type=int, help="Embedding dimension for the model")


# Execution
parser.add_argument("-seed", type=int, help="Random seed for reproducibility")
parser.add_argument("-mask_actions", type=bool, help="Mask invalid actions in the action space")
parser.add_argument("-reusable", type=bool, help="Allow reuse of resources in the environment")
parser.add_argument("-eval_interval", type=int, help="Interval for evaluation")
parser.add_argument("-save_interval", type=int, help="Interval for saving the model")


# Test
parser.add_argument("-pretrained_model_dir", type=str, help="Path to the pretrained model")
parser.add_argument("-prolog_mode", type=str, help='Prolog mode for the test')


# System
parser.add_argument("-save", type=bool, help="Save the generated data")
parser.add_argument("-save_dir", type=str, help="Path to save the generated data")
parser.add_argument("-verbose", type=bool, help="Verbose mode")


# Class and function to handle command line arguments
def get_args():
    """
    Parse command line arguments and return the arguments as a dictionary.
    """
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    return args_dict


@dataclass
class Config:
    # Data
    node_resources = NODE_RESOURCES
    link_resources = LINK_RESOURCES

    # Data (physical network)
    p_net_topology: str = "custom"
    num_nodes: int = 100
    p_net_min_size: int = 50
    p_net_max_size: int = 500
    p_net_min_node_resources: int = 50
    p_net_max_node_resources: int = 100
    p_net_min_link_resources: int = 50
    p_net_max_link_resources: int = 100

    # Data (virtual network)
    num_v_net: int = 1000
    v_net_min_size: int = 2
    v_net_max_size: int = 8
    v_net_min_node_resources: int = 0
    v_net_max_node_resources: int = 20
    v_net_min_link_resources: int = 0
    v_net_max_link_resources: int = 25
    arrival_rate: float = 0.001
    request_avg_lifetime: int = 500

    # Training
    num_workers: int = 1
    num_epochs: int = 100
    lr: float = 1e-3
    clip_grad: float = 0.5
    weight_decay: float = 1e-5

    # Reinforcement Learning
    rl_gamma: float = 0.99
    gae_lambda: float = 0.98
    eps_clip: float = 0.2
    batch_size: int = 128
    target_steps: int = 256
    update_times: int = 10
    entropy_coeff: float = 0.01
    critic_coeff: float = 0.5
    norm_advantage: bool = True

    # Model
    embedding_dim: int = 128
    gcn_num_layers: int = 2
    batch_norm: bool = False
    dropout_prob: float = 0.0

    # Execution
    seed: int = 88
    mask_actions: bool = True
    reusable: bool = True
    eval_interval: int = 10
    save_interval: int = 10

    # Test
    pretrained_model_dir: Optional[str] = None
    prolog_mode: str = "routing"

    # System
    save: bool = True
    save_dir: str = "save"
    verbose: bool = True  
    device: str = "cuda" if torch.cuda.is_available() else "cpu"