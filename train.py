import numpy as np
import fogblend.utils.utils as utils
from fogblend.ppo.agent import PPOAgent
from fogblend.ppo.trainer import PPOTrainer
from fogblend.environment import Environment
from fogblend.config import get_args, Config


if __name__ == "__main__":

    # Get command line arguments and create a configuration object
    args = get_args()
    config = Config(**args)

    # Save the configuration to a YAML file if specified
    if config.save:
        utils.save_config(config)

    # Set the random seed for reproducibility
    np.random.seed(config.seed)

    # Initialize the environment 
    env = Environment(config)

    # Initialize the agent
    agent = PPOAgent(config)

    # Initialize the PPO trainer
    trainer = PPOTrainer(config, agent, env)

    # Train the agent
    trainer.train()
