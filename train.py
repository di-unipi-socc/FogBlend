import numpy as np
from fognetx import Config, get_args
from fognetx.ppo.agent import PPOAgent
from fognetx.ppo.trainer import PPOTrainer
from fognetx.environment import Environment


if __name__ == "__main__":

    # Get command line arguments and create a configuration object
    args = get_args()
    config = Config(**args)

    # Set the random seed for reproducibility
    np.random.seed(config.seed)

    # Initialize the environment 
    env = Environment(config)

    # Initialize the agent
    agent = PPOAgent(config, env)

    # Initialize the PPO trainer
    trainer = PPOTrainer(config, agent, env)

    # Train the agent
    trainer.train()
