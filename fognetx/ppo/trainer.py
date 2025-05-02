# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config, PPOAgent, Environment
# REGULAR IMPORTS
import torch


class PPOTrainer:
    """
    Class to train the agent using the Proximal Policy Optimization (PPO) algorithm.
    """

    def __init__(self, config: Config, agent: PPOAgent, env: Environment):
        self.agent = agent
        self.env = env
        self.config = config


    def train(self):
        """
        Train the agent using the PPO algorithm. The parameters are gathered from the config object.
        """
        for epoch in range(self.config.num_epochs):
            # Create a new environment for each epoch
            self.env.create_env(epoch)

            # Reset buffer
            self.agent.buffer.clear()

            # Log
            print(f"\nStarting epoch {epoch}. Infrastructure: {self.env.p_net.num_nodes} nodes.\n")

            # Iterate through the requests in the environment
            request_elaborated = 0
            while request_elaborated < self.env.requests.num_v_net:
                # Fully elaborate the request
                done = False
                while not done:
                    # Get the current observation
                    obs, mask = self.env.get_observations()

                    with torch.no_grad():
                        # Get the action from the agent
                        high_action, low_action, log_prob = self.agent.act(obs, mask)

                        # Get the value from the agent
                        value = self.agent.evaluate(obs)

                    # Take a step in the environment
                    reward, done = self.env.step(high_action.item(), low_action.item())

                    # Store the transition in the buffer
                    self.agent.buffer.store({'obs': obs, 'mask': mask}, (high_action, low_action), log_prob, reward, done, value)

                # If collected enough samples, update the agent
                if self.agent.buffer.length >= self.config.target_steps:
                    # Compute current value
                    with torch.no_grad():
                        obs, _ = self.env.get_observations()
                        current_value = self.agent.evaluate(obs)
                    # Compute the advantages and returns
                    self.agent.buffer.compute_returns_advantages(current_value, self.config)
                    # Update the agent
                    self.agent_update()
                    # Clear the buffer
                    self.agent.buffer.clear()   

                # Update the request elaborated count
                request_elaborated += 1
            
            # Log the epoch results
            self.env.recorder.log_epoch() 

            # Save the model
            if (epoch + 1) % self.config.save_interval == 0 and self.config.save:
                self.agent.save(epoch + 1)

    
    def agent_update(self):
        """
        Update the agent using the PPO algorithm. The parameters are gathered from the config object.
        """
        for i in range(self.config.update_times):
            for batch in self.agent.buffer.get_batches(self.config.batch_size, self.config.device):
                # Unpack the batch
                obs, masks, high_actions, low_actions, log_probs, returns, advantages = batch

                # Get new log probs and values
                new_log_probs, new_value = self.agent.evaluate_actions(obs, masks, high_actions, low_actions)

                print(f"Done evaluating actions {i}")