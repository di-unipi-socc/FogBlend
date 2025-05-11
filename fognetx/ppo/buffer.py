# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING

import torch
if TYPE_CHECKING: from fognetx.utils.types import Config
# REGULAR IMPORTS
import numpy as np
import fognetx.utils as utils


class PPOBuffer:
    """
    A buffer to store the experience of the agent during training.
    """

    def __init__(self, seed):
        self.length = 0
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = None
        self.advantages = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)


    def store(self, state, action, log_prob, reward, done, value):
        """
        Store the transition in the buffer.
        """
        # Append to lists
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        # Update length
        self.length += 1

    
    def clear(self):
        """
        Clear the buffer by resetting all stored values.
        """
        # Clear all stored values
        self.length = 0
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.returns = None
        self.advantages = None


    def compute_returns_advantages(self, current_value, config: Config) -> None:
        """
        Compute the returns and advantages for the stored transitions.
        """
        # Initialize returns and advantages
        returns = [0] * self.length
        advantages = [0] * self.length
        
        # Start with zero advantage for the last step
        next_advantage = 0
        
        # Compute returns and advantages using Generalized Advantage Estimation (GAE)
        for t in reversed(range(self.length)):
            if t == self.length - 1:
                next_value = current_value
            else:
                next_value = self.values[t + 1]

            delta = self.rewards[t] + config.rl_gamma * next_value * (1 - self.dones[t]) - self.values[t]
            
            # Use the previously calculated advantage instead of accessing out of bounds
            advantages[t] = delta + config.rl_gamma * config.gae_lambda * next_advantage * (1 - self.dones[t])
            next_advantage = advantages[t]
            
            returns[t] = advantages[t] + self.values[t]

        # Normalize advantages (if required)
        if config.norm_advantage:
            advantages = torch.tensor(advantages, dtype=torch.float32, device=config.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Assign computed returns and advantages to the buffer
        self.returns = torch.tensor(returns, dtype=torch.float32, device=config.device)
        self.advantages = advantages


    def get_batches(self, batch_size, device):
        """
        Return an iterator over the batches of the buffer.
        
        Args:
            batch_size (int): The size of each batch.
            device (str): The device to move the data to.
        
        Yields:
            tuple: A tuple containing the batch data (observations, masks, actions, log_probs, rewards, returns, advantages).
        """
        # Shuffle the indices for batch sampling
        indices = np.arange(self.length)
        self.rng.shuffle(indices)

        # Yield batches of data
        for start in range(0, self.length, batch_size):
            end = start + batch_size
            if end > self.length:
                break
            batch_indices = indices[start:end]

            # Collect batch data
            obs = [self.states[i]['obs'] for i in batch_indices]
            masks = [self.states[i]['mask'] for i in batch_indices]
            high_actions = [self.actions[i][0] for i in batch_indices]
            low_actions = [self.actions[i][1] for i in batch_indices]
            log_probs = [self.log_probs[i] for i in batch_indices]
            rewards = [self.rewards[i] for i in batch_indices]
            returns = self.returns[batch_indices]
            advantages = self.advantages[batch_indices]

            # Convert into tensors (returns and advantages are already tensors)
            obs, masks = utils.batch_states(obs, masks, device)
            high_actions = torch.tensor(high_actions, dtype=torch.long, device=device)
            low_actions = torch.tensor(low_actions, dtype=torch.long, device=device)
            log_probs = torch.tensor(log_probs, dtype=torch.float32, device=device)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

            yield (obs, masks, high_actions, low_actions, log_probs, rewards, returns, advantages)