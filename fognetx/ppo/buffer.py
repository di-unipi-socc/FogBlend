
class PPOBuffer:
    """
    A buffer to store the experience of the agent during training.
    """

    def __init__(self):
        self.length = 0
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []


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


    def sample(self):
        # Return tensors or batches
        pass


    def clear(self):
        """
        Clear the buffer by resetting all stored values.
        """
        # Clear all stored values
        self.__init__()
