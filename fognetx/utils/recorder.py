import os
from fognetx.config import Config
from fognetx.environment.solution import Solution


class Recorder:
    """A class to record the results of an epoch."""
    def __init__(self, epoch, num_p_nodes, config: Config):
        self.epoch = epoch
        self.num_p_nodes = num_p_nodes
        self.v_net_count = 0
        self.success_count = 0
        self.place_failure = 0
        self.route_failure = 0
        self.total_r2c_ratio = 0
        self.total_reward = 0
        self.config = config


    def step_update(self, reward, solution: Solution, done):
        """Update the recorder with the results of a step."""
        self.total_reward += reward
        # Update the number of virtual networks processed
        if done:
            self.v_net_count += 1
            if not solution.place_result:
                self.place_failure += 1
            elif not solution.route_result:
                self.route_failure += 1
            else:
                self.success_count += 1
                self.total_r2c_ratio += solution.r2c_ratio


    def log_epoch(self):
        """Log the results of the epoch."""
        # Fields to log
        fields = [
            'epoch', 'num_p_nodes', 'v_net_count', 'success_count',
            'place_failure', 'route_failure', 'avg_r2c_ratio',
            'avg_reward'
        ]

        # Calculate the average r2c ratio and reward
        avg_r2c_ratio = self.total_r2c_ratio / self.success_count if self.success_count > 0 else 0
        avg_reward = self.total_reward / self.v_net_count if self.v_net_count > 0 else 0

        if self.config.verbose:
            print(f"\nEpoch {self.epoch}:")
            print(f"  Number of virtual networks processed: {self.v_net_count}")
            print(f"  Number of successful placements: {self.success_count}")
            print(f"  Placement failures: {self.place_failure}")
            print(f"  Routing failures: {self.route_failure}")
            print(f"  Average r2c ratio: {avg_r2c_ratio:.4f}")
            print(f"  Average reward: {avg_reward:.4f}")

        if self.config.save:
            # Compose the log directory path
            log_dir = os.path.join(self.config.save_dir, 'logs')

            # Create the directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

            # Log the epoch results to a file
            with open(os.path.join(log_dir, 'epochLog.csv'), 'a') as f:
                # Write the header if the file is empty
                if os.stat(os.path.join(log_dir, 'epochLog.csv')).st_size == 0:
                    f.write(','.join(fields) + '\n')
                # Write the epoch results
                f.write(f"{self.epoch},{self.num_p_nodes},{self.v_net_count},"
                        f"{self.success_count},{self.place_failure},"
                        f"{self.route_failure},{avg_r2c_ratio},"
                        f"{avg_reward}\n")
