# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config, PhysicalNetwork, VirtualNetwork
# REGULAR IMPORTS
import os
from typing import OrderedDict


class Solution:
    """A class to represent a solution for a virtual network request in a physical network."""

    def __init__(self, event_id, event_type, p_net: PhysicalNetwork, v_net: VirtualNetwork, config: Config):
        """
        Initialize the solution with a physical network, a virtual network, and environment configuration.

        Args:
            event_id: The event id associated with the solution.
            event_type: The type of the event (arrival or leave).
            p_net: The physical network.
            v_net: The virtual network.
            config: Environment configuration.
        """
        # Store the physical network, virtual network, and environment configuration
        self.p_net = p_net
        self.v_net = v_net
        self.config = config

        # Solution information
        self.event_id = event_id
        self.event_type = event_type
        self.result = True
        self.place_result = True
        self.route_result = True
        self.node_mapping = OrderedDict()   # e.g. {0: (110, {'cpu': 5, 'gpu': 3, 'ram': 2})}
        self.link_mapping = OrderedDict()   # e.g. {(0,1): ([(110,99),(99,50)], {'bandwidth':20})}
        self.cost = 0
        self.revenue = 0
        self.r2c_ratio = 0
        self.violation = 0


    def is_feasible(self):
        """Check if the solution is feasible based on the placement and routing results."""
        return self.place_result and self.route_result


    def compute_info(self):
        """Compute the information of the solution."""
        self.cost = 0
        self.revenue = 0

        # Compute the cost and revenue (sum of physical resources used and sum of virtual resources required)
        for _, resources in self.node_mapping.values():
            self.revenue += sum([resources[res] for res in self.config.node_resources])
        self.cost = self.revenue

        for link_list, resources in self.link_mapping.values():
            self.revenue += sum([resources[res] for res in self.config.link_resources])
            self.cost += sum([resources[res] for res in self.config.link_resources]) * len(link_list)

        # Compute the r2c ratio
        self.r2c_ratio = self.revenue / self.cost if self.cost > 0 else 0


    def log(self):
        """
        Log the solution to a file.

        Args:
            solution: The solution object containing the mapping and routing information.
            env_config: The environment configuration object.
            v_net: The virtual network object.
            p_net: The physical network object.
        """
        # Fields to log
        fields = [
            'event_id', 'event_type', 'v_net_id', 'place_result', 'route_result',
            'node_mapping', 'link_mapping', 'cost', 'revenue',
            'r2c_ratio', 'violation'
        ]

        if self.config.save:
            # Compose the log directory path
            log_dir = os.path.join(self.config.save_dir, self.config.unique_folder, 'logs')

            # Create the directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

            # Convert ordered dict to standard dict string for logging
            node_mapping = '"'+str({k: v for k, v in self.node_mapping.items()})+'"'
            link_mapping = '"'+str({k: v for k, v in self.link_mapping.items()})+'"'
            
            # Log the solution to a file
            with open(os.path.join(log_dir, 'requestLog.csv'), 'a') as f:
                # Write the header if the file is empty
                if os.stat(os.path.join(log_dir, 'requestLog.csv')).st_size == 0:
                    f.write(','.join(fields) + '\n')
                # Write the solution information
                f.write(f"{self.event_id},{self.event_type},{self.v_net.id},"
                        f"{self.place_result},{self.route_result},"
                        f"{node_mapping},{link_mapping},"
                        f"{self.cost},{self.revenue},"
                        f"{self.r2c_ratio},{self.violation}\n"
                        )