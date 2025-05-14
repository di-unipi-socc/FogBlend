# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config, PhysicalNetwork, VirtualNetwork
# REGULAR IMPORTS
import os
from typing import OrderedDict


class Solution:
    """A class to represent a solution for a virtual network request in a physical network."""

    def __init__(self, event_id, event_type, event_time, p_net: PhysicalNetwork, v_net: VirtualNetwork, config: Config):
        """
        Initialize the solution with a physical network, a virtual network, and environment configuration.

        Args:
            event_id: The event id associated with the solution.
            event_type: The type of the event (arrival or leave).
            event_time: The simulation time of the event.
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
        self.event_time = event_time
        self.place_result = True
        self.route_result = True
        self.node_mapping = OrderedDict()   # e.g. {0: (110, {'cpu': 5, 'gpu': 3, 'ram': 2})}
        self.link_mapping = OrderedDict()   # e.g. {(0,1): ([(110,99),(99,50)], {'bandwidth':20})}
        self.cost = 0
        self.revenue = 0
        self.r2c_ratio = 0
        self.running_request = 0
        self.elapsed_time = 0


    def is_feasible(self):
        """Check if the solution is feasible based on the placement and routing results."""
        return self.place_result and self.route_result


    def compute_info(self) -> None:
        """Compute the information of the solution."""
        self.cost = 0
        self.revenue = 0

        # Compute the cost and revenue (sum of physical resources used and sum of virtual resources required)
        for _, resources in self.node_mapping.values():
            self.revenue += sum([resources[res] for res in self.config.node_resources])
        self.cost = self.revenue

        for link_list, resources in self.link_mapping.values():
            self.revenue += sum([resources[res] for res in self.config.link_resources])
            # Check if the link is mapped on the same physical node
            if len(link_list) == 1:
                if link_list[0][0] == link_list[0][1]:
                    # No cost for self-loop links
                    continue
            self.cost += sum([resources[res] for res in self.config.link_resources]) * len(link_list)

        # Compute the r2c ratio
        self.r2c_ratio = self.revenue / self.cost if self.cost > 0 else 0


    def log(self, log_dir = None, file_name=None) -> None:
        """
        Log the solution to a file.

        Args:
            log_dir: The directory where the log file will be saved.
            file_name: The name of the log file.
        """
        # Fields to log
        fields = [
            'event_id', 'event_type', 'event_time', 'v_net_id', 'place_result', 'route_result',
            'node_mapping', 'link_mapping', 'cost', 'revenue', 'r2c_ratio', 'running_request', 'elapsed_time'
        ]

        if self.config.save:
            # Compose the log directory path
            if log_dir is None:
                log_dir = os.path.join(self.config.save_dir, self.config.unique_folder, 'logs', 'requests')

            # Create the directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

            # Convert ordered dict to standard dict string for logging
            node_mapping = '"'+str({k: v for k, v in self.node_mapping.items()})+'"'
            link_mapping = '"'+str({k: v for k, v in self.link_mapping.items()})+'"'

            # Compose the file name
            if file_name is None:
                file_name = 'requestLog.csv'
            
            # Log the solution to a file
            with open(os.path.join(log_dir, file_name), 'a') as f:
                # Write the header if the file is empty
                if os.stat(os.path.join(log_dir, file_name)).st_size == 0:
                    f.write(','.join(fields) + '\n')
                # Write the solution information
                f.write(f"{self.event_id},{self.event_type},{self.event_time},"
                        f"{self.v_net.id},{self.place_result},{self.route_result},"
                        f"{node_mapping},{link_mapping},"
                        f"{self.cost},{self.revenue},"
                        f"{self.r2c_ratio},{self.running_request},{self.elapsed_time}\n"
                        )