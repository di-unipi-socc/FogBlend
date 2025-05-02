# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config
# REGULAR IMPORTS
import numpy as np
import networkx as nx


class PhysicalNetwork():
    """
    Class representing a physical network using NetworkX.
    """

    def __init__(self, config: Config):
        """
        Initialize the PhysicalNetwork with configuration parameters.

        Args:
            config: Configuration object containing parameters for the physical network.
        """
        # Topology
        self.topology = config.p_net_topology
        # Size
        self.num_nodes = None
        self.max_size = config.p_net_max_size
        self.min_size = config.p_net_min_size
        self.sizes = None
        # Resources
        self.min_node_resources = config.p_net_min_node_resources
        self.max_node_resources = config.p_net_max_node_resources
        self.min_link_resources = config.p_net_min_link_resources
        self.max_link_resources = config.p_net_max_link_resources
        self.node_resources = config.node_resources
        self.link_resources = config.link_resources
        # Training
        self.num_epochs = config.num_epochs
        self.iter = 0
        # Network
        self.net = None
        # Random number generator
        self.rng = np.random.default_rng(config.seed)

    
    def generate_p_net(self, size=None):
        """
        Generate a physical network based on the configuration of the class.
        
        Args:
            size (int, optional): Size of the network. If None, a random size is generated.
        """
        if size is not None:
            self.num_nodes = size
        else:
            self.num_nodes = self.get_size(self.min_size, self.max_size, self.iter)
            self.iter += 1

        # Repeat until a connected waxman graph is created
        while True:
            self.net = nx.waxman_graph(self.num_nodes, alpha=0.5, beta=0.2, seed=int(self.rng.integers(0, np.iinfo(np.int32).max)))
            if nx.is_connected(self.net):
                break

        # Vectorize random assignment for nodes
        num_node_resources = len(self.node_resources)

        node_resource_values = self.rng.integers(
            self.min_node_resources, self.max_node_resources, 
            size=(self.num_nodes, num_node_resources),
            endpoint=True
        )

        for i, node in enumerate(self.net.nodes()):
            for j, resource in enumerate(self.node_resources):
                self.net.nodes[node][resource] = node_resource_values[i, j]


        # Vectorize random assignment for edges
        num_edges = self.net.number_of_edges()
        num_link_resources = len(self.link_resources)

        link_resource_values = self.rng.integers(
            self.min_link_resources, self.max_link_resources,
            size=(num_edges, num_link_resources),
            endpoint=True
        )

        for i, edge in enumerate(self.net.edges()):
            for j, resource in enumerate(self.link_resources):
                self.net.edges[edge][resource] = link_resource_values[i, j]


    def get_size(self, min_size, max_size, iter):
        """
        Get the random size of the network based on the iteration number.

        Args:
            min_size (int): Minimum size of the network.
            max_size (int): Maximum size of the network.
            iter (int): Current iteration number.

        Returns:
            int: Size of the network.
        """
        # If none, construct the random array of network sizes
        if self.sizes is None:
            sizes = np.linspace(min_size, max_size, self.num_epochs)
            np.random.shuffle(sizes)
            self.sizes = sizes.astype(int)

        # If the iteration is greater than the number of epochs, return the last size
        if iter >= self.num_epochs:
            return self.sizes[-1]
        
        # Otherwise, return the size for the current iteration
        return self.sizes[iter]