# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config
# REGULAR IMPORTS
import os
import numpy as np
import networkx as nx
from fognetx.utils import utils


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

    
    def generate_p_net(self, size=None) -> None:
        """
        Generate a physical network based on the configuration of the class.
        
        Args:
            size (int, optional): Size of the network. If None, a random size is generated.
        """
        if self.topology == 'waxman':
            if size is not None:
                self.num_nodes = size
            else:
                self.num_nodes = self.get_random_size(self.min_size, self.max_size, self.iter)
                self.iter += 1

            # Repeat until a connected waxman graph is created
            while True:
                self.net = nx.waxman_graph(self.num_nodes, alpha=0.5, beta=0.2, seed=int(self.rng.integers(0, np.iinfo(np.int32).max)))
                if nx.is_connected(self.net):
                    break
        elif self.topology == 'geant':
            self.net = utils.load_geant_topology()
            self.num_nodes = self.net.number_of_nodes()
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        # Vectorize random assignment for nodes
        num_node_resources = len(self.node_resources)

        node_resource_values = self.rng.integers(
            self.min_node_resources, self.max_node_resources, 
            size=(self.num_nodes, num_node_resources),
            endpoint=True
        )

        for i, node in enumerate(self.net.nodes()):
            for j, resource in enumerate(self.node_resources):
                self.net.nodes[node][resource] = int(node_resource_values[i, j])


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
                self.net.edges[edge][resource] = int(link_resource_values[i, j])


    def get_random_size(self, min_size, max_size, iter):
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
    

    def apply_load(self, load) -> None:
        """
        Apply a load to the physical network.

        Args:
            load (float): Load to be applied to the network in percentage (0 to 1).
        """
        # Nodes
        for node in self.net.nodes():
            for resource in self.node_resources:
                self.net.nodes[node][resource] = int(round((1-load) * self.net.nodes[node][resource]))
        # Edges
        for edge in self.net.edges():
            for resource in self.link_resources:
                self.net.edges[edge][resource] = int(round((1-load) * self.net.edges[edge][resource]))


    def save_to_file(self, save_dir, filename) -> None:
        """
        Save the physical network to a file.

        Args:
            save_dir (str): Directory where the file will be saved.
            filename (str): Name of the file.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Save the network to a GML file
        nx.write_gml(self.net, os.path.join(save_dir, filename))


    def load_from_file(self, load_dir, filename) -> None:
        """
        Load the physical network from a file.

        Args:
            load_dir (str): Directory where the file is located.
            filename (str): Name of the file.
        """
        # Check if the file exists
        file_path = os.path.join(load_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist")
        # Load the network from a GML file
        self.net = nx.read_gml(file_path)
        self.net = nx.relabel_nodes(self.net, lambda x: int(x))  # Fix node labels
        # Get the number of nodes
        self.num_nodes = self.net.number_of_nodes()
