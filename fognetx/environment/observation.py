
import torch
from torch_geometric.data import Data, Batch
from fognetx.config import Config
from fognetx.environment.physicalNetwork import PhysicalNetwork
from fognetx.environment.virtualNetworkRequets import VirtualNetwork
import numpy as np
import networkx as nx


class Observation:
    """
    Observation class to represent the state of the environment.
    """

    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, config: Config):
        """
        Initialize the Observation with physical and virtual networks.

        Args:
            p_net: Physical network object.
            v_net: Virtual network object.
        """
        self.config = config
        # Node resources attributes
        self.p_nodes_resources = self.get_nodes_resources(p_net.net, p_net.node_resources)
        self.v_nodes_resources = self.get_nodes_resources(v_net.net, v_net.node_resources)
        self.node_resources_normalization_value = config.p_net_max_node_resources
        # Link resources attributes
        self.p_nodes_link_resources = self.get_links_resources(p_net.net, p_net.link_resources)
        self.v_nodes_link_resources = self.get_links_resources(v_net.net, v_net.link_resources)
        self.link_resources_normalization_value = config.p_net_max_link_resources
        # Link pair 
        self.p_link_pair = self.get_link_pair(p_net.net)
        self.v_link_pair = self.get_link_pair(v_net.net)
        # Node status
        self.p_nodes_status = np.zeros((p_net.net.number_of_nodes(), 1))  # All nodes are empty at the beginning
        self.v_nodes_status = np.zeros((v_net.net.number_of_nodes(), 1))  # All services are not placed at the beginning
        # Physical network distance matrix
        self.p_net_distance_matrix = nx.floyd_warshall_numpy(p_net.net)
        self.average_distance = np.zeros((p_net.net.number_of_nodes(), 1))  # Initially, no nodes in the solution
        # Degree of each node
        self.p_node_degrees = np.array([list(nx.degree_centrality(p_net.net).values())], dtype=np.float32).T
        # Virtual network info
        self.v_net_size = v_net.net.number_of_nodes()
        self.v_num_placed_nodes = 0  # Initially, no nodes in the solution
        # Physical network info
        self.p_net_size = p_net.net.number_of_nodes()


    def get_nodes_resources(self, net: nx.Graph, node_resources: list[str]) -> np.ndarray:
        """
        Get the resources of the nodes 

        Args:
            net: Physical network graph.
            node_resources: Node resources.

        Returns:
            A matrix of node resources [num_nodes x num_resources].
        """
        num_nodes = net.number_of_nodes()
        num_node_resources = len(node_resources)

        # Initialize a matrix to store node resources
        node_resources_matrix = np.zeros((num_nodes, num_node_resources))

        # Fill the matrix with node resources
        for i, node in enumerate(net.nodes()):
            for j, resource in enumerate(node_resources):
                node_resources_matrix[i, j] = net.nodes[node][resource] 

        return node_resources_matrix
    

    def get_links_resources(self, net: nx.Graph, link_resources: list[str]) -> np.ndarray:
        """
        Get maximum, sum and mean resources of the links of each node
        
        Args:
            net: Physical network graph.
            link_resources: Link resources.
        
        Returns:
            A matrix of link resources [num_nodes x num_resources x 3].
        """
        num_nodes = net.number_of_nodes()
        num_link_resources = len(link_resources)

        # Initialize a matrix to store link resources
        link_resources_matrix = np.zeros((num_nodes, num_link_resources, 3))

        # Fill the matrix with link resources
        for i, node in enumerate(net.nodes()):
            for j, resource in enumerate(link_resources):
                # Get the edges connected to the node
                edges = net.edges(node)
                # Get the resources of the edges
                edge_resources = [net.edges[edge][resource] for edge in edges]
                # Calculate max, sum and mean of the resources
                link_resources_matrix[i, j, 0] = np.max(edge_resources) 
                link_resources_matrix[i, j, 1] = np.sum(edge_resources) 
                link_resources_matrix[i, j, 2] = np.mean(edge_resources) 

        return link_resources_matrix
    
    
    def get_link_pair(self, net: nx.Graph) -> np.ndarray:
        """
        Get the link pair of the network considering both direction of the edge.
        
        Args:
            net: Physical network graph.
        
        Returns:
            A numpy array of link pairs [2 x 2*num_edges].
        """
        edges = list(net.edges())
        # Add reverse edges for undirected behavior
        edges += [(v, u) for (u, v) in edges]

        # Remove duplicates (just in case) and convert to np.array
        edges = list(set(edges))
        return np.array(edges, dtype=np.int32).T
    

    def get_observation(self) -> dict:
        """
        Get the observation of the environment as a dictionary of torch_geometric data objects.
        
        Returns:
            A dictionary containing the physical and virtual network observations.
        """
        # Initialize the observation dictionary
        obs = {} 
    
        # Physical network observation
        v_net_size = np.ones((self.p_net_size, 1), dtype=np.float32) * self.v_net_size / self.config.v_net_max_size
        v_num_placed_nodes = np.ones((self.p_net_size, 1), dtype=np.float32) * self.v_num_placed_nodes / self.config.v_net_max_size

        # Normalize the node resources and link resources
        p_nodes_resources = self.p_nodes_resources / self.config.p_net_max_node_resources

        p_nodes_link_max = self.p_nodes_link_resources[:, :, 0] / self.config.p_net_max_link_resources
        p_nodes_link_sum = self.p_nodes_link_resources[:, :, 1] / self.config.p_net_max_link_resources
        p_nodes_link_mean = self.p_nodes_link_resources[:, :, 2] / self.config.p_net_max_link_resources

        x_p_net = np.concatenate((p_nodes_resources, v_net_size, v_num_placed_nodes, self.p_nodes_status, p_nodes_link_max, p_nodes_link_sum, p_nodes_link_mean, self.p_node_degrees, self.average_distance), axis=-1)

        # Convert to torch_geometric data format
        x_p_net = torch.tensor(x_p_net, dtype=torch.float32, device=self.config.device)
        p_link_pair = torch.tensor(self.p_link_pair, device=self.config.device)
        obs['p_net'] = Batch.from_data_list([Data(x=x_p_net, edge_index=p_link_pair)])

        # Virtual network observation
        v_net_size = np.ones((self.v_net_size, 1), dtype=np.float32) * self.v_net_size / self.config.v_net_max_size
        v_num_placed_nodes = np.ones((self.v_net_size, 1), dtype=np.float32) * self.v_num_placed_nodes / self.config.v_net_max_size

        # Normalize the node resources and link resources
        v_nodes_resources = self.v_nodes_resources / self.config.v_net_max_node_resources

        v_nodes_link_max = self.v_nodes_link_resources[:, :, 0] / self.config.v_net_max_link_resources
        v_nodes_link_sum = self.v_nodes_link_resources[:, :, 1] / self.config.v_net_max_link_resources
        v_nodes_link_mean = self.v_nodes_link_resources[:, :, 2] / self.config.v_net_max_link_resources

        x_v_net = np.concatenate((v_nodes_resources, v_net_size, v_num_placed_nodes, self.v_nodes_status, v_nodes_link_max, v_nodes_link_sum, v_nodes_link_mean), axis=-1)

        # Convert to torch_geometric data format
        x_v_net = torch.tensor(x_v_net, dtype=torch.float32, device=self.config.device)
        v_link_pair = torch.tensor(self.v_link_pair, device=self.config.device)
        obs['v_net'] = Batch.from_data_list([Data(x=x_v_net, edge_index=v_link_pair)])

        return obs
