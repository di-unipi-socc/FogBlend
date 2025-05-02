# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: 
    from typing import List
    from fognetx.utils.types import Config, Solution, PhysicalNetwork, VirtualNetwork
# REGULAR IMPORTS
import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Batch


class Observation:
    """
    Observation class to represent the state of the environment.
    """
    def __init__(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, config: Config):
        
        self.config = config
        self.device = config.device

        # Physical network and virtual network
        self.p_net = p_net
        self.v_net = v_net

        # Node resources
        p_node_res = self.get_nodes_resources(p_net.net, p_net.node_resources)
        v_node_res = self.get_nodes_resources(v_net.net, v_net.node_resources)
        self.node_resources_normalization_value = config.p_net_max_node_resources

        self.p_nodes_resources = torch.tensor(
            p_node_res, dtype=torch.float32, device=self.device
        )
        self.v_nodes_resources = torch.tensor(
            v_node_res, dtype=torch.float32, device=self.device
        )

        # Link resources (max, sum, mean) 
        p_link_res = self.get_links_resources(p_net.net, p_net.link_resources)
        v_link_res = self.get_links_resources(v_net.net, v_net.link_resources)
        self.link_resources_normalization_value = config.p_net_max_link_resources

        self.p_nodes_link_resources = torch.tensor(
            p_link_res, dtype=torch.float32, device=self.device
        )
        self.v_nodes_link_resources = torch.tensor(
            v_link_res, dtype=torch.float32, device=self.device
        )

        # Link pair (edge_index format)
        self.p_link_pair = torch.tensor(
            self.get_link_pair(p_net.net), dtype=torch.long, device=self.device
        )
        self.v_link_pair = torch.tensor(
            self.get_link_pair(v_net.net), dtype=torch.long, device=self.device
        )

        # Node status flags (initially 0 = unused/unplaced)
        self.p_nodes_status = torch.zeros(
            (p_net.net.number_of_nodes(), 1), dtype=torch.float32, device=self.device
        )
        self.v_nodes_status = torch.zeros(
            (v_net.net.number_of_nodes(), 1), dtype=torch.float32, device=self.device
        )

        # Distance matrix (Floyd-Warshall)
        self.p_net_distance_matrix = torch.tensor(
            nx.floyd_warshall_numpy(p_net.net), dtype=torch.float32, device=self.device
        )

        # Average distance 
        self.average_distance = torch.zeros(
            (p_net.net.number_of_nodes(), 1), dtype=torch.float32, device=self.device
        )

        # Node degrees (centrality)
        centrality_values = list(nx.degree_centrality(p_net.net).values())
        self.p_node_degrees = torch.tensor(
            np.array(centrality_values, dtype=np.float32).reshape(-1, 1),
            dtype=torch.float32,
            device=self.device
        )

        # Virtual network info
        self.v_net_size = v_net.net.number_of_nodes()
        self.v_num_placed_nodes = 0  # Initially, no nodes in the solution

        # Physical network info
        self.p_net_size = p_net.net.number_of_nodes()


    def get_nodes_resources(self, net: nx.Graph, node_resources: list[str], nodes=None) -> np.ndarray:
        """
        Get the resources of the nodes 

        Args:
            net: Physical network graph.
            node_resources: Node resources.
            nodes: Nodes to consider. If None, all nodes are considered.

        Returns:
            A matrix of node resources [num_nodes x num_resources].
        """
        if nodes is None:
            nodes = sorted(net.nodes())  # consistent ordering

        num_node_resources = len(node_resources)
        node_resources_matrix = np.zeros((len(nodes), num_node_resources), dtype=np.float32)

        for i, node in enumerate(nodes):
            for j, resource in enumerate(node_resources):
                node_resources_matrix[i, j] = net.nodes[node][resource]

        return node_resources_matrix
    

    def get_links_resources(self, net: nx.Graph, link_resources: list[str], nodes=None) -> np.ndarray:
        """
        Get maximum, sum and mean resources of the links of each node
        
        Args:
            net: Physical network graph.
            link_resources: Link resources.
            nodes: Nodes to consider. If None, all nodes are considered.
        
        Returns:
            A matrix of link resources [num_nodes x num_resources x 3].
        """
        if nodes is None:
            nodes = sorted(net.nodes())  # consistent ordering

        num_nodes = len(nodes)
        num_link_resources = len(link_resources)

        # Initialize a matrix to store link resources
        link_resources_matrix = np.zeros((num_nodes, num_link_resources, 3))

        # Fill the matrix with link resources
        for i, node in enumerate(nodes):
            # Get the edges connected to the node
            edges = net.edges(node)
            for j, resource in enumerate(link_resources):
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
        obs = {}

        # Physical network observation
        v_net_size = torch.full(
            (self.p_net_size, 1),
            fill_value=self.v_net_size / self.config.v_net_max_size,
            dtype=torch.float32,
            device=self.device
        )

        v_num_placed_nodes = torch.full(
            (self.p_net_size, 1),
            fill_value=self.v_num_placed_nodes / self.config.v_net_max_size,
            dtype=torch.float32,
            device=self.device
        )

        # Normalize the node resources
        p_nodes_resources = self.p_nodes_resources / self.config.p_net_max_node_resources

        # Normalize the link resources
        p_nodes_link_max = self.p_nodes_link_resources[:, :, 0] / self.config.p_net_max_link_resources
        p_nodes_link_sum = self.p_nodes_link_resources[:, :, 1] / self.config.p_net_max_link_resources
        p_nodes_link_mean = self.p_nodes_link_resources[:, :, 2] / self.config.p_net_max_link_resources

        x_p_net = torch.cat((
            p_nodes_resources,
            v_net_size,
            v_num_placed_nodes,
            self.p_nodes_status,
            p_nodes_link_max,
            p_nodes_link_sum,
            p_nodes_link_mean,
            self.p_node_degrees,
            self.average_distance
        ), dim=-1)

        obs['p_net'] = Batch.from_data_list([
            Data(x=x_p_net.clone(), edge_index=self.p_link_pair)  # Clone to avoid in-place resources modification
        ])

        # Virtual network observation
        v_net_size_v = torch.full(
            (self.v_net_size, 1),
            fill_value=self.v_net_size / self.config.v_net_max_size,
            dtype=torch.float32,
            device=self.device
        )

        v_num_placed_nodes_v = torch.full(
            (self.v_net_size, 1),
            fill_value=self.v_num_placed_nodes / self.config.v_net_max_size,
            dtype=torch.float32,
            device=self.device
        )

        # Normalize the node resources
        v_nodes_resources = self.v_nodes_resources / self.config.p_net_max_node_resources  # Normalize with max physical resources

        # Normalize the link resources
        v_nodes_link_max = self.v_nodes_link_resources[:, :, 0] / self.config.p_net_max_link_resources
        v_nodes_link_sum = self.v_nodes_link_resources[:, :, 1] / self.config.p_net_max_link_resources
        v_nodes_link_mean = self.v_nodes_link_resources[:, :, 2] / self.config.p_net_max_link_resources

        x_v_net = torch.cat((
            v_nodes_resources,
            v_net_size_v,
            v_num_placed_nodes_v,
            self.v_nodes_status,
            v_nodes_link_max,
            v_nodes_link_sum,
            v_nodes_link_mean
        ), dim=-1)

        obs['v_net'] = Batch.from_data_list([
            Data(x=x_v_net.clone(), edge_index=self.v_link_pair)
        ])

        return obs


    def get_mask(self) -> torch.Tensor:
        """
        Get the mask of the environment. The mask is a binary tensor indicating which actions are valid.

        Returns:
            A torch tensor of shape [1, num_v_nodes, num_p_nodes] indicating valid actions.
        """
        # Expand dims for broadcasting
        v_resources_expanded = self.v_nodes_resources.unsqueeze(1)  # [num_v_nodes, 1, num_node_resources]
        p_resources_expanded = self.p_nodes_resources.unsqueeze(0)  # [1, num_p_nodes, num_node_resources]

        # Compare: physical node >= virtual node
        resource_comparison = p_resources_expanded >= v_resources_expanded

        # Valid if all physical resources satisfy virtual demands
        node_constraints_mask = resource_comparison.all(dim=2).float()  # [num_v_nodes, num_p_nodes]

        # Enforce non-reusability of physical nodes if configured
        if not self.config.reusable:
            used_p_nodes = (self.p_nodes_status > 0).squeeze(dim=-1)  # [num_p_nodes]
            node_constraints_mask[:, used_p_nodes] = 0.0

        # Mask already placed virtual nodes
        if self.v_num_placed_nodes > 0:
            placed_v_nodes = (self.v_nodes_status > 0).squeeze(dim=-1)
            node_constraints_mask[placed_v_nodes, :] = 0.0

        # Link constraint using bandwidth
        bw_index = self.config.link_resources.index('bandwidth')
        p_link_max = self.p_nodes_link_resources[:, bw_index, 0]  # [num_p_nodes]
        v_link_max = self.v_nodes_link_resources[:, bw_index, 0]  # [num_v_nodes]

        # Expand for broadcasting
        v_link_expanded = v_link_max.unsqueeze(1)  # [num_v_nodes, 1]
        p_link_expanded = p_link_max.unsqueeze(0)  # [1, num_p_nodes]

        # Link constraint mask
        link_mask = (p_link_expanded >= v_link_expanded).float()  # [num_v_nodes, num_p_nodes]

        # Combine both constraints
        mask = node_constraints_mask * link_mask  # [num_v_nodes, num_p_nodes]

        # Add batch dimension
        return mask.unsqueeze(0)  # [1, num_v_nodes, num_p_nodes]
    

    def update_observation(self, involved_nodes: dict[List, List], solution: Solution, event_type) -> None:
        """
        Update the observation of the environment considering only the involved nodes.

        Args:
            involved_nodes: The set of involved nodes in the solution.
            solution: The solution object associated with the virtual network.
            event_type: The type of the event (arrival or leave).
        
        Returns:
            None
        """
        # Get node resources of the involved nodes
        involved_nodes_resources = self.get_nodes_resources(
            self.p_net.net, self.p_net.node_resources, involved_nodes['place']
        )

        # Update the tensor of involved nodes resources
        self.p_nodes_resources[involved_nodes['place']] = torch.tensor(involved_nodes_resources, dtype=torch.float32, device=self.device)

        # Get link resources of the involved nodes
        involved_node_route = list(involved_nodes['route'])
        involved_nodes_link_resources = self.get_links_resources(
            self.p_net.net, self.p_net.link_resources, involved_node_route
        )

        # Update the tensor of involved nodes link resources
        self.p_nodes_link_resources[involved_node_route] = torch.tensor(involved_nodes_link_resources, dtype=torch.float32, device=self.device)

        if event_type == 'arrival':
            # Update the p status: add one to the node used
            chosen_node = involved_nodes['place'][0]
            self.p_nodes_status[chosen_node][0] += 1.0

            # Update the v status: set to 1 the node placed
            for node in solution.node_mapping.keys():
                self.v_nodes_status[node][0] = 1.0

            # Update the number of placed nodes
            self.v_num_placed_nodes = len(solution.node_mapping)

            # Convert used physical nodes to a tensor
            p_nodes_used = list(set(elem[0] for elem in solution.node_mapping.values()))
            used_p_nodes_tensor = torch.tensor(p_nodes_used, dtype=torch.int, device=self.device)

            # Select distances from all physical nodes to the used physical nodes (shape: [num_p_nodes, num_used_nodes])
            distances_to_used = self.p_net_distance_matrix[:, used_p_nodes_tensor]

            # Compute the mean across the used nodes (dim=1), result shape: [num_p_nodes]
            average_distances = distances_to_used.mean(dim=1)

            # Store as column vector: [num_p_nodes, 1]
            self.average_distance = average_distances.unsqueeze(1)
        else:
            # Update the status: subtract the number of times each node was used
            for node, _ in solution.node_mapping.values():
                self.p_nodes_status[node][0] -= 1.0
            # Other field will be updated in the update_v_net method
            

    def update_v_net(self, v_net: VirtualNetwork) -> None:
        """
        Update the virtual network.

        Args:
            v_net: The new virtual network.
        
        Returns:
            None
        """
        # Update the virtual network and its size
        self.v_net = v_net
        self.v_net_size = v_net.net.number_of_nodes()
        self.v_num_placed_nodes = 0

        # Node resources
        v_node_res = self.get_nodes_resources(v_net.net, v_net.node_resources)
        self.v_nodes_resources = torch.tensor(
            v_node_res, dtype=torch.float32, device=self.device
        )

        # Link resources (max, sum, mean) 
        v_link_res = self.get_links_resources(v_net.net, v_net.link_resources)
        self.v_nodes_link_resources = torch.tensor(
            v_link_res, dtype=torch.float32, device=self.device
        )

        # Link pair (edge_index format)
        self.v_link_pair = torch.tensor(
            self.get_link_pair(v_net.net), dtype=torch.long, device=self.device
        )

        # Node status flags (initially 0 = unplaced)
        self.v_nodes_status = torch.zeros(
            (v_net.net.number_of_nodes(), 1), dtype=torch.float32, device=self.device
        )

        # Reset the average distance
        self.average_distance = torch.zeros(
            (self.average_distance.shape[0], 1), dtype=torch.float32, device=self.device
        )

    
    def release_v_net(self, solution: Solution) -> None:
        """
        Release the virtual network and update the physical network resources.

        Args:
            solution: The solution object associated with the virtual network.
        
        Returns:
            None
        """
        # Find the involved nodes in the solution
        involved_nodes = {}
        involved_nodes['place'] = list(set(elem[0] for elem in solution.node_mapping.values()))
        
        # Extract unique nodes in routing paths
        unique_nodes = set()
        for links, _ in solution.link_mapping.values():
            for u, v in links:
                unique_nodes.add(u)
                unique_nodes.add(v)
        involved_nodes['route'] = list(unique_nodes)

        # Update the observation of the physical network
        self.update_observation(involved_nodes, solution, 'leave')
