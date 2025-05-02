# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config
# REGULAR IMPORTS
import numpy as np
import networkx as nx
from typing import TypedDict


class VirtualNetwork():
    """
    Class representing a virtual network.
    """

    def __init__(self, id, arrival_time, lifetime, node_resources, link_resources, net:nx.Graph):
        """
        Initialize a VirtualNetwork instance.
        """
        self.id = id
        self.arrival_time = arrival_time
        self.lifetime = lifetime
        self.net = net
        self.num_nodes = net.number_of_nodes()
        self.node_resources = node_resources
        self.link_resources = link_resources


class RequestDict(TypedDict):
    """
    Typed dictionary for virtual network requests.
    """
    id: int
    v_net: VirtualNetwork
    event_type: str
    time: int


class VirtualNetworkRequests():
    """
    Class representing a virtual network in the FogNetX environment.
    """

    def __init__(self, config: Config):
        """
        Initialize a VirtualNetwork instance.

        Args:
            config: Configuration object containing network parameters.
        """
        # Size
        self.num_v_net = config.num_v_net 
        self.num_requests = config.num_v_net * 2  # Each virtual network has an arrival and leave event
        self.min_size = config.v_net_min_size
        self.max_size = config.v_net_max_size
        # Resources
        self.min_node_resources = config.v_net_min_node_resources
        self.max_node_resources = config.v_net_max_node_resources
        self.min_link_resources = config.v_net_min_link_resources
        self.max_link_resources = config.v_net_max_link_resources
        self.node_resources = config.node_resources
        self.link_resources = config.link_resources
        # Requests
        self.arrival_rate = config.arrival_rate
        self.request_avg_lifetime = config.request_avg_lifetime
        self.requests = None
        # Random number generator
        self.rng = np.random.default_rng(config.seed)
        
        
    def generate_requests(self):
        """
        Generate a set of virtual network requests based on the configuration.
        """
        # Generate a random number of nodes for each virtual network request
        num_nodes = self.rng.integers(self.min_size, self.max_size, size=self.num_v_net, endpoint=True)

        # Generate the arrival times for each virtual network request (Poisson distribution)
        arrival_times = self.rng.poisson(lam=1/self.arrival_rate, size=self.num_v_net)
        # Generate the lifetimes for each virtual network request (Exponential distribution)
        lifetimes = self.rng.exponential(scale=self.request_avg_lifetime, size=self.num_v_net)
        lifetimes = np.maximum(lifetimes, 1) # Ensure lifetimes are at least 1
        # Compute cumulative arrival times
        cumulative_arrival_times = np.cumsum(arrival_times)

        # Generate the virtual networks
        event_list = []
        for i in range(self.num_v_net):
            v_net = self.generate_v_net(num_nodes[i], int(cumulative_arrival_times[i]), int(lifetimes[i]), i)
            # Add arrival event to the event list
            event_list.append({'id': 0, 'v_net': v_net, 'event_type': 'arrival', 'time': v_net.arrival_time})
            # Add leave event to the event list
            event_list.append({'id': 0, 'v_net': v_net, 'event_type': 'leave', 'time': v_net.arrival_time + v_net.lifetime})
        
        # Sort the event list by time
        event_list.sort(key=lambda x: x['time'])

        # Add a event id to each event
        for i, event in enumerate(event_list):
            event['id'] = i
            
        # Store the event list in the requests attribute
        self.requests = event_list
        

    def generate_v_net(self, num_nodes, arrival_time, lifetime, id) -> VirtualNetwork:
        """
        Generate a virtual network with a specified number of nodes.

        Args:
            num_nodes: Number of nodes in the virtual network.

        Returns:
            A virtualNetwork object representing the generated virtual network.
        """
        # Create a random graph for the virtual network and ensure it is connected
        while True:
            v_net = nx.erdos_renyi_graph(num_nodes, 0.5, directed=False, seed=int(self.rng.integers(0, 99999999)))
            if nx.is_connected(v_net):
                break

        # Assign random resources to the nodes of the virtual network
        num_node_resources = len(self.node_resources)

        node_resource_values = self.rng.integers(
            self.min_node_resources, self.max_node_resources, 
            size=(num_nodes, num_node_resources),
            endpoint=True
        )

        for i, node in enumerate(v_net.nodes()):
            for j, resource in enumerate(self.node_resources):
                v_net.nodes[node][resource] = int(node_resource_values[i, j])

        # Assign random resources to the edges of the virtual network
        num_edges = v_net.number_of_edges()
        num_link_resources = len(self.link_resources)

        link_resource_values = self.rng.integers(
            self.min_link_resources, self.max_link_resources,
            size=(num_edges, num_link_resources),
            endpoint=True
        )

        for i, edge in enumerate(v_net.edges()):
            for j, resource in enumerate(self.link_resources):
                v_net.edges[edge][resource] = int(link_resource_values[i, j])

        return VirtualNetwork(id=id, arrival_time=arrival_time, lifetime=lifetime, node_resources=self.node_resources, link_resources=self.link_resources, net=v_net)


    def get_request_by_id(self, id) -> RequestDict:
        """
        Get a virtual network request by its ID.

        Args:
            id: ID of the virtual network request.

        Returns:
            The virtual network request with the specified ID.
        """
        return self.requests[id]
