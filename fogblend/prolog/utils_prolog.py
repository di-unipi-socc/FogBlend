# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from typing import List, Tuple, Dict; from fogblend.utils.types import VirtualNetwork
# REGULAR IMPORTS
import os
import re
from collections import OrderedDict
from fogblend.prolog import manager_instance
from fogblend.environment.physicalNetwork import PhysicalNetwork


def generate_infr_file(p_net: PhysicalNetwork, save_dir, filename) -> None:
    """
    Generate a Prolog infrastructure file from the physical network.

    Args:
        p_net (PhysicalNetwork): The physical network object.
        save_dir (str): Directory to save the Prolog file.
        filename (str): Name of the Prolog file.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Open the file for writing
    with open(os.path.join(save_dir, filename), 'w') as f:
        # Add all the nodes
        for node, attribute in p_net.net.nodes(data=True):
            cpu = attribute['cpu']
            gpu = attribute['gpu']
            ram = attribute['ram']
            f.write(f"node({node}, [], ({cpu}, {gpu}, {ram}), []).\n")

        # Add all the edges
        for u, v, attribute in p_net.net.edges(data=True):
            bandwidth = attribute['bandwidth']
            f.write(f"link({u}, {v}, 0, {bandwidth}).\n")
            f.write(f"link({v}, {u}, 0, {bandwidth}).\n")


def parse_placement(placement_str: str) -> List[Tuple[int, int]]:
    """
    Convert a Prolog-style placement string to a Python list.

    Example input: "[on(4,29),on(3,49),on(2,49),on(1,49),on(0,49)]"
    Returns: [(4, 29), (3, 49), (2, 49), (1, 49), (0, 49)]
    """
    pattern = r'on\((\d+),(\d+)\)'
    mappings = [(int(v), int(p)) for v, p in re.findall(pattern, placement_str)]
    return mappings[::-1]


def check_bw(allocated_bw, placement):
    if manager_instance.global_manager is None:
        raise ValueError("global_manager not set")
    # Convert the placement string to a dictionary
    placement_dict = parse_placement(placement)
    return manager_instance.global_manager.check_bw(placement_dict)


def convert_placement(placement: List[Tuple], v_net: VirtualNetwork):
    """
    Convert a Python placement list to solution format.
    
    Example input: [(4, 29), (3, 49), (2, 49), (1, 49), (0, 49)]
    
    Returns: {0: (49, {'cpu': 1, 'gpu': 0, 'ram': 0}), ...}
    """

    # Create a dictionary to store the converted placement
    converted_placement = {}
    
    # Iterate through the placement list
    for v, p in placement:
        # Get the virtual node from the virtual network
        v_node = v_net.net.nodes[v]
        
        # Create a new entry in the dictionary
        converted_placement[v] = (p, {
            'cpu': v_node['cpu'],
            'gpu': v_node['gpu'],
            'ram': v_node['ram']
        })
    
    return converted_placement


def apply_solution(p_net: PhysicalNetwork, node_mapping: Dict, link_mapping: Dict) -> None:
    """
    Apply the placement to the physical network.
    
    Args:
        p_net (PhysicalNetwork): The physical network object.
        node_mapping (dict): The node_mapping dictionary.
        link_mapping (dict): The link mapping dictionary.
    """
    for v, (p, resources) in node_mapping.items():
        # Update the physical node with the resources
        p_net.net.nodes[p]['cpu'] -= resources['cpu']
        p_net.net.nodes[p]['gpu'] -= resources['gpu']
        p_net.net.nodes[p]['ram'] -= resources['ram']

    for (u, v), (links, data) in link_mapping.items():
        # Update the physical links with the bandwidth
        for p_u, p_v in links:
            p_net.net.edges[p_u, p_v]['bandwidth'] -= data['bandwidth']


def create_link_mapping(placement: List[Tuple], v_net: VirtualNetwork) -> Dict:
    """
    Create a link mapping from a placement where all virtual nodes are mapped to physical nodes
    that have direct links between them.
    
    Args:
        placement (list): The placement list.
        v_net (VirtualNetwork): The virtual network object.

    Returns:
        dict: The updated link mapping.
    """
    # Create a dictionary to store the link mapping
    link_mapping = OrderedDict()

    if len(placement) <= 1:
        # If there is only one node, return an empty mapping
        return link_mapping
    
    # Convert placement to a dictionary for easier lookup
    placement_dict = {v_node: p_node for v_node, p_node in placement}
    
    # Iterate through virtual links
    for u, v, data in v_net.net.edges(data=True):
        # Check if both nodes are in the placement
        if u in placement_dict and v in placement_dict:
            # Get physical nodes for u and v
            p_u = placement_dict[u]
            p_v = placement_dict[v]

            # Create a new entry in the link mapping
            link_mapping[(u, v)] = ([(p_u, p_v)], {'bandwidth': data['bandwidth']})
    
    return link_mapping
    