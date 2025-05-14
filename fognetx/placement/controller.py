# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from typing import Dict; from fognetx.utils.types import Config, Solution, PhysicalNetwork, VirtualNetwork, Observation
# REGULAR IMPORTS
import networkx as nx
import fognetx.utils as utils
from collections import deque


def place_and_route(v_net: VirtualNetwork, p_net: PhysicalNetwork, v_node_id, p_node_id, 
                    solution: Solution, observations: Observation, config: Config, req_feasibility=True) -> None:
    """Take a step in the environment updating the solution. If placement and routing are 
    successful, update the physical network and the observations.

    Args:
        v_net: The virtual network.
        p_net: The physical network.
        v_node_id: The selected high-level action (virtual node id).
        p_node_id: The selected low-level action (physical node id).
        solution: The solution object updated with the current mapping.
        observations: The observations object containing the current state of the environment.
        config: The environment configuration.
        req_feasibility: If True, check resource feasibility before taking the step.
    """
    # Check if the action is feasible (node resources)   
    placement_feasible = check_feasibility(v_node_id, p_node_id, observations) 

    if not placement_feasible:
        # Failed placement
        solution.place_result = False
        if req_feasibility: 
            return None
        
    # Get links to route
    links_to_route = [    # list of tuples ( (physical link), (virtual link), bandwidth )
        (
            (p_node_id, solution.node_mapping[neighbor][0]) if p_node_id < solution.node_mapping[neighbor][0]
            else (solution.node_mapping[neighbor][0], p_node_id),
            (v_node_id, neighbor),
            v_net.net.edges[v_node_id, neighbor]['bandwidth']
        )
        for neighbor in v_net.net.neighbors(v_node_id)
        if neighbor in solution.node_mapping
    ]

    # Order the links to route by decreasing bandwidth
    links_to_route.sort(key=lambda x: x[2], reverse=True) 

    # Temporary variables
    used_bw = {}
    routing_paths = {}
    involved_nodes = {'place': [p_node_id], 'route': set()}

    # Route each link 
    for (src, dst), v_link, required_bw in links_to_route:
        # Route the link using BFS
        route = route_link_bfs(v_net, p_net, v_link, (src, dst), used_bw)

        if route is None:
            # Failed routing
            solution.route_result = False
            if not req_feasibility:
                # If no feasible route is found but check_feasibility is False, find shortest path. 
                # Graph is connected, path always exists.
                route = nx.dijkstra_path(p_net.net, src, dst)
            else:
                return None
        
        # Add the nodes in the route to the involved nodes
        involved_nodes['route'].update(route)  
        
        # Convert the route to a list of couples
        route = utils.path_to_links(route)
        
        # Successful routing, update routing paths
        routing_paths[v_link] = (route, {'bandwidth': required_bw})   # e.g. {(0,1): ([(110,99),(99,50)], {'bandwidth':20})}
        
        # Updated used bandwidth
        for couple in route:
            # Ignore self-loops
            if couple[0] == couple[1]:
                continue
            # Order the couple to avoid duplicates
            couple = (couple[0], couple[1]) if couple[0] < couple[1] else (couple[1], couple[0])
            # Check if the couple is already in used_bw
            if couple not in used_bw:
                # Add the couple to used_bw with the required bandwidth
                used_bw[couple] = required_bw
            else:
                # Update the bandwidth of the couple
                used_bw[couple] += required_bw

    # Successful placement and routing, update the solution
    resources_used = {resource: v_net.net.nodes[v_node_id][resource] for resource in config.node_resources}
    solution.node_mapping[v_node_id] = (p_node_id, resources_used)  
    solution.link_mapping.update(routing_paths)  
    solution.place_result = solution.place_result and True  # True only if previous steps were successful
    solution.route_result = solution.route_result and True  
    
    # Update node resources in the physical network
    for resource in config.node_resources:
        p_net.net.nodes[p_node_id][resource] -= v_net.net.nodes[v_node_id][resource]
    
    # Update the bandwidth of the physical link
    for couple, bw in used_bw.items():
        p_net.net.edges[couple]['bandwidth'] -= bw

    # Update observations
    observations.update_observation(involved_nodes, solution, 'arrival')

    return None


def route_link_bfs(v_net: VirtualNetwork, p_net: PhysicalNetwork, v_link, p_link, used_bw = None) -> list:
    """Route a link between two nodes in the physical network using BFS.
    
    Args:
        v_net: The virtual network.
        p_net: The physical network.
        v_link: The selected virtual link.
        p_link: The selected physical link.
        used_bw: The bandwidth used in the physical network for the partial solution.
        
    Returns:
        list: A list of nodes representing the path from source to target in the physical network.
    """
    source, target = p_link  # Unpack the physical link

    # Dictionary to track visited nodes
    visited_nodes = {source: True}  

    # Initialize queue with (node, path taken)
    Q = deque()
    Q.append((source, [source]))  # Start with source node and a path containing only itself

    # Start BFS
    while Q:
        current_node, current_path = Q.popleft()  # Pop the first element in the queue
        
        # If we reached the target, return the path
        if current_node == target:
            return current_path
        
        for neighbor in p_net.net.neighbors(current_node):
            # Check if candidate physical link is feasible
            check_result = check_link_feasibility(v_net, p_net, v_link, (current_node, neighbor), used_bw)
            # If the neighbor is not visited and the link is feasible
            if check_result and (neighbor not in visited_nodes): 
                # Add the neighbor to visited nodes and to the queue
                visited_nodes[neighbor] = True
                Q.append((neighbor, current_path + [neighbor]))  # List concatenation create a new list

    # If no path is found, return None
    return None
    
    
def check_feasibility(v_node_id, p_node_id, observations: Observation) -> bool:
    """
    Check if the selected actions are feasible.

    Args:
        v_node_id: The selected high-level action (virtual node id).
        p_node_id: The selected low-level action (physical node id).
        observations: The observations object containing the current state of the environment.

    Returns:
        A boolean indicating if the actions are feasible.
    """
    # Get resources of the selected nodes
    v_node_resources = observations.v_nodes_resources[v_node_id]
    p_node_resources = observations.p_nodes_resources[p_node_id]

    # Check if the selected physical node has enough resources to accommodate the virtual node
    return (p_node_resources >= v_node_resources).all()


def check_link_feasibility(v_net: VirtualNetwork, p_net: PhysicalNetwork, v_link, p_link, used_bw: dict = None) -> bool:
    """
    Check if the selected link is feasible.

    Args:
        v_net: The virtual network.
        p_net: The physical network.
        v_link: The selected virtual link.
        p_link: The selected physical link.
        used_bw: The bandwidth used in the physical network for the partial solution.

    Returns:
        A boolean indicating if the link is feasible.
    """
    # Get already used bandwidth
    if used_bw is None:
        used_bw = {}

    # Sort the physical link to avoid duplicates and get the used bandwidth
    p_link_sorted = (p_link[0], p_link[1]) if p_link[0] < p_link[1] else (p_link[1], p_link[0])
    p_link_used_bw = used_bw.get(p_link_sorted, 0)

    # Get physical and virtual link
    p_link = p_net.net.edges[p_link]
    v_link = v_net.net.edges[v_link]

    # Check if the physical link has enough bandwidth
    if p_link['bandwidth'] - p_link_used_bw >= v_link['bandwidth']:
        return True
    else:
        return False
    

def add_resources_solution(p_net: PhysicalNetwork, solution: Solution) -> None:
    """
    Add resources back to the physical network after a virtual network request is removed.

    Args:
        p_net: The physical network.
        solution: The solution object updated with the current mapping.
    """
    # Iterate over the node mapping and add resources back to the physical network
    for v_node_id, (p_node_id, resources_used) in solution.node_mapping.items():
        # Add resources back to the physical network
        for resource, value in resources_used.items():
            p_net.net.nodes[p_node_id][resource] += value
    
    # Iterate over the link mapping and add bandwidth back to the physical network
    for v_link, (route, resources) in solution.link_mapping.items():
        # Add bandwidth back to the physical network
        for couple in route:
            # Ignore self-loops
            if couple[0] == couple[1]:
                continue
            p_net.net.edges[couple]['bandwidth'] += resources['bandwidth']


def rollback(p_net: PhysicalNetwork, solution: Solution, observation: Observation) -> None:
    """
    Rollback the solution by removing the resources allocated to the virtual network request.

    Args:
        p_net: The physical network.
        solution: The solution object updated with the current mapping.
    """
    # Add resources back to the physical network
    add_resources_solution(p_net, solution)
    
    # Update observations
    observation.release_v_net(solution)


def apply_solution(p_net: PhysicalNetwork, solution: Solution) -> None:
    """
    Apply the solution to the physical network.
    
    Args:
        p_net (PhysicalNetwork): The physical network object.
        solution (Solution): The solution object containing the mapping.
    """
    # Get the node and link mappings from the solution
    node_mapping = solution.node_mapping
    link_mapping = solution.link_mapping

    # Update the physical node with the resources
    for v, (p, resources) in node_mapping.items():
        for resource in resources.keys():
            # Update the physical node with the resources
            p_net.net.nodes[p][resource] -= resources[resource]

    # Update the physical links with the bandwidth
    for (u, v), (links, data) in link_mapping.items():
        for p_u, p_v in links:
            # Ignore self-loops
            if p_u == p_v:
                continue
            # Update the bandwidth of the physical link
            p_net.net.edges[p_u, p_v]['bandwidth'] -= data['bandwidth']


def update_p_net_state(p_net: PhysicalNetwork, solution: Solution, observation: Observation) -> None:
    """
    Update the state of the physical network and observation based on the solution.

    Args:
        p_net (PhysicalNetwork): The physical network object.
        solution (Solution): The solution object containing the mapping.
        observation (Observation): The observation object containing the current state of the environment.
    """
    # Update resources in the physical network
    apply_solution(p_net, solution)

    # Get involved nodes from the solution
    involved_nodes = {}
    involved_nodes['place'] = list(elem[0] for elem in solution.node_mapping.values())
        
    # Extract unique nodes in routing paths
    unique_nodes = set()
    for links, _ in solution.link_mapping.values():
        for u, v in links:
            unique_nodes.add(u)
            unique_nodes.add(v)
    involved_nodes['route'] = list(unique_nodes)

    # Update observations
    observation.p_net = p_net
    observation.update_observation(involved_nodes, solution, 'arrival')
