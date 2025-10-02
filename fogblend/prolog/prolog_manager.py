# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from typing import List, Tuple, Dict; from fogblend.utils.types import PhysicalNetwork, VirtualNetwork, Solution, Config
# REGULAR IMPORTS
import os
import janus_swi as janus  # type: ignore
import fogblend.utils.utils as utils
import fogblend.placement.controller as controller
import fogblend.prolog.utils_prolog as utils_prolog
from collections import defaultdict, OrderedDict
from fogblend.config import FOGBRAINX_DIR
from fogblend.prolog import manager_instance


class PrologManager:

    def __init__(self, config: Config):
        self.p_net: PhysicalNetwork = None
        self.v_net: VirtualNetwork = None
        self.previous_placement: List[Tuple] = None
        self.link_mapping: Dict = OrderedDict()
        self.config = config
        

    def prepare_fogbrainx(self):
        """
        Prepare the Prolog environment for FogBrainX by consulting necessary files containing the Prolog code 
        and the python manager.
        """
        # Compose all the paths to the Prolog files and the python manager
        base_path = os.path.join(FOGBRAINX_DIR, 'fogbrainx.pl')
        placer_path = os.path.join(FOGBRAINX_DIR, 'placers', 'placer-heu.pl')
        check_path = os.path.join(FOGBRAINX_DIR, 'placers', 'requirements-check-py.pl')
        manager_path = os.path.join('fogblend', 'prolog')

        # Heuristic path
        if self.config.heuristic == 'bw':
            heu_path = os.path.join(FOGBRAINX_DIR, 'placers', 'heuristics', 'bw-heu.pl')
        elif self.config.heuristic == 'hw':
            heu_path = os.path.join(FOGBRAINX_DIR, 'placers', 'heuristics', 'hw-heu.pl')
        else:
            raise ValueError(f"Unknown heuristic: {self.config.heuristic}")
    
        # Consult the Prolog files and add the python manager
        janus.query_once(f"consult('{base_path}').")
        janus.query_once(f"consult('{placer_path}').")
        janus.query_once(f"consult('{heu_path}').")
        janus.query_once(f"consult('{check_path}').")
        janus.query_once(f"py_add_lib_dir('{manager_path}'). py_import(utils_prolog, []).")


    def load_infr(self, filepath):
        """
        Consult the Prolog infrastructure file
        
        Args:
            filepath (str): Path to the Prolog infrastructure file.
        """
        # Check if the file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} does not exist.")
            
        # Consult the Prolog infrastructure file
        janus.query_once(f"consult('{filepath}').")


    def update_p_net(self, p_net: PhysicalNetwork):
        """
        Update the Prolog environment with a new physical network.
        
        Args:
            p_net (PhysicalNetwork): The new physical network object.
        """
        self.p_net = p_net

        # Clear the Prolog infrastructure
        janus.query_once("retractall(node(_, _, _, _)).")
        janus.query_once("retractall(link(_, _, _, _)).")

        # Add all the nodes
        for node, attribute in self.p_net.net.nodes(data=True):
            cpu = attribute['cpu']
            gpu = attribute['gpu']
            storage = attribute['storage']
            janus.query_once(f"assert(node({node}, [], ({cpu}, {gpu}, {storage}), [])).")

        # Add all the edges
        for u, v, attribute in self.p_net.net.edges(data=True):
            bandwidth = attribute['bandwidth']
            janus.query_once(f"assert(link({u}, {v}, 0, {bandwidth})).")
            janus.query_once(f"assert(link({v}, {u}, 0, {bandwidth})).")

        
    def update_v_net(self, v_net: VirtualNetwork):
        """
        Update the Prolog environment with a new virtual network.
        """
        self.v_net = v_net

        # Clear placement and link mapping
        self.previous_placement = None
        self.link_mapping = OrderedDict()
        self.current_alloc_bw = defaultdict(lambda: defaultdict(int))

        # Clear the Prolog application
        janus.query_once("retractall(application(_, _)).")
        janus.query_once("retractall(service(_, _, _, _)).")
        janus.query_once("retractall(s2s(_, _, _, _)).")

        # Add all the nodes
        for node, attribute in self.v_net.net.nodes(data=True):
            cpu = attribute['cpu']
            gpu = attribute['gpu']
            storage = attribute['storage']
            janus.query_once(f"assert(service({node}, [], ({cpu}, {gpu}, {storage}), [])).")

        # Add all the links
        for u, v, data in self.v_net.net.edges(data=True):
            bw = data['bandwidth']
            janus.query_once(f"assert(s2s({u}, {v}, 0, {bw})).")

        # Add the application
        all_nodes = list(self.v_net.net.nodes())
        janus.query_once(f"assert(application(id, {all_nodes})).")


    def call_fogbrainx(self):
        """
        Call the FogBrainX Prolog function to perform the placement.
        The placement is returned as a string.
        """
        return janus.query_once("placement(A, _P), term_string(_P, P)")


    def call_hybrid_fogbrainx(self):
        """
        Call the hybrid FogBrainX Prolog function to correct the placement.
        The placement is returned as a string.
        """
        return janus.query_once(f"fogBrainX(id, _P), term_string(_P, P)")
    

    def assert_neural_solution(self, solution: Solution):
        """
        Assert the solution returned by the neural agent into the Prolog environment.

        Args:
            solution: The solution returned by neural agent.
        """
        placement = solution.node_mapping

        # Convert the placement to Prolog format
        prolog_placement = []
        for vir_node, data in placement.items():
            p_node, _ = data 
            prolog_placement.append(f"on({vir_node},{p_node})")
        prolog_placement_str = "[" + ",".join(prolog_placement) + "]"

        # Clear the previous placement
        janus.query_once("retractall(deployment(_, _, _)).")

        # Add the placement to the prolog
        janus.query_once(f"assert(deployment(id,{prolog_placement_str}, ([],[]))).")
    

    def check_bw(self, placement):
        """
        Check if a new placement is valid based on bandwidth constraints.
        Updates link resources only for changed parts.

        Args:
            placement: List of tuples (virtual_node, physical_node)

        Returns:
            1 if placement is valid, 0 otherwise.
        """

        # Early return cases (only one node or all nodes on the same physical node)
        if len(placement) <= 1 or len(set(p[1] for p in placement)) == 1:
            self._restore_bw(0)
            self.current_alloc_bw = defaultdict(lambda: defaultdict(int))
            self.previous_placement = placement
            self.link_mapping = utils_prolog.create_link_mapping(placement, self.v_net)
            return 1
        
        # Find divergence point between old and new placements
        index = 0
        min_length = min(len(self.previous_placement), len(placement))
        
        for i in range(min_length):
            if self.previous_placement[i] != placement[i]:
                index = i
                break
            index = i + 1

        # Restore bandwidth from previously placed virtual nodes beyond the divergence point
        self._restore_bw(index)

        # Truncate the previous placement
        self.previous_placement = self.previous_placement[:index]

        # Track new bandwidth allocations for this round
        temp_alloc_bw = defaultdict(lambda: defaultdict(int))

        for i in range(index, len(placement)):
            vir_node, src_node = placement[i]
            
            # Get relevant edges (connections to already placed nodes)
            relevant_edges = self._get_relevant_edges(vir_node)

            # Order the links to route by decreasing bandwidth
            relevant_edges.sort(key=lambda x: x[2], reverse=True)

            # Route each relevant virtual edge
            for v_link, tgt_node, bw_required in relevant_edges:

                if src_node == tgt_node:
                    self.link_mapping[v_link] = ([(src_node, tgt_node)], {'bandwidth': bw_required})
                    # No need to route to the same node
                    continue
                    
                # Order the link to route (to ensure same behavior as the neural agent)
                p_link = (src_node, tgt_node) if src_node < tgt_node else (tgt_node, src_node)

                # Search for a path in the physical network
                path = controller.route_link_bfs(self.v_net, self.p_net, v_link, p_link)

                # Check if a path was found
                if path is None:
                    # Rollback all temporary allocations
                    for vn, link_bws in temp_alloc_bw.items():
                        for (u, v), bw in link_bws.items():
                            self.p_net.net.edges[u, v]['bandwidth'] += bw
                    return 0
                
                # Update the link mapping
                self.link_mapping[v_link] = (utils.path_to_links(path), {'bandwidth': bw_required})

                # Apply bandwidth reduction
                for u, v in utils.path_to_links(path):
                    self.p_net.net.edges[u, v]['bandwidth'] -= bw_required
                    temp_alloc_bw[vir_node][(u, v)] += bw_required
        
        # Update the current allocation bw
        for vir_node, allocations in temp_alloc_bw.items():
            for (u, v), bw in allocations.items():
                self.current_alloc_bw[vir_node][(u, v)] += bw
        # Update previous placement
        self.previous_placement.extend(placement[index:])

        return 1
    

    def _restore_bw(self, index):
        """
        Restore bandwidth allocations for outdated placements.
        
        Args:
            index: Index where old and new placements start to differ
        """
        if self.previous_placement is None:
            return
        for i in range(index, len(self.previous_placement)):
            vir_node = self.previous_placement[i][0]
            if vir_node in self.current_alloc_bw:
                for (u, v), bw in self.current_alloc_bw[vir_node].items():
                    self.p_net.net.edges[u, v]['bandwidth'] += bw
                del self.current_alloc_bw[vir_node]


    def _get_relevant_edges(self, vir_node):
        """
        Identify all virtual edges originating from the given virtual node that connect to already placed virtual nodes.
        
        Args:
            vir_node: The virtual node currently being placed.
        
        Returns:
            A list of tuples in the format (v_link, target_p_node, bandwidth), where:
            - v_link is the virtual edge (vir_node, neighbor).
            - target_p_node is the physical node assigned to the virtual neighbor.
            - bandwidth is the bandwidth requirement of the virtual edge.
        """
        relevant_edges = []
        
        # Create lookup dict for quick access
        placement_dict = {v: p for v, p in self.previous_placement}
        
        # Check all neighbors of the virtual node
        for neighbor in self.v_net.net.neighbors(vir_node):
            if neighbor in placement_dict:
                v_link = (vir_node, neighbor)
                target_p_node = placement_dict[neighbor]
                bw = self.v_net.net.edges[vir_node, neighbor]['bandwidth']
                relevant_edges.append((v_link, target_p_node, bw))
        
        return relevant_edges
    

    def set_global_manager(self):
        """
        Set the global Prolog manager instance.
        """
        manager_instance.global_manager = self