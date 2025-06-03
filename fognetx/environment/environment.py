# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config, VirtualNetwork
# REGULAR IMPORTS
import fognetx.placement.controller as controller
from fognetx.utils.recorder import Recorder
from fognetx.placement.solution import Solution
from fognetx.environment.observation import Observation
from fognetx.environment.physicalNetwork import PhysicalNetwork
from fognetx.environment.virtualNetworkRequets import VirtualNetworkRequests


class Environment():
    """
    Class representing the environment. It contains the physical network, virtual network requests, 
    and the observation.
    """
    
    def __init__(self, env_config: Config):
        self.env_config = env_config
        self.p_net = PhysicalNetwork(self.env_config)
        self.requests = VirtualNetworkRequests(self.env_config)
        self.request_index = 0
        self.running_request = 0
        self.observations = {}
        self.epoch = None
        self.recoder: Recorder = None
        self.solutions: dict[int, Solution] = {}


    def create_env(self, epoch) -> None:
        """Create the environment generating the physical network and virtual network requests
        
        Args:
            epoch: The current epoch of the training.
        """
        # Generate the physical network and virtual network requests
        self.p_net.generate_p_net()
        self.requests.generate_requests(self.p_net.num_nodes)

        # Reset the request index
        self.request_index = 0

        # Initialize the first request
        request = self.requests.get_request_by_id(0)
        self.current_v_net = request['v_net']
        self.event_type = request['event_type']
        self.event_time = request['time']

        # Initialize observations
        self.observations = Observation(self.p_net, self.current_v_net, self.env_config)

        # Initialize epoch and recorder
        self.epoch = epoch
        self.recorder = Recorder(epoch, self.p_net.num_nodes, self.requests.arrival_rate, self.env_config)

        # Initialize solution
        self.solutions = {}
        self.solutions[self.current_v_net.id] = Solution(self.request_index, self.event_type, request['time'], self.p_net, self.current_v_net, self.env_config)


    def get_observations(self):
        """Get the current observations of the environment
        
        Returns:
            obs: The current observations of the environment.
            mask: The action mask if applicable.
        """
        return self.observations.get_observation(), self.observations.get_mask()
        

    def step(self, v_node_id, p_node_id, check_feasibility=True):
        """Take a step in the environment

        Args:
            v_node_id: The selected high-level action (virtual node id).
            p_node_id: The selected low-level action (physical node id).
            check_feasibility: If True, check resource feasibility before taking the step.

        Returns:
            reward: The reward received after taking the action.
            done: A boolean indicating if the request handling is terminated (failed or completed).
        """
        # Get current partial solution
        solution = self.solutions[self.current_v_net.id]

        # Try to place and route the selected virtual node in the physical network
        controller.place_and_route(self.current_v_net, self.p_net, v_node_id, p_node_id, 
                                            solution, self.observations, self.env_config, check_feasibility)

        # Check if the request is fully placed and routed
        fully_placed = True if len(solution.node_mapping.keys()) == self.current_v_net.num_nodes else False

        # Check if the request handling is terminated (failed or completed)
        if check_feasibility and not solution.is_feasible():
            # Rollback the solution
            controller.rollback(self.p_net, solution, self.observations)
            done = True
        else:
            done = fully_placed

        # If the request is done, compute the information and log the solution
        if done:
            # If solution is feasible, increase the running request count
            if solution.is_feasible():
                self.running_request += 1
            # Compute information for the solution
            solution.compute_info()
            # Log the solution (arrival request)
            solution.running_request = self.running_request
            solution.log(file_name=f"requestLog_{self.epoch}.csv")

        # Calculate the reward
        reward = self.calculate_reward(solution, fully_placed)

        # Update recorder
        self.recorder.step_update(reward, solution, done)    

        # If the request is done, move to the next request
        if done:
            # Process all leaving requests until the next arrival
            self.go_next_arrival()
           
        return reward, done
    

    def go_next_arrival(self, log_leave = True) -> None:
        """Move to the next arrival request. All leaving requests are processed.

        Args:
            log_leave: If True, log the leaving requests.
        """
        while True:
            # Procede to the next request
            self.request_index += 1
            # Check if next request exists
            if self.request_index >= self.requests.num_requests:
                return None
            # Get the next request
            request = self.requests.get_request_by_id(self.request_index)
            current_v_net = request['v_net']
            event_type = request['event_type']
            event_time = request['time']

            # If the request is an arrival, update state and add the solution
            if event_type == 'arrival':
                self.observations.update_v_net(current_v_net)
                self.current_v_net = current_v_net
                self.event_type = event_type
                self.event_time = event_time
                self.solutions[current_v_net.id] = Solution(self.request_index, event_type, event_time, self.p_net, current_v_net, self.env_config)
                return None
            
            # If the request is a leave, but was not placed, delete it without add resources back to the physical network
            if event_type == 'leave' and not self.solutions[current_v_net.id].is_feasible():
                del self.solutions[current_v_net.id]
                continue

            # Create a new solution for the leaving request
            solution = Solution(self.request_index, event_type, event_time, self.p_net, current_v_net, self.env_config)
            
            # Decrease the running request count
            self.running_request -= 1

            # Log the solution (leaving request)
            if log_leave:
                solution.running_request = self.running_request
                solution.log(file_name=f"requestLog_{self.epoch}.csv")

            # If the request is a leave, add resources back to the physical network
            controller.add_resources_solution(self.p_net, self.solutions[current_v_net.id])

            # Update the observations
            self.observations.release_v_net(self.solutions[current_v_net.id])

            # Remove the solution from the solutions dictionary
            del self.solutions[current_v_net.id]
        
   
    def calculate_reward(self, solution: Solution, done):
        """Calculate the reward based on the result of the action taken

        Args:
            solution: The current solution object.
            done: A boolean indicating if the episode is done.

        Returns:
            reward: The calculated reward.
        """
        # If single step is successful
        if solution.is_feasible():
            reward = 1 / self.current_v_net.num_nodes
        else:
            reward = -1 / self.current_v_net.num_nodes
        
        # If the request is fully placed and routed, give a bonus
        if done:
            reward += solution.r2c_ratio

        return reward
    

class TestEnvironment(Environment):
    """
    Class representing the test environment. It contains the physical network, virtual network requests, 
    and the observation.
    """
    
    def __init__(self, p_net: PhysicalNetwork, requests: VirtualNetworkRequests, env_config: Config):
        super().__init__(env_config)
        self.p_net = p_net
        self.requests = requests
        # Initialize the first request
        request = self.requests.get_request_by_id(0)
        self.current_v_net = request['v_net']
        self.event_type = request['event_type']
        self.event_time = request['time']
        self.solutions[self.current_v_net.id] = Solution(self.request_index, self.event_type, self.event_time, self.p_net, self.current_v_net, self.env_config)
        # Initialize observations
        self.observations = Observation(self.p_net, self.current_v_net, self.env_config)


    def step(self, v_node_id, p_node_id):
        """Take a step in the environment

        Args:
            v_node_id: The selected high-level action (virtual node id).
            p_node_id: The selected low-level action (physical node id).

        Returns:
            done: A boolean indicating if the request is fully placed and routed.
            solution: The solution found for the request.
        """
        # Get current partial solution
        solution = self.solutions[self.current_v_net.id]

        # Try to place and route the selected virtual node in the physical network
        controller.place_and_route(self.current_v_net, self.p_net, v_node_id, p_node_id, 
                                            solution, self.observations, self.env_config, req_feasibility=False)

        # Check if the request is fully placed and routed
        fully_placed = True if len(solution.node_mapping.keys()) == self.current_v_net.num_nodes else False

        # Check if the request handling is terminated 
        if fully_placed:
            # Compute information for the solution
            solution.compute_info()

            # If not feasible, rollback the solution (ghost solution)
            if not solution.is_feasible():
                controller.rollback(self.p_net, solution, self.observations)
            else:
                # If feasible, increase the running request count
                self.running_request += 1

            solution.running_request = self.running_request

            # Test logic must call go_next_arrival
            
            return True, solution
        
        # Request not fully placed and routed
        return False, solution
    

    def go_next_arrival(self, log_leave = True, update_obs = True) -> None:
        """Move to the next arrival request. All leaving requests are processed.

        Args:
            log_leave: If True, log the leaving requests.
        """
        while True:
            # Procede to the next request
            self.request_index += 1
            # Check if next request exists
            if self.request_index >= self.requests.num_requests:
                return None
            # Get the next request
            request = self.requests.get_request_by_id(self.request_index)
            current_v_net = request['v_net']
            event_type = request['event_type']
            event_time = request['time']

            # If the request is an arrival, update state and add the solution
            if event_type == 'arrival':
                if update_obs:
                    self.observations.update_v_net(current_v_net)
                self.current_v_net = current_v_net
                self.event_type = event_type
                self.event_time = event_time
                self.solutions[current_v_net.id] = Solution(self.request_index, event_type, event_time, self.p_net, current_v_net, self.env_config)
                return None
            
            # If the request is a leave, but was not placed, delete it without add resources back to the physical network
            if event_type == 'leave' and not self.solutions[current_v_net.id].is_feasible():
                del self.solutions[current_v_net.id]
                continue

            # Create a new solution for the leaving request
            solution = Solution(self.request_index, event_type, event_time, self.p_net, current_v_net, self.env_config)
            
            # Decrease the running request count
            self.running_request -= 1

            # Log the solution (leaving request)
            if log_leave:
                solution.running_request = self.running_request
                solution.log(file_name=f"requestLog_{self.epoch}.csv")

            # If the request is a leave, add resources back to the physical network
            controller.add_resources_solution(self.p_net, self.solutions[current_v_net.id])

            # Update the observations
            if update_obs:
                self.observations.release_v_net(self.solutions[current_v_net.id])

            # Remove the solution from the solutions dictionary
            del self.solutions[current_v_net.id]
    

    def apply_prolog_solution(self, p_net: PhysicalNetwork, solution: Solution) -> None:
        """Update the state of the physical network using the Prolog solution
        
        Args:
            p_net: The physical network to be updated.
        """
        # Apply the Prolog solution to the physical network
        controller.update_p_net_state(p_net, solution, self.observations)

        # Update solution
        self.solutions[solution.v_net_id] = solution

        # Update running request count
        self.running_request += 1
        
        # Update p_net
        self.p_net = p_net
