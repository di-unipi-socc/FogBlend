import fognetx.environment.controller as controller
from fognetx.config import Config
from fognetx.utils.recorder import Recorder
from fognetx.environment.solution import Solution
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
        self.observations = {}
        self.recoder: Recorder = None
        self.solutions: dict[int, Solution] = {}


    def create_env(self, epoch):
        """Create the environment generating the physical network and virtual network requests
        
        Args:
            epoch: The current epoch of the training.
        """
        # Generate the physical network and virtual network requests
        self.p_net.generate_p_net()
        self.requests.generate_requests()

        # Reset the request index
        self.request_index = 0

        # Initialize the first request
        request = self.requests.get_request_by_id(0)
        self.current_v_net = request['v_net']
        self.event_type = request['event_type']

        # Initialize observations
        self.observations = Observation(self.p_net, self.current_v_net, self.env_config)

        # Initialize recorder
        self.recorder = Recorder(epoch, self.p_net.num_nodes, self.env_config)

        # Initialize solution
        self.solutions = {}
        self.solutions[self.current_v_net.id] = Solution(self.request_index, self.event_type, self.p_net, self.current_v_net, self.env_config)


    def get_observations(self):
        """Get the current observations of the environment
        
        Returns:
            obs: The current observations of the environment.
            mask: The action mask if applicable.
        """
        mask = self.observations.get_mask() if self.env_config.mask_actions else None
        return self.observations.get_observation(), mask
        

    def step(self, v_node_id, p_node_id, check_feasibility=True):
        """Take a step in the environment

        Args:
            v_node_id: The selected high-level action (virtual node id).
            p_node_id: The selected low-level action (physical node id).
            check_feasibility: If True, check resource feasibility before taking the step.

        Returns:
            reward: The reward received after taking the action.
            done: A boolean indicating if the episode is done.
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

        # If the request is fully placed and routed, compute the information and log the solution
        if done:
            # Compute information for the solution
            solution.compute_info()
            # Log the solution (arrival request)
            solution.log()

        # Calculate the reward
        reward = self.calculate_reward(solution, fully_placed)

        # Update recorder
        self.recorder.step_update(reward, solution, done)    

        # If the request is fully placed and routed, move to the next request
        if done:
            # Process all leaving requests until the next arrival
            self.current_v_net, self.event_type = self.go_next_arrival()
            # Create new solution for the next request (if exists)
            if self.current_v_net is not None:
                self.solutions[self.current_v_net.id] = Solution(self.request_index, self.event_type, self.p_net, self.current_v_net, self.env_config)
        
        return reward, done
    

    def go_next_arrival(self):
        """Move to the next arrival request. All leaving requests are processed.
        
        Returns:
            current_v_net: The virtual network in the next arrival request.
            event_type: The type of the event (arrival).
        """
        while True:
            # Procede to the next request
            self.request_index += 1
            # Check if next request exists
            if self.request_index >= self.requests.num_requests:
                return None, None
            # Get the next request
            request = self.requests.get_request_by_id(self.request_index)
            current_v_net = request['v_net']
            event_type = request['event_type']

            # If the request is an arrival, update the current virtual network and return
            if event_type == 'arrival':
                self.observations.update_v_net(current_v_net)
                return current_v_net, event_type
            
            # If the request is a leave, but was not placed, skip it
            if event_type == 'leave' and current_v_net.id not in self.solutions:
                continue

            # Create a new solution for the leaving request
            solution = Solution(self.request_index, event_type, self.p_net, current_v_net, self.env_config)

            # Log the solution (leaving request)
            solution.log()

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
            result: The result of the action taken.
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