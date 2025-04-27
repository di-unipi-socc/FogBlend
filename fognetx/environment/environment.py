from fognetx.environment.physicalNetwork import PhysicalNetwork
from fognetx.environment.virtualNetworkRequets import VirtualNetworkRequests
from fognetx.environment.observation import Observation


class Environment():
    """
    Class representing the environment. It contains the physical network, virtual network requests, 
    and the observation.
    """
    
    def __init__(self, env_config: dict):
        self.env_config = env_config
        self.p_net = PhysicalNetwork(self.env_config)
        self.requests = VirtualNetworkRequests(self.env_config)
        self.request_index = 0
        self.observations = {}

    def create_env(self):
        """Create the environment"""
        # Generate the physical network and virtual network requests
        self.p_net.generate_p_net()
        self.requests.generate_requests()
        # Initialize observations
        self.observations = Observation(self.p_net, self.requests.get_request_by_id(0)['v_net'], self.env_config)

    def get_observations(self):
        """Get the current observations of the environment"""
        return self.observations.get_observation()
        

    def step(self, high_action, low_action):
        """Take a step in the environment"""
        return None, None