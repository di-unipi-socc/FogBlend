from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fognetx.config import Config
    from fognetx.environment.solution import Solution
    from fognetx.environment.physicalNetwork import PhysicalNetwork
    from fognetx.environment.virtualNetworkRequets import VirtualNetwork
    from fognetx.environment.observation import Observation
    from fognetx.environment.environment import Environment
    from fognetx.ppo.agent import PPOAgent