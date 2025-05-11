from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fognetx.config import Config
    from fognetx.ppo.agent import PPOAgent
    from fognetx.placement.solution import Solution
    from fognetx.environment.observation import Observation
    from fognetx.environment.physicalNetwork import PhysicalNetwork
    from fognetx.environment.virtualNetworkRequets import VirtualNetwork
    from fognetx.environment.environment import Environment, TestEnvironment