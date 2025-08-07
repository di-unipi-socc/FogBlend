from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fogblend.config import Config
    from fogblend.ppo.agent import PPOAgent
    from fogblend.placement.solution import Solution
    from fogblend.environment.observation import Observation
    from fogblend.environment.physicalNetwork import PhysicalNetwork
    from fogblend.environment.virtualNetworkRequets import VirtualNetwork
    from fogblend.environment.environment import Environment, TestEnvironment