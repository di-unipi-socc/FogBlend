# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from typing import Tuple; from fognetx.utils.types import Config, PPOAgent, TestEnvironment
# REGULAR IMPORTS
import time
import copy
import fognetx.placement.controller as controller
from tqdm import tqdm
from multiprocessing import Process, Queue
from fognetx.prolog import utils_prolog
from fognetx.placement.solution import Solution
from fognetx.prolog.prolog_manager import PrologManager


class AgentInference:
    """
    Class to handle the inference of the agent.
    """
    def __init__(self, agent: PPOAgent, env: TestEnvironment, config: Config):
        self.agent = agent
        self.env = env  # TestEnvironment must be already initialized
        self.config = config
        self.solutions = []
        self.pbar = tqdm(total=env.requests.num_v_net, desc="Running RL Agent", unit="request") if config.test == 'simulation' else None


    def run_inference(self) -> list[Solution]:
        """
        Run the inference process.
        """
        for _ in range(self.env.requests.num_v_net):
            done = False

            start_time = time.time()

            while not done:
                # Get the current observation
                obs, mask = self.env.get_observations()

                # Get the action from the agent
                high_action, low_action, _ = self.agent.act(obs, mask, sample=False)

                # Take a step in the environment
                done, solution = self.env.step(high_action.item(), low_action.item())

                if done:
                    # Update time taken for the request
                    solution_time = time.time() - start_time
                    solution.elapsed_time = solution_time
                    # Store the solution
                    self.solutions.append(solution)
                    # Update the progress bar
                    self.pbar.update(1) if self.pbar is not None else None
                    # Go to the next request
                    self.env.go_next_arrival(log_leave=False)
                
        return self.solutions


class PrologInference:
    """
    Class to handle inference with Prolog pipeline.
    """
    def __init__(self, env: TestEnvironment, config: Config):
        self.env = env  # TestEnvironment must be already initialized
        self.config = config
        self.solutions = []
        self.pbar = tqdm(total=env.requests.num_v_net, desc="Running Prolog", unit="request") if config.test == 'simulation' else None


    def run_inference(self) -> list[Solution]:
        """
        Run the inference process.
        """
        # Create PrologManager instance
        prolog_manager = PrologManager()

        # Set the prolog manager to the global variable (so that Prolog can call class methods)
        prolog_manager.set_global_manager()

        # Prepare the FogBrainX
        prolog_manager.prepare_fogbrainx()

        # Get the physical network from the environment
        p_net = self.env.p_net

        # Get first request
        v_net = self.env.current_v_net
        event_type = self.env.event_type
        event_time = self.env.event_time

        for i in range(self.env.requests.num_v_net):
            # Update the infrastructure
            prolog_manager.update_p_net(p_net)            

            # Update request in Prolog
            prolog_manager.update_v_net(v_net)

            # Run Prolog inference
            def run(prolog_manager: PrologManager, queue: Queue):
                
                # Call the Prolog inference
                response = prolog_manager.call_fogbrainx()

                # Parse the response
                result = {}
                result['placement'] = response['P']
                result['link_mapping'] = prolog_manager.link_mapping

                # Add the solution to the queue
                queue.put(result)
                return None
            
            # Run process
            queue = Queue()
            p = Process(target=run, args=(prolog_manager, queue))
            
            # Measure the time taken for the process
            start_time = time.time()
            # Start the process
            p.start()
            # Wait timeout
            p.join(self.config.timeout)
            elapsed_time = time.time() - start_time

            # Create a solution object
            solution = Solution(i, event_type, event_time, p_net, v_net, self.config)

            # Update the elapsed time
            solution.elapsed_time = elapsed_time           
            
            # Check if process is still alive
            if p.is_alive():
                p.terminate()
                p.join()
                # Failed solution
                solution.place_result = False
                solution.route_result = False
            else:
                # Get the result from the queue
                result = queue.get()
                if result is None or result['placement'] is None:
                    # Failed solution
                    solution.place_result = False
                    solution.route_result = False
                else:
                    # Successful solution
                    solution.place_result = True
                    solution.route_result = True

                    # Update the solution with the result
                    placement = utils_prolog.parse_placement(result['placement'])
                    solution.node_mapping = utils_prolog.convert_placement(placement, v_net)
                    solution.link_mapping = result['link_mapping']

                    # If the solution is feasible, update running request count
                    self.env.running_request += 1
                    solution.running_request = self.env.running_request

                    # Consume physical resources
                    controller.apply_solution(p_net, solution)

                    # Compute the information of the solution
                    solution.compute_info()

            # Update environment with the solution
            self.env.solutions[solution.v_net.id] = solution

            # Store the solution
            self.solutions.append(solution)

            # Update the progress bar
            self.pbar.update(1) if self.pbar is not None else None
            
            # Go to the next request
            self.env.go_next_arrival(log_leave=False)
            v_net = self.env.current_v_net
            event_type = self.env.event_type
            event_time = self.env.event_time

        return self.solutions
                


class HybridInference:
    """
    Class to handle inference with RL-Prolog pipeline.
    """
    def __init__(self, agent: PPOAgent, env: TestEnvironment, config: Config):
        self.agent = agent
        self.env = env  # TestEnvironment must be already initialized
        self.config = config
        self.solutions_rl = []
        self.solutions_prolog = []
        self.pbar = tqdm(total=env.requests.num_v_net, desc="Running Hybrid", unit="request") if config.test == 'simulation' else None


    def run_inference(self) -> list[Tuple[Solution, Solution]]:
        """
        Run the inference process.
        """
        # Create PrologManager instance
        prolog_manager = PrologManager()

        # Set the prolog manager to the global variable (so that Prolog can call class methods)
        prolog_manager.set_global_manager()

        # Prepare the FogBrainX
        prolog_manager.prepare_fogbrainx()

        for i in range(self.env.requests.num_v_net):
            # Copy the physical network
            p_net_original = copy.deepcopy(self.env.p_net)

            # Get v_net from the environment
            v_net = self.env.current_v_net

            # RL PHASE
            done = False

            start_time = time.time()

            while not done:
                # Get the current observation
                obs, mask = self.env.get_observations()

                # Get the action from the agent
                high_action, low_action, _ = self.agent.act(obs, mask, sample=False)

                # Take a step in the environment
                done, solution_rl = self.env.step(high_action.item(), low_action.item())

                if done:
                    # Update time taken for the request
                    solution_time = time.time() - start_time
                    solution_rl.elapsed_time = solution_time
                    # Store the solution
                    self.solutions_rl.append(solution_rl)

            # PROLOG PHASE    
            # Update the infrastructure
            prolog_manager.update_p_net(p_net_original)

            # Update request in Prolog
            prolog_manager.update_v_net(v_net)

            # Assert the RL solution in Prolog
            prolog_manager.assert_rl_solution(solution_rl)

            # Run Prolog inference
            def run(prolog_manager: PrologManager, queue: Queue):
                
                # Call the Prolog inference
                response = prolog_manager.call_hybrid_fogbrainx()

                # Parse the response
                result = {}
                result['placement'] = response['P']
                result['link_mapping'] = prolog_manager.link_mapping

                # Add the solution to the queue
                queue.put(result)
                return None
            
            # Run process
            queue = Queue()
            p = Process(target=run, args=(prolog_manager, queue))
            
            # Measure the time taken for the process
            start_time = time.time()
            # Start the process
            p.start()
            # Wait timeout
            p.join(self.config.timeout)
            elapsed_time = time.time() - start_time

            # Create a solution object
            solution_prolog = Solution(self.env.request_index, self.env.event_type, self.env.event_time, p_net_original, v_net, self.config)

            # Update the elapsed time
            solution_prolog.elapsed_time = elapsed_time           
            
            # Check if process is still alive
            if p.is_alive():
                p.terminate()
                p.join()
                # Failed solution
                solution_prolog.place_result = False
                solution_prolog.route_result = False
            else:
                # Get the result from the queue
                result = queue.get()
                if result is None or result['placement'] is None:
                    # Failed solution
                    solution_prolog.place_result = False
                    solution_prolog.route_result = False
                else:
                    # Successful solution
                    solution_prolog.place_result = True
                    solution_prolog.route_result = True

                    # Update the solution with the result
                    placement = utils_prolog.parse_placement(result['placement'])
                    solution_prolog.node_mapping = utils_prolog.convert_placement(placement, v_net)
                    solution_prolog.link_mapping = result['link_mapping']

                    # If the RL solution is not feasible, apply the Prolog solution
                    if not solution_rl.is_feasible():
                        self.env.apply_prolog_solution(p_net_original, solution_prolog)

                    solution_prolog.running_request = self.env.running_request

                    # Compute the information of the solution
                    solution_prolog.compute_info()

            # Store the solution
            self.solutions_prolog.append(solution_prolog)

            # Update the progress bar
            self.pbar.update(1) if self.pbar is not None else None

            # Go to the next request
            self.env.go_next_arrival(log_leave=False)

        return self.solutions_rl, self.solutions_prolog