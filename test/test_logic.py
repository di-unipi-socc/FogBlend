# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fognetx.utils.types import Config, PPOAgent
# REGULAR IMPORTS
import os
import copy
import numpy as np
import test.utils_test as utils_test
from tqdm import tqdm
from fognetx.config import INFR_DIR
from multiprocessing import Process, Queue
from fognetx.placement.inference import AgentInference, PrologInference, HybridInference
from fognetx.environment.environment import TestEnvironment, PhysicalNetwork, VirtualNetworkRequests


def run_load_based_tests(agent, config: Config) -> None:
    """
    Run load-based tests for the RL agent, Prolog agent, and hybrid agent.
    The tests are executed on various infrastructure with different loads.
    
    Args:
        agent (PPOAgent): The RL agent to be tested.
        config (Config): Configuration object containing test parameters.
        save_dir (str): Directory to save the test results.
    """
    # Settings
    num_nodes = config.num_nodes
    num_iterations = config.num_iterations
    loads = [0.30, 0.40, 0.50, 0.60, 0.70]
    config.num_v_net = 1  # Forced to be 1 for load test

    # Random generator
    rng = np.random.default_rng(config.seed)
    seeds = rng.integers(0, 2**32 - 1, size=num_iterations)

    # Execute the test
    for load in loads:
        print(f"\nTesting with infrastructure load: {load}")
        pbar = tqdm(total=num_iterations, desc='Testing', unit='iteration')

        # Initialize result variables
        test_result = utils_test.initialize_result_dict(load, num_nodes, num_iterations)

        for i in range(num_iterations):
            # Get random seed
            config.seed = seeds[i]

            # Create path to physical network
            p_net_path = os.path.join(INFR_DIR.format(num_nodes=num_nodes), str(load).replace('.', '_'))

            # Load the physical network
            p_net = PhysicalNetwork(config)
            p_net.load_from_file(p_net_path, f'p_net_{i}.gml')

            # Create requests
            requests = VirtualNetworkRequests(config)
            requests.generate_requests(p_net.num_nodes)

            # Create Test Environment
            env = TestEnvironment(p_net, requests, config)

            # Clone environment for different tests
            env_rl = copy.deepcopy(env)
            env_prolog = utils_test.convert_env_prolog(env)
            env_hybrid = copy.deepcopy(env)

            # Test the Prolog agent in a separate process
            prolog_queue = Queue()
            prolog_process = Process(target=_test_prolog_worker, args=(env_prolog, config, load, prolog_queue))
            prolog_process.start()
           
            # Test the RL agent
            result_rl = test_agent(agent, env_rl, config, load)

            # Test the Hybrid agent
            result_hybrid = test_hybrid(agent, env_hybrid, config, load)

            # Wait for Prolog process to finish and get the result
            prolog_process.join()
            result_prolog = prolog_queue.get()

            # Update results
            utils_test.update_results(test_result, result_rl, result_prolog, result_hybrid)

            # Update progress bar
            pbar.update(1)

        pbar.close()
        print()

        # Compute averages
        utils_test.finalize_results(test_result, num_iterations)

        # Save results to CSV
        file_name = f"summary_results.csv"
        utils_test.save_results_to_csv(test_result, config.save_dir, file_name)


def run_simulation_based_tests(agent, config: Config) -> None:
    """
    Run simulation-based tests for the RL agent, Prolog agent, and hybrid agent.
    The tests are executed on the same infrastructure starting without load.

    Args:
        agent (PPOAgent): The RL agent to be tested.
        config (Config): Configuration object containing test parameters.
        save_dir (str): Directory to save the test results.
    """
    # Settings
    num_nodes = config.num_nodes
    num_iterations = 1
    load = 0 

    # Initialize result variables
    test_result = utils_test.initialize_result_dict(load, num_nodes, num_iterations)

    # Create a physical network
    p_net = PhysicalNetwork(config)
    p_net.generate_p_net(num_nodes)

    # Create requests
    requests = VirtualNetworkRequests(config)
    requests.generate_requests(num_nodes)

    # Create Test Environment
    env = TestEnvironment(p_net, requests, config)

    # Clone environment for different tests
    env_rl = copy.deepcopy(env)
    env_prolog = utils_test.convert_env_prolog(env)
    env_hybrid = copy.deepcopy(env)

    # Create queues to collect Prolog result
    prolog_queue = Queue()

    # Launch Prolog in parallel process
    env_prolog = utils_test.convert_env_prolog(env)
    prolog_process = Process(target=_test_prolog_worker, args=(env_prolog, config, load, prolog_queue))
    prolog_process.start()

    # Launch RL agent in main process
    result_rl = test_agent(agent, env_rl, config, load)

    # Launch Hybrid in main process
    result_hybrid = test_hybrid(agent, env_hybrid, config, load)

    # Wait for Prolog process to finish and get the result
    prolog_process.join()
    result_prolog = prolog_queue.get()

    # Update results
    utils_test.update_results(test_result, result_rl, result_prolog, result_hybrid)

    # Compute averages
    utils_test.finalize_results(test_result, config.num_v_net)

    # Save results to CSV
    file_name = f"summary_results.csv"
    utils_test.save_results_to_csv(test_result, config.save_dir, file_name)
    

# Wrapper function to run in subprocesses
def _test_prolog_worker(env: TestEnvironment, config: Config, load: float, result_queue: Queue):
    result = test_prolog(env, config, load)
    result_queue.put(result)


def test_agent(agent: PPOAgent, env: TestEnvironment, config: Config, load: float = None) -> dict:
    """
    Test the RL agent on the given environment.
    
    Args:
        agent (PPOAgent): The RL agent to be tested.
        env (TestEnvironment): The environment in which the agent will be tested.
        config (Config): Configuration object containing test parameters.
        load (float): Load factor for the test.
    """    
    # Create instance of AgentInference
    agent_inference = AgentInference(agent, env, config)

    # Run the RL agent
    rl_solution = agent_inference.run_inference()

    # Elaborate the results
    result = {}
    result['request_count_rl'] = 0
    result['success_count_rl'] = 0
    result['avg_time_failure_rl'] = 0
    result['avg_time_success_rl'] = 0
    result['avg_r2c_ratio_rl'] = 0

    file_name = f"rl_solution_{str(load).replace('.', '_')}.csv" if load else 'rl_solution.csv'

    for solution in rl_solution:
        # Log individual solutions
        solution.log(log_dir=config.save_dir, file_name=file_name)
        # Aggregate results
        if solution.is_feasible():
            result['success_count_rl'] += 1
            result['avg_time_success_rl'] += solution.elapsed_time
            result['avg_r2c_ratio_rl'] += solution.r2c_ratio
        else:
            result['avg_time_failure_rl'] += solution.elapsed_time
        result['request_count_rl'] += 1

    return result


def test_prolog(env: TestEnvironment, config: Config, load: float = None) -> dict:
    """
    Test the Prolog agent on the given environment.

    Args:
        env (TestEnvironment): The environment in which the Prolog agent will be tested.
        config (Config): Configuration object containing test parameters.
        load (float): Load factor for the test.
    """
    # Create instance of PrologInference
    prolog_inference = PrologInference(env, config)

    # Run the Prolog agent
    prolog_solution = prolog_inference.run_inference()

    # Elaborate the results
    result = {}
    result['request_count_prolog'] = 0
    result['success_count_prolog'] = 0
    result['avg_time_failure_prolog'] = 0
    result['avg_time_success_prolog'] = 0
    result['avg_r2c_ratio_prolog'] = 0

    file_name = f"prolog_solution_{str(load).replace('.', '_')}.csv" if load else 'prolog_solution.csv'

    for solution in prolog_solution:
        # Log individual solutions
        solution.log(log_dir=config.save_dir, file_name=file_name)
        # Aggregate results
        if solution.is_feasible():
            result['success_count_prolog'] += 1
            result['avg_time_success_prolog'] += solution.elapsed_time
            result['avg_r2c_ratio_prolog'] += solution.r2c_ratio
        else:
            result['avg_time_failure_prolog'] += solution.elapsed_time
        result['request_count_prolog'] += 1

    return result


def test_hybrid(agent: PPOAgent, env: TestEnvironment, config: Config, load: float = None) -> dict:
    """
    Test the RL-Prolog hybrid agent on the given environment.

    Args:
        agent (PPOAgent): The RL agent to be tested.
        env (TestEnvironment): The environment in which the hybrid agent will be tested.
        config (Config): Configuration object containing test parameters.
        load (float): Load factor for the test.
    """
    # Create instance of HybridInference
    hybrid_inference = HybridInference(agent, env, config)

    # Run the hybrid agent
    hybrid_solutions = hybrid_inference.run_inference()

    # Elaborate the results
    result = {}
    result['request_count_hybrid'] = 0
    result['success_count_rl_phase'] = 0
    result['success_count_prolog_phase'] = 0
    result['avg_time_failure_rl_phase'] = 0
    result['avg_time_failure_prolog_phase'] = 0
    result['avg_time_success_rl_phase'] = 0
    result['avg_time_success_prolog_phase'] = 0
    result['avg_r2c_ratio_hybrid'] = 0

    # Unpack the hybrid solutions
    solution_rl_list, solution_prolog_list = hybrid_solutions

    file_name_rl = f"hybrid_solution_rl_phase_{str(load).replace('.', '_')}.csv" if load else 'hybrid_solution_rl_phase.csv'
    file_name_prolog = f"hybrid_solution_prolog_phase_{str(load).replace('.', '_')}.csv" if load else 'hybrid_solution_prolog_phase.csv'

    # Iterate over the solutions
    for solution_rl_phase, solution_prolog_phase in zip(solution_rl_list, solution_prolog_list):
        # Log individual solutions
        solution_rl_phase.log(log_dir=config.save_dir, file_name=file_name_rl)
        solution_prolog_phase.log(log_dir=config.save_dir, file_name=file_name_prolog)
        # Aggregate results
        if solution_rl_phase.is_feasible():
            result['success_count_rl_phase'] += 1
            result['avg_time_success_rl_phase'] += solution_rl_phase.elapsed_time
        else:
            result['avg_time_failure_rl_phase'] += solution_rl_phase.elapsed_time
        if solution_prolog_phase.is_feasible():
            # This correspond to the success of the entire hybrid solution
            result['success_count_prolog_phase'] += 1
            result['avg_time_success_prolog_phase'] += solution_prolog_phase.elapsed_time
            result['avg_r2c_ratio_hybrid'] += solution_prolog_phase.r2c_ratio
        else:
            # This correspond to the failure of the entire hybrid solution
            result['avg_time_failure_prolog_phase'] += solution_prolog_phase.elapsed_time
        result['request_count_hybrid'] += 1

    return result