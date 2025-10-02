# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from fogblend.utils.types import Config, PPOAgent
# REGULAR IMPORTS
import copy
import numpy as np
import test.utils_test as utils_test
from tqdm import tqdm
from multiprocessing import Process, Queue
from fogblend.placement.inference import AgentInference, PrologInference, HybridInference
from fogblend.environment.environment import TestEnvironment, PhysicalNetwork, VirtualNetworkRequests


def run_load_based_tests(agent, config: Config) -> None:
    """
    Run load-based tests for the neural agent, symbolic agent, and hybrid agent.
    The tests are executed on various infrastructure with different loads.
    
    Args:
        agent (PPOAgent): The neural agent to be tested.
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

            # Generate and apply load to the physical network
            p_net = PhysicalNetwork(config)
            p_net.generate_p_net(num_nodes)
            p_net.apply_load(load=load)

            # Create requests
            requests = VirtualNetworkRequests(config)
            requests.generate_requests(p_net.num_nodes)

            # Create Test Environment
            env = TestEnvironment(p_net, requests, config)

            # Clone environment for different tests
            env_neural = copy.deepcopy(env) if config.test_neural else None
            env_symbolic = utils_test.convert_env_symbolic(env) if config.test_symbolic else None
            env_hybrid = copy.deepcopy(env) if config.test_hybrid else None

            # Test the symbolic agent in a separate process
            if config.test_symbolic:
                symbolic_queue = Queue()
                symbolic_process = Process(target=_test_symbolic_worker, args=(env_symbolic, config, load, symbolic_queue))
                symbolic_process.start()
           
            # Test the Neural agent
            result_neural = test_agent(agent, env_neural, config, load) if config.test_neural else {}

            # Test the Hybrid agent
            result_hybrid = test_hybrid(agent, env_hybrid, config, load) if config.test_hybrid else {}

            # Wait for symbolic process to finish and get the result
            if config.test_symbolic:
                symbolic_process.join()
                result_symbolic = symbolic_queue.get()
            else:
                result_symbolic = {}

            # Update results
            utils_test.update_results(test_result, result_neural, result_symbolic, result_hybrid)

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
    Run simulation-based tests for the neural agent, symbolic agent, and hybrid agent.
    The tests are executed on the same infrastructure starting without load.

    Args:
        agent (PPOAgent): The neural agent to be tested.
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
    env_neural = copy.deepcopy(env) if config.test_neural else None
    env_symbolic = utils_test.convert_env_symbolic(env) if config.test_symbolic else None
    env_hybrid = copy.deepcopy(env) if config.test_hybrid else None

    # Test the symbolic agent in a separate process
    if config.test_symbolic:
        symbolic_queue = Queue()
        symbolic_process = Process(target=_test_symbolic_worker, args=(env_symbolic, config, load, symbolic_queue))
        symbolic_process.start()

    # Launch Neural agent in main process
    result_neural = test_agent(agent, env_neural, config, load) if config.test_neural else {}

    # Launch Hybrid in main process
    result_hybrid = test_hybrid(agent, env_hybrid, config, load) if config.test_hybrid else {}

    # Wait for symbolic process to finish and get the result
    if config.test_symbolic:
        symbolic_process.join()
        result_symbolic = symbolic_queue.get()
    else:
        result_symbolic = {}

    # Update results
    utils_test.update_results(test_result, result_neural, result_symbolic, result_hybrid)

    # Compute averages
    utils_test.finalize_results(test_result, config.num_v_net)

    # Save results to CSV
    file_name = f"summary_results.csv"
    utils_test.save_results_to_csv(test_result, config.save_dir, file_name)
    

# Wrapper function to run in subprocesses
def _test_symbolic_worker(env: TestEnvironment, config: Config, load: float, result_queue: Queue):
    result = test_symbolic(env, config, load)
    result_queue.put(result)


def test_agent(agent: PPOAgent, env: TestEnvironment, config: Config, load: float = None) -> dict:
    """
    Test the neural agent on the given environment.
    
    Args:
        agent (PPOAgent): The neural agent to be tested.
        env (TestEnvironment): The environment in which the agent will be tested.
        config (Config): Configuration object containing test parameters.
        load (float): Load factor for the test.
    """    
    # Create instance of AgentInference
    agent_inference = AgentInference(agent, env, config)

    # Run the neural agent
    neural_solution = agent_inference.run_inference()

    # Elaborate the results
    result = {}
    result['request_count_neural'] = 0
    result['success_count_neural'] = 0
    result['avg_time_failure_neural'] = 0
    result['avg_time_success_neural'] = 0
    result['avg_r2c_ratio_neural'] = 0
    lt_revenue = 0
    lt_cost = 0
    result['lt_r2c_ratio_neural'] = 0

    file_name = f"neural_solution_{str(load).replace('.', '_')}.csv" if load else 'neural_solution.csv'

    for solution in neural_solution:
        # Log individual solutions
        solution.log(log_dir=config.save_dir, file_name=file_name)
        # Aggregate results
        if solution.is_feasible():
            result['success_count_neural'] += 1
            result['avg_time_success_neural'] += solution.elapsed_time
            result['avg_r2c_ratio_neural'] += solution.r2c_ratio
            lt_revenue += solution.longterm_revenue
            lt_cost += solution.longterm_cost
        else:
            result['avg_time_failure_neural'] += solution.elapsed_time
        result['request_count_neural'] += 1

    # Compute long-term r2c ratio
    result['lt_r2c_ratio_neural'] = lt_revenue / lt_cost if lt_cost > 0 else 0

    return result


def test_symbolic(env: TestEnvironment, config: Config, load: float = None) -> dict:
    """
    Test the symbolic agent on the given environment.

    Args:
        env (TestEnvironment): The environment in which the symbolic agent will be tested.
        config (Config): Configuration object containing test parameters.
        load (float): Load factor for the test.
    """
    # Create instance of symbolic inference
    symbolic_inference = PrologInference(env, config)

    # Run the symbolic agent
    symbolic_solution = symbolic_inference.run_inference()

    # Elaborate the results
    result = {}
    result['request_count_symbolic'] = 0
    result['success_count_symbolic'] = 0
    result['avg_time_failure_symbolic'] = 0
    result['avg_time_success_symbolic'] = 0
    result['avg_r2c_ratio_symbolic'] = 0
    lt_revenue = 0
    lt_cost = 0
    result['lt_r2c_ratio_symbolic'] = 0

    file_name = f"symbolic_solution_{str(load).replace('.', '_')}.csv" if load else 'symbolic_solution.csv'

    for solution in symbolic_solution:
        # Log individual solutions
        solution.log(log_dir=config.save_dir, file_name=file_name)
        # Aggregate results
        if solution.is_feasible():
            result['success_count_symbolic'] += 1
            result['avg_time_success_symbolic'] += solution.elapsed_time
            result['avg_r2c_ratio_symbolic'] += solution.r2c_ratio
            lt_revenue += solution.longterm_revenue
            lt_cost += solution.longterm_cost
        else:
            result['avg_time_failure_symbolic'] += solution.elapsed_time
        result['request_count_symbolic'] += 1

    # Compute long-term r2c ratio
    result['lt_r2c_ratio_symbolic'] = lt_revenue / lt_cost if lt_cost > 0 else 0

    return result


def test_hybrid(agent: PPOAgent, env: TestEnvironment, config: Config, load: float = None) -> dict:
    """
    Test the Neuro-Symbolic hybrid agent on the given environment.

    Args:
        agent (PPOAgent): The neural agent to be tested.
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
    result['success_count_neural_phase'] = 0
    result['success_count_symbolic_phase'] = 0
    result['avg_time_failure_neural_phase'] = 0
    result['avg_time_failure_symbolic_phase'] = 0
    result['avg_time_success_neural_phase'] = 0
    result['avg_time_success_symbolic_phase'] = 0
    result['avg_r2c_ratio_hybrid'] = 0
    lt_revenue = 0
    lt_cost = 0
    result['lt_r2c_ratio_hybrid'] = 0

    # Unpack the hybrid solutions
    solution_neural_list, solution_symbolic_list = hybrid_solutions

    file_name_neural = f"hybrid_solution_neural_phase_{str(load).replace('.', '_')}.csv" if load else 'hybrid_solution_neural_phase.csv'
    file_name_symbolic = f"hybrid_solution_symbolic_phase_{str(load).replace('.', '_')}.csv" if load else 'hybrid_solution_symbolic_phase.csv'

    # Iterate over the solutions
    for solution_neural_phase, solution_symbolic_phase in zip(solution_neural_list, solution_symbolic_list):
        # Log individual solutions
        solution_neural_phase.log(log_dir=config.save_dir, file_name=file_name_neural)
        solution_symbolic_phase.log(log_dir=config.save_dir, file_name=file_name_symbolic)
        # Aggregate results
        if solution_neural_phase.is_feasible():
            result['success_count_neural_phase'] += 1
            result['avg_time_success_neural_phase'] += solution_neural_phase.elapsed_time
        else:
            result['avg_time_failure_neural_phase'] += solution_neural_phase.elapsed_time
        if solution_symbolic_phase.is_feasible():
            # This correspond to the success of the entire hybrid solution
            result['success_count_symbolic_phase'] += 1
            result['avg_time_success_symbolic_phase'] += solution_symbolic_phase.elapsed_time
            result['avg_r2c_ratio_hybrid'] += solution_symbolic_phase.r2c_ratio
            lt_revenue += solution_symbolic_phase.longterm_revenue
            lt_cost += solution_symbolic_phase.longterm_cost
        else:
            # This correspond to the failure of the entire hybrid solution
            result['avg_time_failure_symbolic_phase'] += solution_symbolic_phase.elapsed_time
        result['request_count_hybrid'] += 1

    # Compute long-term r2c ratio
    result['lt_r2c_ratio_hybrid'] = lt_revenue / lt_cost if lt_cost > 0 else 0

    return result