# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from typing import Dict, Optional; from fognetx.utils.types import TestEnvironment
# REGULAR IMPORTS
import os
import copy

def initialize_result_dict(load: float, num_nodes: int, num_iterations: int) -> Dict:
    """
    Initialize the result dictionary with default values.
    
    Args:
        load (float): The infrastructure load.
        num_nodes (int): The number of nodes in the network.
        num_iterations (int): The number of iterations for the test.
    Returns:
        dict: A dictionary containing the initialized result values.
    """
    return {
        'success_count_rl': 0, 'avg_time_success_rl': 0, 'avg_time_failure_rl': 0, 'avg_r2c_ratio_rl': 0,

        'success_count_prolog': 0, 'avg_time_success_prolog': 0, 'avg_time_failure_prolog': 0, 'avg_r2c_ratio_prolog': 0,

        'success_count_rl_phase': 0, 'success_count_prolog_phase': 0, 
        'avg_time_success_rl_phase': 0, 'avg_time_success_prolog_phase': 0,
        'avg_time_failure_rl_phase': 0, 'avg_time_failure_prolog_phase': 0, 
        'avg_r2c_ratio_hybrid': 0,

        'num_requests': 0,
        'infr_load': load,
        'num_iterations': num_iterations,
        'num_nodes': num_nodes
    }


def update_results(accum: Dict, result_rl: Optional[Dict] = None, result_prolog: Optional[Dict] = None, result_hybrid: Optional[Dict] = None) -> None:
    """
    Update the accumulated results with the new results from the current iteration.
    
    Args:
        accum (dict): The accumulated results dictionary.
        result_rl (dict, optional): The results from the RL agent.
        result_prolog (dict, optional): The results from the Prolog agent.
        result_hybrid (dict, optional): The results from the hybrid agent.
    """
    # Update requests count with any non none result
    if result_rl:
        accum['num_requests'] += result_rl['request_count_rl']
    elif result_prolog:
        accum['num_requests'] += result_prolog['request_count_prolog']
    elif result_hybrid:
        accum['num_requests'] += result_hybrid['request_count_hybrid']

    # Update other fields
    if result_rl:
        accum['success_count_rl'] += result_rl['success_count_rl']
        accum['avg_time_success_rl'] += result_rl['avg_time_success_rl']
        accum['avg_time_failure_rl'] += result_rl['avg_time_failure_rl']
        accum['avg_r2c_ratio_rl'] += result_rl['avg_r2c_ratio_rl']

    if result_prolog:
        accum['success_count_prolog'] += result_prolog['success_count_prolog']
        accum['avg_time_success_prolog'] += result_prolog['avg_time_success_prolog']
        accum['avg_time_failure_prolog'] += result_prolog['avg_time_failure_prolog']
        accum['avg_r2c_ratio_prolog'] += result_prolog['avg_r2c_ratio_prolog']

    if result_hybrid:
        accum['success_count_prolog_phase'] += result_hybrid['success_count_prolog_phase']
        accum['success_count_rl_phase'] += result_hybrid['success_count_rl_phase']
        accum['avg_time_success_rl_phase'] += result_hybrid['avg_time_success_rl_phase']
        accum['avg_time_success_prolog_phase'] += result_hybrid['avg_time_success_prolog_phase']
        accum['avg_time_failure_rl_phase'] += result_hybrid['avg_time_failure_rl_phase']
        accum['avg_time_failure_prolog_phase'] += result_hybrid['avg_time_failure_prolog_phase']
        accum['avg_r2c_ratio_hybrid'] += result_hybrid['avg_r2c_ratio_hybrid']


def convert_env_prolog(env: TestEnvironment) -> TestEnvironment:
    """
    Convert the environment to Prolog format removing CUDA if present.
    
    Args:
        env (TestEnvironment): The environment to be converted.
    
    Returns:
        TestEnvironment: The converted environment.
    """
    # Deep copy the environment to avoid modifying the original
    env_copy = copy.deepcopy(env)
    # Remove CUDA from the environment
    env_copy.env_config.device = 'cpu'
    # Remove observation (Tensor)
    env_copy.observations = None
    return env_copy


def finalize_results(result: Dict, total_requests: int) -> None:
    """
    Finalize the results by computing average times and ratios for each agent and hybrid phases.
    
    Args:
        result (dict): The result dictionary containing accumulated test values.
        total_requests (int): The total number of requests processed.
    """
    agent_types = ['rl', 'prolog']
    
    for agent in agent_types:
        success_count = result[f'success_count_{agent}']
        failure_count = total_requests - success_count

        if success_count > 0:
            result[f'avg_time_success_{agent}'] /= success_count
            result[f'avg_r2c_ratio_{agent}'] /= success_count
        else:
            result[f'avg_time_success_{agent}'] = 0
            result[f'avg_r2c_ratio_{agent}'] = 0

        if failure_count > 0:
            result[f'avg_time_failure_{agent}'] /= failure_count
        else:
            result[f'avg_time_failure_{agent}'] = 0

    # Handle hybrid (split in rl_phase and prolog_phase)
    for phase in ['rl_phase', 'prolog_phase']:
        success_count = result.get(f'success_count_{phase}', 0)
        failure_count = total_requests - success_count

        success_key = f'avg_time_success_{phase}'
        failure_key = f'avg_time_failure_{phase}'

        if success_count > 0:
            result[success_key] /= success_count
        else:
            result[success_key] = 0

        if failure_count > 0:
            result[failure_key] /= failure_count
        else:
            result[failure_key] = 0

    # Hybrid success is equal to prolog phase success 
    hybrid_success = result.get('success_count_prolog_phase', 0)
    if hybrid_success > 0:
        result['avg_r2c_ratio_hybrid'] /= hybrid_success
    else:
        result['avg_r2c_ratio_hybrid'] = 0


def save_results_to_csv(result, save_dir, file_name) -> None:
    """
    Save the results to a CSV file.

    Args:
        result (dict): The result dictionary containing the test results.
        save_dir (str): The directory where the CSV file will be saved.
        file_name (str): The name of the CSV file.
    """
    # Compose the full path for the CSV file and create the directory if it doesn't exist
    test_result_file = os.path.join(save_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(test_result_file):
        with open(test_result_file, 'w') as f:
            header = [
                'infr_load', 'num_iterations', 'num_requests', 'num_nodes', 
                'success_count_rl', 'avg_time_success_rl', 'avg_time_failure_rl', 'avg_r2c_ratio_rl',
                'success_count_prolog', 'avg_time_success_prolog', 'avg_time_failure_prolog', 'avg_r2c_ratio_prolog',
                'success_count_rl_phase', 'avg_time_success_rl_phase', 'avg_time_failure_rl_phase', 
                'success_count_prolog_phase', 'avg_time_success_prolog_phase', 'avg_time_failure_prolog_phase',
                'avg_r2c_ratio_hybrid'
            ]
            f.write(','.join(header) + '\n')

    with open(test_result_file, 'a') as f:
        f.write(','.join(map(str, [
            result['infr_load'],
            result['num_iterations'],
            result['num_requests'],
            result['num_nodes'],
            result['success_count_rl'],
            result['avg_time_success_rl'],
            result['avg_time_failure_rl'],
            result['avg_r2c_ratio_rl'],
            result['success_count_prolog'],
            result['avg_time_success_prolog'],
            result['avg_time_failure_prolog'],
            result['avg_r2c_ratio_prolog'],
            result['success_count_rl_phase'],
            result['avg_time_success_rl_phase'],
            result['avg_time_failure_rl_phase'],
            result['success_count_prolog_phase'],
            result['avg_time_success_prolog_phase'],
            result['avg_time_failure_prolog_phase'],
            result['avg_r2c_ratio_hybrid']
        ])) + '\n')