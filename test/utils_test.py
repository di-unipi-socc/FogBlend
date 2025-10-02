# TYPE CHECKING IMPORTS
from __future__ import annotations; from typing import TYPE_CHECKING
if TYPE_CHECKING: from typing import Dict, Optional; from fogblend.utils.types import TestEnvironment
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
        'success_count_neural': 0, 'avg_time_success_neural': 0, 'avg_time_failure_neural': 0, 'avg_r2c_ratio_neural': 0, 'lt_r2c_ratio_neural': 0,

        'success_count_symbolic': 0, 'avg_time_success_symbolic': 0, 'avg_time_failure_symbolic': 0, 'avg_r2c_ratio_symbolic': 0, 'lt_r2c_ratio_symbolic': 0,

        'success_count_neural_phase': 0, 'success_count_symbolic_phase': 0, 
        'avg_time_success_neural_phase': 0, 'avg_time_success_symbolic_phase': 0,
        'avg_time_failure_neural_phase': 0, 'avg_time_failure_symbolic_phase': 0, 
        'avg_r2c_ratio_hybrid': 0, 'lt_r2c_ratio_hybrid': 0,

        'num_requests': 0,
        'infr_load': load,
        'num_iterations': num_iterations,
        'num_nodes': num_nodes
    }


def update_results(accum: Dict, result_neural: Optional[Dict] = None, result_symbolic: Optional[Dict] = None, result_hybrid: Optional[Dict] = None) -> None:
    """
    Update the accumulated results with the new results from the current iteration.
    
    Args:
        accum (dict): The accumulated results dictionary.
        result_neural (dict, optional): The results from the neural agent.
        result_symbolic (dict, optional): The results from the symbolic agent.
        result_hybrid (dict, optional): The results from the hybrid agent.
    """
    # Update requests count with any non none result
    if result_neural:
        accum['num_requests'] += result_neural['request_count_neural']
    elif result_symbolic:
        accum['num_requests'] += result_symbolic['request_count_symbolic']
    elif result_hybrid:
        accum['num_requests'] += result_hybrid['request_count_hybrid']

    # Update other fields
    if result_neural:
        accum['success_count_neural'] += result_neural['success_count_neural']
        accum['avg_time_success_neural'] += result_neural['avg_time_success_neural']
        accum['avg_time_failure_neural'] += result_neural['avg_time_failure_neural']
        accum['avg_r2c_ratio_neural'] += result_neural['avg_r2c_ratio_neural']
        accum['lt_r2c_ratio_neural'] += result_neural['lt_r2c_ratio_neural']

    if result_symbolic:
        accum['success_count_symbolic'] += result_symbolic['success_count_symbolic']
        accum['avg_time_success_symbolic'] += result_symbolic['avg_time_success_symbolic']
        accum['avg_time_failure_symbolic'] += result_symbolic['avg_time_failure_symbolic']
        accum['avg_r2c_ratio_symbolic'] += result_symbolic['avg_r2c_ratio_symbolic']
        accum['lt_r2c_ratio_symbolic'] += result_symbolic['lt_r2c_ratio_symbolic']

    if result_hybrid:
        accum['success_count_symbolic_phase'] += result_hybrid['success_count_symbolic_phase']
        accum['success_count_neural_phase'] += result_hybrid['success_count_neural_phase']
        accum['avg_time_success_neural_phase'] += result_hybrid['avg_time_success_neural_phase']
        accum['avg_time_success_symbolic_phase'] += result_hybrid['avg_time_success_symbolic_phase']
        accum['avg_time_failure_neural_phase'] += result_hybrid['avg_time_failure_neural_phase']
        accum['avg_time_failure_symbolic_phase'] += result_hybrid['avg_time_failure_symbolic_phase']
        accum['avg_r2c_ratio_hybrid'] += result_hybrid['avg_r2c_ratio_hybrid']
        accum['lt_r2c_ratio_hybrid'] += result_hybrid['lt_r2c_ratio_hybrid']


def convert_env_symbolic(env: TestEnvironment) -> TestEnvironment:
    """
    Convert the environment to symbolic format removing CUDA if present.
    
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
    agent_types = ['neural', 'symbolic']
    
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

    # Handle hybrid (split in neural_phase and symbolic_phase)
    for phase in ['neural_phase', 'symbolic_phase']:
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

    # Hybrid success is equal to symbolic phase success 
    hybrid_success = result.get('success_count_symbolic_phase', 0)
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
                'success_count_neural', 'avg_time_success_neural', 'avg_time_failure_neural', 'avg_r2c_ratio_neural', 'lt_r2c_ratio_neural',
                'success_count_symbolic', 'avg_time_success_symbolic', 'avg_time_failure_symbolic', 'avg_r2c_ratio_symbolic', 'lt_r2c_ratio_symbolic',
                'success_count_neural_phase', 'avg_time_success_neural_phase', 'avg_time_failure_neural_phase', 
                'success_count_symbolic_phase', 'avg_time_success_symbolic_phase', 'avg_time_failure_symbolic_phase',
                'avg_r2c_ratio_hybrid', 'lt_r2c_ratio_hybrid'
            ]
            f.write(','.join(header) + '\n')

    with open(test_result_file, 'a') as f:
        f.write(','.join(map(str, [
            result['infr_load'],
            result['num_iterations'],
            result['num_requests'],
            result['num_nodes'],
            result['success_count_neural'],
            result['avg_time_success_neural'],
            result['avg_time_failure_neural'],
            result['avg_r2c_ratio_neural'],
            result['lt_r2c_ratio_neural'],
            result['success_count_symbolic'],
            result['avg_time_success_symbolic'],
            result['avg_time_failure_symbolic'],
            result['avg_r2c_ratio_symbolic'],
            result['lt_r2c_ratio_symbolic'],
            result['success_count_neural_phase'],
            result['avg_time_success_neural_phase'],
            result['avg_time_failure_neural_phase'],
            result['success_count_symbolic_phase'],
            result['avg_time_success_symbolic_phase'],
            result['avg_time_failure_symbolic_phase'],
            result['avg_r2c_ratio_hybrid'],
            result['lt_r2c_ratio_hybrid']
        ])) + '\n')