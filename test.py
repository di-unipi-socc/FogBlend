import os
import yaml
import test.test_logic as tl
import fogblend.utils.utils as utils
from dataclasses import asdict
from fogblend.ppo.agent import PPOAgent    
from fogblend.config import get_args, Config, TEST_RESULT_DIR, SAVE_DIR, UNIQUE_FOLDER


if __name__ == "__main__":
    # Get command line arguments and create a configuration object
    args = get_args()
    config = Config(**args)

    # In case of default save_dir, set it to TEST_RESULT_DIR
    config.save_dir = TEST_RESULT_DIR if config.save_dir == SAVE_DIR else os.path.join(config.save_dir, UNIQUE_FOLDER)

    # Create the save directory if it doesn't exist
    os.makedirs(config.save_dir, exist_ok=True)

    # Load the model   
    agent = PPOAgent(config)
    agent.load(config.pretrained_model_path)

    # Set agent to evaluation mode
    agent.evaluation_mode()

    # Warm up the agent
    utils.warmup_agent(agent, config)

    # Save the configuration to a YAML file
    with open(os.path.join(config.save_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(asdict(config), f, default_flow_style=False, sort_keys=False)

    # Run tests based on the specified test type
    if config.test == 'load':
        tl.run_load_based_tests(agent, config)
    elif config.test == 'simulation':
        tl.run_simulation_based_tests(agent, config)
    else:
        raise ValueError("Invalid test type. Use 'load' or 'simulation'.")