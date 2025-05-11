import os
import fognetx.utils.utils as utils
import test.test_logic as tl
from fognetx.ppo.agent import PPOAgent    
from fognetx.config import get_args, Config, TEST_RESULT_DIR, SAVE_DIR


if __name__ == "__main__":
    # Get command line arguments and create a configuration object
    args = get_args()
    config = Config(**args)

    # In case of default save_dir, set it to TEST_RESULT_DIR
    save_dir = TEST_RESULT_DIR if config.save_dir == SAVE_DIR else config.save_dir

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Load the model   
    agent = PPOAgent(config)
    agent.load(config.pretrained_model_path)

    # Set agent to evaluation mode
    agent.evaluation_mode()

    # Warm up the agent
    utils.warmup_agent(agent, config)

    if config.test == 'load':
        tl.run_load_based_tests(agent, config, save_dir)
    elif config.test == 'simulation':
        tl.run_simulation_based_tests(agent, config, save_dir)
    else:
        raise ValueError("Invalid test type. Use 'load' or 'simulation'.")