import os
import copy
import fogblend.prolog.utils_prolog as utils_prolog
from tqdm import tqdm
from fogblend.config import get_args, Config, INFR_DIR
from fogblend.environment.physicalNetwork import PhysicalNetwork


if __name__ == "__main__":
    # Get command line arguments and create a configuration object
    args = get_args()
    config = Config(**args)

    # Settings
    num_nodes = config.num_nodes
    start_index = 0
    num_generations = 100
    loads = [0.30, 0.40, 0.50, 0.60, 0.70]
    save_dir = INFR_DIR.format(num_nodes=num_nodes)

    # Generation
    os.makedirs(save_dir, exist_ok=True)
    p_net = PhysicalNetwork(config)

    print(f'Generating {num_generations} physical networks with {num_nodes} nodes.')
    pbar = tqdm(total=num_generations, desc='Generating physical networks', unit='network')

    for i in range(start_index, start_index + num_generations):
        # Generate a physical network
        p_net.generate_p_net(size=num_nodes)

        for load in loads:
            # Copy the physical network
            p_net_loaded = copy.deepcopy(p_net)

            # Apply load to physical network
            p_net_loaded.apply_load(load=load)

            # Create directory and filename for saving
            save_dir_loaded = os.path.join(save_dir, str(load).replace('.', '_'))
            filename = f'p_net_{i+start_index}.gml'
            filename_prolog = f'p_net_{i+start_index}.txt'

            # Save the physical network
            p_net_loaded.save_to_file(save_dir=save_dir_loaded, filename=filename)

            # Save Prolog infrastructure
            # utils_prolog.generate_infr_file(p_net_loaded, save_dir=save_dir_loaded, filename=filename_prolog)

        # Update progress bar
        pbar.update(1)

    pbar.close()