<p align="center">
  <img src="save/img/logo.png" alt="FogBlend Logo" width="400"/>
</p>

# FogBlend

FogBlend is a neuro-symbolic approach to the Cloud-Edge application placement problem, combining the strengths of neural inference and symbolic reasoning. It leverages an enhanced version of FlagVNE (FlagVNE+) as the neural agent to generate initial placement solutions, and extends FogBrainX (FogBrainX+) as the symbolic engine to correct invalid allocations through continuous reasoning.

This repository contains the full implementation of FogBlend along with a Python-based simulator for running experiments and benchmarks.

## Contents

- [Installation](#installation)
- [Training Commands](#training-commands)
- [Experiments Commands](#experiments-commands)
- [Plotting Commands](#plotting-commands)


## Installation

In the repository is provided a Dockerfile to build a container including all the library requirements.

### Prerequisites

**Required:**
- **Docker:** Version 19.03 or later
- **Operating System:** Linux, macOS, or Windows with WSL2

**Optional (for GPU acceleration):**
- NVIDIA GPU with CUDA support
- NVIDIA Driver: Version 450.80.02 or later
- NVIDIA Container Toolkit

### Building the Container

Clone the repository and build the Docker image:

```bash
git clone <repository-url>
cd <repository-name>
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t fogblend .
```

The build arguments ensure proper file permissions by matching the container user with your host user ID.

### Running the Container

**With GPU acceleration:**
```bash
docker run --gpus all -it --rm -v $(pwd):/workspace fogblend
```

**CPU-only (no NVIDIA GPU required):**
```bash
docker run -it --rm -v $(pwd):/workspace fogblend
```

### Verifying the Installation

Once inside the container, verify the installation:

```bash
# Check Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check SWI-Prolog
swipl --version

# Check other key libraries
python -c "import torch_geometric; import numpy; import pandas; print('All libraries loaded successfully')"
```

## Training Commands

To train the neural agent with Reinforcement Learning use the following command:

```bash
python train.py <PARAMETERS>
```

### Key Parameters

- **num_epochs**: The number of training epochs. Default is `100`.
- **p_net_min_size**: The minimum size of the infrastructures generated for training. Default is `50`.
- **p_net_max_size**: The maximum size of the infrastructures generated for training. Default is `500`.
- **num_nodes**: If set, this parameter overrides `p_net_min_size` and `p_net_max_size` and fixes the size of all training infrastructures.
- **topology**: The topology of the infrastructures used for training (`Waxman` or `GÉANT`). If `GÉANT` is selected, the `num_nodes`, `p_net_min_size`, and `p_net_max_size` parameters are overridden. Default is `Waxman`.
- **arrival_rate**: Arrival rate of requests during the training simulations. If not indicated, for each epoch it will be generated according to the size of the infrastructure used.
- **architecture**: Architecture of the model to train, where `original` corresponds to FlagVNE while `new` corresponds to FlagVNE+. Default is `new`.
- **reusable**: Whether to allow multiple components of the same application to be mapped to the same node
- **mask_actions**: Whether to mask invalid actions in the action space

### Training Notes
The `new` architecture has been designed to provide a model capable of making inference on any infrastructure size. For this reason, it is suggested to avoid setting `num_nodes` and `topology` parameters for this architecture option in order to use the default training strategy.

By default, the model info and training results will be saved under a folder containing the date of when the training has started in the `save/` directory.

The complete set of parameters, including all model hyperparameters, is available in the `config.py` file.

### Example

```bash
python train.py -architecture original -topology geant -arrival_rate 0.001 -num_epochs 50
```


## Experiments Commands

Two different types of tests can be carried out to cover the full spectrum of operational scenarios, ensuring a complete performance analysis:

- **Load Test**: Evaluates the performance of placing a single application on an infrastructure with a predetermined percentage of already allocated resources.
- **Lifelike Simulation**: Assesses long-term performance by modeling the progressive arrival of application placement requests and the subsequent release of resources at the end of their lifetime.

### Running Tests

To execute a **load test**, use the following command:

```bash
python test.py -test load <LOAD_TEST_PARAMETERS>
```

To execute a **simulation test**, use the following command:

```bash
python test.py -test simulation <SIMULATION_TEST_PARAMETERS>
```

### Test Parameters

#### Load Test Only

- **num_iterations**: Number of iterations for the load test. Each iteration consists of placing a random application on a random infrastructure (or GÉANT if selected) under load conditions of `30%`, `40%`, `50%`, `60%`, and `70%`. A load condition of `X%` means that X% of each node hardware and link bandwidth resources of the infrastructure are considered already allocated. Default is `100`.

#### Simulation Test Only

- **num_v_net**: Number of application placement requests in the simulation.
- **arrival_rate**: Arrival rate of requests in the simulation.

#### Common Parameters (Both Tests)

- **timeout**: The maximum number of seconds for Prolog execution in both symbolic and hybrid approaches. If the timeout expires, the placement fails. Default is `300`.
- **test_neural**: Whether to test the neural approach. Default is `True`.
- **test_symbolic**: Whether to test the symbolic approach (FogBrainX+). Default is `True`.
- **test_hybrid**: Whether to test the hybrid approach (neural + symbolic). Default is `True`.
- **save_dir**: Path to save the generated results. Default is `test/results`.
- **pretrained_model_path**: Path to the pretrained neural model to use in the test. This parameter is **required** if `test_neural` and/or `test_hybrid` is `True`.
- **topology**: The topology of the infrastructures used for test (`Waxman` or `GÉANT`). Default is `Waxman`.
- **num_nodes**: The number of nodes in the infrastructures generated for the test. This parameter is **required** if GÉANT is not the selected `topology`.
- **seed**: Seed used for the random generation in the test; same seed guarantees reproducibility. Default is `88`.

### Configuration Notes

The parameters used to train the neural model (if tested) are automatically retrieved from the `config.yaml` file present in the parent directory of the model. If the file is not found, the training parameters used must be provided together with the test parameters.

In both cases, it is possible to change all parameters that have no effect on the model architecture (for instance, `reusable` and `mask_actions`) to study the behavior of the model under different conditions with respect to those used in training.

### Example Commands

**Load test with only symbolic approach:**
```bash
python test.py -test load -save_dir test/results/load_100 -num_nodes 100 -test_neural False -test_hybrid False
```

**Simulation test with all three approaches:**
```bash
python test.py -test simulation -pretrained_model_path save/fogblend/model/model.pkl -save_dir test/results/simulation_geant -topology geant -arrival_rate 0.006
```


## Plotting Commands

The `plot.py` file is provided in the `test/` folder to generate plots of the experiment results.

### Configuration

At the beginning of the file, several static variables can be configured:

#### Plot Behavior

- **TOTAL_TIME_PLOT**: If `True`, the execution time plots will consider both successes and failures to compute the inference average time; otherwise, only successes will be considered.
- **USE_PAPER_RESULTS**: If `True`, the available success rate results from the paper (simulation on GÉANT and 100-node infrastructures) will be included in the plots for comparison. Otherwise, if provided, the `original_dir` files will be used to compute the success rate of the original architecture (FlagVNE).
- **SYMBOLIC_EXEC_THRESHOLD**: An array containing the thresholds to be considered for generating the plot of simulation success rates for different execution time thresholds.

#### Visualization Customization

- **NAME_MAPPING**: A dictionary to personalize the method names displayed in the plots.
- **COLORS**: A dictionary to personalize the colors assigned to each method in the plots.
- **{X}_FONTSIZE**: Variables to set the font size of various plot components (title, legend, axes, etc.).

### Selecting Data for Plotting

To select the data to consider for plotting, modify the `FOLDER_LIST_LOAD` and `FOLDER_LIST_SIMULATION` arrays. Each entry has the form:

```python
(folder_dir, topology, original_dir)
```

Where:

- **folder_dir**: The directory of the target folder run (for load test) or the directory of the folder containing one or more simulation runs (if multiple runs are present, results will be aggregated).
- **topology**: The topology used in the experiments (e.g., `GÉANT`, `Waxman`).
- **original_dir**: Same as **folder_dir** but considering results obtained with another neural model (e.g., FlagVNE). If provided, the results will be added to the plot for comparison; if `None`, it will be ignored.

### Generating Plots

To generate the plots, execute the following command from the root directory:

```bash
python test/plot.py
```