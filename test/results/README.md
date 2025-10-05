## Results Folder

This folder contains experimental results demonstrating that **FogBlend** successfully combines the speed of neural inference with the reliability of symbolic reasoning to achieve a high percentage of valid placements, providing improved placement quality and scalability compared to purely symbolic or purely neural approaches.

The experiments are divided into two categories:

- **Load Test**: evaluates the performance of placing a single application on an random infrastructure with a predetermined percentage of pre-allocated resources (each inference is independent).
- **Lifelike Simulation**: assesses long-term performance by modeling the progressive arrival of application placement requests and the subsequent release of resources at the end of their lifetime (each inference influences the rest of the simulation and a single infrastructure is used).

### Folder Naming Convention

Each folder follows this naming pattern:
```bash
[test_type]_[topology]_<original>
```

where:

- **test_type**: indicates whether the results are from "load" or "simulation" tests
- **topology**: specifies the infrastructure topology used in the experiments (GÉANT, Waxman 100 nodes, Waxman 500 nodes)
- **original**: (optional) indicates that results were obtained using the neural model with the original architecture introduced by [FlagVNE](https://github.com/GeminiLight/flag-vne)

### Run Directory Structure

Each configuration folder contains experimental runs, organized in directories following this pattern:
```bash
run_[date_of_execution]
```
**Note for Load Tests**: load test folders contain only a single run, as each run already performs multiple inferences on different randomly generated infrastructures, providing comprehensive performance statistics for the experimental configuration.

### Files in Each Run Directory

Each run directory contains the following files:

- **config.yaml**: contains all parameter values used for the experiment
- **summary_result.csv**: provides an overview of results highlighting key metrics (average execution time, success rate, etc.)
- **neural_solution.csv**: contains detailed information about each inference performed by the neural model (e.g., chosen placement, allocated resources, elapsed time, etc.)
- **symbolic_solution.csv**: contains detailed information about each inference performed by the symbolic model
- **hybrid_solution_neural_phase.csv**: contains detailed information about each inference performed by the neural phase (FlagVNE+) of **FogBlend**
- **hybrid_solution_symbolic_phase.csv**: contains detailed information about each inference performed by the symbolic phase (FogBrainX+) of **FogBlend**

**Note for Load Tests**: Files are labeled with the infrastructure target load (i.e., 0_3 = 30%, 0_4 = 40%, etc.).

All results for different models within a single run were executed on the same infrastructures with identical placement requests, ensuring direct comparability.

### Visualization

For each experimental configuration is present a "plot" folders containing comparison charts of the various methods. In case of simulation tests the results provided represent the aggregation of all executed runs.