import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

# SETTINGS
LABELS_FONTSIZE = 15
AXIS_FONTSIZE = 18
TITLE_FONTSIZE = 20
TICKS_FONTSIZE = 15
LEGEND_FONTSIZE = 15
TOTAL_TIME_PLOT = False  # Set to True to plot total time, False for success time only

LOADS = [0.30, 0.40, 0.50, 0.60, 0.70]
SYMBOLIC_EXEC_THRESHOLD = [5, 10, 30, 60, 120, 180, 300]

NAME_MAPPING = {
    'original': 'FlagVNE',
    'neural': 'FlagVNE\u207A',
    'symbolic': 'FogBrainX\u207A',
    'hybrid': 'FogBlend'
}

COLORS = {
    NAME_MAPPING['original']: '#A1C9F4', 
    NAME_MAPPING['neural']: '#1f77b4', 
    NAME_MAPPING['symbolic']: '#ff7f0e', 
    NAME_MAPPING['hybrid']: '#2ca02c'
}

PAPER_SIM_RESULTS = {
    'GÉANT': 75.4,
    '100': 80.4 
}

USE_PAPER_RESULTS = True  # Set to True to include paper results in simulation plots (success rate)

# DATA FOLDERS
FOLDER_LIST_LOAD = [
        ("test/results/load_100/run_2025-07-13_22h05m09s", "Waxman 100", "test/results/load_100_original/run_2025-08-04_20h21m11s"),
        ("test/results/load_500/run_2025-07-13_23h59m10s", "Waxman 500", "test/results/load_500_original/run_2025-08-04_21h54m58s"),
]

FOLDER_LIST_SIMULATION = [
        ("test/results/simulations_geant", "GÉANT", "test/results/simulations_geant_original"),
        ("test/results/simulations_100", "Waxman 100", "test/results/simulations_100_original"),
        ("test/results/simulations_500", "Waxman 500", "test/results/simulations_500_original"),
]


# UTILITY FUNCTIONS
def compute_success_with_threshold(folder, thresholds):
    """Compute the success rate for the symbolic and hybrid methods with the given execution time thresholds"""
    all_data = []
    run_dirs = [d for d in glob(os.path.join(folder, '*')) if os.path.isdir(d) and not d.endswith('plots')]

    for run_dir in run_dirs:
        # Symbolic method success rate with threshold
        symbolic_file = os.path.join(run_dir, "symbolic_solution.csv")
        if os.path.exists(symbolic_file):
            symbolic_df = pd.read_csv(symbolic_file)
            success = symbolic_df["place_result"] & symbolic_df["route_result"]
            
            for threshold in thresholds:
                all_data.append({
                    'threshold': threshold,
                    'method': 'symbolic',
                    'success_rate': (success & (symbolic_df["elapsed_time"] <= threshold)).sum() / len(symbolic_df) * 100
                })
        
        # Hybrid method success rate with threshold
        neural_file = os.path.join(run_dir, "hybrid_solution_neural_phase.csv")
        hybrid_symbolic_file = os.path.join(run_dir, "hybrid_solution_symbolic_phase.csv")
        
        if os.path.exists(neural_file) and os.path.exists(hybrid_symbolic_file):
            neural_df = pd.read_csv(neural_file)
            symbolic_df = pd.read_csv(hybrid_symbolic_file)
            
            # Calculate total execution time for hybrid approach
            succ_neural = neural_df["place_result"] & neural_df["route_result"]
            succ_symbolic = symbolic_df["place_result"] & symbolic_df["route_result"]
            
            total_time = np.where(
                succ_neural,
                neural_df["elapsed_time"],
                np.where(succ_symbolic, 
                        neural_df["elapsed_time"] + symbolic_df["elapsed_time"],
                        np.inf)
            )
            
            for threshold in thresholds:
                all_data.append({
                    'threshold': threshold,
                    'method': 'hybrid',
                    'success_rate': (total_time <= threshold).sum() / len(neural_df) * 100
                })
    
    return pd.DataFrame(all_data)

    
def compute_hybrid_times(neural_phase_file, symbolic_phase_file):
    """Compute the average success and failure times for the hybrid approach."""
    # Check if files exist
    if not os.path.exists(neural_phase_file) or not os.path.exists(symbolic_phase_file):
        return 0, 0
    # Read the data for the given load
    neural_phase_df = pd.read_csv(neural_phase_file)
    symbolic_phase_df = pd.read_csv(symbolic_phase_file)

    # Merge the two dataframes on index
    merged_df = pd.merge(neural_phase_df, symbolic_phase_df, left_index=True, right_index=True, suffixes=('_neural', '_symbolic'))

    # Compute if placement and routing were successful in each phase
    succ_neural = merged_df["place_result_neural"] & merged_df["route_result_neural"]
    succ_symbolic = merged_df["place_result_symbolic"] & merged_df["route_result_symbolic"]

    # Compute hybrid success and failure times 
    success_times_neural = merged_df.loc[succ_neural, "elapsed_time_neural"]

    success_times_symbolic = (merged_df.loc[~succ_neural, "elapsed_time_neural"] 
                            + merged_df.loc[succ_symbolic, "elapsed_time_symbolic"])

    failure_times = (merged_df.loc[~(succ_neural | succ_symbolic), "elapsed_time_neural"] 
                    + merged_df.loc[~(succ_neural | succ_symbolic), "elapsed_time_symbolic"])

    # Compute averages, handling empty series
    avg_success_hybrid = np.mean(pd.concat([success_times_neural, success_times_symbolic])) if not pd.concat([success_times_neural, success_times_symbolic]).empty else 0
    avg_failure_hybrid = failure_times.mean() if not failure_times.empty else 0

    # Return the results
    return avg_success_hybrid, avg_failure_hybrid


def compute_load_hybrid_times(dir, loads):
    """Compute hybrid average success and failure times for different loads."""
    hybrid_data = []

    for load in loads:
        # Build the file paths
        string_load = str(load).replace('.', '_')
        neural_phase_file = os.path.join(dir, f"hybrid_solution_neural_phase_{string_load}.csv")
        symbolic_phase_file = os.path.join(dir, f"hybrid_solution_symbolic_phase_{string_load}.csv")

        # Skip if files do not exist
        if not os.path.exists(neural_phase_file) or not os.path.exists(symbolic_phase_file):
            continue
        
        avg_success_hybrid, avg_failure_hybrid = compute_hybrid_times(neural_phase_file, symbolic_phase_file)

        # Append the results
        hybrid_data.append({
            "infr_load": load,
            "avg_time_success_hybrid": avg_success_hybrid,
            "avg_time_failure_hybrid": avg_failure_hybrid
        })
    
    return pd.DataFrame(hybrid_data)


def get_simulation_results(folder):
    """Return all simulation results from a given folder"""
    all_data = []
    run_dirs = [d for d in glob(os.path.join(folder, '*')) if os.path.isdir(d) and not d.endswith('plots')]

    for run_dir in run_dirs:
        summary_file = os.path.join(run_dir, "summary_results.csv")
        if not os.path.exists(summary_file):
            continue
        
        df = pd.read_csv(summary_file)
        
        # Compute hybrid times once
        avg_success_hybrid, avg_failure_hybrid = compute_hybrid_times(
            os.path.join(run_dir, "hybrid_solution_neural_phase.csv"),
            os.path.join(run_dir, "hybrid_solution_symbolic_phase.csv"),
        )
        
        # Process all methods at once
        for _, row in df.iterrows():
            num_req = row['num_requests']
            if num_req == 0:
                continue
                
            methods_data = {
                'neural': {
                    'success': row['success_count_neural'],
                    'time_success': row['avg_time_success_neural'],
                    'time_failure': row['avg_time_failure_neural'],
                    'r2c': row['avg_r2c_ratio_neural']
                },
                'symbolic': {
                    'success': row['success_count_symbolic'],
                    'time_success': row['avg_time_success_symbolic'],
                    'time_failure': row['avg_time_failure_symbolic'],
                    'r2c': row['avg_r2c_ratio_symbolic']
                },
                'hybrid': {
                    'success': row['success_count_symbolic_phase'],
                    'time_success': avg_success_hybrid,
                    'time_failure': avg_failure_hybrid,
                    'r2c': row['avg_r2c_ratio_hybrid']
                }
            }
            
            for method, data in methods_data.items():
                failures = num_req - data['success']
                all_data.append({
                    'method': method,
                    'success_rate': data['success'] / num_req * 100,
                    'time_success': data['time_success'],
                    'time_total': (data['time_success'] * data['success'] + data['time_failure'] * failures) / num_req,
                    'r2c_ratio': data['r2c']
                })

    return pd.DataFrame(all_data)


# PLOTTING FUNCTIONS
def load_plot(folder, topology, original):
    """Plot load results from a given folder
    
    Args:
        folder (str): Path to the folder containing the results
        topology (str or int): Topology or number of nodes of the test infrastructure
        original (str): Path to the folder containing the results obtained with the original model architecture. If None, the plot will not include the original results.
    """
    # Create the folder to save the plots
    plot_dir = os.path.join(folder, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Load the data
    summary_df = pd.read_csv(os.path.join(folder, "summary_results.csv"))

    # Compute and add the hybrid average success and failure time columns 
    summary_df = summary_df.merge(
        compute_load_hybrid_times(folder, LOADS),
        on="infr_load",
        how="left"
    )

    # If original is provided, load the summary and merge it 
    if original:
        summary_df_original = pd.read_csv(os.path.join(original, "summary_results.csv"))
        summary_df = summary_df.merge(
            summary_df_original[['infr_load', 'avg_time_success_neural', 'avg_time_failure_neural', 'success_count_neural', 'avg_r2c_ratio_neural', 'num_requests']],
            on="infr_load",
            how="left",
            suffixes=('', '_original')
        )
    

    # SUCCESS RATE PLOT
    summary_df['success_rate_neural'] = summary_df['success_count_neural'] / summary_df['num_requests'] * 100
    summary_df['success_rate_symbolic'] = summary_df['success_count_symbolic'] / summary_df['num_requests'] * 100
    summary_df['success_rate_hybrid'] = summary_df['success_count_symbolic_phase'] / summary_df['num_requests'] * 100
    if original:
        summary_df['success_rate_neural_original'] = summary_df['success_count_neural_original'] / summary_df['num_requests_original'] * 100

    # Assign names to methods 
    method_cols_success = {}
    if original: 
        method_cols_success['success_rate_neural_original'] = NAME_MAPPING['original'] 
    method_cols_success.update({
    'success_rate_neural': NAME_MAPPING['neural'],
    'success_rate_symbolic': NAME_MAPPING['symbolic'],
    'success_rate_hybrid': NAME_MAPPING['hybrid']
    })
    
    # Plot
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=summary_df.melt(
            id_vars='infr_load',
            value_vars=list(method_cols_success.keys()),
            var_name='method',
            value_name='success_rate'
        ).replace({'method': method_cols_success}),
        x='infr_load',
        y='success_rate',
        hue='method',
        palette=COLORS
    )

    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%d%%', label_type='edge', padding=3, fontsize=LABELS_FONTSIZE, rotation=90)

    # Customize the plot
    if type(topology) is int:
        plt.title(f'Success Rate ({topology} Nodes - Load)', fontsize=TITLE_FONTSIZE)
    else:
        plt.title(f'Success Rate ({topology} - Load)', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Infrastructure Load (%)', fontsize=AXIS_FONTSIZE)
    plt.xticks([i for i in range(len(LOADS))], [str(int(l*100)) + "%" for l in LOADS], fontsize=TICKS_FONTSIZE)
    plt.ylim(0, 115)
    plt.ylabel('Success Rate (%)', fontsize=AXIS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.legend(title='Method', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'success_rate_by_load.png'))
    plt.close()
    

    # EXECUTION TIME PLOT
    methods_cols_time = {}
    if TOTAL_TIME_PLOT:
        # Compute average total time (success and failure) if required
        summary_df['avg_time_neural'] = (summary_df['avg_time_success_neural'] * summary_df['success_count_neural'] + summary_df['avg_time_failure_neural'] * (summary_df['num_requests'] - summary_df['success_count_neural'])) / summary_df['num_requests']
        summary_df['avg_time_symbolic'] = (summary_df['avg_time_success_symbolic'] * summary_df['success_count_symbolic'] + summary_df['avg_time_failure_symbolic'] * (summary_df['num_requests'] - summary_df['success_count_symbolic'])) / summary_df['num_requests']
        summary_df['avg_time_hybrid'] = (summary_df['avg_time_success_hybrid'] * summary_df['success_count_symbolic_phase'] + summary_df['avg_time_failure_hybrid'] * (summary_df['num_requests'] - summary_df['success_count_symbolic_phase'])) / summary_df['num_requests']
        methods_cols_time = {
            'avg_time_neural': NAME_MAPPING['neural'],
            'avg_time_symbolic': NAME_MAPPING['symbolic'],
            'avg_time_hybrid': NAME_MAPPING['hybrid']
        }
    else:
        # Otherwise, plot average success time only
        methods_cols_time = {
            'avg_time_success_neural': NAME_MAPPING['neural'],
            'avg_time_success_symbolic': NAME_MAPPING['symbolic'],
            'avg_time_success_hybrid': NAME_MAPPING['hybrid']
        }

    # Plot
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=summary_df.melt(
            id_vars='infr_load',
            value_vars=list(methods_cols_time.keys()),
            var_name='method',
            value_name='avg_time'
        ).replace({'method': methods_cols_time}),
        x='infr_load',
        y='avg_time',
        hue='method',
        palette=COLORS
    )

    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f s', label_type='edge', padding=3, fontsize=LABELS_FONTSIZE, rotation=90)
    
    # Customize the plot
    plt.xticks([i for i in range(len(LOADS))], [str(int(l*100)) + "%" for l in LOADS], fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.xlabel('Infrastructure Load (%)', fontsize=AXIS_FONTSIZE)
    plt.ylabel('Execution Time (s)', fontsize=AXIS_FONTSIZE)
    plt.yscale('log')
    plt.ylim(1e-2, 1e2)
    if type(topology) is int:
        plt.title(f'Execution Time ({topology} Nodes - Load)', fontsize=TITLE_FONTSIZE)
    else:
        plt.title(f'Execution Time ({topology} - Load)', fontsize=TITLE_FONTSIZE)
    plt.legend(title='Method', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'execution_time_by_load.png'))
    plt.close()


    # R2C RATIO PLOT
    methods_cols_r2c = {}
    if original:
        methods_cols_r2c['avg_r2c_ratio_neural_original'] = NAME_MAPPING['original']
    methods_cols_r2c.update({
        'avg_r2c_ratio_neural': NAME_MAPPING['neural'],
        'avg_r2c_ratio_symbolic': NAME_MAPPING['symbolic'],
        'avg_r2c_ratio_hybrid': NAME_MAPPING['hybrid']
    })

    # Plot
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(
        data=summary_df.melt(
            id_vars='infr_load',
            value_vars=list(methods_cols_r2c.keys()),
            var_name='method',
            value_name='avg_r2c_ratio'
        ).replace({'method': methods_cols_r2c}),
        x='infr_load',
        y='avg_r2c_ratio',
        hue='method',
        palette=COLORS
    )

    # Add value labels on top of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge', padding=3, fontsize=LABELS_FONTSIZE, rotation=90)

    # Customize the plot
    if type(topology) is int:
        plt.title(f'R2C Ratio ({topology} Nodes - Load)', fontsize=TITLE_FONTSIZE)
    else:
        plt.title(f'R2C Ratio ({topology} - Load)', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Infrastructure Load (%)', fontsize=AXIS_FONTSIZE)
    plt.xticks([i for i in range(len(LOADS))], [str(int(l*100)) + "%" for l in LOADS], fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.ylim(0.5, 1.8)
    plt.ylabel('R2C Ratio', fontsize=AXIS_FONTSIZE)
    plt.legend(title='Method', loc='best', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, ncols=2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'r2c_ratio_by_load.png'))
    plt.close()


def simulation_plot(folder, topology, original):
    """
    Plot simulation results from a given folder
    
    Args:
        folder (str): Path to the folder containing the results
        topology (str or int): Topology or number of nodes of the test infrastructure
        original (str): Path to the folder containing the results obtained with the original model architecture. If None, the plot will not include the original results.
    """
    # Create the folder to save the plots
    plot_dir = os.path.join(folder, "plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Load the data from all runs
    merged_df = get_simulation_results(folder)

    if original:
        merged_df_original = get_simulation_results(original)
        merged_df = pd.concat([
            merged_df,
            merged_df_original[merged_df_original['method'] == 'neural'].assign(method='original')
        ], ignore_index=True)    

    # Rename methods for better visualization
    merged_df['method'] = merged_df['method'].map(NAME_MAPPING)
    method_order = list(NAME_MAPPING.values()) if original else list(NAME_MAPPING.values())[1:]
    merged_df['method'] = pd.Categorical(merged_df['method'], categories=method_order, ordered=True) 

    # SUCCESS RATE PLOT
    success_df = merged_df.copy()

    # If specified, include paper results for comparison (replaces original method success rate)
    if USE_PAPER_RESULTS and str(topology) in PAPER_SIM_RESULTS:
        paper_value = PAPER_SIM_RESULTS[str(topology)]
        if original:
            # Update first occurence of original method success rate with the paper value and replace the others with NaN (avoid SD bar)
            first_index = success_df[success_df['method'] == NAME_MAPPING['original']].index[0]
            success_df.loc[first_index, 'success_rate'] = paper_value
            success_df.loc[(success_df['method'] == NAME_MAPPING['original']) & (success_df.index != first_index), 'success_rate'] = np.nan
        else:
            # Add a new row with the paper results
            success_df = pd.concat([
                pd.DataFrame([{
                    'method': NAME_MAPPING['original'],
                    'success_rate': paper_value,
                    'time_success': np.nan,
                    'time_total': np.nan,
                    'r2c_ratio': np.nan
                }]),
                success_df
            ], ignore_index=True)  

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=success_df,
        x='method',
        y='success_rate',
        hue='method',
        palette=COLORS,
        errorbar='sd',
        estimator='mean',
        capsize=0.05,
        err_kws={'linewidth':1.3}
    )
    
    # Add value labels on top of each bar 
    for container in ax.containers:
        labels = ax.bar_label(
            container,
            fmt='%.1f%%',
            label_type='edge',
            padding=2,
            fontsize=LABELS_FONTSIZE,
            rotation=0
        )
        # Shift each label slightly to the right
        for label in labels:
            label.set_x(label.get_position()[0] + 35)

    # Customize the plot
    if type(topology) is int:
        plt.title(f'Success Rate ({topology} Nodes - Simulation)', fontsize=TITLE_FONTSIZE)
    else:
        plt.title(f'Success Rate ({topology} - Simulation)', fontsize=TITLE_FONTSIZE)
    plt.ylabel('Success Rate (%)', fontsize=AXIS_FONTSIZE)
    plt.xlabel('', fontsize=AXIS_FONTSIZE)
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'success_rate_simulations.png'))
    plt.close()


    # EXECUTION TIME PLOT
    time_col = 'time_total' if TOTAL_TIME_PLOT else 'time_success'
    
    # Exclude original method from execution time plot (if present)
    plot_data = merged_df[merged_df['method'] != NAME_MAPPING['original']].copy()
    plot_data['method'] = pd.Categorical(plot_data['method'], categories=list(NAME_MAPPING.values())[1:], ordered=True)

    plt.figure(figsize=(7, 6))
    ax = sns.barplot(
        data=plot_data,
        x='method',
        y=time_col,
        hue='method',
        palette=COLORS,
        errorbar='sd',
        estimator='mean',
        capsize=0.05,
        err_kws={'linewidth':1.3}
    )

    # Add value labels on top of each bar
    for container in ax.containers:
        labels = ax.bar_label(
            container,
            fmt='%.2f s',
            label_type='edge',
            padding=2,
            fontsize=LABELS_FONTSIZE,
            rotation=0
        )
        # Shift each label slightly to the right
        for label in labels:
            label.set_x(label.get_position()[0] + 35)

    # Customize the plot
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.xlabel('', fontsize=AXIS_FONTSIZE)
    plt.ylabel('Execution Time (s)', fontsize=AXIS_FONTSIZE)

    if TOTAL_TIME_PLOT:
        plt.yscale('log')
        plt.ylim(1e-2, 1e2)
    else:
        plt.yscale('linear')
        stats = plot_data.groupby('method', observed=True)[time_col].agg(['mean', 'std'])
        plt.ylim(0, max(1.21, (stats["mean"] + stats["std"]).max() + 0.2))

    if type(topology) is int:
        plt.title(f'Execution Time ({topology} Nodes - Simulation)', fontsize=TITLE_FONTSIZE)
    else:
        plt.title(f'Execution Time ({topology} - Simulation)', fontsize=TITLE_FONTSIZE)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'execution_time_simulations.png'))
    plt.close()


    # R2C RATIO PLOT
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
        data=merged_df,
        x='method',
        y='r2c_ratio',
        hue='method',
        palette=COLORS,
        errorbar='sd',
        estimator='mean',
        capsize=0.05,
        err_kws={'linewidth':1.3}
    )

    # Add value labels on top of each bar
    for container in ax.containers:
        labels = ax.bar_label(
            container,
            fmt='%.2f',
            label_type='edge',
            padding=2,
            fontsize=LABELS_FONTSIZE,
            rotation=0
        )
        # Shift each label slightly to the right
        for label in labels:
            label.set_x(label.get_position()[0] + 30)
    
    # Customize the plot
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.ylabel('R2C Ratio', fontsize=AXIS_FONTSIZE)
    plt.xlabel('', fontsize=AXIS_FONTSIZE)

    if type(topology) is int:
        plt.title(f'R2C Ratio ({topology} Nodes - Simulation)', fontsize=TITLE_FONTSIZE)
    else:
        plt.title(f'R2C Ratio ({topology} - Simulation)', fontsize=TITLE_FONTSIZE)

    stats = merged_df.groupby('method', observed=True)['r2c_ratio'].agg(['mean', 'std']).fillna(0)
    plt.ylim(0.5, max(stats["mean"] + stats["std"]) + 0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'r2c_ratio_simulations.png'))
    plt.close()


    # SUCCESS RATE VS THRESHOLD PLOT
    threshold_df = compute_success_with_threshold(folder, SYMBOLIC_EXEC_THRESHOLD)
    threshold_df['method'] = threshold_df['method'].map(NAME_MAPPING)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=threshold_df,
        x='threshold',
        y='success_rate',
        hue='method',
        palette=COLORS,
        errorbar='sd',
        estimator='mean',
        capsize=0.05,
        err_kws={'linewidth': 1.5}
    )

    # Add value labels on top of each bar
    for i, container in enumerate(ax.containers):
        # i=0 is first method (symbolic), i=1 is second method (hybrid)
        if i == 0:  # Symbolic - labels below the bar top
            labels = ax.bar_label(
                container,
                fmt='%.1f%%',
                label_type='edge',  # or 'edge' with negative padding
                padding=-30,
                fontsize=LABELS_FONTSIZE,
                rotation=0
            )
        else:  # Hybrid - labels above the bar top
            labels = ax.bar_label(
                container,
                fmt='%.1f%%',
                label_type='edge',
                padding=15,
                fontsize=LABELS_FONTSIZE,
                rotation=0
            )

    # Get max success mean for each method between all thresholds
    stats = threshold_df.groupby(['method', 'threshold'], observed=True)['success_rate'].agg(['mean', 'std']).reset_index()
    max_success = stats.loc[stats.groupby('method')['mean'].idxmax()]

    # Add reference horizontal dotted lines from highest bars
    for _, row in max_success.iterrows():
        plt.axhline(y=row['mean'], color=COLORS[row['method']], linestyle='--', linewidth=1, alpha=0.7)

    # Customize the plot
    plt.xticks(fontsize=TICKS_FONTSIZE)
    plt.yticks(fontsize=TICKS_FONTSIZE)
    plt.legend(loc='upper left', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, ncols=2)
    # plt.legend(loc='upper left', title='Method', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, ncols=2)

    # Set ylim and remove 110 tick
    ax = plt.gca()
    ax.set_ylim(50, 113)
    ticks = ax.get_yticks()
    ticks = [t for t in ticks if round(t) != 110]
    ax.set_yticks(ticks)
    ax.set_ylim(50, 113)

    if isinstance(topology, int):
        plt.title(f'Success Rate vs Execution Time Threshold ({topology} Nodes - Simulation)', fontsize=TITLE_FONTSIZE-2)
    else:
        plt.title(f'Success Rate vs Execution Time Threshold ({topology} - Simulation)', fontsize=TITLE_FONTSIZE-2)

    plt.xlabel('Execution Time Threshold (s)', fontsize=AXIS_FONTSIZE)
    plt.ylabel('Success Rate (%)', fontsize=AXIS_FONTSIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'success_threshold_simulation.png'))
    plt.close()
    


if __name__ == "__main__":
    
    # For every load results folder
    for folder, topology, original in FOLDER_LIST_LOAD:
        if os.path.isdir(folder):
            print(f"Plotting load {folder} with topology {topology}")
            # Call the plot function
            load_plot(folder, topology, original)

    # For every simulation results folder
    for folder, topology, original in FOLDER_LIST_SIMULATION:
        if os.path.isdir(folder):
            print(f"Plotting simulation {folder} with topology {topology}")
            # Call the plot function
            simulation_plot(folder, topology, original)