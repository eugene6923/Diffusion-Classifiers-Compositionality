"""Script to analyze saved scores from a specific wandb run.
Loads results from .pt files and computes various statistics about timestep influence.
"""
import torch
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import json
from collections import defaultdict
from typing import List, Dict, Optional
import pickle

import seaborn as sns
sns.reset_defaults()
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

# global sizes of font 8
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8
plt.rcParams['axes.titlesize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

def find_wandb_dir(base_dir, wandb_id):
    """Find the directory containing the wandb ID by searching immediate subdirs.
    
    Args:
        base_dir (str): Base directory to search in
        wandb_id (str): The wandb run ID to find
        
    Returns:
        str: Full path to the wandb run directory or None if not found
    """
    # Check immediate subdirectories
    for subdir in os.listdir(base_dir):
        full_subdir = os.path.join(base_dir, subdir)
        if not os.path.isdir(full_subdir):
            continue
            
        # Check if wandb_id exists in this subdir
        potential_path = os.path.join(full_subdir, wandb_id)
        if os.path.exists(potential_path):
            return potential_path
            
    return None

def load_metadata(wandb_id):
    """Load metadata for a specific wandb run.
    
    Args:
        wandb_id (str): The wandb run ID
        
    Returns:
        dict: Metadata for the run or None if not found
    """
    metadata_dir = '/mnt/lustre/work/oh/owl661/compositional-vaes/cached_wandb_metadata'
    all_metadata_path = os.path.join(metadata_dir, "all_runs_metadata.pkl")
    
    try:
        with open(all_metadata_path, 'rb') as f:
            data = pickle.load(f)
            
        # Get the run data
        run_data = data['runs'].get(wandb_id)
        if run_data is None:
            print(f"Warning: No metadata found for {wandb_id}")
            return None
            
        # Extract config and summary
        config = {}
        summary = {}
        if isinstance(run_data, dict):
            if 'config' in run_data:
                raw_config = run_data['config']
                if hasattr(raw_config, 'items'):
                    config = {k: v for k, v in raw_config.items()}
                elif isinstance(raw_config, dict):
                    config = raw_config
            if 'summary' in run_data:
                summary = run_data['summary']
        
        version = config.get('version')
        # For SD 3-m, append resize info to version
        if version == '3-m':
            sd3_resize = config.get('sd3_resize', False)
            version = f"3-m {'(resize)' if sd3_resize else '(no-resize)'}"
            
            # if sd3_resize:
            #     raise ValueError("SD 3-m with resize is not supported yet.")
        
        # Return processed metadata
        return {
            'task': config.get('task'),
            'geneval_version': config.get('geneval_version'),
            'version': version,
            'sd_version': version,  # alias for version
            'model_type': config.get('model_type', version),  # fallback to version if model_type not present
            'use_normed_classifier': config.get('use_normed_classifier', True),
            'accuracy': summary.get('accuracy'),
            'best_accuracy': summary.get('best_accuracy'),
            'final_accuracy': summary.get('final_accuracy'),
            'sampling_steps': config.get('sampling_steps', 5000)
        }
        
    except Exception as e:
        print(f"Error loading metadata for {wandb_id}: {e}")
        return None

def load_and_analyze_scores(base_dir, wandb_id, similarity_method='l2'):
    """Load and analyze scores from all batches in a wandb run.
    
    Args:
        base_dir (str): Base directory containing score results
        wandb_id (str): The wandb run ID to analyze
        similarity_method (str): Method to use for similarity calculation
    """
    # Load metadata first
    metadata = load_metadata(wandb_id)
    if metadata is not None:
        print("\nRun Metadata:")
        print(f"Task: {metadata.get('task', 'N/A')}")
        print(f"Model Type: {metadata.get('model_type', 'N/A')}")
        print(f"SD Version: {metadata.get('sd_version', 'N/A')}")
        print(f"Geneval Version: {metadata.get('geneval_version', 'N/A')}\n")
    
    # First find the correct directory containing the wandb ID
    score_dir = find_wandb_dir(base_dir, wandb_id)
    if score_dir is None:
        print(f"Could not find wandb ID {wandb_id} in any subdirectory of {base_dir}")
        return None
        
    print(f"Found wandb run in: {score_dir}")
    
    # Get all score files
    score_files = sorted(glob.glob(os.path.join(score_dir, '*_scores.pt')))
    
    if not score_files:
        print(f"No score files found in {score_dir}")
        return None
    
    print(f"Found {len(score_files)} score files")
    
    # Load all data and concatenate
    all_scores_with_timestep = []  # Will be [num_files, num_timesteps, num_prompts, batch_size]
    all_correct_indices = []       # Will be [num_files * batch_size]
    all_timesteps = None          # Will store timesteps (should be same for all files)
    
    print("Loading all data...")
    for score_file in tqdm(score_files):
        data = torch.load(score_file)
        scores_with_timestep = data['scores_with_timestep'].numpy()  # [num_timesteps, num_prompts, batch_size]
        correct_indices = data['correct_indices']
        
        all_scores_with_timestep.append(scores_with_timestep)
        all_correct_indices.extend(correct_indices)
        
        if all_timesteps is None:
            all_timesteps = data['all_timesteps_used']
        else:
            assert np.array_equal(all_timesteps, data['all_timesteps_used']), "Timesteps differ between files!"
    
    # Concatenate along batch dimension
    scores = np.concatenate(all_scores_with_timestep, axis=2).astype(np.float32)  # [num_timesteps, num_prompts, total_samples]
    correct_indices = np.array(all_correct_indices)
    
    # get average accuracy as well.
    preds_mean = np.mean(scores, axis=0)  # [num_prompts, total_samples]
    # Create boolean mask for all indices, then invert it to get remaining indices
    correct_scores = preds_mean[correct_indices, np.arange(preds_mean.shape[1])]
    num_smaller = (preds_mean < correct_scores).sum(axis=0)
    acc_mean = np.mean(num_smaller == 0)
    print(f"\nAnalyzing {scores.shape[2]} total samples across {scores.shape[0]} timesteps")
    print(f"Average accuracy: {acc_mean:.4f}")
    
    # get softmax over the whole scores
    # Convert scores to tensor and apply softmax across all samples
    scores_tensor = torch.from_numpy(scores).to(torch.float32).mean(axis=0)       # [num_timesteps, num_prompts, total_samples]
    softmax_scores = torch.nn.functional.softmax(-scores_tensor, dim=0).numpy()  # Negative since lower scores are better
    max_softmax_scores = np.max(softmax_scores, axis=0)
    avg_max_softmax_scores = np.mean(max_softmax_scores)
    top_1_probs = []
    
    # Calculate per-timestep statistics
    timestep_stats = {}
    for t in range(scores.shape[0]):
        timestep_scores = scores[t]  # [num_prompts, total_samples]
        
        # Calculate true vs incorrect errors
        true_errors = []
        incorrect_errors = []
        accuracies = []
        influences = []
        influence_stds = []
        
        for sample_idx in range(scores.shape[2]):
            correct_idx = correct_indices[sample_idx]
            true_error = timestep_scores[correct_idx, sample_idx]
            incorrect_error = np.mean([
                timestep_scores[i, sample_idx] 
                for i in range(scores.shape[1]) 
                if i != correct_idx
            ])
            true_errors.append(true_error)
            incorrect_errors.append(incorrect_error)
        
        # Calculate accuracy
        min_scores = np.min(timestep_scores, axis=0, keepdims=True)
        is_min = timestep_scores == min_scores
        has_tie = np.sum(is_min, axis=0) > 1
        predictions = np.argmax(timestep_scores, axis=0)
        predictions[has_tie] = -1
        accuracy = np.mean(predictions == correct_indices)
        
        # get softmax over the whole timestep_scores using safe version
        timestep_scores_tensor = torch.from_numpy(timestep_scores)
        softmax_scores = torch.nn.functional.softmax(-timestep_scores_tensor, dim=0).numpy().max(axis=0)
        
        # top_1_probs.append(np.mean(softmax_scores))
        
        # Calculate influence (difference between true and incorrect errors)
        influence = np.mean(np.array(incorrect_errors) - np.array(true_errors))
        influence_std = np.std(np.array(incorrect_errors) - np.array(true_errors))
        
        timestep_stats[t] = {
            'true_errors': np.mean(true_errors),
            'incorrect_errors': np.mean(incorrect_errors),
            'accuracy': accuracy,
            'influence': influence,
            'influence_std': influence_std,
            'top_1_probs': np.mean(softmax_scores)
        }
    
    # # Create visualizations
    os.makedirs('figures_002', exist_ok=True)
    
    # # Plot true vs incorrect errors
    # plt.figure(figsize=(10, 6))
    # plt.plot(all_timesteps, [stats['true_errors'] for stats in timestep_stats.values()], 
    #          label='True Errors')
    # plt.plot(all_timesteps, [stats['incorrect_errors'] for stats in timestep_stats.values()], 
    #          label='Incorrect Errors', linestyle='--')
    # plt.xlabel('Timestep Value')
    # plt.ylabel('Error')
    # plt.title(f'Error Analysis (Run: {wandb_id})')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'figures_002/error_analysis_{wandb_id}.pdf')
    # plt.close()
    
    # # Plot accuracies
    # plt.figure(figsize=(10, 6))
    # plt.plot(all_timesteps, [stats['accuracy'] for stats in timestep_stats.values()], 'o-')
    # plt.xlabel('Timestep Value')
    # plt.ylabel('Accuracy')
    # plt.title(f'Accuracy vs Timestep (Run: {wandb_id})')
    # plt.grid(True)
    # plt.savefig(f'figures_002/accuracy_vs_timestep_{wandb_id}.pdf')
    # plt.close()
    
    # # Plot influences
    # plt.figure(figsize=(10, 6))
    # influences = [stats['influence'] for stats in timestep_stats.values()]
    # influence_stds = [stats['influence_std'] for stats in timestep_stats.values()]
    # plt.plot(all_timesteps, influences, 'o-')
    # plt.fill_between(all_timesteps, 
    #                 np.array(influences) - np.array(influence_stds),
    #                 np.array(influences) + np.array(influence_stds),
    #                 alpha=0.2)
    # plt.xlabel('Timestep Value')
    # plt.ylabel('Influence')
    # plt.title(f'Timestep Influence (Run: {wandb_id})')
    # plt.grid(True)
    # plt.savefig(f'figures_002/timestep_influence_{wandb_id}.pdf')
    # plt.close()
    
    return {
        'timestep_stats': timestep_stats,
        'all_timesteps': all_timesteps,
        'scores_shape': scores.shape,
        'acc_mean': acc_mean,
        'avg_max_softmax_scores': avg_max_softmax_scores
    }

def analyze_multiple_runs(base_dir: str, wandb_ids: List[str], 
                        row_key: str = 'geneval_version', 
                        col_key: str = 'task'):
    """Analyze multiple runs and create grid plots based on metadata categories.
    
    Args:
        base_dir (str): Base directory containing score results
        wandb_ids (List[str]): List of wandb run IDs to analyze
        row_key (str): Metadata key to use for organizing rows
        col_key (str): Metadata key to use for organizing columns
    """
    # First collect all results and metadata
    all_results = {}
    all_metadata = {}
    row_values = set()
    col_values = set()
    sd_versions = set()
    
    show_softmax=True
    
    print("Analyzing runs...")
    for wandb_id in tqdm(wandb_ids):
        results = load_and_analyze_scores(base_dir, wandb_id)
        metadata = load_metadata(wandb_id)
        
        if results is not None and metadata is not None:
            all_results[wandb_id] = results
            all_metadata[wandb_id] = metadata
            row_values.add(metadata.get(row_key, 'N/A'))
            col_values.add(metadata.get(col_key, 'N/A'))
            sd_versions.add(metadata.get('sd_version', 'N/A'))
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Sort the values for consistent ordering
    row_values = sorted(list(row_values), key=str)
    col_values = sorted(list(col_values), key=str) 
    sd_versions = sorted(list(sd_versions), key=str)
    
    # Create grid of plots
    n_rows = len(row_values)
    n_cols = len(col_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create colorblind-friendly color map for sd versions
    colors = sns.color_palette("colorblind", n_colors=len(sd_versions))
    sd_version_colors = dict(zip(sd_versions, colors))
    
    # Group runs by row and column categories
    grouped_runs = defaultdict(lambda: defaultdict(list))
    for wandb_id, metadata in all_metadata.items():
        row_val = metadata.get(row_key, 'N/A')
        col_val = metadata.get(col_key, 'N/A')
        sd_version = metadata.get('sd_version', 'N/A')
        grouped_runs[(row_val, col_val)][sd_version].append(wandb_id)
    
    # Plot each cell in the grid
    for i, row_val in enumerate(row_values):
        for j, col_val in enumerate(col_values):
            ax = axes[i, j]
            sd_version_runs = grouped_runs.get((row_val, col_val), {})
            
            if sd_version_runs:
                # Plot each SD version as a separate line
                for sd_version, runs in sd_version_runs.items():
                    # Collect accuracies for all runs of this SD version
                    all_accuracies = []
                    all_timesteps = None
                    
                    if len(runs) > 1:
                        print("WARNING SOMETIHNG WORNG FOUND 2 RUNS W?HAT!!!")

                    for wandb_id in runs:
                        results = all_results[wandb_id]
                        timestep_stats = results['timestep_stats']
                        if show_softmax:
                            accuracies = [stats['top_1_probs'] for stats in timestep_stats.values()]
                        else:
                            accuracies = [stats['accuracy'] for stats in timestep_stats.values()]
                        all_accuracies.append(accuracies)
                        
                        if all_timesteps is None:
                            all_timesteps = results['all_timesteps']
                    
                    if all_accuracies:
                        # Average accuracies across runs
                        mean_accuracies = np.mean(all_accuracies, axis=0)
                        std_accuracies = np.std(all_accuracies, axis=0)
                        
                        # Plot mean with error bands
                        color = sd_version_colors[sd_version]
                        ax.plot(all_timesteps, mean_accuracies, '-', 
                               label=f'SD {sd_version}', color=color, alpha=0.8)
                        # ax.fill_between(all_timesteps, 
                        #               mean_accuracies - std_accuracies,
                        #               mean_accuracies + std_accuracies,
                        #               color=color, alpha=0.2)
                        
                        if show_softmax:
                            # show as dashed line the average max softmax scores
                            ax.axhline(results['avg_max_softmax_scores'], color=color, linestyle='--', alpha=0.8)
                        else:
                            # show as dashed line the average accuracy (shoudl only be 1)
                            ax.axhline(results['acc_mean'], color=color, linestyle='--', alpha=0.8)
                            
                        
                        
            
            ax.set_xlabel('Timestep Value')
            if show_softmax:
                ax.set_ylabel('Top 1 Probability')
            else:
                ax.set_ylabel('Accuracy')
            ax.grid(True)
            
            if i == 0:
                ax.set_title(f'{col_key}={col_val}')
            if j == 0:
                if show_softmax:
                    ax.set_ylabel(f'{row_key}={row_val}\nTop 1 Probability')
                else:
                    ax.set_ylabel(f'{row_key}={row_val}\nAccuracy')
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    # Create common legend
    handles, labels = [], []
    for sd_version in sd_versions:
        color = sd_version_colors[sd_version]
        handles.append(plt.Line2D([0], [0], color=color, alpha=0.8))
        labels.append(f'SD {sd_version}')
    
    # Add legend below the subplots
    fig.legend(handles, labels, 
              loc='center', 
              bbox_to_anchor=(0.5, 0.02),
              ncol=4,
              frameon=False)
    
    # Adjust bottom margin to accommodate legend
    plt.subplots_adjust(bottom=0.15)
    
    os.makedirs('figures_002', exist_ok=True)
    plt.savefig(f'figures_002/grid_comparison_{row_key}_vs_{col_key}_softmax_{show_softmax}.pdf')
    plt.close()
    
    # Print summary statistics for each group
    print("\nSummary Statistics by Group:")
    for row_val in row_values:
        for col_val in col_values:
            sd_version_runs = grouped_runs.get((row_val, col_val), {})
            if sd_version_runs:
                print(f"\n{row_key}={row_val}, {col_key}={col_val}:")
                for sd_version, runs in sd_version_runs.items():
                    print(f"\n  SD Version: {sd_version}")
                    for wandb_id in runs:
                        results = all_results[wandb_id]
                        timestep_stats = results['timestep_stats']
                        all_timesteps = results['all_timesteps']
                        
                        # Get best accuracy and most influential timestep
                        best_accuracy_t = max(timestep_stats.items(), 
                                           key=lambda x: x[1]['accuracy'])
                        most_influential_t = max(timestep_stats.items(), 
                                              key=lambda x: x[1]['influence'])
                        
                        print(f"    Run: {wandb_id}")
                        print(f"    Best accuracy: {best_accuracy_t[1]['accuracy']:.4f} at timestep {all_timesteps[best_accuracy_t[0]]}")
                        print(f"    Most influential timestep: {all_timesteps[most_influential_t[0]]} (influence: {most_influential_t[1]['influence']:.4f})")

if __name__ == "__main__":
    # Base directory for score results
    BASE_DIR = '/mnt/lustre/work/oh/owl661/compositional-vaes/score_results'
    
    # List of wandb IDs to analyze
    WANDB_IDS = """https://wandb.ai/oshapio/diffusion-itm/runs/zrwirsyp
https://wandb.ai/oshapio/diffusion-itm/runs/bl7fm3pg
https://wandb.ai/oshapio/diffusion-itm/runs/ic2a0sgv
https://wandb.ai/oshapio/diffusion-itm/runs/w6t6nzl5
https://wandb.ai/oshapio/diffusion-itm/runs/4c28wxjf
https://wandb.ai/oshapio/diffusion-itm/runs/n8s9tam6



https://wandb.ai/oshapio/diffusion-itm/runs/iqu5fynu
https://wandb.ai/oshapio/diffusion-itm/runs/zfawifi9
https://wandb.ai/oshapio/diffusion-itm/runs/cqdhg22k
https://wandb.ai/oshapio/diffusion-itm/runs/junv7psv
https://wandb.ai/oshapio/diffusion-itm/runs/zy0ojl6s
https://wandb.ai/oshapio/diffusion-itm/runs/d7txlos7



https://wandb.ai/oshapio/diffusion-itm/runs/y7d8gbo8
https://wandb.ai/oshapio/diffusion-itm/runs/zh96em4j
https://wandb.ai/oshapio/diffusion-itm/runs/ksx9w111
https://wandb.ai/oshapio/diffusion-itm/runs/d1emcf0f
https://wandb.ai/oshapio/diffusion-itm/runs/sdfkhlzp
https://wandb.ai/oshapio/diffusion-itm/runs/9r93c5ju





https://wandb.ai/oshapio/diffusion-itm/runs/uglkshs3
https://wandb.ai/oshapio/diffusion-itm/runs/6nyo6yzk
https://wandb.ai/oshapio/diffusion-itm/runs/ncnnfhv9
https://wandb.ai/oshapio/diffusion-itm/runs/5p92ij8p
https://wandb.ai/oshapio/diffusion-itm/runs/fi1nfmw6
https://wandb.ai/oshapio/diffusion-itm/runs/f2mgeyda



https://wandb.ai/oshapio/diffusion-itm/runs/xk4s53n6
https://wandb.ai/oshapio/diffusion-itm/runs/edc8skul
https://wandb.ai/oshapio/diffusion-itm/runs/cfm8cp1w
https://wandb.ai/oshapio/diffusion-itm/runs/vdcithmx
https://wandb.ai/oshapio/diffusion-itm/runs/mcij8llj
https://wandb.ai/oshapio/diffusion-itm/runs/y52r3i7h



https://wandb.ai/oshapio/diffusion-itm/runs/pbd1r9hj
https://wandb.ai/oshapio/diffusion-itm/runs/n7k1f3up
https://wandb.ai/oshapio/diffusion-itm/runs/3qoxlqg3
https://wandb.ai/oshapio/diffusion-itm/runs/3wi84ft9
https://wandb.ai/oshapio/diffusion-itm/runs/fh8z1zlg
https://wandb.ai/oshapio/diffusion-itm/runs/9l4c64i0



https://wandb.ai/oshapio/diffusion-itm/runs/qrxo4qsd
https://wandb.ai/oshapio/diffusion-itm/runs/9cc86yeo
https://wandb.ai/oshapio/diffusion-itm/runs/d18s64d4
https://wandb.ai/oshapio/diffusion-itm/runs/ffvujtpe
https://wandb.ai/oshapio/diffusion-itm/runs/8p2uh8iv
https://wandb.ai/oshapio/diffusion-itm/runs/4m2n2m22



https://wandb.ai/oshapio/diffusion-itm/runs/6l9h48vb
https://wandb.ai/oshapio/diffusion-itm/runs/3fcj575m
https://wandb.ai/oshapio/diffusion-itm/runs/gpki7ab2
https://wandb.ai/oshapio/diffusion-itm/runs/zujzvqfa
https://wandb.ai/oshapio/diffusion-itm/runs/b55o6dzj
https://wandb.ai/oshapio/diffusion-itm/runs/11vkksd2



https://wandb.ai/oshapio/diffusion-itm/runs/ewc3rcez
https://wandb.ai/oshapio/diffusion-itm/runs/208jqzbt
https://wandb.ai/oshapio/diffusion-itm/runs/i6wobrtj
https://wandb.ai/oshapio/diffusion-itm/runs/nrw8axi7
https://wandb.ai/oshapio/diffusion-itm/runs/gsyk06zt
https://wandb.ai/oshapio/diffusion-itm/runs/v9ihg2wd

https://wandb.ai/oshapio/diffusion-itm/runs/qnd422z0
https://wandb.ai/oshapio/diffusion-itm/runs/3csiyiwj
https://wandb.ai/oshapio/diffusion-itm/runs/2ogb4dsr
https://wandb.ai/oshapio/diffusion-itm/runs/d0f1zktp
https://wandb.ai/oshapio/diffusion-itm/runs/kcv77bsz
https://wandb.ai/oshapio/diffusion-itm/runs/mwjlfx6v
https://wandb.ai/oshapio/diffusion-itm/runs/pyt1r0tk
https://wandb.ai/oshapio/diffusion-itm/runs/opxjon02
https://wandb.ai/oshapio/diffusion-itm/runs/ot61afj9
https://wandb.ai/oshapio/diffusion-itm/runs/rc6ejogc
https://wandb.ai/oshapio/diffusion-itm/runs/b7ugg7ni
https://wandb.ai/oshapio/diffusion-itm/runs/vv8u01mr
https://wandb.ai/oshapio/diffusion-itm/runs/47mw60j7
https://wandb.ai/oshapio/diffusion-itm/runs/pofs27kj
https://wandb.ai/oshapio/diffusion-itm/runs/s7pqiyl1
https://wandb.ai/oshapio/diffusion-itm/runs/ip0drr6k
https://wandb.ai/oshapio/diffusion-itm/runs/u31r9mbb
https://wandb.ai/oshapio/diffusion-itm/runs/j4eryxha"""


    WANDB_IDS = """
https://wandb.ai/oshapio/diffusion-itm/runs/xc91lbl4
https://wandb.ai/oshapio/diffusion-itm/runs/wainv9ia
https://wandb.ai/oshapio/diffusion-itm/runs/5heqmzaa
https://wandb.ai/oshapio/diffusion-itm/runs/rby9c34l
https://wandb.ai/oshapio/diffusion-itm/runs/jw0n5mh3
https://wandb.ai/oshapio/diffusion-itm/runs/i1oceqe0
https://wandb.ai/oshapio/diffusion-itm/runs/rolhpq66
https://wandb.ai/oshapio/diffusion-itm/runs/3eswfles
https://wandb.ai/oshapio/diffusion-itm/runs/40k279yr
https://wandb.ai/oshapio/diffusion-itm/runs/6e6yjkdb
https://wandb.ai/oshapio/diffusion-itm/runs/5u46ok76
https://wandb.ai/oshapio/diffusion-itm/runs/5yr1cswj
https://wandb.ai/oshapio/diffusion-itm/runs/a2l2tzh9
https://wandb.ai/oshapio/diffusion-itm/runs/i5vhvr5d
https://wandb.ai/oshapio/diffusion-itm/runs/65ej1cc1
https://wandb.ai/oshapio/diffusion-itm/runs/g8nx1mr8
https://wandb.ai/oshapio/diffusion-itm/runs/wmjragri
https://wandb.ai/oshapio/diffusion-itm/runs/wopg1rql
https://wandb.ai/oshapio/diffusion-itm/runs/so8t2hzg
https://wandb.ai/oshapio/diffusion-itm/runs/iz4or0ni
https://wandb.ai/oshapio/diffusion-itm/runs/m3e0zl2y
https://wandb.ai/oshapio/diffusion-itm/runs/b90gcpfh
https://wandb.ai/oshapio/diffusion-itm/runs/szzhl92b
https://wandb.ai/oshapio/diffusion-itm/runs/tol806zv
https://wandb.ai/oshapio/diffusion-itm/runs/h6xep1mr
https://wandb.ai/oshapio/diffusion-itm/runs/riplahgz
https://wandb.ai/oshapio/diffusion-itm/runs/ulc10gu3
https://wandb.ai/oshapio/diffusion-itm/runs/6xktwzdj
https://wandb.ai/oshapio/diffusion-itm/runs/jyam2r7g
https://wandb.ai/oshapio/diffusion-itm/runs/0m2ku1m1
https://wandb.ai/oshapio/diffusion-itm/runs/02z5dvjr
https://wandb.ai/oshapio/diffusion-itm/runs/g7l80p48
https://wandb.ai/oshapio/diffusion-itm/runs/ioue7vpe
https://wandb.ai/oshapio/diffusion-itm/runs/1ivvzc2y
https://wandb.ai/oshapio/diffusion-itm/runs/dlpn7278
https://wandb.ai/oshapio/diffusion-itm/runs/5d27oyfv
https://wandb.ai/oshapio/diffusion-itm/runs/xobikc0u
https://wandb.ai/oshapio/diffusion-itm/runs/d1gribr8
https://wandb.ai/oshapio/diffusion-itm/runs/072kjyl4
https://wandb.ai/oshapio/diffusion-itm/runs/1dm0485w
https://wandb.ai/oshapio/diffusion-itm/runs/qzbxlhph
https://wandb.ai/oshapio/diffusion-itm/runs/3dmdmfp3
https://wandb.ai/oshapio/diffusion-itm/runs/yuuzn8y7
https://wandb.ai/oshapio/diffusion-itm/runs/zdn5iq84
https://wandb.ai/oshapio/diffusion-itm/runs/3d8eofvi
https://wandb.ai/oshapio/diffusion-itm/runs/rm5rl56r
https://wandb.ai/oshapio/diffusion-itm/runs/yn99jezw
https://wandb.ai/oshapio/diffusion-itm/runs/l0l63dyi"""


    # Parse wandb IDs from URLs by taking last part after /
    WANDB_IDS = [url.strip().split('/')[-1] for url in WANDB_IDS.strip().split('\n') if url.strip()]
    
    # Analyze runs and create grid plots
    # You can change row_key and col_key to any metadata fields you want to compare
    analyze_multiple_runs(BASE_DIR, WANDB_IDS, 
                        # row_key='geneval_version', 
                        row_key='sampling_steps', 
                        col_key='task')


