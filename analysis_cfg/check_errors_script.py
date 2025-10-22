""" Simple idea: check variances of errors at each timestep. If the variance is high, that step infleunces the error more. Therefore timestep weighting makes sense. 
"""
import torch
import os
import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

def load_and_evaluate_noises(noise_dir, wandb_run_id, similarity_method='l2'):
    """
    Load and evaluate noises from a specific directory.
    First index in selected_indices should be the correct one.
    Args:
       n noise_dir: Directory containing noise files
        wandb_run_id: WandB run ID for saving plots
        similarity_method: Either 'l2' or 'cosine' for distance calculation
    """
    # First try to load multiple target gaussian noises
    target_noise_paths = sorted(glob.glob(os.path.join(noise_dir, 'target_gaussian_noises_batch*.pt')))
    if len(target_noise_paths) > 1:
        target_noise_paths = sorted(target_noise_paths, key=lambda x: int(x.split('batch')[-1].split('.')[0]))
    device='cuda' if torch.cuda.is_available() else 'cpu'
    
    saved_all_timesteps_for_target = False
    
    if target_noise_paths:
        # Multiple target files found - concatenate them
        target_gaussian_noises_list = []
        for target_path in target_noise_paths:
            target_noise_data = torch.load(target_path)
            target_gaussian_noises_list.append(target_noise_data['target_gaussian_noises'])
        # Concatenate along batch dimension (dim=1)
        target_gaussian_noises = torch.cat(target_gaussian_noises_list, dim=1).to(device)  # shape: [timesteps, total_batches, ...]
    else:
        # Try legacy single file
        target_noise_path = os.path.join(noise_dir, 'target_gaussian_noises.pt')
        if not os.path.exists(target_noise_path):
            print(f"No target gaussian noise files found at {noise_dir}")
            return None
        target_noise_data = torch.load(target_noise_path)
        target_gaussian_noises = target_noise_data['target_gaussian_noises'].to(device) # shape: [timesteps, 1, ...]
    
    # Get all noise files except target_gaussian_noises*.pt
    noise_files = sorted([f for f in glob.glob(os.path.join(noise_dir, "*.pt")) 
                  if not any(x in f for x in ["target_gaussian_noises.pt", "target_gaussian_noises_batch"])],
                  key=lambda x: int(x.split('batch')[-1].split('_')[0]))
    
    # Load and concatenate all noise files
    all_cond_noises = []
    all_uncond_noises = []
    for noise_file in tqdm(noise_files, desc=f"Loading noise files from {os.path.basename(noise_dir)}"):
        data = torch.load(noise_file)
        all_cond_noises.append(data['conditional_noises'])  # shape: [num_selected, timesteps, batch_size, ...]
        all_uncond_noises.append(data['unconditional_noises'])  # [1, timesteps, batch_size, ...]
    
    # Concatenate along batch dimension
    
    cond_noises = torch.cat(all_cond_noises, dim=2).to(device)  # shape: [num_selected, timesteps, total_batch_size, ...]
    uncond_noises = torch.cat(all_uncond_noises, dim=2).to(device)  # [1, timesteps, total_batch_size, ...]
    
    if len(cond_noises.shape) == 6:
        cond_noises = cond_noises.reshape(cond_noises.shape[0], cond_noises.shape[1], cond_noises.shape[2], -1)
        uncond_noises = uncond_noises.reshape(uncond_noises.shape[0], uncond_noises.shape[1], uncond_noises.shape[2], -1)
        
        target_gaussian_noises = target_gaussian_noises.reshape(target_gaussian_noises.shape[0], target_gaussian_noises.shape[1], -1)
        saved_all_timesteps_for_target = True
        
    # Print shapes for debugging
    print(f"Initial shapes:")
    print(f"cond_noises: {cond_noises.shape}")
    print(f"uncond_noises: {uncond_noises.shape}")
    print(f"target_gaussian_noises: {target_gaussian_noises.shape}")
    
    # For each timestep, calculate error variances across all samples
    num_timesteps = cond_noises.shape[1]
    num_samples = cond_noises.shape[2]
    num_classes = cond_noises.shape[0]
    
    # Store errors for each timestep and sample
    true_class_errors = []  # Mean errors for class 0 (true class)
    true_class_stds = []    # Standard deviations for class 0
    incorrect_class_errors = []  # Mean errors for other classes
    incorrect_class_stds = []    # Standard deviations for other classes
    accuracies_per_timestep = []  # Track accuracy at each timestep
    all_errors = []  # Store all errors for aggregation [timestep, num_classes, batch_size]
    
    # For each timestep
    for t in range(num_timesteps):
        # Get target noise for this timestep
        if saved_all_timesteps_for_target:
            target_t = target_gaussian_noises[t]  # [total_batches, ...]
        else:
            target_t = target_gaussian_noises[t]  # [1, features]
            
        # Get conditional noise predictions for this timestep
        cond_t = cond_noises[:, t]  # [num_selected, total_batch_size, ...]
        
        print(f"\nShapes at timestep {t}:")
        print(f"cond_t: {cond_t.shape}")
        print(f"target_t: {target_t.shape}")
        
        # Calculate errors for each class prediction
        if similarity_method == 'l2':
            # L2 distance (lower is better)
            if not saved_all_timesteps_for_target:  # If spatial dimensions present
                cond_t = cond_t.reshape(cond_t.shape[0], cond_t.shape[1], -1)  # [num_classes, total_batch_size, flattened_features]
                target_t = target_t.reshape(target_t.shape[0], -1)  # [total_batch_size, flattened_features]
                
                print(f"After reshape:")
                print(f"cond_t: {cond_t.shape}")
                print(f"target_t: {target_t.shape}")
                
                # Expand target_t to match cond_t's shape for broadcasting
                target_t = target_t.unsqueeze(0).expand(num_classes, -1, -1)  # [num_classes, total_batch_size, flattened_features]
                
                print(f"After expansion:")
                print(f"target_t: {target_t.shape}")
            else:
                target_t = target_t.unsqueeze(0)
            dists = torch.norm(cond_t.to(torch.float32) - target_t.to(torch.float32), p=2, dim=-1)  # [num_classes, total_batch_size]
            errors = dists
            
        else:  # cosine similarity
            if len(cond_t.shape) == 3:  # If spatial dimensions present
                cond_t = cond_t.reshape(cond_t.shape[0], cond_t.shape[1], -1)  # [num_classes, total_batch_size, flattened_features]
                target_t = target_t.reshape(target_t.shape[0], -1)  # [total_batch_size, flattened_features]
                
                # Expand target_t to match cond_t's shape for broadcasting
                target_t = target_t.unsqueeze(0).expand(num_classes, -1, -1)  # [num_classes, total_batch_size, flattened_features]
            
            cond_t_norm = torch.nn.functional.normalize(cond_t, p=2, dim=-1)
            target_t_norm = torch.nn.functional.normalize(target_t, p=2, dim=-1)
            similarity = (cond_t_norm * target_t_norm).sum(dim=-1)  # [num_classes, total_batch_size]
            errors = -similarity  # Convert to error (lower is better)
        
        # Store raw errors for aggregation
        all_errors.append(errors)
        
        # Calculate per-timestep accuracy (for visualization)
        correct_predictions = (errors[0] < errors[1:].min(dim=0)[0])
        accuracies_per_timestep.append(correct_predictions.float().mean().item())
        
        # Calculate mean and std for true class
        true_class_mean = errors[0].mean().item()
        true_class_std = errors[0].std().item()
        true_class_errors.append(true_class_mean)
        true_class_stds.append(true_class_std)
        
        # Calculate mean and std for incorrect classes
        incorrect_mean = errors[1:].mean().item()
        incorrect_std = errors[1:].std().item()
        incorrect_class_errors.append(incorrect_mean)
        incorrect_class_stds.append(incorrect_std)
    
    # Stack all errors and calculate total accuracy
    all_errors = torch.stack(all_errors)
    summed_errors = all_errors.sum(dim=0)
    correct_predictions_total = (summed_errors[0] < summed_errors[1:].min(dim=0)[0])
    mean_accuracy = correct_predictions_total.float().mean().item()
    
    # Calculate influence of each timestep
    influences = []  # Store minimum delta needed for each timestep
    influence_stds = []  # Store standard deviation of deltas for each timestep
    
    # For each timestep
    for t in range(num_timesteps):
        # Get current timestep and other timesteps contributions
        current_timestep = all_errors[t]  # [num_classes, batch_size]
        other_timesteps_sum = summed_errors - current_timestep  # [num_classes, batch_size]
        
        # For each sample, find minimum delta needed
        sample_deltas = []
        for sample_idx in range(num_samples):
            # Get current errors from this timestep and others
            other_errors = other_timesteps_sum[:, sample_idx].cpu()  # [num_classes]
            current_errors = current_timestep[:, sample_idx].cpu()  # [num_classes]
            
            # Current prediction based on all timesteps
            total_errors = other_errors + current_errors
            current_pred = total_errors.argmin().item()
            
            # Find minimum delta needed to change prediction
            min_delta = float('inf')
            for target_class in range(num_classes):
                if target_class == current_pred:
                    continue
                
                # For target_class to win over current_pred, we need:
                # (other_errors[target_class] + (current_errors[target_class] + delta)) < (other_errors[current_pred] + current_errors[current_pred])
                # So: delta < (other_errors[current_pred] + current_errors[current_pred]) - (other_errors[target_class] + current_errors[target_class])
                margin = (other_errors[current_pred] + current_errors[current_pred]) - (other_errors[target_class] + current_errors[target_class])
                
                # How much do we need to change current timestep's contribution?
                # We need to change it by more than the margin
                delta_needed = abs(margin)
                min_delta = min(min_delta, delta_needed)
            
            sample_deltas.append(min_delta)
        
        # Store average influence and std for this timestep
        influences.append(np.mean(sample_deltas))
        influence_stds.append(np.std(sample_deltas))
    
    influences = np.array(influences)
    influence_stds = np.array(influence_stds)
    
    # Convert to numpy arrays for plotting
    true_errors = np.array(true_class_errors)
    true_stds = np.array(true_class_stds)
    incorrect_errors = np.array(incorrect_class_errors)
    incorrect_stds = np.array(incorrect_class_stds)
    accuracies = np.array(accuracies_per_timestep)
    timesteps = np.arange(num_timesteps)
    
    # Find the matching run info for this wandb_run_id
    run_info = next((run for run in run_ids if run['id'] == wandb_run_id), None)
    if run_info:
        title_info = f"{run_info['gen']} → {run_info['eval']} ({run_info['type']})"
    else:
        title_info = wandb_run_id
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Plot 1: Mean errors with std shading
    plt.figure(figsize=(10, 6))
    
    # Create primary axis for errors
    ax1 = plt.gca()
    # Plot true class errors with std shading
    ln1 = ax1.plot(timesteps, true_errors, label='True Class Error', color='blue')
    ax1.fill_between(timesteps, true_errors - true_stds, true_errors + true_stds,
                     alpha=0.3, color='blue', label='True Class Std')
    
    # Plot incorrect classes errors with std shading
    ln2 = ax1.plot(timesteps, incorrect_errors, label='Incorrect Classes Error', color='red')
    ax1.fill_between(timesteps, incorrect_errors - incorrect_stds, incorrect_errors + incorrect_stds,
                     alpha=0.3, color='red', label='Incorrect Classes Std')
    
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel(f'Error ({similarity_method})')
    ax1.tick_params(axis='y')
    
    # Create secondary axis for accuracy
    ax2 = ax1.twinx()
    ln3 = ax2.plot(timesteps, accuracies, label='Per-timestep Accuracy', color='green', linestyle='--')
    ax2.set_ylabel('Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combine legends
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right')
    
    plt.title(f'Error and Accuracy per Timestep\n{title_info}\nTotal Accuracy (all timesteps): {mean_accuracy:.3f}')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'figures/{wandb_run_id}_errors_and_accuracy.pdf')
    plt.close()
    
    # Plot 2: Standard deviations only
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, true_stds, label='True Class Std', color='blue')
    plt.plot(timesteps, incorrect_stds, label='Incorrect Classes Std', color='red')
    plt.xlabel('Timestep')
    plt.ylabel(f'Standard Deviation of {similarity_method} Error')
    plt.title(f'Error Standard Deviations per Timestep\n{title_info}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'figures/{wandb_run_id}_error_stds.pdf')
    plt.close()
    
    # Create correlation plots
    # Collect all errors per sample for each timestep
    errors_per_sample_timestep = []  # Will be [num_timesteps, num_samples]
    for t in range(num_timesteps):
        if saved_all_timesteps_for_target:
            target_t = target_gaussian_noises[t]
        else:
            target_t = target_gaussian_noises[t]
            
        cond_t = cond_noises[:, t]
        
        if similarity_method == 'l2':
            if not saved_all_timesteps_for_target:
                cond_t = cond_t.reshape(cond_t.shape[0], cond_t.shape[1], -1)
                target_t = target_t.reshape(target_t.shape[0], -1)
                target_t = target_t.unsqueeze(0).expand(num_classes, -1, -1)
            else:
                target_t = target_t.unsqueeze(0)
            dists = torch.norm(cond_t.to(torch.float32) - target_t.to(torch.float32), p=2, dim=-1)
            errors = dists
        else:
            if len(cond_t.shape) == 3:
                cond_t = cond_t.reshape(cond_t.shape[0], cond_t.shape[1], -1)
                target_t = target_t.reshape(target_t.shape[0], -1)
                target_t = target_t.unsqueeze(0).expand(num_classes, -1, -1)
            
            cond_t_norm = torch.nn.functional.normalize(cond_t, p=2, dim=-1)
            target_t_norm = torch.nn.functional.normalize(target_t, p=2, dim=-1)
            similarity = (cond_t_norm * target_t_norm).sum(dim=-1)
            errors = -similarity
            
        # Store raw errors for correlation analysis
        errors_per_sample_timestep.append(errors.cpu().numpy())
    
    # Convert to numpy array [num_timesteps, num_classes, num_samples]
    errors_array = np.stack(errors_per_sample_timestep)
    
    # Calculate correlations for true class (class 0)
    true_class_errors_per_sample = errors_array[:, 0, :]  # [num_timesteps, num_samples]
    true_class_pearson_corr = np.corrcoef(true_class_errors_per_sample)
    
    # Calculate Spearman rank correlation for true class
    true_class_spearman_corr = np.zeros((num_timesteps, num_timesteps))
    for i in range(num_timesteps):
        for j in range(num_timesteps):
            rho, _ = scipy.stats.spearmanr(true_class_errors_per_sample[i], true_class_errors_per_sample[j])
            true_class_spearman_corr[i, j] = rho
    
    # Calculate correlations for incorrect classes (average across classes 1+)
    incorrect_class_errors_per_sample = errors_array[:, 1:, :].mean(axis=1)  # [num_timesteps, num_samples]
    incorrect_class_pearson_corr = np.corrcoef(incorrect_class_errors_per_sample)
    
    # Calculate Spearman rank correlation for incorrect classes
    incorrect_class_spearman_corr = np.zeros((num_timesteps, num_timesteps))
    for i in range(num_timesteps):
        for j in range(num_timesteps):
            rho, _ = scipy.stats.spearmanr(incorrect_class_errors_per_sample[i], incorrect_class_errors_per_sample[j])
            incorrect_class_spearman_corr[i, j] = rho
    
    # Plot correlation heatmaps (2x2 grid: Pearson and Spearman for both true and incorrect)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # True class correlations - Pearson
    sns.heatmap(true_class_pearson_corr, ax=ax1, cmap='RdBu_r', vmin=-1, vmax=1,
                xticklabels=timesteps[::5], yticklabels=timesteps[::5])
    ax1.set_title('True Class Pearson Correlations')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Timestep')
    
    # True class correlations - Spearman
    sns.heatmap(true_class_spearman_corr, ax=ax2, cmap='RdBu_r', vmin=-1, vmax=1,
                xticklabels=timesteps[::5], yticklabels=timesteps[::5])
    ax2.set_title('True Class Spearman Rank Correlations')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Timestep')
    
    # Incorrect class correlations - Pearson
    sns.heatmap(incorrect_class_pearson_corr, ax=ax3, cmap='RdBu_r', vmin=-1, vmax=1,
                xticklabels=timesteps[::5], yticklabels=timesteps[::5])
    ax3.set_title('Incorrect Classes Pearson Correlations')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Timestep')
    
    # Incorrect class correlations - Spearman
    sns.heatmap(incorrect_class_spearman_corr, ax=ax4, cmap='RdBu_r', vmin=-1, vmax=1,
                xticklabels=timesteps[::5], yticklabels=timesteps[::5])
    ax4.set_title('Incorrect Classes Spearman Rank Correlations')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Timestep')
    
    plt.suptitle(f'Error Correlations Between Timesteps\n{title_info}')
    plt.tight_layout()
    plt.savefig(f'figures/{wandb_run_id}_error_correlations.pdf')
    plt.close()
    
    # Plot influences with std shading
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, influences, label='Mean Timestep Influence', color='purple')
    plt.fill_between(timesteps, influences - influence_stds, influences + influence_stds,
                     alpha=0.3, color='purple', label='Influence Std')
    plt.xlabel('Timestep')
    plt.ylabel('Average Influence (min delta needed)')
    plt.title(f'Timestep Influence on Final Prediction\n{title_info}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'figures/{wandb_run_id}_timestep_influence.pdf')
    plt.close()
    
    # Return statistics
    return {
        'true_errors': true_errors,
        'true_stds': true_stds,
        'incorrect_errors': incorrect_errors,
        'incorrect_stds': incorrect_stds,
        'accuracies': accuracies,
        'timesteps': timesteps,
        'num_samples': num_samples,
        'mean_accuracy': mean_accuracy,
        'summed_errors': summed_errors.cpu().numpy(),
        'influences': influences,
        'influence_stds': influence_stds,
        'true_class_pearson_correlations': true_class_pearson_corr,
        'true_class_spearman_correlations': true_class_spearman_corr,
        'incorrect_class_pearson_correlations': incorrect_class_pearson_corr,
        'incorrect_class_spearman_correlations': incorrect_class_spearman_corr
    }

def create_accuracy_comparison_plot(all_stats, run_ids):
    """Create plots comparing per-timestep and overall accuracy for SD1.5, SD2, and SD3."""
    task_types = ['counting', 'position', 'color_attr']
    w, h = 3.25, 1.5
    fig, axes = plt.subplots(1, 3, figsize=(w, h), sharey=True)
    
    # Get colorblind-friendly colors
    colors = sns.color_palette("colorblind")
    color1, color2, color3 = colors[0], colors[1], colors[2]
    
    for task_idx, task_type in enumerate(task_types):
        ax = axes[task_idx]
        
        # Get runs for this task type
        sd15_run = next(run for run in run_ids if run['eval'] == 'sd1.5' and run['type'] == task_type)
        sd2_run = next(run for run in run_ids if run['eval'] == 'sd2' and run['type'] == task_type)
        sd3_run = next(run for run in run_ids if run['eval'] == 'sd3' and run['type'] == task_type)
        
        # Get stats
        sd15_stats = all_stats[sd15_run['id']]
        sd2_stats = all_stats[sd2_run['id']]
        sd3_stats = all_stats[sd3_run['id']]
        
        # Plot per-timestep accuracies
        ax.plot(sd15_stats['timesteps'] / 30, sd15_stats['accuracies'], 
                label='SD1.5', color=color1, linestyle='-')
        ax.plot(sd2_stats['timesteps'] / 30, sd2_stats['accuracies'],
                label='SD2', color=color2, linestyle='-')
        ax.plot(sd3_stats['timesteps'] / 30, sd3_stats['accuracies'],
                label='SD3', color=color3, linestyle='-')
        
        # Add vertical lines for overall accuracies
        ax.axhline(y=sd15_stats['mean_accuracy'], color=color1, linestyle='--', alpha=0.5)
        ax.axhline(y=sd2_stats['mean_accuracy'], color=color2, linestyle='--', alpha=0.5)
        ax.axhline(y=sd3_stats['mean_accuracy'], color=color3, linestyle='--', alpha=0.5)
        
        ax.set_title(f'{task_type.replace("_", " ").title()}')
        ax.set_xlabel('Timestep')
        if task_idx == 0:
            ax.set_ylabel('Accuracy')
        if task_idx == 1:
            ax.legend(fontsize=7, frameon=False)
    
    # Adjust figure size and spacing
    fig.set_size_inches(w, h)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.25, wspace=0.1)
    plt.savefig('figures/accuracy_comparison.pdf', pad_inches=0)
    plt.close()

def create_aggregate_comparison_plot(all_stats, run_ids):
    """Create aggregate plots comparing SD2 and SD3 evaluations for each task type."""
    task_types = ['counting', 'position', 'color_attr']
    w, h = 3.25, 1.5
    fig, axes = plt.subplots(1, 3, figsize=(w, h), sharey=True)
    
    # Get colorblind-friendly colors
    colors = sns.color_palette("colorblind")
    color1, color2 = colors[0], colors[1]
    
    for task_idx, task_type in enumerate(task_types):
        ax = axes[task_idx]
        
        # Get runs for this task type
        sd2_run = next(run for run in run_ids if run['eval'] == 'sd2' and run['type'] == task_type)
        sd3_run = next(run for run in run_ids if run['eval'] == 'sd3' and run['type'] == task_type)
        
        # Get stats and normalize each model's errors to [0,1] range
        sd2_stats = all_stats[sd2_run['id']]
        sd3_stats = all_stats[sd3_run['id']]
        
        # Normalize SD2 errors
        sd2_errors = sd2_stats['true_errors']
        sd2_stds = sd2_stats['true_stds']
        sd2_errors_norm = (sd2_errors - sd2_errors.min()) / (sd2_errors.max() - sd2_errors.min())
        sd2_stds_norm = sd2_stds / (sd2_errors.max() - sd2_errors.min())
        
        # Normalize SD3 errors
        sd3_errors = sd3_stats['true_errors']
        sd3_stds = sd3_stats['true_stds']
        sd3_errors_norm = (sd3_errors - sd3_errors.min()) / (sd3_errors.max() - sd3_errors.min())
        sd3_stds_norm = sd3_stds / (sd3_errors.max() - sd3_errors.min())
        
        # Plot SD2 evaluation
        ax.plot(sd2_stats['timesteps'] / 30, sd2_errors_norm, 
                label='SD2', color=color1, linestyle='-')
        ax.fill_between(sd2_stats['timesteps'] / 30, 
                       sd2_errors_norm - sd2_stds_norm,
                       sd2_errors_norm + sd2_stds_norm,
                       alpha=0.2, color=color1, edgecolor=None)
        
        # Plot SD3 evaluation
        ax.plot(sd3_stats['timesteps'] / 30, sd3_errors_norm,
                label='SD3', color=color2, linestyle='-')
        ax.fill_between(sd3_stats['timesteps'] / 30,
                       sd3_errors_norm - sd3_stds_norm,
                       sd3_errors_norm + sd3_stds_norm,
                       alpha=0.2, color=color2, edgecolor=None)
        
        ax.set_title(f'{task_type.replace("_", " ").title()}')
        ax.set_xlabel('Timestep')
        if task_idx == 0:
            ax.set_ylabel('Normalized Error')
        if task_idx == 1:
            ax.legend(fontsize=7, frameon=False)
    
    fig.set_size_inches(w, h)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.25, wspace=0.1)
    plt.savefig('figures/aggregate_comparison.pdf', pad_inches=0)
    plt.close()

# Take only SD3 runs of few categories
run_ids = [
    # {"gen": "sd3", "eval":"sd3", "type": "whatsup_A", "id": "2zfmbdgb"},
    # {"gen": "sd3", "eval":"sd3", "type": "whatsup_B", "id": "72srd0uj"},
    
    {"gen": "sd3", "eval":"sd2", "type": "position", "id": "aut3pbwh"},
    {"gen": "sd3", "eval":"sd2", "type": "counting", "id": "lxu0ohji"},
    {"gen": "sd3", "eval":"sd2", "type": "color_attr", "id": "8axobt1l"},
    # # {"gen": "sd3", "eval":"sd2", "type": "single", "id": "wwieih3x"},
    
    {"gen": "sd3", "eval":"sd1.5", "type": "position", "id": "curxv9va"},
    {"gen": "sd3", "eval":"sd1.5", "type": "counting", "id": "9j8saxms"},
    {"gen": "sd3", "eval":"sd1.5", "type": "color_attr", "id": "ok41tk28"},
    # # {"gen": "sd3", "eval":"sd1.5", "type": "single", "id": "4xuy2o92"},
    
    {"gen": "sd3", "eval":"sd3", "type": "counting", "id": "1tcxc2rz"},
    {"gen": "sd3", "eval":"sd3", "type": "color_attr", "id": "znbz2uhj"},
    {"gen": "sd3", "eval":"sd3", "type": "position", "id": "4xuy2o92"},
    
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mnt/lustre/work/oh/owl661/compositional-vaes/noise_results',
                      help='Base directory containing noise results')
    parser.add_argument('--run_id', type=str, default=None,
                      help='Specific run ID to evaluate. If not provided, will evaluate all runs in run_ids list.')
    parser.add_argument('--similarity', type=str, choices=['l2', 'cosine'], default='l2',
                      help='Method to compute similarity between noises (l2 or cosine)')
    args = parser.parse_args()
    
    if args.run_id:
        # Find the directory containing this run_id
        run_dirs = []
        for root, dirs, files in os.walk(args.base_dir):
            if args.run_id in dirs:
                run_dirs.append(os.path.join(root, args.run_id))
        
        if not run_dirs:
            print(f"No directory found for run_id: {args.run_id}")
            exit(1)
            
        # Process the specified run
        for run_dir in run_dirs:
            print(f"\nEvaluating run: {args.run_id}")
            stats = load_and_evaluate_noises(run_dir, args.run_id, args.similarity)
            if stats is None:
                continue
            
            print(f"\nResults for {args.run_id}:")
            print(f"Total samples: {stats['num_samples']}")
            print(f"Mean error across all timesteps: {stats['true_errors'].mean():.4f}")
            print(f"Mean std across all timesteps: {stats['incorrect_errors'].mean():.4f}")
            print(f"Max std at timestep {stats['timesteps'][stats['incorrect_errors'].argmax()]}: {stats['incorrect_errors'].max():.4f}")
            print(f"Min std at timestep {stats['timesteps'][stats['incorrect_errors'].argmin()]}: {stats['incorrect_errors'].min():.4f}")
            print(f"Mean accuracy: {stats['mean_accuracy']:.3f}")
    else:
        # Process all runs in run_ids list and collect stats
        all_stats = {}
        for run_info in run_ids:
            run_id = run_info['id']
            run_dir = None
            
            # Find the directory for this run_id
            for root, dirs, files in os.walk(args.base_dir):
                if run_id in dirs:
                    run_dir = os.path.join(root, run_id)
                    break
            
            if run_dir is None:
                print(f"No directory found for run_id: {run_id}")
                continue
                
            print(f"\nEvaluating run: {run_id} ({run_info['gen']}->{run_info['eval']}, {run_info['type']})")
            stats = load_and_evaluate_noises(run_dir, run_id, args.similarity)
            if stats is None:
                continue
            
            # Store stats for this run
            all_stats[run_id] = stats
            
            print(f"Total samples: {stats['num_samples']}")
            print(f"Mean error across all timesteps: {stats['true_errors'].mean():.4f}")
            print(f"Mean std across all timesteps: {stats['incorrect_errors'].mean():.4f}")
            print(f"Max std at timestep {stats['timesteps'][stats['incorrect_errors'].argmax()]}: {stats['incorrect_errors'].max():.4f}")
            print(f"Min std at timestep {stats['timesteps'][stats['incorrect_errors'].argmin()]}: {stats['incorrect_errors'].min():.4f}")
            print(f"Mean accuracy: {stats['mean_accuracy']:.3f}")
        
        # Create aggregate comparison plot
        create_aggregate_comparison_plot(all_stats, run_ids)
        
        # Create accuracy comparison plot
        create_accuracy_comparison_plot(all_stats, run_ids)

