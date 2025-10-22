import os
import torch
import glob
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
        # enumerate them using {i} for i in range
        # Human sort the target noise paths based on batch number
        # target_gaussian_noises_batch0.pt
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
        saved_all_timesteps_for_target = True
    else:
        # Try legacy single file
        target_noise_path = os.path.join(noise_dir, 'target_gaussian_noises.pt')
        if not os.path.exists(target_noise_path):
            print(f"No target gaussian noise files found at {noise_dir}")
            return None, None
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
    
    # Generate CFG values and ensure 1.0 is included
    cfg_range = np.concatenate([np.arange(-2, 1.0, 0.1), [1.0], np.arange(1.03, 19.0, 0.3)])
    cfg_values = np.sort(np.unique(cfg_range))  # Ensure unique and sorted values
    
    # Track probabilities for all 4 classes
    results = {cfg: {
        'total_correct': 0, 
        'class_probs': [[] for _ in range(4)],  # List for each class
        'all_scores': [], 
        'all_margins': []
    } for cfg in cfg_values}
    total_samples = 0
    
    # For each item in the batch
    for batch_idx in tqdm(range(cond_noises.shape[2]), desc="Processing batches"):
        # Get noises for this batch item
        batch_cond_noises = cond_noises[:, :, batch_idx]  # [num_selected, timesteps, ...]
        batch_uncond_noises = uncond_noises[:, :, batch_idx]  # [1, timesteps, ...]
        if saved_all_timesteps_for_target:
            target_noise = target_gaussian_noises[:, batch_idx]  # [timesteps, total_batches, ...]
        else:
            target_noise = target_gaussian_noises.unsqueeze(0)
        
        if len(batch_cond_noises.shape) == 5:
            # flatten last 3 dims
            batch_cond_noises = batch_cond_noises.reshape(batch_cond_noises.shape[0], batch_cond_noises.shape[1], -1)
            batch_uncond_noises = batch_uncond_noises.reshape(batch_uncond_noises.shape[0], batch_uncond_noises.shape[1], -1)
            target_noise = target_noise.reshape(target_noise.shape[0], -1)
            
        if not saved_all_timesteps_for_target:
            target_noise = target_noise.squeeze(2)
        
        for cfg in cfg_values:
            # Apply CFG
            if cfg == 0.0:
                noise_pred = batch_uncond_noises
            else:
                noise_pred = batch_uncond_noises + cfg * (batch_cond_noises - batch_uncond_noises)
            
            # Calculate similarity based on method
            if similarity_method == 'l2':
                # L2 distance (lower is better, so we negate)
                dists = (torch.norm(noise_pred.to(torch.float32) - target_noise.to(torch.float32), p=2, dim=-1)).mean(dim=-1)
                scores = -dists
            else:  # cosine similarity
                # Normalize vectors for cosine similarity
                noise_pred_norm = torch.nn.functional.normalize(noise_pred, p=2, dim=-1)
                target_noise_norm = torch.nn.functional.normalize(target_noise, p=2, dim=-1)
                # Compute cosine similarity (higher is better)
                scores = (noise_pred_norm * target_noise_norm).to(torch.float32).sum(dim=-1).mean(dim=-1)
            
            # Calculate softmax probabilities and store for all classes
            probs = torch.nn.functional.softmax(scores, dim=0)
            for class_idx in range(len(probs)):
                results[cfg]['class_probs'][class_idx].append(probs[class_idx].item())
            
            # The first index should be the correct one (using raw scores)
            pred = torch.argmax(scores).item()
            num_preds = torch.sum(scores == scores[0]).item()
            correct = (pred == 0)  # 0 is the correct index
            if num_preds > 1:
                correct = False
            
            results[cfg]['total_correct'] += correct
            
            # Save statistics
            results[cfg]['all_scores'].append(scores.tolist())
            # Calculate margin (difference between correct and second best)
            correct_score = scores[0]
            other_best = torch.max(scores[1:], dim=0)[0] if len(scores) > 1 else torch.tensor(-float('inf'))
            margin = (correct_score - other_best).item()
            results[cfg]['all_margins'].append(margin)
        
        total_samples += 1
    
    # Calculate final statistics
    final_results = {}
    for cfg in cfg_values:
        accuracy = results[cfg]['total_correct'] / total_samples if total_samples > 0 else 0
        mean_class_probs = [np.mean(class_probs) if class_probs else 0 for class_probs in results[cfg]['class_probs']]
        mean_margin = np.mean(results[cfg]['all_margins']) if results[cfg]['all_margins'] else 0
        std_margin = np.std(results[cfg]['all_margins']) if results[cfg]['all_margins'] else 0
        
        final_results[cfg] = {
            'accuracy': accuracy,
            'mean_class_probs': mean_class_probs,
            'mean_margin': mean_margin,
            'std_margin': std_margin,
            'all_scores': results[cfg]['all_scores'],
            'all_margins': results[cfg]['all_margins']
        }
    
    final_results['total_samples'] = total_samples
    
    # quick plot - save as pdf
    plt.figure(figsize=(8, 6))
    
    # Create primary axis for accuracy
    ax1 = plt.gca()
    accuracies = [final_results[cfg]['accuracy'] for cfg in cfg_values]
    ln1 = ax1.plot(cfg_values, accuracies, '-', color=sns.color_palette("colorblind")[0], 
                   marker='.', markersize=3, label='Accuracy')[0]  # Get the line object
    ax1.set_xlabel('CFG Value')
    ax1.set_ylabel('Accuracy', color=sns.color_palette("colorblind")[0])
    ax1.tick_params(axis='y', labelcolor=sns.color_palette("colorblind")[0])
    
    # Create secondary axis for class probabilities
    ax2 = ax1.twinx()
    class_probs = [[final_results[cfg]['mean_class_probs'][i] for cfg in cfg_values] for i in range(4)]
    
    # Plot each class probability with different colors
    lns = [ln1]
    colors = sns.color_palette("colorblind")[1:5]  # Get 4 colors for the classes
    for i, (probs, color) in enumerate(zip(class_probs, colors)):
        ln = ax2.plot(cfg_values, probs, '-', color=color, 
                     marker='.', markersize=3, label=f'Class {i} Prob')[0]  # Get the line object
        lns.append(ln)
    
    ax2.set_ylabel('Class Probabilities', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Calculate deltas
    max_acc = max(accuracies)
    acc_at_1 = accuracies[np.abs(cfg_values - 1.0).argmin()]
    delta_pp = (max_acc - acc_at_1) * 100
    
    # Calculate delta in probabilities at max accuracy point
    max_acc_idx = accuracies.index(max_acc)
    correct_prob_at_max = class_probs[0][max_acc_idx]  # Class 0 is correct class
    correct_prob_at_1 = class_probs[0][np.abs(cfg_values - 1.0).argmin()]
    delta_correct_prob = correct_prob_at_max - correct_prob_at_1
    
    plt.title(f'CFG vs Metrics ({similarity_method}, n={total_samples})\nΔ Acc: {delta_pp:+.1f}pp, Δ Correct: {delta_correct_prob:+.3f}')
    plt.grid(True)
    
    # Add vertical line at x=1
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
    
    # Add legend
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right')
    
    # Save plot in the same directory as the noise files with method prefix
    plot_path = f'{similarity_method}_{wandb_run_id}_cfg_vs_accuracy.pdf'
    plt.savefig(plot_path)
    plt.close()
    
    return final_results, cfg_values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mnt/lustre/work/oh/owl661/compositional-vaes/noise_results',
                      help='Base directory containing noise results')
    parser.add_argument('--run_id', type=str, default="aut3pbwh",
                      help='Specific run ID to evaluate. If not provided, will evaluate all runs recursively.')
    parser.add_argument('--similarity', type=str, choices=['l2', 'cosine'], default='l2',
                      help='Method to compute similarity between noises (l2 or cosine)')
    args = parser.parse_args()
    
    # Store results for final plots
    all_cfgs = []
    all_accuracies = []
    all_run_ids = []  # Add run_id tracking
    best_cfgs = []
    
    if args.run_id:
        # Find the directory containing this run_id (could be in any subdirectory)
        run_dirs = []
        for root, dirs, files in os.walk(args.base_dir):
            if args.run_id in dirs:
                run_dirs.append(os.path.join(root, args.run_id))
        
        if not run_dirs:
            print(f"No directory found for run_id: {args.run_id}")
            return
            
        # Should typically only find one, but process all if multiple exist
        for run_dir in run_dirs:
            print(f"\nEvaluating run: {args.run_id}")
            results, cfg_values = load_and_evaluate_noises(run_dir, args.run_id, args.similarity)
            if results is None:
                continue
                
            # Store results for plotting
            accuracies = [results[cfg]['accuracy'] for cfg in cfg_values]
            all_cfgs.extend(cfg_values)
            all_accuracies.extend(accuracies)
            all_run_ids.extend([args.run_id] * len(cfg_values))  # Add run_id for each point
            best_cfgs.append(cfg_values[np.argmax(accuracies)])
            
            print(f"\nResults for {args.run_id}:")
            print(f"Total samples: {results['total_samples']}")
            print("\nResults per CFG value:")
            for cfg in cfg_values:
                cfg_results = results[cfg]
                print(f"\nCFG = {cfg}:")
                print(f"  Accuracy: {cfg_results['accuracy']:.4f}")
                print(f"  Mean margin: {cfg_results['mean_margin']:.4f} ± {cfg_results['std_margin']:.4f}")
    else:
        # Evaluate all runs found recursively
        for root, dirs, files in os.walk(args.base_dir):
            # Only process directories that contain .pt files
            if any(f.endswith('.pt') for f in files):
                run_id = os.path.basename(root)
                print(f"\nEvaluating run: {run_id}")
                results, cfg_values = load_and_evaluate_noises(root, run_id, args.similarity)
                if results is None:
                    continue
                    
                # Store results for plotting
                accuracies = [results[cfg]['accuracy'] for cfg in cfg_values]
                all_cfgs.extend(cfg_values)
                all_accuracies.extend(accuracies)
                all_run_ids.extend([run_id] * len(cfg_values))  # Add run_id for each point
                best_cfgs.append(cfg_values[np.argmax(accuracies)])
                
                print(f"\nResults for {run_id}:")
                print(f"Total samples: {results['total_samples']}")
                print("\nResults per CFG value:")
                for cfg in cfg_values:
                    cfg_results = results[cfg]
                    print(f"\nCFG = {cfg}:")
                    print(f"  Accuracy: {cfg_results['accuracy']:.4f}")
                    print(f"  Mean margin: {cfg_results['mean_margin']:.4f} ± {cfg_results['std_margin']:.4f}")
    
    # Create final plots if we have data
    if all_cfgs:
        # Scatter plot of all CFG vs accuracy points
        plt.figure(figsize=(8, 6))
        plt.scatter(all_cfgs, all_accuracies, c=sns.color_palette("colorblind")[0], s=10, alpha=0.5)
        plt.xlabel('CFG Value (symlog scale)')
        plt.ylabel('Accuracy')
        plt.title(f'CFG vs Accuracy (all runs)')
        plt.grid(True)
        plt.xscale('symlog', linthresh=0.1)
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        plt.savefig('all_runs_cfg_vs_accuracy.pdf')
        plt.close()
        
        # Histogram of best CFG values
        plt.figure(figsize=(8, 6))
        plt.hist(best_cfgs, bins=20, color=sns.color_palette("colorblind")[0], alpha=0.7)
        plt.xlabel('Best CFG Value')
        plt.ylabel('Count')
        plt.title(f'Distribution of Best CFG Values (n={len(best_cfgs)} runs)')
        plt.grid(True)
        plt.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        plt.savefig('best_cfg_histogram.pdf')
        plt.close()
        
        # Distribution plot per CFG value
        plt.figure(figsize=(12, 6))
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame({
            'cfg': all_cfgs, 
            'accuracy': all_accuracies,
            'run_id': all_run_ids
        })
        
        # Create color palette for runs
        unique_runs = df['run_id'].unique()
        n_runs = len(unique_runs)
        color_palette = sns.color_palette("husl", n_runs)  # Generate distinct colors
        run_colors = dict(zip(unique_runs, color_palette))
        
        # Add points colored by run with jitter
        for run_id in unique_runs:
            run_data = df[df['run_id'] == run_id]
            # Add jitter to x positions
            x_jittered = [df['cfg'].unique().tolist().index(x) + np.random.uniform(-0.3, 0.3) for x in run_data['cfg']]
            plt.scatter(x_jittered, run_data['accuracy'], 
                       c=[run_colors[run_id]], alpha=0.7, s=30, 
                       label=run_id)
        
        plt.xlabel('CFG Value')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Distribution per CFG Value ({n_runs} runs)')
        plt.grid(True, axis='y')
        plt.axvline(x=df['cfg'].unique().tolist().index(1.0), color='gray', linestyle='--', alpha=0.5)
        plt.xticks(range(len(df['cfg'].unique())), [f'{x:.2f}' for x in sorted(df['cfg'].unique())], rotation=45)
        
        # # Add legend if there are multiple runs
        # if n_runs > 1:
        #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
            
        plt.tight_layout()
        plt.savefig('cfg_accuracy_distribution.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    main()