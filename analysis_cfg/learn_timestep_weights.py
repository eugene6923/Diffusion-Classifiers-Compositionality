import os
import torch
import glob
from tqdm import tqdm
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from datetime import datetime

def get_polynomial_weights(coefficients, num_timesteps, device, positive_weights=False):
    """Convert polynomial coefficients to timestep weights"""
    # Create normalized timesteps from -1 to 1
    t = torch.linspace(-1, 1, num_timesteps, device=device)
    # Compute polynomial values: a_n * t^n + ... + a_1 * t + a_0
    weights = torch.zeros_like(t)
    for i, coeff in enumerate(coefficients):
        weights += coeff * (t ** i)
    if positive_weights:
        weights = weights.exp()
    return weights  # Return raw weights without abs()

def load_and_evaluate_noises(noise_dir, wandb_run_id, num_steps=100, lr=0.01, use_uncond=False, poly_degree=None, use_linear_solver=False, l1_lambda=0.0, positive_weights=False, train_ratio=1.0, val_ratio=0.1):
    """
    Load noises or scores and learn:
    1. Timestep weights for MSE calculation (either direct or polynomial)
    2. CFG weights per timestep for noise prediction (only if use_uncond=True and using noises)
    
    Args:
        noise_dir: Directory containing either noise files or score files
        wandb_run_id: ID of the wandb run
        poly_degree: If not None, fit a polynomial of this degree instead of learning individual weights
        use_linear_solver: If True, solve for weights using linear regression instead of gradient descent
        l1_lambda: Strength of L1 regularization (default: 0.0)
        train_ratio: Ratio of remaining data (after 50% test split) to use for training (default: 1.0)
        val_ratio: Ratio of remaining data (after 50% test split) to use for validation (default: 0.1)
    """
    results_dict = {
        'wandb_run_id': wandb_run_id,
        'learning_rate': lr,
        'num_steps': num_steps,
        'use_uncond': use_uncond,
        'poly_degree': poly_degree,
        'use_linear_solver': use_linear_solver,
        'l1_lambda': l1_lambda,
        'timestamp': datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
    }
    
    # Set device    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if we're using scores or noises
    score_files = sorted(glob.glob(os.path.join(noise_dir, '*_scores.pt')))
    using_scores = len(score_files) > 0
    
    if using_scores:
        print("Using score-based optimization...")
        # Load all score files
        all_scores = []
        all_correct_indices = []
        all_timesteps = None
        
        for score_file in tqdm(score_files, desc="Loading score files"):
            data = torch.load(score_file)
            scores = data['scores_with_timestep']  # [num_timesteps, num_prompts, batch_size]
            correct_indices = data['correct_indices']
            
            all_scores.append(scores)
            all_correct_indices.extend(correct_indices)
            
            if all_timesteps is None:
                all_timesteps = data['all_timesteps_used']
            else:
                assert np.array_equal(all_timesteps, data['all_timesteps_used']), "Timesteps differ between files!"
        
        # Concatenate along batch dimension
        scores = torch.cat(all_scores, dim=2).to(device)  # [num_timesteps, num_prompts, total_samples]
        correct_indices = torch.tensor(all_correct_indices, device=device)
        
        num_timesteps = scores.shape[0]
        total_samples = scores.shape[2]
        
        # Create random permutation for splitting with fixed seed
        torch.manual_seed(42)  # Fixed seed for consistent splits
        indices = torch.randperm(total_samples)
        
        # Handle splitting based on train_ratio
        if train_ratio is None:
            # Use all data for everything (no splitting)
            train_indices = indices
            val_indices = indices
            test_indices = indices
            
            num_train = total_samples
            num_val = total_samples
            num_test = total_samples
        else:
            # Always take 50% for test set
            num_test = total_samples // 2
            remaining_samples = total_samples - num_test
            test_indices = indices[remaining_samples:]
            
            # Split remaining data into train/val based on ratios
            if train_ratio == 0:
                # Use all remaining data for both train and val
                train_indices = indices[:remaining_samples]
                val_indices = indices[:remaining_samples]  # Same as train
                
                num_train = remaining_samples
                num_val = remaining_samples
            else:
                num_train = int(remaining_samples * train_ratio)
                num_val = remaining_samples - num_train
                
                train_indices = indices[:num_train]
                val_indices = indices[num_train:remaining_samples]

        print(f"Data split: Train={num_train}, Val={num_val}, Test={num_test}")
        
        # Calculate initial accuracy using mean scores
        def compute_accuracy(indices, weights):
            with torch.no_grad():
                weighted_scores = (scores[..., indices] * weights.view(-1, 1, 1)).mean(dim=0)  # [num_prompts, num_samples]
                min_scores = weighted_scores.min(dim=0, keepdim=True)[0]
                is_min = weighted_scores == min_scores
                has_tie = is_min.sum(dim=0) > 1
                predictions = weighted_scores.argmin(dim=0)
                predictions[has_tie] = -1
                return ((predictions == correct_indices[indices]) & ~has_tie).float().mean().item()
        
        # Calculate initial accuracies with uniform weights
        uniform_weights = torch.ones(num_timesteps, device=device)
        train_accuracy_before = compute_accuracy(train_indices, uniform_weights)
        val_accuracy_before = compute_accuracy(val_indices, uniform_weights)
        test_accuracy_before = compute_accuracy(test_indices, uniform_weights)
        print(f"Initial accuracies - Train: {train_accuracy_before:.4f}, Val: {val_accuracy_before:.4f}, Test: {test_accuracy_before:.4f}")
        
        # Initialize learnable weights
        if poly_degree is not None:
            init_coeffs = torch.zeros(poly_degree + 1, device=device)
            init_coeffs[0] = 1.0  # Start with constant term = 1
            poly_coeffs = torch.nn.Parameter(init_coeffs)
            timestep_weights = get_polynomial_weights(poly_coeffs, num_timesteps, device, positive_weights)
            optimizer_params = [poly_coeffs]
        else:
            poly_coeffs = None
            timestep_weights = torch.zeros(num_timesteps, device=device, requires_grad=True)
            optimizer_params = [timestep_weights]
        
        optimizer = torch.optim.Adam(optimizer_params, lr=lr)
        
        # Training loop
        best_accuracy = 0.0  # Initialize with 0 accuracy
        best_val_accuracy = 0.0
        accuracy_clipped = 0.0
        best_weights = {
            'timestep': torch.ones_like(timestep_weights).cpu(),
            'train_accuracy': train_accuracy_before,
            'val_accuracy': val_accuracy_before,
            'test_accuracy': test_accuracy_before,
            'accuracy': best_accuracy,
            'accuracy_clipped': accuracy_clipped
        }
        
        for step in tqdm(range(num_steps), desc="Optimizing weights"):
            optimizer.zero_grad()
            
            # Update timestep weights if using polynomial
            if poly_degree is not None:
                timestep_weights = get_polynomial_weights(poly_coeffs, num_timesteps, device, positive_weights)
                
                weights_use = timestep_weights
            else:
                if positive_weights:
                    weights_use = timestep_weights.exp()
                else:
                    weights_use = timestep_weights
            
            # Calculate weighted scores for training set
            weighted_scores = (scores[..., train_indices] * weights_use.view(-1, 1, 1)).mean(dim=0)
            
        
            # Use cross entropy loss with L1 regularization
            targets = correct_indices[train_indices]
            ce_loss = torch.nn.functional.cross_entropy(-weighted_scores.t(), targets)
            l1_loss = l1_lambda * torch.norm(weights_use, p=1)
            loss = ce_loss + l1_loss
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Track accuracies
            with torch.no_grad():
                train_accuracy = compute_accuracy(train_indices, weights_use)
                val_accuracy = compute_accuracy(val_indices, weights_use)
                test_accuracy = compute_accuracy(test_indices, weights_use)
                
                # Legacy accuracy computation (on all data)
                weighted_scores_all = (scores * weights_use.view(-1, 1, 1)).mean(dim=0)
                min_scores = weighted_scores_all.min(dim=0, keepdim=True)[0]
                is_min = weighted_scores_all == min_scores
                has_tie = is_min.sum(dim=0) > 1
                predictions = weighted_scores_all.argmin(dim=0)
                predictions[has_tie] = -1
                current_accuracy = ((predictions == correct_indices) & ~has_tie).float().mean().item()
                
                # Evaluate with negative weights clipped to 0
                clipped_weights = weights_use.clamp(min=0)
                clipped_weighted_scores = (scores * clipped_weights.view(-1, 1, 1)).mean(dim=0)
                min_scores_clipped = clipped_weighted_scores.min(dim=0, keepdim=True)[0]
                has_tie_clipped = (clipped_weighted_scores == min_scores_clipped).sum(dim=0) > 1
                predictions_clipped = clipped_weighted_scores.argmin(dim=0)
                clipped_accuracy = ((predictions_clipped == correct_indices) & ~has_tie_clipped).float().mean().item()
                
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    best_weights = {
                        'timestep': weights_use.detach().cpu().clone(),
                        'train_accuracy': train_accuracy,
                        'val_accuracy': val_accuracy,
                        'test_accuracy': test_accuracy,
                        'accuracy': current_accuracy,
                        'accuracy_clipped': clipped_accuracy,
                        'clipped_weights': clipped_weights.detach().cpu().clone()
                    }
                    if poly_degree is not None:
                        best_weights['poly_coeffs'] = poly_coeffs.detach().cpu().clone()
            
            if step % 10 == 0:
                print(f"Step {step}, Train: {train_accuracy:.4f}, Val: {val_accuracy:.4f}, Test: {test_accuracy:.4f}, Legacy: {current_accuracy:.4f} (Clipped: {clipped_accuracy:.4f}), CE Loss: {ce_loss.item():.4f}, L1: {l1_loss.item():.4f}")
        
        # Store results
        results_dict.update({
            'train_accuracy_before': float(train_accuracy_before),
            'val_accuracy_before': float(val_accuracy_before),
            'test_accuracy_before': float(test_accuracy_before),
            'train_accuracy': float(best_weights['train_accuracy']),
            'val_accuracy': float(best_weights['val_accuracy']),
            'test_accuracy': float(best_weights['test_accuracy']),
            'accuracy_original': float(train_accuracy_before),  # Legacy
            'accuracy_best': float(best_weights['accuracy']),  # Legacy
            'accuracy_clipped': float(best_weights['accuracy_clipped']),  # Legacy
            'timestep_weights': best_weights['timestep'].cpu().numpy(),
        })
        
        if 'poly_coeffs' in best_weights:
            results_dict['poly_coeffs'] = best_weights['poly_coeffs'].cpu().numpy()
    
    else:
        # First try to load multiple target gaussian noises
        target_noise_paths = sorted(glob.glob(os.path.join(noise_dir, 'target_gaussian_noises_batch*.pt')))
        if len(target_noise_paths) > 1:
            # Human sort the target noise paths based on batch number
            target_noise_paths = sorted(target_noise_paths, key=lambda x: int(x.split('batch')[-1].split('.')[0]))
        
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
                print(f"Target gaussian noise file not found at {target_noise_path}")
                return None
            target_noise_data = torch.load(target_noise_path)
            target_gaussian_noises = target_noise_data['target_gaussian_noises'].to(device)  # shape: [timesteps, 1, ...]
        
        num_timesteps = target_gaussian_noises.shape[0]
        
        # Initialize learnable weights
        if poly_degree is not None:
            # Initialize polynomial coefficients (without in-place operations)
            init_coeffs = torch.zeros(poly_degree + 1)
            init_coeffs[0] = 1.0  # Start with constant term = 1
            poly_coeffs = torch.nn.Parameter(init_coeffs.to(device))
            timestep_weights = get_polynomial_weights(poly_coeffs, num_timesteps, device)
            optimizer_params = [poly_coeffs]
        else:
            poly_coeffs = None
            timestep_weights = torch.ones(num_timesteps, device=device, requires_grad=True)
            optimizer_params = [timestep_weights]

        cfg_weights = torch.ones(num_timesteps, device=device, requires_grad=use_uncond)
        if use_uncond:
            optimizer_params.append(cfg_weights)
        
        optimizer = torch.optim.Adam(optimizer_params, lr=lr)
        
        # Get all noise files except target_gaussian_noises*.pt
        noise_files = sorted([f for f in glob.glob(os.path.join(noise_dir, "*.pt")) 
                      if not any(x in f for x in ["target_gaussian_noises.pt", "target_gaussian_noises_batch"])],
                      key=lambda x: int(x.split('batch')[-1].split('_')[0]))
        
        # Pre-load and concatenate all noise data
        print("Loading all noise data...")
        all_cond_noises = []
        all_uncond_noises = []
        
        for noise_file in tqdm(noise_files, desc="Loading noise files"):
            data = torch.load(noise_file)
            all_cond_noises.append(data['conditional_noises'])  # [num_selected, timesteps, batch_size, ...]
            if use_uncond:
                all_uncond_noises.append(data['unconditional_noises'])  # [1, timesteps, batch_size, ...]
        
        # Concatenate along batch dimension (dim=2)
        all_cond_noises = torch.cat(all_cond_noises, dim=2).to(device)  # [num_selected, timesteps, total_batch, ...]
        if use_uncond:
            all_uncond_noises = torch.cat(all_uncond_noises, dim=2).to(device)  # [1, timesteps, total_batch, ...]
        
        # Calculate total number of unique samples
        total_samples = all_cond_noises.shape[2]  # total_batch
        samples_per_group = all_cond_noises.shape[0]  # num_selected
        print(f"Total samples: {total_samples}, Candidates per sample: {samples_per_group}")
        
        # Split data into train/val/test sets
        num_train = int(total_samples * train_ratio)
        num_val = int(total_samples * val_ratio)
        num_test = total_samples - num_train - num_val

        # Create random permutation for splitting with fixed seed
        torch.manual_seed(42)  # Fixed seed for consistent splits
        indices = torch.randperm(total_samples)
        
        # Handle splitting based on train_ratio
        if train_ratio is None:
            # Use all data for everything (no splitting)
            train_indices = indices
            val_indices = indices
            test_indices = indices
            
            num_train = total_samples
            num_val = total_samples
            num_test = total_samples
        else:
            # Always take 50% for test set
            num_test = total_samples // 2
            remaining_samples = total_samples - num_test
            test_indices = indices[remaining_samples:]
            
            # Split remaining data into train/val based on ratios
            if train_ratio == 0:
                # Use all remaining data for both train and val
                train_indices = indices[:remaining_samples]
                val_indices = indices[:remaining_samples]  # Same as train
                
                num_train = remaining_samples
                num_val = remaining_samples
            else:
                num_train = int(remaining_samples * train_ratio)
                num_val = remaining_samples - num_train
                
                train_indices = indices[:num_train]
                val_indices = indices[num_train:remaining_samples]

        print(f"Data split: Train={num_train}, Val={num_val}, Test={num_test}")

        if len(target_gaussian_noises.shape) == 5:
            target_gaussian_noises = target_gaussian_noises.reshape(target_gaussian_noises.shape[0], target_gaussian_noises.shape[1], -1)
            all_cond_noises = all_cond_noises.reshape(all_cond_noises.shape[0], all_cond_noises.shape[1], all_cond_noises.shape[2], -1)
            if use_uncond:
                all_uncond_noises = all_uncond_noises.reshape(all_uncond_noises.shape[0], all_uncond_noises.shape[1], all_uncond_noises.shape[2], -1)

        # Training loop
        best_accuracy = 0.0  # Initialize with 0 accuracy
        best_val_accuracy = 0.0
        accuracy_clipped = 0.0
        best_weights = {
            'timestep': torch.ones_like(timestep_weights).cpu(),
            'train_accuracy': train_accuracy_before,
            'val_accuracy': val_accuracy_before,
            'test_accuracy': test_accuracy_before,
            'accuracy': best_accuracy,
            'accuracy_clipped': accuracy_clipped
        }

        def compute_accuracy(indices, weights_t, weights_cfg=None):
            if use_uncond:
                cfg_weights_expanded = weights_cfg.view(1, -1, 1, 1)
                noise_pred = all_uncond_noises[..., indices, :] + cfg_weights_expanded * (all_cond_noises[..., indices, :] - all_uncond_noises[..., indices, :])
            else:
                noise_pred = all_cond_noises[..., indices, :]

            if saved_all_timesteps_for_target:
                target_expanded = target_gaussian_noises[..., indices, :].unsqueeze(0)
            else:
                target_expanded = target_gaussian_noises.unsqueeze(0).expand(-1, -1, len(indices), -1)

            timestep_weights_expanded = weights_t.view(1, -1, 1, 1)
            dists = ((noise_pred - target_expanded).to(torch.float16).norm(p=2, dim=-1))
            dists = (timestep_weights_expanded.squeeze(-1) * dists).mean(dim=1)
            scores = -dists.t()

            max_scores, _ = torch.max(scores, dim=1, keepdim=True)
            num_max = (scores == max_scores).sum(dim=1)
            correct = (num_max == 1) & (scores[:, 0] == max_scores.squeeze())
            return correct.float().mean().item()

        for step in tqdm(range(num_steps), desc="Optimizing weights"):
            optimizer.zero_grad()
            
            # Update timestep weights if using polynomial
            if poly_degree is not None:
                timestep_weights = get_polynomial_weights(poly_coeffs, num_timesteps, device)
            
            # Apply CFG per timestep only if using unconditional
            if use_uncond:
                cfg_weights_expanded = cfg_weights.view(1, -1, 1, 1)
                noise_pred = all_uncond_noises[..., train_indices, :] + cfg_weights_expanded * (all_cond_noises[..., train_indices, :] - all_uncond_noises[..., train_indices, :])
            else:
                noise_pred = all_cond_noises[..., train_indices, :]
            
            # Handle target noise shape based on whether we have per-batch targets
            if saved_all_timesteps_for_target:
                target_expanded = target_gaussian_noises[..., train_indices, :].unsqueeze(0)
            else:
                target_expanded = target_gaussian_noises.unsqueeze(0).expand(-1, -1, num_train, -1)
            
            timestep_weights_expanded = timestep_weights.view(1, -1, 1, 1)
            
            # Calculate L2 distances with timestep weights
            dists = ((noise_pred - target_expanded).to(torch.float16).norm(p=2, dim=-1))
            dists = (timestep_weights_expanded.squeeze(-1) * dists).mean(dim=1)
            dists = dists.t()
            scores = -dists
            
            # Use cross entropy loss with L1 regularization
            targets = torch.zeros(num_train, dtype=torch.long, device=device)
            ce_loss = torch.nn.functional.cross_entropy(scores, targets)
            l1_loss = l1_lambda * torch.norm(timestep_weights, p=1)
            loss = ce_loss + l1_loss
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Compute accuracies on all sets
            with torch.no_grad():
                train_accuracy = compute_accuracy(train_indices, timestep_weights, cfg_weights if use_uncond else None)
                val_accuracy = compute_accuracy(val_indices, timestep_weights, cfg_weights if use_uncond else None)
                test_accuracy = compute_accuracy(test_indices, timestep_weights, cfg_weights if use_uncond else None)
                
            # Track best weights based on validation accuracy
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_weights = {
                    'timestep': timestep_weights.detach().cpu().clone(),
                    'cfg': cfg_weights.detach().cpu().clone(),
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'test_accuracy': test_accuracy
                }
                if poly_degree is not None:
                    best_weights['poly_coeffs'] = poly_coeffs.detach().cpu().clone()
            
            if step % 10 == 0:
                print(f"Step {step}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, Test Acc: {test_accuracy:.4f}, CE Loss: {ce_loss.item():.4f}, L1 Loss: {l1_loss.item():.4f}")

        # Store results
        results_dict.update({
            'train_accuracy': float(best_weights['train_accuracy']),
            'val_accuracy': float(best_weights['val_accuracy']),
            'test_accuracy': float(best_weights['test_accuracy']),
            'timestep_weights': best_weights['timestep'].cpu().numpy(),
        })
        
        if 'poly_coeffs' in best_weights:
            results_dict['poly_coeffs'] = best_weights['poly_coeffs'].cpu().numpy()
        
        if use_uncond:
           results_dict['cfg_weights'] = best_weights['cfg'].cpu().numpy()
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 6))
        
        # Plot 1: Weights
        ax1 = plt.subplot(121)
        
        # Plot timestep weights on left y-axis
        color1 = sns.color_palette("colorblind")[0]
        ax1.plot(best_weights['timestep'].numpy(), label='Timestep Weights', color=color1, alpha=0.7)
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Timestep Weight', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        if poly_degree is not None:
            # Plot the polynomial fit points
            t = np.linspace(-1, 1, num_timesteps)
            poly_fit = np.zeros_like(t)
            for i, coeff in enumerate(best_weights['poly_coeffs'].numpy()):
                poly_fit += coeff * (t ** i)
            ax1.plot(np.abs(poly_fit), '--', color=color1, alpha=0.5, label=f'Polynomial (deg={poly_degree})')
        
        # Plot CFG weights on right y-axis if using unconditional
        if use_uncond:
            ax1_twin = ax1.twinx()
            color2 = sns.color_palette("colorblind")[1]
            ax1_twin.plot(best_weights['cfg'].numpy(), label='CFG Weights', color=color2, alpha=0.7)
            ax1_twin.set_ylabel('CFG Weight', color=color2)
            ax1_twin.tick_params(axis='y', labelcolor=color2)
            ax1_twin.ticklabel_format(useOffset=False, style='plain', axis='y')
        
        # Calculate correlation if using unconditional
        if use_uncond:
            corr = np.corrcoef(best_weights['timestep'].numpy(), best_weights['cfg'].numpy())[0, 1]
            title = f'Learned Weights per Timestep (n={total_samples})\n{best_accuracy:.3f} accuracy (corr: {corr:.2f})'
        else:
            title = f'Learned Weights per Timestep (n={total_samples})\n{best_accuracy:.3f} accuracy'
        
        if poly_degree is not None:
            title += f'\nPolynomial coefficients: {best_weights["poly_coeffs"].numpy().round(3)}'
        ax1.set_title(title)
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        if use_uncond:
            lines2, labels2 = ax1_twin.get_legend_handles_labels()
            ax1_twin.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(lines1, labels1, loc='upper right')
        
        # Plot 2: Softmax probability distributions
        ax2 = plt.subplot(122)
        
        # Plot histograms
        bins = np.linspace(0, 1, 50)
        ax2.hist(probs_before, bins=bins, alpha=0.5, label='Before Weighting', density=True)
        ax2.hist(probs_after, bins=bins, alpha=0.5, label='After Weighting', density=True)
        
        ax2.set_xlabel('Softmax Probability of Correct Class')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Classification Probabilities\nBefore and After Weighting')
        ax2.legend()
        
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join('learned_weights', f'{wandb_run_id}.pdf'))
        plt.close()
        
    # Save results
    os.makedirs('processed_runs', exist_ok=True)
    results_file = os.path.join('processed_runs', f'{wandb_run_id}_train{train_ratio}_val{val_ratio:.1f}_results_{positive_weights}_l1_{l1_lambda}_{poly_degree}_consistent.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results_dict, f)

    # Save weights
    os.makedirs('weights_data', exist_ok=True)
    torch.save(best_weights, os.path.join('weights_data', f'{wandb_run_id}_train{train_ratio}_val{val_ratio}_poly{poly_degree}_consistent.pt'))
    
    return best_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, 
                       default='/mnt/lustre/work/oh/owl661/compositional-vaes/score_results',
                       help='Base directory containing noise results')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Specific run ID to evaluate. If not provided, will process all run IDs recursively.')
    parser.add_argument('--num_steps', type=int, default=5000,
                       help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--use_uncond', action='store_true',
                       help='Whether to use unconditional noise mixing (default: False)')
    parser.add_argument('--poly_degree', type=int, default=None,
                       help='If provided, fit a polynomial of this degree instead of learning individual weights')
    parser.add_argument('--no_linear_solver', action='store_true',
                       help='Disable linear regression solver (enabled by default) and use gradient descent instead')
    parser.add_argument('--l1_lambda', type=float, default=0,
                       help='L1 regularization strength (default: 0.01)')
    parser.add_argument('--positive_weights', default=False,
                       help='Whether to use positive weights only (default: False)')
    parser.add_argument('--skip_processed', type=bool, default=True,
                       help='Skip runs that have already been processed (default: True)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Ratio of data to use for validation (default: 0.1)')
    args = parser.parse_args()
    
    args.no_linear_solver = True
    
    def cleanup_cuda_memory():
        """Helper function to clean up CUDA memory"""
        try:
            # Empty CUDA cache
            torch.cuda.empty_cache()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            # Force garbage collection
            import gc
            gc.collect()
        except Exception as e:
            print(f"Warning: Error during CUDA cleanup: {str(e)}")

    def process_run_id(run_dir, run_id, train_ratio=None, val_ratio=0.2):
        """Helper function to process a single run ID with error handling"""
        # Check if results already exist
        results_file = os.path.join('processed_runs', f'{run_id}_train{str(train_ratio)}_val{val_ratio:.1f}_results_{args.positive_weights}_l1_{args.l1_lambda}_{args.poly_degree}_consistent.pkl')
        if args.skip_processed and os.path.exists(results_file):
            print(f"Skipping already processed run: {run_id}")
            return True

        try:
            print(f"\nProcessing run: {run_id}")
            best_weights = load_and_evaluate_noises(run_dir, run_id, 
                                                  num_steps=args.num_steps, lr=args.lr,
                                                  use_uncond=args.use_uncond, poly_degree=args.poly_degree,
                                                  use_linear_solver=not args.no_linear_solver,
                                                  l1_lambda=args.l1_lambda,
                                                  positive_weights=args.positive_weights,
                                                  train_ratio=train_ratio,
                                                  val_ratio=val_ratio)
            # if best_weights is not None:
            #     print(f"\nFinal Results for {run_id}:")
            #     print(f"Best Accuracy: {best_weights['accuracy']:.4f}")
            #     # Load and print original accuracy from pickle file
            #     results_file = os.path.join('processed_runs', f'{run_id}_train{train_ratio}_val{val_ratio:.1f}_results.pkl')
            #     if os.path.exists(results_file):
            #         with open(results_file, 'rb') as f:
            #             results = pickle.load(f)
            #             print(f"Original Accuracy: {results['accuracy_original']:.4f}")
            #             print(f"Improvement: {(results['accuracy_best'] - results['accuracy_original'])*100:.2f}%")
            return True
        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory error for run {run_id}. Attempting cleanup...")
            cleanup_cuda_memory()
            # Try one more time after cleanup
            try:
                best_weights = load_and_evaluate_noises(run_dir, run_id, 
                                                      num_steps=args.num_steps, lr=args.lr,
                                                      use_uncond=args.use_uncond, poly_degree=args.poly_degree,
                                                      use_linear_solver=not args.no_linear_solver,
                                                      l1_lambda=args.l1_lambda,
                                                      positive_weights=args.positive_weights,
                                                      train_ratio=train_ratio,
                                                      val_ratio=args.val_ratio)
                if best_weights is not None:
                    print(f"\nFinal Results for {run_id} (after cleanup):")
                    print(f"Best Accuracy: {best_weights['accuracy']:.4f}")
                return True
            except Exception as retry_e:
                print(f"Failed to process run {run_id} even after memory cleanup: {str(retry_e)}")
                cleanup_cuda_memory()  # Clean up again after failure
                return False
        except Exception as e:
            print(f"Error processing run {run_id}: {str(e)}. Skipping...")
            cleanup_cuda_memory()  # Clean up after any error
            return False

    if args.run_id:
        # Process specific run ID
        run_dirs = []
        for root, dirs, files in os.walk(args.base_dir):
            if args.run_id in dirs:
                run_dirs.append(os.path.join(root, args.run_id))
        
        if not run_dirs:
            print(f"No directory found for run_id: {args.run_id}")
            return
        
        # Process the specific run
        for run_dir in run_dirs:
            process_run_id(run_dir, args.run_id)
    else:
        # Process all run IDs recursively
        processed_runs = set()
        successful_runs = 0
        failed_runs = 0
        for train_ratio in [0.01, 0.02, 0.03, 0.04, 0.05, None]:
            print(f"Processing all run IDs recursively with train ratio: {train_ratio}")
            
            # Get all directories and their creation times
            dir_info = []
            for root, dirs, files in os.walk(args.base_dir):
                for dir_name in dirs:
                    if len(dir_name) >= 8:
                        run_dir = os.path.join(root, dir_name)
                        # Get directory creation time
                        creation_time = os.path.getctime(run_dir)
                        dir_info.append((run_dir, dir_name, creation_time))
            
            # Sort directories by creation time (most recent first)
            dir_info.sort(key=lambda x: x[2], reverse=True)
            
            # Process directories in sorted order
            for run_dir, dir_name, _ in dir_info:
                if dir_name not in processed_runs:
                    processed_runs.add(dir_name)
                    
                    # Check if directory contains expected files
                    if any(fname.endswith('_scores.pt') or fname.startswith('target_gaussian_noises') for fname in os.listdir(run_dir)):
                        success = process_run_id(run_dir, dir_name, train_ratio=train_ratio, val_ratio=args.val_ratio)
                        if success:
                            successful_runs += 1
                        else:
                            failed_runs += 1
            
            print(f"\nProcessing complete!")
            print(f"Successfully processed: {successful_runs} runs")
            print(f"Failed to process: {failed_runs} runs")
            print(f"Total runs found: {len(processed_runs)}")
            processed_runs = set()
            successful_runs = 0
            failed_runs = 0

if __name__ == '__main__':
    main() 