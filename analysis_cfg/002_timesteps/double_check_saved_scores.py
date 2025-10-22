"""Script to analyze saved scores from a specific wandb run.
Loads results from .pt files and computes various statistics.
"""
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_and_analyze_scores(base_dir, wandb_id):
    """Load and analyze scores from all batches in a wandb run.
    
    Args:
        base_dir (str): Base directory containing score results
        wandb_id (str): The wandb run ID to analyze
    """
    # First find the correct directory containing the wandb ID
    score_dir = find_wandb_dir(base_dir, wandb_id)
    if score_dir is None:
        print(f"Could not find wandb ID {wandb_id} in any subdirectory of {base_dir}")
        return
        
    print(f"Found wandb run in: {score_dir}")
    
    # Get all score files
    score_files = sorted(glob(os.path.join(score_dir, '*_scores.pt')))
    
    if not score_files:
        print(f"No score files found in {score_dir}")
        return
    
    print(f"Found {len(score_files)} score files")
    
    # First, load all data and concatenate
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
    
    print(f"\nAnalyzing {scores.shape[2]} total samples across {scores.shape[0]} timesteps")
    
    # Calculate accuracy for each timestep
    timestep_accuracies = {}
    timestep_predictions = {}
    for t in range(scores.shape[0]):
        timestep_scores = scores[t]  # [num_prompts, total_samples]
        # Handle ties by marking them as incorrect
        min_scores = np.min(timestep_scores, axis=0, keepdims=True)  # [1, total_samples]
        is_min = timestep_scores == min_scores
        has_tie = np.sum(is_min, axis=0) > 1  # [total_samples]
        predictions = np.argmin(timestep_scores, axis=0)  # [total_samples]
        # Set predictions with ties to an invalid index to make them incorrect
        predictions[has_tie] = -1
        accuracy = np.mean(predictions == correct_indices)
        timestep_accuracies[t] = accuracy
        timestep_predictions[t] = predictions
    
    # Calculate overall accuracy (using average scores across timesteps)
    avg_scores = np.mean(scores, axis=0)  # [num_prompts, total_samples]
    # Handle ties by marking them as incorrect
    min_scores = np.min(avg_scores, axis=0, keepdims=True)  # [1, total_samples] 
    is_min = avg_scores == min_scores
    has_tie = np.sum(is_min, axis=0) > 1  # [total_samples]
    predictions = np.argmin(avg_scores, axis=0)  # [total_samples]
    # Set predictions with ties to an invalid index to make them incorrect
    predictions[has_tie] = -1
    correct_ones = predictions == correct_indices
    overall_accuracy = np.mean(correct_ones)
    
    print("\nOverall Statistics:")
    print(f"Mean Accuracy: {overall_accuracy:.4f}")
    
    # Plot accuracy vs timestep
    plt.figure(figsize=(10, 6))
    timesteps = sorted(timestep_accuracies.keys())
    accuracies = [timestep_accuracies[t] for t in timesteps]
    
    plt.plot(all_timesteps, accuracies, 'o-')
    plt.xlabel('Timestep Value')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Timestep (Run: {wandb_id})')
    plt.grid(True)
    
    # Save plot
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/accuracy_vs_timestep_{wandb_id}.pdf', bbox_inches='tight')
    plt.close()
    
    # Score distribution analysis
    plt.figure(figsize=(10, 6))
    sns.histplot(scores.flatten(), bins=50)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title(f'Score Distribution (Run: {wandb_id})')
    plt.savefig(f'figures/score_distribution_{wandb_id}.pdf', bbox_inches='tight')
    plt.close()
    
    # Additional analysis: Consistency across timesteps
    agreement_matrix = np.zeros((scores.shape[0], scores.shape[0]))
    for i in range(scores.shape[0]):
        for j in range(scores.shape[0]):
            agreement = np.mean(timestep_predictions[i] == timestep_predictions[j])
            agreement_matrix[i, j] = agreement
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(agreement_matrix, cmap='viridis', 
                xticklabels=all_timesteps, yticklabels=all_timesteps)
    plt.xlabel('Timestep Value')
    plt.ylabel('Timestep Value')
    plt.title('Prediction Agreement Between Timesteps')
    plt.savefig(f'figures/timestep_agreement_{wandb_id}.pdf', bbox_inches='tight')
    plt.close()
    
    return {
        'overall_accuracy': overall_accuracy,
        'timestep_accuracies': timestep_accuracies,
        'agreement_matrix': agreement_matrix,
        'scores_shape': scores.shape,
        'all_timesteps': all_timesteps
    }

if __name__ == "__main__":
    # Base directory for score results
    BASE_DIR = '/mnt/lustre/work/oh/owl661/compositional-vaes/score_results'
    
    # Specific wandb run ID to analyze
    WANDB_ID = '149qi202'
    
    # Run analysis
    results = load_and_analyze_scores(BASE_DIR, WANDB_ID)
    
    if results is not None:
        # Print some additional insights
        print("\nAdditional Insights:")
        print(f"Best timestep: {max(results['timestep_accuracies'].items(), key=lambda x: x[1])}")
        print(f"Worst timestep: {min(results['timestep_accuracies'].items(), key=lambda x: x[1])}")
        print(f"\nAverage agreement between consecutive timesteps: {np.mean([results['agreement_matrix'][i,i+1] for i in range(results['agreement_matrix'].shape[0]-1)]):.4f}")
    else:
        print(f"\nFailed to find or analyze wandb run {WANDB_ID}")
