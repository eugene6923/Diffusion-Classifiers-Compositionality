import os
import torch
import glob
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
import wandb

# Flag to ensure keys are printed only once
first_file_keys_printed_global = False

def login_wandb():
    # api_key = "c743143a34e5f47f22c8d98469cfb33387573055"
    # api_key = "local-addff5ac065ec24ca8d463a7188760e623c531d7"
    # api_key = "local-9e68aa2925ebe8ee4e638fc0e33a0d47922c20f6"
    # api_key = "local-1d3a330785de0daa6c5e0dd683723eabe9d80d35"

    # api_key = "local-9e68aa2925ebe8ee4e638fc0e33a0d47922c20f6"
    # add timeout period longer os
    import os

    os.environ["WANDB__SERVICE_WAIT"] = "30000"
    api_key = "9876773f72d210923a9694ae701f8d71c9d30381"

    os.environ["WANDB_API_KEY"] = api_key
    # os.environ["WANDB_BASE_URL"] = "http://185.80.128.108:8080"
    # os.environ["WANDB_BASE_URL"] = "http://176.118.198.12:8080"

login_wandb()

def compute_accuracy_with_weights(scores_data, correct_indices_data, weights, device):
    """
    Computes accuracy using provided timestep weights.
    Assumes scores_data: [num_timesteps, num_prompts, num_samples]
    correct_indices_data: [num_samples]
    weights: [num_timesteps]
    """
    with torch.no_grad():
        correct_indices_data = correct_indices_data.long().to(device)
        weighted_scores = (scores_data * weights.view(-1, 1, 1)).mean(dim=0)
        
        if weighted_scores.numel() == 0 or correct_indices_data.numel() == 0:
            return 0.0
        
        if weighted_scores.shape[1] != correct_indices_data.shape[0]:
             print(f"Warning: Sample size mismatch in compute_accuracy. Scores: {weighted_scores.shape[1]}, Indices: {correct_indices_data.shape[0]}")
             return 0.0

        min_scores_vals, _ = weighted_scores.min(dim=0, keepdim=True)
        is_min = weighted_scores == min_scores_vals
        has_tie = is_min.sum(dim=0) > 1
        
        predictions = torch.full_like(correct_indices_data, -1, dtype=torch.long, device=device)
        if weighted_scores.shape[0] > 0:
            predictions = weighted_scores.argmin(dim=0)
        
        predictions[has_tie] = -1
        correct_preds = (predictions == correct_indices_data) & ~has_tie
        return correct_preds.float().mean().item() if correct_preds.numel() > 0 else 0.0

def get_stats_for_run(run_dir_path, wandb_run_id, device, wandb_api):
    """
    Loads score data, fetches wandb config, and calculates uniform and exp-weighted accuracy.
    """
    global first_file_keys_printed_global
    score_files = sorted(glob.glob(os.path.join(run_dir_path, '*_scores.pt')))
    
    all_scores_list = []
    all_correct_indices_list = []
    num_timesteps = None

    print(f"Processing run: {wandb_run_id} from {run_dir_path}")
    for i, score_file in enumerate(tqdm(score_files, desc=f"Loading scores for {wandb_run_id}", leave=False)):
        try:
            data = torch.load(score_file, map_location='cpu')
            if not first_file_keys_printed_global:
                print(f"INFO: Keys in the first loaded score file ({score_file}): {list(data.keys())}")
                first_file_keys_printed_global = True
                
            scores_data_file = data['scores_with_timestep'].to(device)
            correct_indices_data_file = torch.tensor(data['correct_indices'], device=device, dtype=torch.long)
            
            current_timesteps = scores_data_file.shape[0]
            if num_timesteps is None:
                num_timesteps = current_timesteps
            elif num_timesteps != current_timesteps:
                print(f"Warning: Timestep mismatch in {score_file} for run {wandb_run_id}. Expected {num_timesteps}, got {current_timesteps}. Skipping file.")
                continue
            
            all_scores_list.append(scores_data_file)
            all_correct_indices_list.append(correct_indices_data_file)
        except Exception as e:
            print(f"Error loading or processing file {score_file}: {e}. Skipping file.")
            continue
            
    if not all_scores_list:
        print(f"No valid score data loaded for run {wandb_run_id}.")
        # Still try to fetch wandb config even if local files are missing/problematic
        # return None # Modified to allow config fetching even if scores fail

    run_version = 'N/A'
    run_task = 'N/A'
    if wandb_api:
        try:
            # Using the project path provided by the user
            run = wandb_api.run(f"oshapio/diffusion-itm/{wandb_run_id}")
            run_version = run.config.get('version', 'N/A')
            run_task = run.config.get('task', 'N/A')
            print(f"Successfully fetched wandb config for {wandb_run_id}: version='{run_version}', task='{run_task}'")
        except Exception as e:
            print(f"Warning: Could not fetch wandb config for run {wandb_run_id} from project oshapio/diffusion-itm: {e}")
    else:
        print(f"Skipping wandb config fetch for {wandb_run_id} as API is not available.")

    if not all_scores_list: # If score loading failed earlier
        return {
            'run_id': wandb_run_id,
            'uniform_accuracy': 'N/A (score data error)',
            'exp_weighted_accuracy': 'N/A (score data error)',
            'total_samples': 'N/A',
            'version': run_version,
            'task': run_task
        }

    try:
        scores_all_samples = torch.cat(all_scores_list, dim=2)
        correct_indices_all_samples = torch.cat(all_correct_indices_list, dim=0)
    except Exception as e:
        print(f"Error concatenating score data for run {wandb_run_id}: {e}. Skipping run accuracy calc.")
        return {
            'run_id': wandb_run_id,
            'uniform_accuracy': 'N/A (concat error)',
            'exp_weighted_accuracy': 'N/A (concat error)',
            'total_samples': 'N/A',
            'version': run_version,
            'task': run_task
        }

    total_samples = scores_all_samples.shape[2]
    if total_samples == 0:
        print(f"No samples found after concatenating for run {wandb_run_id}.")
        return {'run_id': wandb_run_id, 'uniform_accuracy': 'N/A (0 samples)', 'exp_weighted_accuracy': 'N/A (0 samples)', 'total_samples': 0, 'version': run_version, 'task': run_task}
    
    if scores_all_samples.shape[1] == 0: 
        print(f"Scores data has 0 prompts for run {wandb_run_id}. Cannot compute accuracy.")
        return {'run_id': wandb_run_id, 'uniform_accuracy': 'N/A (0 prompts)', 'exp_weighted_accuracy': 'N/A (0 prompts)', 'total_samples': total_samples, 'version': run_version, 'task': run_task}

    uniform_accuracy = 0.0
    exp_weighted_accuracy = 0.0
    if scores_all_samples.numel() > 0 and correct_indices_all_samples.numel() > 0 and num_timesteps is not None and num_timesteps > 0:
        # Uniform weights
        uniform_weights = torch.ones(num_timesteps, device=device, dtype=scores_all_samples.dtype)
        uniform_accuracy = compute_accuracy_with_weights(
            scores_all_samples, correct_indices_all_samples, uniform_weights, device
        )

        # Exponential weights: exp(-7*t) for t in [0, 1]
        t = torch.linspace(0, 1, num_timesteps, device=device, dtype=scores_all_samples.dtype)
        exp_weights = torch.exp(-7 * t)
        exp_weighted_accuracy = compute_accuracy_with_weights(
            scores_all_samples, correct_indices_all_samples, exp_weights, device
        )
    else:
        print(f"Data is empty, num_timesteps not set ({num_timesteps}), or 0 for run {wandb_run_id}. Accuracies are N/A.")
        
    return {
        'run_id': wandb_run_id,
        'uniform_accuracy': uniform_accuracy,
        'exp_weighted_accuracy': exp_weighted_accuracy,
        'total_samples': total_samples,
        'version': run_version,
        'task': run_task
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate and display uniform and exp-weighted accuracy for given wandb runs, including wandb config.")
    parser.add_argument('--base_dir', type=str, 
                        default='/mnt/lustre/work/oh/owl661/compositional-vaes/score_results',
                        help='Base directory containing score results for runs.')
    parser.add_argument('--run_ids', type=str, default="1gsnsa2o,dv7mhpzz,hx60elvb,gc605ar8,hwcn73mr,29q5d2xx,b7gx30uz,961j3xwe,l8g13ki0,wjuhppfn,9bw4cnl1,6g5mut9l,b4um7r6f,pluicytc,uwc7fyhe,k37mxru1,9zqho8xf,7ogbhzcr,5qs0cajc,qcsggx25,2ixwlnhn,tvq9it9p,fjzp9aw6,o5d3z2se,em6aaazu,wthwlvaf,hp2g1663,vy0kaed9,vlpu4ohf,6bmm64m2,screz7kc,gxtgq8xy,41wb1olj,vig7x23t,s3ztqxhw,23qsl093,q067qlrs,pi7bq42z,0lfvzysc,vx1poyp1,qru1ckzg,3zekw8fn,5erx4y12,u84yqwo7,k4woiq6b,1vm0mbq0,s9bo9zen,igvywro9,au2xmxaj,yaie1x38,0uy3ew0t,cwz050vu,n5gpqwrk,dia1lx1m",
                        help='Comma-separated string of wandb run IDs to process.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb_api = None
    try:
        wandb_api = wandb.Api()
        print("Successfully initialized wandb.Api().")
    except Exception as e:
        print(f"Warning: Failed to initialize wandb.Api(). Will not fetch configs. Error: {e}")
        print("Please ensure you have the 'wandb' library installed and are logged in ('wandb login').")

    results_list = []
    run_ids_list = [run_id.strip() for run_id in args.run_ids.split(',') if run_id.strip()]

    for run_id in run_ids_list:
        print(f"\nSearching for run ID: {run_id} in base directory: {args.base_dir}")
        run_found_path = None
        
        potential_path = os.path.join(args.base_dir, run_id)
        if os.path.isdir(potential_path):
            run_found_path = potential_path
        else:
            for root, dirs, files in os.walk(args.base_dir):
                if run_id in dirs:
                    run_found_path = os.path.join(root, run_id)
                    print(f"Found {run_id} at {run_found_path}")
                    break 
        
        if run_found_path and os.path.isdir(run_found_path):
            stats = get_stats_for_run(run_found_path, run_id, device, wandb_api)
            if stats:
                results_list.append(stats)
            else: 
                 results_list.append({
                    'run_id': run_id,
                    'uniform_accuracy': 'N/A (get_stats_for_run error)',
                    'exp_weighted_accuracy': 'N/A (get_stats_for_run error)',
                    'total_samples': 'N/A',
                    'version': 'N/A',
                    'task': 'N/A'
                })
        else:
            print(f"Directory for run ID {run_id} not found in {args.base_dir} or its subdirectories.")
            results_list.append({
                'run_id': run_id,
                'uniform_accuracy': 'N/A (run dir not found)',
                'exp_weighted_accuracy': 'N/A (run dir not found)',
                'total_samples': 'N/A',
                'version': 'N/A',
                'task': 'N/A'
            })
    
    if not results_list:
        print("No results to display.")
        return

    results_df = pd.DataFrame(results_list)
    
    cols_order = ['run_id', 'version', 'task', 'uniform_accuracy', 'exp_weighted_accuracy', 'total_samples']
    for col in cols_order:
        if col not in results_df.columns:
            results_df[col] = 'N/A'
            
    results_df = results_df[cols_order]

    if not results_df.empty:
        results_df = results_df.sort_values(by='run_id').reset_index(drop=True)
        print("\n--- Accuracy Results (All Samples) with Wandb Config ---")
        print(results_df.to_string(index=False))
    else:
        print("No data processed successfully.")

if __name__ == '__main__':
    main() 