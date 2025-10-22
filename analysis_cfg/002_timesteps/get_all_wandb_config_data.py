"""Script to scan wandb runs and save their configs and summaries in pickle files.
Similar to show_results.py but focusing on metadata only."""

import os
import pickle
import wandb
from tqdm import tqdm
import sys
import multiprocessing
from multiprocessing import Pool
from functools import partial

def login_wandb():
    """Login to wandb with API key."""
    api_key = "9876773f72d210923a9694ae701f8d71c9d30381"
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB__SERVICE_WAIT"] = "30000"

def extract_safe_value(obj):
    """Safely extract value from wandb objects to avoid recursion."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [extract_safe_value(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: extract_safe_value(v) for k, v in obj.items()}
    else:
        # For wandb objects, try to get their value or convert to string
        try:
            return obj.value if hasattr(obj, 'value') else str(obj)
        except:
            return str(obj)

def save_run_metadata(run_id, run_data, base_dir, cache_dir):
    """Save metadata for a single run, maintaining directory structure."""
    # Find the run's subdirectory in base_dir
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path) and run_id in os.listdir(subdir_path):
            # Create corresponding structure in cache_dir
            cache_subdir = os.path.join(cache_dir, subdir)
            os.makedirs(cache_subdir, exist_ok=True)
            
            # Save metadata file
            metadata_path = os.path.join(cache_subdir, f"{run_id}_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(run_data, f)
            return True
    return False

def process_single_run(run_id, base_dir, cache_dir, project_name):
    """Process a single run and return its metadata."""
    try:
        # Initialize wandb API (needed for each process)
        api = wandb.Api()
        
        # Try to get the run from wandb
        run = api.run(f"{api.default_entity}/{project_name}/{run_id}")
        
        # Safely extract config and summary
        config = {k: extract_safe_value(v) for k, v in run.config.items()}
        summary = {k: extract_safe_value(v) for k, v in run.summary.items() 
                  if not k.startswith('_')}
        
        # Create run data
        run_data = {
            'config': config,
            'summary': summary,
            'name': str(run.name),
            'state': str(run.state),
            'created_at': str(run.created_at)
        }
        
        # Save individual run metadata maintaining directory structure
        if save_run_metadata(run_id, run_data, base_dir, cache_dir):
            return run_id, run_data, None
        else:
            return run_id, None, f"Could not find directory structure for run {run_id}"
            
    except Exception as e:
        return run_id, None, str(e)

def get_runs_metadata(base_dir, cache_dir, project_name="diffusion-itm"):
    """Get metadata for all runs in the base directory using parallel processing."""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Get all immediate subdirectories (depth 1)
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # For each model directory, get all run directories
    all_run_ids = []
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        run_dirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
        all_run_ids.extend(run_dirs)
    
    print(f"Found {len(all_run_ids)} total runs in {base_dir}")
    
    # Calculate number of processes to use (all CPUs minus 4)
    num_processes = max(1, min(8, multiprocessing.cpu_count() - 4))
    print(f"Using {num_processes} processes for parallel processing")
    
    # Create partial function with fixed arguments
    process_run = partial(process_single_run, 
                         base_dir=base_dir,
                         cache_dir=cache_dir,
                         project_name=project_name)
    
    # Process runs in parallel
    metadata = {}
    failed_runs = []
    
    with Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_run, all_run_ids),
            total=len(all_run_ids),
            desc="Processing runs"
        ))
    
    # Process results
    for run_id, run_data, error in results:
        if run_data is not None:
            metadata[run_id] = run_data
        else:
            failed_runs.append((run_id, error))
    
    if failed_runs:
        print(f"\nFailed to process {len(failed_runs)} runs:")
        for run_id, error in failed_runs:
            print(f"  - {run_id}: {error}")
    
    # Save combined metadata as backup
    all_metadata_path = os.path.join(cache_dir, "all_runs_metadata.pkl")
    with open(all_metadata_path, 'wb') as f:
        pickle.dump({
            'runs': metadata,
            'run_ids': all_run_ids,
            'failed_runs': [run_id for run_id, _ in failed_runs]
        }, f)
    
    print(f"\nSuccessfully processed {len(metadata)} runs")
    print(f"Individual metadata files saved in: {cache_dir}")
    print(f"Combined backup saved as: {all_metadata_path}")
    
    # Print some statistics about the data
    print("\nConfig fields found:")
    all_config_keys = set()
    for run_data in metadata.values():
        all_config_keys.update(run_data['config'].keys())
    for key in sorted(all_config_keys):
        print(f"  - {key}")
    
    return metadata

if __name__ == "__main__":
    # Base directory containing score results
    BASE_DIR = '/mnt/lustre/work/oh/owl661/compositional-vaes/score_results'
    
    # Directory to save cached metadata
    CACHE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'cached_wandb_metadata')
    
    # Login to wandb
    login_wandb()
    
    # Get and save metadata
    metadata = get_runs_metadata(BASE_DIR, CACHE_DIR)
