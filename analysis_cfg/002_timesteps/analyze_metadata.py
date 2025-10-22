"""Script to analyze wandb metadata and create a summary table."""

import os
import pickle
import pandas as pd
from tabulate import tabulate
import sys
import numpy as np

def normalize_bool(val, default=True):
    """Normalize any value to boolean, with a default."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() != 'false'
    return default

def load_metadata(cache_dir):
    """Load the combined metadata file."""
    all_metadata_path = os.path.join(cache_dir, "all_runs_metadata.pkl")
    try:
        with open(all_metadata_path, 'rb') as f:
            data = pickle.load(f)
            # Convert wandb objects to plain dictionaries if needed
            runs = {}
            for run_id, run_data in data['runs'].items():
                try:
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
                    runs[run_id] = {'config': config, 'summary': summary}
                except Exception as e:
                    print(f"Warning: Could not process run {run_id}: {str(e)}")
                    continue
            return {'runs': runs, 'run_ids': data.get('run_ids', list(runs.keys()))}
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return {'runs': {}, 'run_ids': []}

def extract_fields(metadata):
    """Extract relevant fields from metadata and create a DataFrame."""
    rows = []
    for run_id, run_data in metadata['runs'].items():
        try:
            config = run_data.get('config', {})
            summary = run_data.get('summary', {})
            if not config:
                print(f"Warning: No config found for run {run_id}")
                continue
                
            # Calculate total steps * batch size
            steps = config.get('max_steps', config.get('num_steps', 0))
            batch_size = config.get('batch_size', 1)
            total_samples = steps * batch_size
            
            # Extract the fields we're interested in
            row = {
                'run_id': run_id,
                'sd3_resize': config.get('sd3_resize', None),
                'task': config.get('task', None),
                'geneval_version': config.get('geneval_version', None),
                'version': config.get('version', None),
                'use_normed_classifier': normalize_bool(config.get('use_normed_classifier', True)),
                # Add accuracy from summary
                'accuracy': summary.get('accuracy', None),
                'best_accuracy': summary.get('best_accuracy', None),
                'final_accuracy': summary.get('final_accuracy', None),
                # Add total samples for filtering
                'total_samples': total_samples,
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing run {run_id}: {str(e)}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    return df

def create_summary_table(df):
    """Create a summary table grouping by the specified fields."""
    # Remove rows where all specified fields are None
    config_fields = ['geneval_version', 'task', 'version', 'sd3_resize', 'use_normed_classifier']  # Reordered fields
    df_clean = df.dropna(subset=config_fields, how='all')
    
    # For each group, keep only the run with the largest total_samples
    df_filtered = df_clean.sort_values('total_samples', ascending=False).groupby(config_fields).first().reset_index()
    
    # Create summary with the filtered data
    grouped = df_filtered.groupby(config_fields).agg({
        'accuracy': ['mean', 'count'],
        'best_accuracy': ['mean', 'count'],
        'final_accuracy': ['mean', 'count'],
        'total_samples': 'first'  # Keep the total_samples for reference
    }).reset_index()
    
    # Flatten column names
    grouped.columns = [
        col[0] if col[1] == '' else f"{col[0]}_{col[1]}" 
        for col in grouped.columns
    ]
    
    # Sort hierarchically
    grouped = grouped.sort_values(
        ['geneval_version', 'task', 'version', 'best_accuracy_mean'],
        ascending=[True, True, True, False]
    )
    
    # Format accuracy columns to percentage with 2 decimal places
    for col in grouped.columns:
        if 'accuracy_mean' in col:
            grouped[col] = grouped[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")
    
    return grouped

def safe_sort_values(values):
    """Safely sort mixed type values by converting to strings and handling None."""
    def key_func(x):
        if pd.isna(x):
            return ('', '')  # None values come first
        if isinstance(x, bool):
            return ('bool', str(x))
        if isinstance(x, (int, float)):
            return ('num', str(x).zfill(20))  # pad numbers for proper string sorting
        return ('str', str(x))
    
    return sorted(values, key=key_func)

if __name__ == "__main__":
    # Directory containing cached metadata
    CACHE_DIR = '/mnt/lustre/work/oh/owl661/compositional-vaes/cached_wandb_metadata'
    
    if not os.path.exists(CACHE_DIR):
        print(f"Error: Cache directory not found: {CACHE_DIR}")
        sys.exit(1)
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata(CACHE_DIR)
    
    if not metadata['runs']:
        print("Error: No valid metadata found")
        sys.exit(1)
    
    print(f"Successfully loaded {len(metadata['runs'])} runs")
    
    # Extract fields and create DataFrame
    print("\nExtracting fields...")
    df = extract_fields(metadata)
    
    if len(df) == 0:
        print("Error: No valid data extracted")
        sys.exit(1)
    
    # Create summary table
    print("\nCreating summary table...")
    summary_table = create_summary_table(df)
    
    # Print table
    print("\nSummary Table:")
    print(tabulate(summary_table, headers='keys', tablefmt='grid', showindex=False))
    
    # Save summary table to CSV
    summary_csv_path = os.path.join(CACHE_DIR, 'metadata_summary.csv')
    summary_table.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary to: {summary_csv_path}")
    
    # Create and save detailed statistics
    print("\nCreating detailed statistics...")
    detailed_stats = []
    
    # First group by geneval_version
    for geneval_version in safe_sort_values(df['geneval_version'].unique()):
        geneval_df = df[df['geneval_version'] == geneval_version]
        
        # Then by task
        for task in safe_sort_values(geneval_df['task'].unique()):
            task_df = geneval_df[geneval_df['task'] == task]
            
            # Then by version and other fields
            for version in safe_sort_values(task_df['version'].unique()):
                version_df = task_df[task_df['version'] == version]
                
                for field in ['sd3_resize', 'use_normed_classifier']:
                    for val in safe_sort_values(version_df[field].unique()):
                        subset = version_df[version_df[field] == val]
                        stats = {
                            'geneval_version': geneval_version,
                            'task': task,
                            'version': version,
                            'field': field,
                            'value': val,
                            'count': len(subset),
                            'mean_accuracy': subset['accuracy'].mean() if pd.notnull(subset['accuracy'].mean()) else None,
                            'mean_best_accuracy': subset['best_accuracy'].mean() if pd.notnull(subset['best_accuracy'].mean()) else None,
                            'mean_final_accuracy': subset['final_accuracy'].mean() if pd.notnull(subset['final_accuracy'].mean()) else None,
                            'std_accuracy': subset['accuracy'].std() if pd.notnull(subset['accuracy'].std()) else None,
                            'std_best_accuracy': subset['best_accuracy'].std() if pd.notnull(subset['best_accuracy'].std()) else None,
                            'std_final_accuracy': subset['final_accuracy'].std() if pd.notnull(subset['final_accuracy'].std()) else None,
                        }
                        detailed_stats.append(stats)
    
    detailed_stats_df = pd.DataFrame(detailed_stats)
    # Convert accuracies to percentages
    for col in detailed_stats_df.columns:
        if 'accuracy' in col:
            detailed_stats_df[col] = detailed_stats_df[col].apply(lambda x: x * 100 if pd.notnull(x) else None)
    
    # Sort the detailed stats
    detailed_stats_df = detailed_stats_df.sort_values(['geneval_version', 'task', 'version', 'field', 'mean_best_accuracy'], 
                                                    ascending=[True, True, True, True, False])
    
    # Save detailed statistics to CSV
    detailed_csv_path = os.path.join(CACHE_DIR, 'metadata_detailed_stats.csv')
    detailed_stats_df.to_csv(detailed_csv_path, index=False)
    print(f"Saved detailed statistics to: {detailed_csv_path}")
    
    # Print statistics hierarchically
    print("\nStatistics by Hierarchy:")
    print(f"Total number of runs: {len(df)}")
    
    for geneval_version in safe_sort_values(df['geneval_version'].unique()):
        geneval_df = df[df['geneval_version'] == geneval_version]
        print(f"\nGenEval Version: {geneval_version}")
        print(f"Total runs: {len(geneval_df)}")
        
        for task in safe_sort_values(geneval_df['task'].unique()):
            task_df = geneval_df[geneval_df['task'] == task]
            mean_acc = task_df['best_accuracy'].mean()
            print(f"\n  Task: {task}")
            print(f"  Runs: {len(task_df)}")
            print(f"  Mean best accuracy: {mean_acc*100:.2f}%" if pd.notnull(mean_acc) else "  Mean best accuracy: N/A")
            
            for version in safe_sort_values(task_df['version'].unique()):
                version_df = task_df[task_df['version'] == version]
                mean_acc = version_df['best_accuracy'].mean()
                print(f"    Version: {version}")
                print(f"    Runs: {len(version_df)}")
                print(f"    Mean best accuracy: {mean_acc*100:.2f}%" if pd.notnull(mean_acc) else "    Mean best accuracy: N/A")