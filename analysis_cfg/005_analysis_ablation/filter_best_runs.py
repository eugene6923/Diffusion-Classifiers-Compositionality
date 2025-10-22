"""Script to filter and analyze wandb runs based on flexible criteria.
Builds on get_all_wandb_config_data.py to find best performing runs."""

import os
import pickle
from typing import Dict, List, Any, Optional
from collections import defaultdict
import pandas as pd
import numpy as np
from tabulate import tabulate
from datetime import datetime

def load_metadata(cache_dir: str) -> Dict:
    """Load the combined metadata file with error handling."""
    metadata_path = os.path.join(cache_dir, "all_runs_metadata.pkl")
    try:
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            return data['runs']
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def get_metric_value(summary: Dict, metric_keys: List[str], default=None) -> float:
    """Get metric value trying multiple possible keys."""
    for key in metric_keys:
        value = summary.get(key)
        if value is not None:
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
            if isinstance(value, (int, float)):
                return float(value)
    return default

def get_config_value(config: Dict, key: str, default=None) -> Any:
    """Get configuration value handling lists and None."""
    value = config.get(key, default)
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value

def normalize_bool(val: Any) -> bool:
    """Normalize any value to boolean. If value exists and isn't explicitly False, return True."""
    if val is None:  # Key doesn't exist
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() != 'false'
    return True  # Any other value is considered True

def safe_sort_values(values):
    """Sort values handling mixed types by converting all to strings."""
    return sorted(values, key=lambda x: str(x) if x is not None else '')

class AblationAnalyzer:
    def __init__(self, cache_dir: str):
        """Initialize analyzer with path to cached metadata."""
        self.cache_dir = cache_dir
        self.metadata = load_metadata(cache_dir)
        if not self.metadata:
            raise ValueError("No metadata found!")
        
        self.results_dir = os.path.join(os.path.dirname(cache_dir), 'ablation_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Convert to DataFrame for easier analysis
        self.df = self._create_dataframe()
        
        print(f"\nLoaded {len(self.metadata)} runs")
        print(f"Results will be saved to: {self.results_dir}")
        
        # Print unique values for debugging
        print("\nUnique values found:")
        for col in ['task', 'geneval_version', 'version', 'sampling_steps', 'model_precision']:
            values = self.df[col].unique()
            print(f"\n{col}:")
            for v in values:
                print(f"  {v} (type: {type(v)})")
        
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert metadata to DataFrame with consistent columns."""
        rows = []
        for run_id, run_data in self.metadata.items():
            config = run_data['config']
            summary = run_data['summary']
            
            # Get metrics with fallbacks
            accuracy = get_metric_value(summary, ['accuracy', 'best_accuracy', 'final_accuracy'])
            
            # Normalize use_normed_classifier to boolean - if key exists and isn't explicitly False, it's True
            use_normed = normalize_bool(config.get('use_normed_classifier'))
            
            # Get sampling steps
            sampling_steps = get_config_value(config, 'sampling_steps')
            
            # Only include runs with 30 or 100 steps
            if sampling_steps not in [30, 100]:
                continue
                
            # Get model precision and SD3 resize info
            model_precision = get_config_value(config, 'model_precision', 'float32')
            sd3_resize = normalize_bool(get_config_value(config, 'sd3_resize', False))
                
            # Convert versions to strings for consistent handling
            geneval_version = str(get_config_value(config, 'geneval_version'))
            base_version = str(get_config_value(config, 'version'))
            
            # Skip if no geneval version
            if not geneval_version or geneval_version.lower() == 'none':
                continue
            
            # For SD3, append resolution info
            if base_version == '3-m':
                version = f"3-m ({'resize' if sd3_resize else 'no-resize'})"
            else:
                version = base_version
                
            row = {
                'run_id': run_id,
                'task': get_config_value(config, 'task'),
                'geneval_version': geneval_version,
                'version': version,
                'base_version': base_version,
                'sampling_steps': sampling_steps,
                'model_precision': model_precision,
                'sd3_resize': sd3_resize,
                'accuracy': accuracy,
                'use_normed_classifier': use_normed
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        return df
    
    def create_ablation_table(self, steps: int) -> pd.DataFrame:
        """Create a comprehensive ablation table for specific number of steps."""
        # Filter for specific steps
        df_steps = self.df[self.df['sampling_steps'] == steps]
        
        if len(df_steps) == 0:
            return pd.DataFrame()  # Return empty DataFrame if no data
            
        # Get unique values using safe sorting
        tasks = safe_sort_values(df_steps['task'].unique())
        geneval_versions = safe_sort_values(df_steps['geneval_version'].unique())
        versions = safe_sort_values(df_steps['version'].unique())
        precisions = safe_sort_values(df_steps['model_precision'].unique())
        
        # Create multi-level table
        rows = []
        for task in tasks:
            for gv in geneval_versions:
                for v in versions:
                    for prec in precisions:
                        # Get accuracies for both normed and non-normed
                        subset = df_steps[
                            (df_steps['task'] == task) & 
                            (df_steps['geneval_version'] == gv) & 
                            (df_steps['version'] == v) &
                            (df_steps['model_precision'] == prec)
                        ]
                        
                        if len(subset) == 0:
                            continue  # Skip if no data for this combination
                        
                        normed_acc = subset[subset['use_normed_classifier']]['accuracy'].mean()
                        non_normed_acc = subset[~subset['use_normed_classifier']]['accuracy'].mean()
                        
                        normed_count = len(subset[subset['use_normed_classifier']])
                        non_normed_count = len(subset[~subset['use_normed_classifier']])
                        
                        # Only include if we have both normed and non-normed data
                        if normed_count == 0 or non_normed_count == 0:
                            continue
                        
                        row = {
                            'Task': task,
                            'GenEval Version': gv,
                            'Model Version': v,
                            'Precision': prec,
                            'Normed Acc': f"{normed_acc*100:.2f}% (n={normed_count})" if pd.notnull(normed_acc) else "N/A",
                            'Non-Normed Acc': f"{non_normed_acc*100:.2f}% (n={non_normed_count})" if pd.notnull(non_normed_acc) else "N/A",
                            'Diff (Normed - Non)': f"{(normed_acc - non_normed_acc)*100:.2f}%" if (pd.notnull(normed_acc) and pd.notnull(non_normed_acc)) else "N/A"
                        }
                        rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_sd3_comparison_table(self, steps: int) -> pd.DataFrame:
        """Create a comparison table for SD3-m variants with normed classifier."""
        # Filter for specific steps and SD3-m with normed classifier
        df_steps = self.df[
            (self.df['sampling_steps'] == steps) & 
            (self.df['base_version'] == '3-m') &
            (self.df['use_normed_classifier'] == True)
        ]
        
        if len(df_steps) == 0:
            return pd.DataFrame()
            
        # Get unique values
        tasks = safe_sort_values(df_steps['task'].unique())
        geneval_versions = safe_sort_values(df_steps['geneval_version'].unique())
        precisions = safe_sort_values(df_steps['model_precision'].unique())
        
        # Create table
        rows = []
        for task in tasks:
            for gv in geneval_versions:
                for prec in precisions:
                    # Get accuracies for both resize variants
                    subset = df_steps[
                        (df_steps['task'] == task) & 
                        (df_steps['geneval_version'] == gv) & 
                        (df_steps['model_precision'] == prec)
                    ]
                    
                    if len(subset) == 0:
                        continue
                    
                    resize_acc = subset[subset['sd3_resize']]['accuracy'].mean()
                    no_resize_acc = subset[~subset['sd3_resize']]['accuracy'].mean()
                    
                    resize_count = len(subset[subset['sd3_resize']])
                    no_resize_count = len(subset[~subset['sd3_resize']])
                    
                    # Only include if we have at least one variant
                    if resize_count == 0 and no_resize_count == 0:
                        continue
                    
                    row = {
                        'Task': task,
                        'GenEval Version': gv,
                        'Precision': prec,
                        'Resize Acc': f"{resize_acc*100:.2f}% (n={resize_count})" if pd.notnull(resize_acc) and resize_count > 0 else "N/A",
                        'No-Resize Acc': f"{no_resize_acc*100:.2f}% (n={no_resize_count})" if pd.notnull(no_resize_acc) and no_resize_count > 0 else "N/A",
                        'Diff (Resize - No)': f"{(resize_acc - no_resize_acc)*100:.2f}%" if (pd.notnull(resize_acc) and pd.notnull(no_resize_acc) and resize_count > 0 and no_resize_count > 0) else "N/A"
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_timestep_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table between 30 and 100 timesteps."""
        # Get unique values
        tasks = safe_sort_values(self.df['task'].unique())
        geneval_versions = safe_sort_values(self.df['geneval_version'].unique())
        versions = safe_sort_values(self.df['version'].unique())
        precisions = safe_sort_values(self.df['model_precision'].unique())
        
        # Create table
        rows = []
        for task in tasks:
            for gv in geneval_versions:
                for v in versions:
                    for prec in precisions:
                        # Get accuracies for both timestep variants
                        subset = self.df[
                            (self.df['task'] == task) & 
                            (self.df['geneval_version'] == gv) & 
                            (self.df['version'] == v) &
                            (self.df['model_precision'] == prec)
                        ]
                        
                        if len(subset) == 0:
                            continue
                        
                        acc_30 = subset[subset['sampling_steps'] == 30]['accuracy'].mean()
                        acc_100 = subset[subset['sampling_steps'] == 100]['accuracy'].mean()
                        
                        count_30 = len(subset[subset['sampling_steps'] == 30])
                        count_100 = len(subset[subset['sampling_steps'] == 100])
                        
                        # Skip if we don't have both 30 and 100 step data
                        if count_30 == 0 or count_100 == 0 or pd.isna(acc_30) or pd.isna(acc_100):
                            continue
                        
                        row = {
                            'Task': task,
                            'GenEval Version': gv,
                            'Model Version': v,
                            'Precision': prec,
                            '30 Steps': f"{acc_30*100:.2f}% (n={count_30})",
                            '100 Steps': f"{acc_100*100:.2f}% (n={count_100})",
                            'Diff (100 - 30)': f"{(acc_100 - acc_30)*100:.2f}%"
                        }
                        rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_ablation_analysis(self):
        """Save comprehensive ablation analysis for both 30 and 100 steps."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for steps in [30, 100]:
            # First table: Original normed vs non-normed analysis
            results = self.create_ablation_table(steps)
            if len(results) > 0:
                # Filter for geneval tasks only
                results = results[results['Task'].str.contains('geneval')]
                
                filename = f"normed_classifier_ablation_{steps}_steps_{timestamp}"
                
                # Create LaTeX table
                latex_path = os.path.join(self.results_dir, f"{filename}.tex")
                with open(latex_path, 'w') as f:
                    f.write("\\begin{table}[h]\n")
                    f.write("\\centering\n")
                    f.write("\\small\n")  # Make table slightly smaller to fit better
                    f.write(f"\\caption{{Normed vs Non-normed Classifier Performance ({steps} steps)}}\n")
                    f.write("\\label{tab:normed_comparison}\n")
                    f.write("\\begin{tabular}{@{}llcccr@{}}\n")  # Align numbers better
                    f.write("\\toprule\n")
                    f.write("Task & Model Ver. & Normed & Non-normed & Diff & GenEval Ver. \\\\\n")
                    f.write("\\midrule\n")
                    
                    # Sort by task name and then by improvement
                    results = results.copy()
                    results['Improvement'] = results['Diff (Normed - Non)'].str.rstrip('%').astype(float)
                    results = results.sort_values(['Task', 'Improvement'], ascending=[True, False])
                    
                    current_task = None
                    for _, row in results.iterrows():
                        # Clean up the task name
                        task = row['Task'].replace('geneval_', '')
                        task = task.replace('_', ' ').title()
                        
                        # Add midrule between different tasks
                        if current_task is not None and current_task != task:
                            f.write("\\midrule\n")
                        current_task = task
                        
                        # Clean up the numbers
                        normed = row['Normed Acc'].replace('%', '').split(' ')[0]
                        non_normed = row['Non-Normed Acc'].replace('%', '').split(' ')[0]
                        diff = row['Diff (Normed - Non)'].replace('%', '')
                        
                        # Format the row (task only shown if changed)
                        task_display = task if current_task == task else ""
                        latex_row = f"{task_display} & {row['Model Version']} & {normed}\\% & {non_normed}\\% & {diff}\\% & {row['GenEval Version']} \\\\\n"
                        f.write(latex_row)
                    
                    f.write("\\bottomrule\n")
                    f.write("\\end{tabular}\n")
                    f.write("\\end{table}\n")
                
                print(f"\nResults for {steps} steps saved to:")
                print(f"- LaTeX: {latex_path}")
                
                print(f"\nAblation Results ({steps} steps):")
                print(tabulate(results, headers='keys', tablefmt='grid', showindex=False))
            
            # Calculate statistics only for geneval tasks
            df_steps = self.df[
                (self.df['sampling_steps'] == steps) & 
                self.df['task'].str.contains('geneval')
            ]
            
            # Print overall statistics
            print(f"\nOverall Statistics for {steps} steps (Geneval Tasks Only):")
            normed = df_steps[df_steps['use_normed_classifier']]['accuracy'].mean()
            non_normed = df_steps[~df_steps['use_normed_classifier']]['accuracy'].mean()
            normed_count = len(df_steps[df_steps['use_normed_classifier']])
            non_normed_count = len(df_steps[~df_steps['use_normed_classifier']])
            
            print(f"\nNormed Classifier: {normed*100:.2f}% (n={normed_count})")
            print(f"Non-normed Classifier: {non_normed*100:.2f}% (n={non_normed_count})")
            print(f"Overall Improvement: {(normed - non_normed)*100:.2f}%")

    def save_timestep_analysis(self):
        """Save timestep comparison analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = self.create_timestep_comparison_table()
        if len(results) == 0:
            print("\nNo valid comparisons found!")
            return
            
        # Filter for geneval tasks only
        results = results[results['Task'].str.contains('geneval')]
        
        # Drop the Precision column
        results = results.drop(columns=['Precision'])
        
        filename = f"timestep_comparison_{timestamp}"
        
        # Save as CSV
        csv_path = os.path.join(self.results_dir, f"{filename}.csv")
        results.to_csv(csv_path, index=False)
        
        # Create markdown table
        md_path = os.path.join(self.results_dir, f"{filename}.md")
        with open(md_path, 'w') as f:
            f.write("# Timestep Comparison Analysis\n\n")
            f.write("Analysis comparing 30 vs 100 timesteps for normed classifier with geneval versions.\n\n")
            f.write(tabulate(results, headers='keys', tablefmt='pipe', showindex=False))
        
        # Create LaTeX table
        latex_path = os.path.join(self.results_dir, f"{filename}.tex")
        with open(latex_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Comparison of 30 vs 100 Timesteps Performance}\n")
            f.write("\\label{tab:timestep_comparison}\n")
            f.write("\\begin{tabular}{lllrrr}\n")
            f.write("\\toprule\n")
            f.write("Task & GenEval Ver. & Model Ver. & 30 Steps & 100 Steps & Diff \\\\\n")
            f.write("\\midrule\n")
            
            for _, row in results.iterrows():
                # Clean up the task name
                task = row['Task'].replace('geneval_', '')
                task = task.replace('_', ' ').title()
                
                # Clean up the numbers
                steps_30 = row['30 Steps'].replace('%', '').split(' ')[0]
                steps_100 = row['100 Steps'].replace('%', '').split(' ')[0]
                diff = row['Diff (100 - 30)'].replace('%', '')
                
                # Format the row
                latex_row = f"{task} & {row['GenEval Version']} & {row['Model Version']} & {steps_30}\\% & {steps_100}\\% & {diff}\\% \\\\\n"
                f.write(latex_row)
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"\nResults saved to:")
        print(f"- CSV: {csv_path}")
        print(f"- Markdown: {md_path}")
        print(f"- LaTeX: {latex_path}")
        
        print("\nTimestep Comparison Results:")
        print(tabulate(results, headers='keys', tablefmt='grid', showindex=False))
        
        # Calculate statistics only for geneval tasks
        df_geneval = self.df[self.df['task'].str.contains('geneval')]
        
        # Print overall statistics
        print("\nOverall Statistics (Geneval Tasks Only):")
        acc_30 = df_geneval[df_geneval['sampling_steps'] == 30]['accuracy'].mean()
        acc_100 = df_geneval[df_geneval['sampling_steps'] == 100]['accuracy'].mean()
        count_30 = len(df_geneval[df_geneval['sampling_steps'] == 30])
        count_100 = len(df_geneval[df_geneval['sampling_steps'] == 100])
        
        print(f"\nOverall 30 Steps: {acc_30*100:.2f}% (n={count_30})")
        print(f"Overall 100 Steps: {acc_100*100:.2f}% (n={count_100})")
        print(f"Overall Improvement: {(acc_100 - acc_30)*100:.2f}%")

def example_usage():
    """Example usage of the AblationAnalyzer."""
    cache_dir = os.path.join(os.path.dirname('/mnt/lustre/work/oh/owl661/compositional-vaes/score_results'), 
                            'cached_wandb_metadata')
    
    # Initialize analyzer
    analyzer = AblationAnalyzer(cache_dir)
    
    # Run both analyses
    analyzer.save_timestep_analysis()
    analyzer.save_ablation_analysis()

if __name__ == "__main__":
    example_usage()