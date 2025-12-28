import json
import glob
import pandas as pd

def generate_table(output_file="experiments.md"):
    """
    Generate markdown table from results.json files (manual markdown)
    """
    results_files = glob.glob("results/*/results.json")
    all_runs = []
    
    for f in results_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
            
            # Extract relevant fields
            hist = data.get('history', {})
            val_acc_hist = hist.get('val_acc', [])
            val_loss_hist = hist.get('val_loss', [])
            
            best_val_acc = max(val_acc_hist) if val_acc_hist else 0.0
            last_val_loss = val_loss_hist[-1] if val_loss_hist else -1
            
            run_data = {
                'Experiment': data.get('experiment_name', 'unknown'),
                'Backbone': data.get('config', {}).get('backbone', 'unknown'),
                'Augmentation': str(data.get('config', {}).get('use_augmentation', False)),
                'LR Schedule': data.get('config', {}).get('lr_schedule', 'unknown'),
                'Val Accuracy': f"{best_val_acc:.4f}",
                'Val Loss': f"{last_val_loss:.4f}"
            }
            all_runs.append(run_data)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not all_runs:
        print("No results found.")
        return

    # Sort
    all_runs.sort(key=lambda x: float(x['Val Accuracy']), reverse=True)
    
    # Manual Markdown
    headers = ['Experiment', 'Backbone', 'Augmentation', 'LR Schedule', 'Val Accuracy', 'Val Loss']
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(['---']*len(headers)) + " |")
    
    for run in all_runs:
        row = [str(run[h]) for h in headers]
        lines.append("| " + " | ".join(row) + " |")
        
    md_table = "\n".join(lines)
    
    with open(output_file, 'w') as f:
        f.write("# Experiment Results\n\n")
        f.write(md_table)
        
    print(f"Table written to {output_file}")

if __name__ == "__main__":
    generate_table()
