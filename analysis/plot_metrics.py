import json
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def plot_metrics(output_dir="analysis/plots"):
    """
    Plot metrics from results.json files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_files = glob.glob("results/*/results.json")
    
    plt.figure(figsize=(10, 6))
    
    for f in results_files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                
            name = data.get('experiment_name', 'unknown')
            history = data.get('history', {})
            val_acc = history.get('val_acc', [])
            
            if val_acc:
                epochs = range(1, len(val_acc) + 1)
                plt.plot(epochs, val_acc, label=f"{name} (Best: {max(val_acc):.4f})")
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "val_acc_comparison.png")
    print(f"Saved plot to {output_dir / 'val_acc_comparison.png'}")
    plt.close()

if __name__ == "__main__":
    plot_metrics()
