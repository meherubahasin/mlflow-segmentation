# Segmentation Experiments Walkthrough

This document outlines how to run the segmentation experiments and analyze the results.

## 1. Running Experiments

The experiments are located in the `experiments/` directory. You can run them individually using `python`.

**Note:** Ensure you are in the project root (`/Users/betopia/segmentation`) and have `PYTHONPATH=.` set if needed (though the scripts should handle relative imports).

### Baseline
Train the standard ResNet18 model on CIFAR10.
```bash
python experiments/baseline.py
```

### Augmentation Study
Compare validation accuracy with different augmentation intensities (Light, Medium, Heavy).
```bash
python experiments/augmentation_study.py
```

### Learning Rate Schedule Study
Compare different LR schedules (Step, Cosine, OneCycle).
```bash
python experiments/lr_schedule_study.py
```

### Backbone Study
Compare different model backbones (ResNet18, ResNet50, EfficientNetB0, MobileNetV3).
```bash
python experiments/backbone_study.py
```

## 2. Generating Deliverables

Once the experiments are complete, results are stored in the `results/` directory as `results.json` files. You can generate the analysis deliverables using the scripts in `analysis/`.

### Experiment Results Table
Generates `experiments.md` containing a table of all runs, sorted by validation accuracy.
```bash
python analysis/generate_tables.py
```
**Output:** [experiments.md](file:///Users/betopia/segmentation/experiments.md)

### Metric Plots
Generates a comparison plot of validation accuracy over epochs for all experiments.
```bash
python analysis/plot_metrics.py
```
**Output:** `analysis/plots/val_acc_comparison.png`

### Failure Analysis
Visualizes misclassified examples from the best model of the baseline experiment.
```bash
python analysis/failure_analysis.py
```
**Output:** `analysis/failures/`

## 3. Configuration

You can adjust experiment settings in `config/base_config.py`.
- **Debug Mode:** Set `debug = True` and `debug_subset_size = 0.01` (1%) for fast verification runs. Set `debug = False` for full training.
- **Multiprocessing:** `num_workers` is currently set to `0` to avoid issues on macOS. You can increase this on Linux.

## 4. Troubleshooting
- **Missing Results:** Ensure the experiments finished successfully and `results.json` exists in each experiment's folder in `results/`.
- **Performance:** If verification is too slow, ensure `debug` mode is on.
