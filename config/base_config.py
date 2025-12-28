class BaseConfig:
    
    # Dataset
    dataset_name = "CIFAR10"  # or "VOC2012", "COCO", etc.
    data_root = "./data"
    num_classes = 10
    input_size = (224, 224)
    
    # Training
    batch_size = 64
    num_epochs = 1
    num_workers = 0
    device = "cuda"
    seed = 42
    
    # Optimization
    optimizer = "adam"
    lr = 1e-3
    weight_decay = 1e-4
    momentum = 0.9
    
    # LR Schedule (baseline)
    lr_schedule = "constant"
    
    # Model
    save_best_only = False
    backbone = "resnet18"
    pretrained = False
    
    # Augmentation (baseline - minimal)
    use_augmentation = False
    
    # Logging
    log_interval = 10
    save_dir = "./results"
    experiment_name = "baseline"
    
    # Evaluation
    eval_interval = 1
    
    # Debug
    debug = True
    debug_subset_size = 0.01  # 1% of data