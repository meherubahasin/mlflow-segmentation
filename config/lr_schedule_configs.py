from config.base_config import BaseConfig

class StepLRConfig(BaseConfig):
    """Step LR schedule"""
    experiment_name = "lr_step"
    lr_schedule = "step"
    lr_step_size = 30
    lr_gamma = 0.1


class CosineAnnealingConfig(BaseConfig):
    """Cosine annealing LR schedule"""
    experiment_name = "lr_cosine"
    lr_schedule = "cosine"
    lr_min = 1e-6
    

class OneCycleLRConfig(BaseConfig):
    """One cycle LR schedule"""
    experiment_name = "lr_onecycle"
    lr_schedule = "onecycle"
    lr_max = 1e-2
    pct_start = 0.3


class WarmupCosineConfig(BaseConfig):
    """Warmup + Cosine annealing"""
    experiment_name = "lr_warmup_cosine"
    lr_schedule = "warmup_cosine"
    warmup_epochs = 5
    lr_min = 1e-6
