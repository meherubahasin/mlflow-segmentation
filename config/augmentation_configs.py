from config.base_config import BaseConfig

class LightAugConfig(BaseConfig):
    """Light augmentation configuration"""
    experiment_name = "aug_light"
    use_augmentation = True
    
    # Augmentation parameters
    random_crop = True
    random_flip = True
    color_jitter = False
    rotation = 0
    

class MediumAugConfig(BaseConfig):
    """Medium augmentation configuration"""
    experiment_name = "aug_medium"
    use_augmentation = True
    
    random_crop = True
    random_flip = True
    color_jitter = True
    color_jitter_params = {
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
        'hue': 0.1
    }
    rotation = 15


class HeavyAugConfig(BaseConfig):
    """Heavy augmentation configuration"""
    experiment_name = "aug_heavy"
    use_augmentation = True
    
    random_crop = True
    random_flip = True
    color_jitter = True
    color_jitter_params = {
        'brightness': 0.4,
        'contrast': 0.4,
        'saturation': 0.4,
        'hue': 0.2
    }
    rotation = 30
    random_erasing = True
    mixup_alpha = 0.2
    cutmix_alpha = 1.0