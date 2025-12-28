import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def get_class_name(class_idx):
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    return classes[class_idx]

def build_model(config_dict, device):
    """Rebuild model from config dict"""
    num_classes = config_dict.get('num_classes', 10)
    backbone = config_dict.get('backbone', 'resnet18')
    pretrained = config_dict.get('pretrained', False)
    
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif backbone == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif backbone == 'mobilenet_v3':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
        
    return model.to(device)

def run_failure_analysis(checkpoint_path, data_root='./data', output_dir='analysis/failures', num_failures=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Rebuild model
    model = build_model(config, device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load validation data
    input_size = config.get('input_size', (224, 224))
    data_root = config.get('data_root', './data')
    dataset_name = config.get('dataset_name', 'CIFAR10')
    
    transform = transforms.Compose([
        transforms.Resize(tuple(input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if dataset_name == 'CIFAR10':
        val_dataset = datasets.CIFAR10(root=data_root, train=False,
                                     download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Inverse transform for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    failures = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            
            if preds != targets:
                img_tensor = inv_normalize(inputs[0]).cpu()
                failures.append({
                    'img': img_tensor,
                    'true': targets.item(),
                    'pred': preds.item(),
                    'idx': i
                })
                
                if len(failures) >= num_failures:
                    break
    
    # Plot failures
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, failure in enumerate(failures):
        if i >= len(axes): break
        
        img = failure['img'].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax = axes[i]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"True: {get_class_name(failure['true'])}\nPred: {get_class_name(failure['pred'])}")
        
    plt.tight_layout()
    plt.savefig(output_dir / "failure_cases.png")
    print(f"Saved failure analysis to {output_dir / 'failure_cases.png'}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ckpt_path = sys.argv[1]
    else:
        # Default to baseline best checkpoint if available
        ckpt_path = "results/baseline/best_checkpoint.pth"
        
    if Path(ckpt_path).exists():
        run_failure_analysis(ckpt_path)
    else:
        print(f"Checkpoint not found: {ckpt_path}")
