import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
from config.base_config import BaseConfig
from utils.metrics import MetricsTracker

from utils.logger import MLflowLogger


class ExperimentRunner:

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Setup directories
        self.exp_dir = Path(config.save_dir) / config.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Data loaders
        self.train_loader, self.val_loader = self._build_dataloaders()

        # Initialize components
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Tracking
        self.metrics_tracker = MetricsTracker(config.num_classes)
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        self.best_val_acc = 0.0

        # MLflow Logger
        self.logger = MLflowLogger(config.experiment_name, config.save_dir)
        
    def _build_model(self):
        """Build model based on config"""
        if self.config.backbone == 'resnet18':
            model = models.resnet18(pretrained=self.config.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        elif self.config.backbone == 'resnet50':
            model = models.resnet50(pretrained=self.config.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.config.num_classes)
        elif self.config.backbone == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.config.pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 
                                           self.config.num_classes)
        elif self.config.backbone == 'mobilenet_v3':
            model = models.mobilenet_v3_small(pretrained=self.config.pretrained)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, 
                                           self.config.num_classes)
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")
        
        return model.to(self.device)
    
    def _build_optimizer(self):
        """Build optimizer based on config"""
        if self.config.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), 
                            lr=self.config.lr,
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(),
                           lr=self.config.lr,
                           momentum=self.config.momentum,
                           weight_decay=self.config.weight_decay)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(self.model.parameters(),
                             lr=self.config.lr,
                             weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.lr_schedule == 'constant':
            return None
        elif self.config.lr_schedule == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_schedule == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.lr_min
            )
        elif self.config.lr_schedule == 'onecycle':
            return optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.lr_max,
                steps_per_epoch=len(self.train_loader),
                epochs=self.config.num_epochs,
                pct_start=self.config.pct_start
            )
        elif self.config.lr_schedule == 'warmup_cosine':
            # Custom warmup + cosine
            return self._get_warmup_cosine_scheduler()
        else:
            raise ValueError(f"Unknown LR schedule: {self.config.lr_schedule}")
    
    def _get_warmup_cosine_scheduler(self):
        """Custom warmup + cosine annealing scheduler"""
        def lr_lambda(epoch):
            if epoch < self.config.warmup_epochs:
                return epoch / self.config.warmup_epochs
            else:
                progress = (epoch - self.config.warmup_epochs) / \
                          (self.config.num_epochs - self.config.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _build_dataloaders(self):
        """Build train and validation dataloaders"""
        # Base transforms
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training transforms
        train_transforms = [transforms.Resize(self.config.input_size)]
        
        if self.config.use_augmentation:
            if self.config.random_crop:
                train_transforms.append(transforms.RandomCrop(
                    self.config.input_size, padding=4
                ))
            if self.config.random_flip:
                train_transforms.append(transforms.RandomHorizontalFlip())
            if hasattr(self.config, 'color_jitter') and self.config.color_jitter:
                params = self.config.color_jitter_params
                train_transforms.append(transforms.ColorJitter(**params))
            if hasattr(self.config, 'rotation') and self.config.rotation > 0:
                train_transforms.append(transforms.RandomRotation(
                    self.config.rotation
                ))
        
        train_transforms.extend([
            transforms.ToTensor(),
            normalize
        ])
        
        # Validation transforms (no augmentation)
        val_transforms = transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            normalize
        ])
        
        train_transform = transforms.Compose(train_transforms)
        
        # Load dataset
        if self.config.dataset_name == 'CIFAR10':
            train_dataset = datasets.CIFAR10(
                root=self.config.data_root,
                train=True,
                transform=train_transform,
                download=True
            )
            val_dataset = datasets.CIFAR10(
                root=self.config.data_root,
                train=False,
                transform=val_transforms,
                download=True
            )
        else:
            raise ValueError(f"Dataset {self.config.dataset_name} not implemented")

        if getattr(self.config, 'debug', False):
            # Subset dataset
            idx = np.arange(len(train_dataset))
            subset_len = int(len(train_dataset) * self.config.debug_subset_size)
            train_dataset = Subset(train_dataset, idx[:subset_len])
            
            idx = np.arange(len(val_dataset))
            subset_len = int(len(val_dataset) * self.config.debug_subset_size)
            val_dataset = Subset(val_dataset, idx[:subset_len])
            
            print(f"Debug mode: Reducing dataset to {self.config.debug_subset_size*100}%")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.metrics_tracker.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config.num_epochs}')
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            preds = outputs.argmax(dim=1)
            self.metrics_tracker.update(preds, targets, loss.item())

            # Log batch-level metrics
            self.logger.log_metrics({
                'train_batch_loss': loss.item(),
            }, step=epoch * len(self.train_loader) + batch_idx)
            
            # Update progress bar
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            # Step scheduler if OneCycle
            if self.config.lr_schedule == 'onecycle':
                self.scheduler.step()
        
        epoch_time = time.time() - epoch_start
        metrics = self.metrics_tracker.compute()
        
        return metrics, epoch_time
    
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        self.metrics_tracker.reset()
        
        for inputs, targets in tqdm(self.val_loader, desc='Validating'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            preds = outputs.argmax(dim=1)
            self.metrics_tracker.update(preds, targets, loss.item())
        
        metrics = self.metrics_tracker.compute()
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': vars(self.config),
            'history': self.history
        }
        
        # Save last checkpoint
        ckpt_path = self.exp_dir / 'last_checkpoint.pth'
        torch.save(checkpoint, ckpt_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.exp_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint (acc: {metrics['accuracy']:.4f})")
    
    def train(self):
        """Complete training loop"""
        print(f"\n{'='*60}")
        print(f"Starting Experiment: {self.config.experiment_name}")
        print(f"{'='*60}\n")
        
        print(f"Model: {self.config.backbone}")
        print(f"Dataset: {self.config.dataset_name}")
        print(f"Device: {self.device}")
        print(f"Augmentation: {self.config.use_augmentation}")
        print(f"LR Schedule: {self.config.lr_schedule}")

        # Log parameters
        self.logger.log_params(vars(self.config))
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Parameters: {num_params/1e6:.2f}M\n")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            train_metrics, epoch_time = self.train_epoch(epoch)
            
            # Validate
            if epoch % self.config.eval_interval == 0:
                val_metrics = self.validate()
                
                # Update history
                self.history['train_loss'].append(train_metrics['avg_loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['avg_loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

                # Log epoch metrics to MLflow
                self.logger.log_metrics({
                    'train_loss': train_metrics['avg_loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['avg_loss'],
                    'val_acc': val_metrics['accuracy'],
                    'lr': self.optimizer.param_groups[0]['lr']
                }, step=epoch)
                
                # Print metrics
                print(f"\nEpoch {epoch}/{self.config.num_epochs} "
                      f"({epoch_time:.1f}s)")
                print(f"Train - Loss: {train_metrics['avg_loss']:.4f}, "
                      f"Acc: {train_metrics['accuracy']:.4f}")
                print(f"Val   - Loss: {val_metrics['avg_loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}")
                
                # Save checkpoint
                is_best = val_metrics['accuracy'] > self.best_val_acc
                if is_best:
                    self.best_val_acc = val_metrics['accuracy']
                
                if not self.config.save_best_only or is_best:
                    self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Step scheduler (except OneCycle)
            if self.scheduler and self.config.lr_schedule != 'onecycle':
                self.scheduler.step()
        
        # Save final results
        self.save_results()
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.4f}")
        print(f"{'='*60}\n")
        
        return self.best_val_acc
    
    def save_results(self):
        """Save experiment results"""
        results = {
            'experiment_name': self.config.experiment_name,
            'best_val_acc': self.best_val_acc,
            'config': vars(self.config),
            'history': self.history,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = self.exp_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Log artifacts
        self.logger.log_artifact(str(results_path))
        if (self.exp_dir / 'best_checkpoint.pth').exists():
            self.logger.log_artifact(str(self.exp_dir / 'best_checkpoint.pth'))
        
        self.logger.close()
