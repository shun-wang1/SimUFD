from torch import nn
import torch
import time
import os
from matplotlib import pyplot as plt
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict

class ModelTrainer:
    """Base trainer class that can be extended for different model types"""

    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = device
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model = None
        self.model_save_path = os.path.join(args.save_path, f'{args.model}_model.pth')

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        additional_metrics = {}

        for batch in train_loader:
            # To be implemented by subclasses
            pass

        return total_loss / len(train_loader), additional_metrics

    def validate_epoch(self, valid_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in valid_loader:
                # To be implemented by subclasses
                pass

        return total_loss / len(valid_loader)

    def train(self, train_loader, valid_loader, optimizer, scheduler):
        """Train the model"""
        start_time = time.time()
        train_losses = []
        valid_losses = []
        additional_metrics_history = {}

        os.makedirs(self.args.save_path, exist_ok=True)
        model_save_path = os.path.join(self.args.save_path, f'{self.args.model}_model.pth')

        for epoch in range(self.args.epochs):
            # Training phase
            train_loss, additional_metrics = self.train_epoch(train_loader, optimizer)
            train_losses.append(train_loss)

            # Update additional metrics history
            for key, value in additional_metrics.items():
                if key not in additional_metrics_history:
                    additional_metrics_history[key] = []
                additional_metrics_history[key].append(value)

            # Validation phase
            valid_loss = self.validate_epoch(valid_loader)
            valid_losses.append(valid_loss)

            # Logging
            current_lr = optimizer.param_groups[0]['lr']
            log_message = f'Epoch [{epoch + 1}/{self.args.epochs}] Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, LR: {current_lr:.6f}'

            # Add additional metrics to log message
            for key, value in additional_metrics.items():
                log_message += f', {key}: {value:.4f}'

            print(log_message)

            # Model checkpoint and early stopping
            if valid_loss < self.best_loss - self.args.early_stopping_delta:
                self.best_loss = valid_loss
                self.patience_counter = 0
                self.best_model = self.model.state_dict().copy() if not isinstance(self.model, dict) else {
                    k: v.state_dict().copy() for k, v in self.model.items()
                }
                self.save_checkpoint(epoch, train_loss, valid_loss, optimizer)
            else:
                self.patience_counter += 1

            # Learning rate adjustment
            if self.args.lr_scheduler:
                scheduler.step()

            if self.args.early_stopping and self.patience_counter >= self.args.early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

        # Restore best model
        if self.best_model is not None:
            self.load_best_model()

        # Final logging
        training_time = (time.time() - start_time) / 60
        print(f"Training completed in {training_time:.2f} minutes")

        # Plot training history
        self.plot_training_history(train_losses, valid_losses, additional_metrics_history)

        return train_losses, valid_losses, additional_metrics_history

    def load_best_model(self):
        if isinstance(self.model, dict):
            for name, model in self.model.items():
                if name in self.best_model:
                    model.load_state_dict(self.best_model[name])
        else:
            self.model.load_state_dict(self.best_model)

    def save_checkpoint(self, epoch, train_loss, valid_loss, optimizer):
        if isinstance(self.model, dict):
            save_dict = {
                'epoch': epoch,
                'args': self.args,
                'train_loss': train_loss,
                'valid_loss': valid_loss
            }

            for name, model in self.model.items():
                save_dict[f'{name}_state_dict'] = model.state_dict()
                if optimizer and name in optimizer:
                    save_dict[f'{name}_optimizer_state_dict'] = optimizer[name].state_dict()
        else:
            save_dict = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'args': self.args
            }

        torch.save(save_dict, self.model_save_path)

    def calculate_threshold_rule(self, scores, method='sigma', sigma_multiplier=3, percentile=95):
        """Calculate anomaly threshold using specified method"""
        if method == 'sigma':
            return np.mean(scores) + sigma_multiplier * np.std(scores)
        elif method == 'percentile':
            return np.percentile(scores, percentile)
        elif method == 'iqr':
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            iqr = q3 - q1
            return q3 + 1.5 * iqr
        else:
            raise ValueError(f"Unknown threshold method: {method}")

    def load_checkpoint(self, path=None):
        path = path or self.model_save_path
        checkpoint = torch.load(path, weights_only=False)

        if isinstance(self.model, dict):
            for name, model in self.model.items():
                if f'{name}_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint[f'{name}_state_dict'])
        else:
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])

        return checkpoint


    def plot_training_history(self, train_losses, valid_losses, additional_metrics_history):
        """Plot training history"""
        # Base implementation that can be overridden by subclasses
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'{self.args.model} Training History')
        plt.savefig(os.path.join(self.args.save_path, f'{self.args.model}_training_history.png'))

        # Plot additional metrics if available
        for key, values in additional_metrics_history.items():
            plt.figure(figsize=(12, 6))
            plt.plot(values, label=key)
            plt.xlabel('Epochs')
            plt.ylabel(key)
            plt.legend()
            plt.title(f'{self.args.model} {key}')
            plt.savefig(os.path.join(self.args.save_path, f'{self.args.model}_{key}.png'))

        plt.close('all')

    def plot_health_index(self, train_data, val_data, test_data, threshold, figsize=(6, 6)):
        train_timesteps, train_scores = train_data
        val_timesteps, val_scores = val_data
        test_timesteps, test_scores = test_data

        # Find first threshold crossing point
        all_timesteps = np.concatenate([train_timesteps, val_timesteps, test_timesteps])
        all_scores = np.concatenate([train_scores, val_scores, test_scores])
        threshold_crossings = np.where(all_scores > threshold)[0]

        if len(threshold_crossings) > 0:
            first_crossing_idx = threshold_crossings[0]
            crossing_time = all_timesteps[first_crossing_idx]
            crossing_score = all_scores[first_crossing_idx]

        # Set style for academic publication
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('white')

        # Plot data points
        ax.scatter(train_timesteps, train_scores,
                   c='#2E86C1', label='Training',
                   alpha=0.6, s=20, marker='o')
        ax.scatter(val_timesteps, val_scores,
                   c='#28B463', label='Validation',
                   alpha=0.6, s=20, marker='s')
        ax.scatter(test_timesteps, test_scores,
                   c='#CB4335', label='Testing',
                   alpha=0.6, s=20, marker='^')

        # Plot threshold
        ax.axhline(y=threshold, color='#424949',
                   linestyle='--', linewidth=1.5,
                   label=f'Threshold ({self.args.threshold_method})')

        # Mark first threshold crossing point if exists
        if len(threshold_crossings) > 0:
            ax.scatter(crossing_time, crossing_score,
                       c='#F1C40F', marker='*', s=200,
                       label='Detected Fault Point',
                       zorder=5, edgecolor='black', linewidth=1)

            # Add annotation
            ax.annotate(f'Fault Point\nt={int(crossing_time)}',
                        xy=(crossing_time, crossing_score),
                        xytext=(-40, 40), textcoords='offset points',
                        ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

        # Customize plot
        ax.set_title(self.args.data_type + ' Health Index Monitoring',
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('File Number', fontsize=12, labelpad=10)
        ax.set_ylabel('Health Index Score', fontsize=12, labelpad=10)

        ax.grid(False)
        ax.minorticks_off()

        # Customize ticks
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Customize legend
        ax.legend(frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=12,
                  loc='upper left',  #
                  bbox_to_anchor=(0.02, 0.98),  #
                  borderaxespad=0.)  #

        # Customize spines
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)

        # Add minor ticks
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', alpha=0.2)

        # Tight layout
        plt.tight_layout()

        # Save plot
        save_path = os.path.join(self.args.save_path, f'{self.args.model}_health_index.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()

    @abstractmethod
    def calculate_threshold(self, train_loader, valid_loader):
        pass

    @abstractmethod
    def validate_and_visualize(self, all_loader, threshold):
        pass

    @abstractmethod
    def collect_predictions(self, data_loader):
        pass
