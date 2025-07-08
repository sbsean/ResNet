import os
import sys
import random
import inspect
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F_t
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def create_checkpoint_folder(base_path='checkpoints', log=False):
    setting_number = 1

    while True:
        if not log:
            folder_name = f'setting_#{setting_number}'
        else:
            folder_name = f'setting_#{setting_number}/logs'
        path = os.path.join(base_path, folder_name)

        if not os.path.exists(path):
            os.makedirs(path)
            return path

        setting_number += 1


class Trainer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.model = kwargs.get('model') if 'model' in kwargs else ValueError('Model is required.')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lr = kwargs.get('lr', 1e-3)
        self.batch_size = kwargs.get('batch_size', 128)
        self.num_epochs = kwargs.get('num_epochs', 200)
        self.num_workers = kwargs.get('num_workers', 4)

        self.criterion = kwargs.get('criterion')

        self.train_data = kwargs.get('train_data')
        self.val_data = kwargs.get('val_data')

        self.seed = kwargs.get('seed', 42)
        self.checkpoint_dir = create_checkpoint_folder(base_path=kwargs.get('checkpoint_dir', './checkpoints'))
        self.log_dir = create_checkpoint_folder(base_path=kwargs.get('checkpoint_dir', './checkpoints'), log=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self._set_seed()
        self.model.to(self.device)

        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]

    @staticmethod
    def _set_seed(seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

    def _init_optimizer(self):
        opt = self.kwargs.get('optimizer')
        if opt is not None and self.model is not None:
            optimizer_class = getattr(torch.optim, opt)
            optimizer_params = inspect.signature(optimizer_class).parameters

            valid_kwargs = {k: v for k, v in self.kwargs.items() if k in optimizer_params}

            self.optimizer = optimizer_class(filter(lambda p: p.requires_grad, self.model.parameters()), **valid_kwargs)

    def _init_scheduler(self):
        sch = self.kwargs.get('scheduler')
        if sch is not None and self.optimizer is not None:
            scheduler_class = getattr(torch.optim.lr_scheduler, sch)
            scheduler_params = inspect.signature(scheduler_class).parameters

            valid_kwargs = {k: v for k, v in self.kwargs.items() if k in scheduler_params and k != 'optimizer'}

            self.scheduler = scheduler_class(self.optimizer, **valid_kwargs)
        else:
            self.scheduler = None

    def _init_data(self):
        if not self.train_data:
            raise ValueError("Training data is empty or not initialized.")
        if not self.val_data:
            raise ValueError("Validation data is empty or not initialized.")

        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def calculate_accuracy(self, outputs, targets):
        """Calculate classification accuracy for CrossEntropy loss"""
        _, predicted = torch.max(outputs.data, 1)
        total = targets.size(0)
        correct = (predicted == targets).sum().item()
        return 100 * correct / total

    def plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training curves for loss and accuracy"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss curves
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy curves
        ax2.plot(train_accs, label='Train Accuracy', color='blue')
        ax2.plot(val_accs, label='Validation Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        os.makedirs(os.path.join(self.checkpoint_dir, 'img'), exist_ok=True)
        fig.savefig(os.path.join(self.checkpoint_dir, 'img', 'training_curves.png'), dpi=150)
        plt.close()

    def train(self):
        os.makedirs(self.checkpoint_dir + '/ckpt', exist_ok=True)

        self._init_optimizer()
        self._init_scheduler()
        self._init_data()

        epoch_train_losses = []
        epoch_val_losses = []
        epoch_train_accs = []
        epoch_val_accs = []

        best_val_acc = 0.0
        min_val_loss = float('inf')

        steps_per_epoch = len(self.train_loader)
        total_steps = steps_per_epoch * self.num_epochs

        pbar = tqdm(total=total_steps, desc="Training", unit="step", dynamic_ncols=True)

        for epoch in range(self.num_epochs):
            # Training phase
            train_loss, train_acc = self.train_step(epoch, pbar)
            epoch_train_losses.append(train_loss)
            epoch_train_accs.append(train_acc)

            # Validation phase
            val_loss, val_acc = self.evaluate()
            epoch_val_losses.append(val_loss)
            epoch_val_accs.append(val_acc)

            if self.scheduler is not None:
                self.scheduler.step()

            # Log to tensorboard
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Save best model
            save_best = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_best = True

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                save_best = True

            if save_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_loss': train_loss,
                    'train_acc': train_acc
                }, os.path.join(self.checkpoint_dir + '/ckpt', 'best_model.pth'))

            # Plot training curves
            self.plot_training_curves(epoch_train_losses, epoch_val_losses, 
                                    epoch_train_accs, epoch_val_accs)

            tqdm.write(
                f"{{'epoch': {epoch}, 'train_loss': {train_loss:.4f}, 'train_acc': {train_acc:.2f}%, "
                f"'val_loss': {val_loss:.4f}, 'val_acc': {val_acc:.2f}%, 'lr': {self.optimizer.param_groups[0]['lr']:.1e}}}")

        # Save final model
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir + '/ckpt', 'final_model.pth'))
        pbar.close()

        print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")

    def train_step(self, epoch, pbar):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_steps = len(self.train_loader)

        for i, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            # random horizontal flip
            if torch.rand(1).item() < 0.5:
                data = F_t.hflip(data)

            self.optimizer.zero_grad()

            outputs = self.model(data)
            
            # CrossEntropy Loss 
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()


            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            total_correct += (predicted == targets).sum().item()

            total_loss += loss.item()

            # Update progress bar
            current_acc = 100 * total_correct / total_samples
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{current_acc:.2f}%",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.1e}"
            })
            pbar.update(1)

           
            global_step = epoch * total_steps + i
            self.writer.add_scalar('Train/Step_Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/Step_Accuracy', current_acc, global_step)

        avg_loss = total_loss / total_steps
        avg_acc = 100 * total_correct / total_samples

        return avg_loss, avg_acc

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            raise ValueError("Validation DataLoader is not initialized.")

        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_steps = len(self.val_loader)

        with tqdm(total=total_steps, desc="Evaluation", leave=False, unit='step', dynamic_ncols=True) as pbar:
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                outputs = self.model(data)
                
                # CrossEntropy Loss 
                loss = self.criterion(outputs, targets)

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()

                total_loss += loss.item()

                current_acc = 100 * total_correct / total_samples
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{current_acc:.2f}%"
                })
                pbar.update(1)

        avg_loss = total_loss / total_steps
        avg_acc = 100 * total_correct / total_samples

        return avg_loss, avg_acc