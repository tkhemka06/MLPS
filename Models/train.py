import torch 
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as utils
from torch.amp import GradScaler, autocast  # Fixed import
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data

class train:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device, num_epochs=60):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs  # Store num_epochs as instance variable
    
    def train(self):  # Removed unused 'epochs' parameter
        torch.manual_seed(42)  # For reproducibility
        scaler = GradScaler(device_type='cuda')  # Fixed GradScaler initialization
        
        for epoch in range(self.num_epochs):  # Use self.num_epochs
            # ========== Training Phase ==========
            self.model.train()  # Use self.model
            running_loss = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
            
            for X_batch, y_batch in progress_bar:
                self.optimizer.zero_grad()  # Use self.optimizer
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Mixed precision training
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = self.model(X_batch)
                    loss = self.loss_fn(outputs, y_batch)  # Use self.loss_fn instead of criterion
                
                scaler.scale(loss).backward()
                
                # Add unscale before gradient clipping to fix the error
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                scaler.step(self.optimizer)
                scaler.update()
                
                running_loss += loss.item()
                progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

            avg_train_loss = running_loss / len(self.train_loader)

            # ========== Validation Phase ==========
            self.model.eval()
            val_loss = 0.0
            progress_bar_val = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
            
            with torch.no_grad():
                for X_val, y_val in progress_bar_val:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(X_val)
                        loss = self.loss_fn(outputs, y_val)
                    
                    val_loss += loss.item()
                    progress_bar_val.set_postfix({'val_loss': f"{loss.item():.4f}"})

            avg_val_loss = val_loss / len(self.val_loader)
            
            # Learning rate scheduling - Check if scheduler exists and handle different types
            if hasattr(self, 'scheduler'):
                if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            print(f"\nEpoch {epoch+1}/{self.num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
