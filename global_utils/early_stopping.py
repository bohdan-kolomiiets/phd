import os
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, acceptable_change_percentage=0.0, verbose=False, path='checkpoint.pt'):
        """
        delta_pct (float): Minimum percentage improvement (e.g., 0.01 means 1%).
        """
        self.patience = patience
        self.delta_pct = acceptable_change_percentage
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self._save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss * (1 - self.delta_pct):
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self._save_checkpoint(val_loss, model)
            self.counter = 0

    def _save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model...")
        folder_path = os.path.dirname(self.path)
        os.makedirs(folder_path, exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss