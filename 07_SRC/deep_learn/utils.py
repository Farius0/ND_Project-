# ==================================================
# ===============  MODULE: utils  ==================
# ==================================================
from __future__ import annotations

import torch, numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Sequence

# Public API
__all__ = ["EarlyStopping", "plot_training"]

# ==================================================
# ================== EarlyStopping =================
# ==================================================

class EarlyStopping:
    def __init__(
        self,
        save_path: str,
        patience: int = 5,
        min_delta: float = 1e-3,
        save_model: bool = True,
        params: Optional[Dict[str, Any]] = None,
    ) -> None: 
        """
        Args:
            save_path (str): Path to save the best model checkpoint.
            patience (int): Number of epochs with no improvement before stopping.
            min_delta (float): Minimum required improvement to consider as progress.
            save_model (bool): Whether to save the model when improvement occurs.
            params (dict): Dictionary of hyperparameters.
        """
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer.")
        if not (isinstance(min_delta, (int, float)) and min_delta >= 0.0):
            raise ValueError("min_delta must be a non-negative float.")
        if not isinstance(save_path, str) or not save_path:
            raise ValueError("save_path must be a non-empty string.")   
             
        self.patience = patience
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.best_epoch: int = -1
        self.save_model = bool(save_model)
        self.save_path = save_path
        self.params = dict(params) if params is not None else None

    def __call__(
        self,
        val_loss: float,
        model: Optional[torch.nn.Module] = None,
        epoch: Optional[int] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        weights_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update early-stopping state.

        Returns
        -------
        bool
            True if training should stop, False otherwise.
        """

        improved = (self.best_loss is None) or (val_loss < self.best_loss - self.min_delta)

        if improved:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0

            if self.save_model and model is not None:
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

                checkpoint: Dict[str, Any] = {
                    "epoch": epoch,
                    "model_state": state_dict,
                    "best_val_loss": self.best_loss
                }

                if optimizer is not None:
                    checkpoint["optimizer_state"] = optimizer.state_dict()

                if scheduler is not None:
                    state = getattr(scheduler, "state_dict", None)
                    if callable(state):
                        checkpoint["scheduler_state"] = scheduler.state_dict()

                if self.params is not None:
                    if weights_params is not None:
                        self.params.update(weights_params)
                    checkpoint["params"] = self.params

                torch.save(checkpoint, self.save_path)
                print(f"Best model saved at epoch {epoch} with val_loss = {val_loss:.4f}")
        else:
            self.counter += 1
            print(f"No improvement detected ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                return True
        return False


def plot_training(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    train_accuracies: Optional[Sequence[float]] = None,
    val_accuracies: Optional[Sequence[float]] = None,
) -> None:
    """
    Displays training and validation loss curves, and accuracy curves if available.

    Args:
        train_losses (list of float): Training losses.
        val_losses (list of float): Validation losses.
        train_accuracies (list of float, optional): Training accuracies (in %).
        val_accuracies (list of float, optional): Validation accuracies (in %).
    """
    has_accuracy = train_accuracies is not None and val_accuracies is not None
    
    assert type(train_losses) == list and type(val_losses) == list
    assert len(train_losses) == len(val_losses)
    if has_accuracy:
        assert type(train_accuracies) == list and type(val_accuracies) == list
        assert len(train_accuracies) == len(val_accuracies)

    # ---- Figure ----
    n_cols = 2 if has_accuracy else 1
    fig, axs = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))

    # Normalize axs to a list
    axs = [axs] if n_cols == 1 else list(axs)

    # Loss plot
    axs[0].plot(train_losses, label="Train Loss")
    axs[0].plot(val_losses, label="Validation Loss")
    axs[0].set_title("Loss per Epoch")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Accuracy plot (if provided)
    if has_accuracy:
        axs[1].plot(train_accuracies, label="Train Accuracy")
        axs[1].plot(val_accuracies, label="Validation Accuracy")
        axs[1].set_title("Accuracy per Epoch")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy (%)")
        axs[1].legend()
    else:
        print("No accuracy data provided — only loss curves will be displayed.")

    fig.tight_layout()
    plt.show()
