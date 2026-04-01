# -*- coding: utf-8 -*-
# File: utils/pytorch_pipeline.py

import time
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from numpy import floating
from torch import nn
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class PyTorchPipeline(object):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler] = None,
        device: Optional[torch.device] = None,
        save_full: bool = False,
        precision: int = 3,
        project: str = "training",
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device or self.get_device()
        self._save_full = save_full
        self._precision = precision
        self._project = project
        self._name = name
        self._kwargs = kwargs
        self._output_dir: Optional[Path] = None
        self._learning_rates: list[float] = []
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._best_loss: Optional[float] = None

    def start(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
    ) -> str:
        if val_dataloader is None and self._scheduler is not None:
            assert not isinstance(
                self._scheduler, ReduceLROnPlateau
            ), "If `val_dataloader` is None, then you cannot use ReduceLROnPlateau as your learning rate scheduler."

        if self._output_dir is None:
            self._output_dir = Path(self._project) / (self._name or self.get_datetime())
        self._output_dir.mkdir(parents=True, exist_ok=False)

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Retrieve learning rate for current epoch
            lr = self._optimizer.param_groups[0]["lr"]
            self._learning_rates.append(lr)

            # Compute losses for training and validation sets
            train_loss = self.train(train_dataloader, epoch, epochs)
            self._train_losses.append(train_loss)
            val_loss: Optional[float] = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader, epoch, epochs)
                self._val_losses.append(val_loss)

            # Update learning rate
            if self._scheduler is not None:
                self._scheduler.step(val_loss if isinstance(self._scheduler, ReduceLROnPlateau) else None)

            # Save current and best-performing (lowest loss) models
            self.save_checkpoint("checkpoint_last")
            if val_loss is not None and (self._best_loss is None or val_loss < self._best_loss):
                self._best_loss = val_loss
                self.save_checkpoint("checkpoint_best")

            # Plot training curves (losses and learning rates)
            self.plot()

            epoch_end_time = time.time()

            # Display epoch information
            print(
                f"{self.get_epoch_str(epoch, epochs)} -",
                f"train_loss: {train_loss:.{self._precision}f},",
                f"{f'val_loss: {val_loss:.{self._precision}f},' if val_loss is not None else ''}",
                f"lr: {lr:.{self._precision}e},",
                f"epoch_time: {self.format_time_elapsed(epoch_end_time - epoch_start_time)},",
                f"ETR: {self.get_etr(epoch, epochs, epoch_end_time - start_time)}",
                flush=True,
            )

        print(f"Total time: {self.format_time_elapsed(time.time() - start_time)}")

        return str(self._output_dir)

    def train(
        self,
        dataloader: DataLoader,
        epoch: int,
        epochs: int,
    ) -> floating:
        self._model.train()

        losses = []
        for X, y in tqdm(dataloader, desc=f"{self.get_epoch_str(epoch, epochs)} Training", leave=False):
            # Load a batch of data
            X, y = X.to(self._device), y.to(self._device)

            # Feedforward
            pred = self._model(X)

            # Loss
            loss = self._criterion(pred, y)
            losses.append(loss.item())

            # Backpropagation
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        return np.mean(losses)

    @torch.no_grad()  # Turn off gradient descent
    def validate(
        self,
        dataloader: DataLoader,
        epoch: int,
        epochs: int,
    ) -> floating:
        self._model.eval()

        losses = []
        for X, y in tqdm(dataloader, desc=f"{self.get_epoch_str(epoch, epochs)} Validating", leave=False):
            # Load a batch of data
            X, y = X.to(self._device), y.to(self._device)

            # Feedforward
            pred = self._model(X)

            # Loss
            loss = self._criterion(pred, y)
            losses.append(loss.item())

        return np.mean(losses)

    def save_checkpoint(self, name: str = "checkpoint") -> None:
        checkpoint = {"model_state_dict": self._model.state_dict()}
        checkpoint.update(self._kwargs)
        if self._save_full:
            checkpoint.update(
                {
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "train_losses": self._train_losses,
                    "val_losses": self._val_losses,
                    "learning_rates": self._learning_rates,
                }
            )
        torch.save(checkpoint, self._output_dir / f"{name}.pt")

    def plot(self) -> None:
        x = list(range(1, len(self._train_losses) + 1))
        for curve in ("Loss", "Learning Rate"):
            plt.figure()

            if curve == "Loss":
                plt.plot(x, self._train_losses, "b-", label="Train")
                if len(self._val_losses) > 0:
                    plt.plot(x, self._val_losses, "r-", label="Val")
                plt.legend()
            else:
                plt.plot(x, self._learning_rates, "g-")

            # Show only integers on the x-axis
            ax = plt.gca()
            ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

            plt.title(f"{curve} vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(curve)
            plt.tight_layout()
            plt.savefig(self._output_dir / f"{curve.lower().replace(' ', '_')}.png", dpi=300)
            plt.close()

    @cached_property
    def n_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    @cached_property
    def n_trainable_params(self) -> int:
        return sum(p.numel() for p in self._model.parameters() if p.requires_grad)

    @staticmethod
    def get_device() -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def format_time_elapsed(time_elapsed: float) -> str:
        assert time_elapsed >= 0

        d = int(time_elapsed // 86400)
        time_elapsed -= d * 86400
        h = int(time_elapsed // 3600)
        time_elapsed -= h * 3600
        m = int(time_elapsed // 60)
        time_elapsed -= m * 60
        s = round(time_elapsed)

        return f"{f'{d}:' if d > 0 else ''}{h:02d}:{m:02d}:{s:02d}"

    @staticmethod
    def get_etr(
        progress: int,
        total: int,
        time_elapsed: float,
    ) -> str:
        assert total > 0, f"{total} > 0. `total` must be a positive integer."
        assert (
            0 < progress <= total
        ), f"0 < {progress} <= {total}. `progress` must be a positive integer not greater than `total`."

        etr = time_elapsed * ((total / progress) - 1)
        return PyTorchPipeline.format_time_elapsed(etr)

    @staticmethod
    def get_datetime(sep: str = "-") -> str:
        return datetime.now().strftime(f"%Y%m%d{sep}%H%M%S")

    @staticmethod
    def get_epoch_str(epoch: int, epochs: int) -> str:
        return f"Epoch {epoch:{len(str(epochs))}d}/{epochs}"


# For testing purposes
if __name__ == "__main__":
    from torch import Tensor
    from torch.utils.data import Dataset

    DIM = 200
    EPOCHS = 20
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3

    class DummyDataset(Dataset):
        def __init__(self, size: int, dim: int) -> None:
            super().__init__()
            self.x = torch.rand(size, dim)
            self.y = torch.where(self.x.sum(dim=1) >= dim / 2, 1, 0)

        def __getitem__(self, index: int) -> Tensor:
            return self.x[index], self.y[index].unsqueeze(0).float()

        def __len__(self) -> int:
            return len(self.x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = DummyDataset(BATCH_SIZE * 100, DIM)
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = DummyDataset(BATCH_SIZE * 50, DIM)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = nn.Sequential(
        nn.Linear(DIM, DIM * 10),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(DIM * 10, DIM * 5),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(DIM * 5, DIM),
        nn.GELU(),
        nn.Dropout(),
        nn.Linear(DIM, 1),
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=EPOCHS // 2, gamma=0.1)

    pipeline = PyTorchPipeline(model, criterion, optimizer, scheduler=scheduler, device=device)
    print(f"Trainable parameters: {pipeline.n_parameters:,}")
    pipeline.start(EPOCHS, train_dataloader, val_dataloader=val_dataloader)
