import os
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False

    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cuda.matmul.allow_tf32 = True
        return device


def setup_log_directory(root_log_dir: str):
    get_number = lambda path: path.replace(".pt", "").replace("version_", "")

    if os.path.exists(os.path.join(root_log_dir, "version_0")):
        folder_numbers = [
            int(get_number(folder)) for folder in os.listdir(root_log_dir)
        ]

        last_version_number = max(folder_numbers)

        version_name = f"version_{last_version_number + 1}"
    else:
        version_name = f"version_0"

    checkpoint_path = os.path.join(root_log_dir, version_name)

    os.makedirs(checkpoint_path, exist_ok=True)

    return checkpoint_path, version_name


def train_model(
    model,
    config,
    scheduler,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    train_metric,
    valid_metric,
    early_stopping,
    device,
    model_name=None,
    checkpoint_path=None,
    break_after_it=None,
):
    train_results = []
    valid_results = []

    N = config.EPOCHS * len(train_dataloader)
    progress = tqdm(total=N, desc="Training Progress", leave=True)

    scaler = torch.amp.GradScaler(device)  # type: ignore

    model = model.to(device)

    for epoch in range(config.EPOCHS):
        model.train()

        train_metric.reset()
        for i, batch in enumerate(train_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            if device == "cuda":
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    y_hat = model(x)
                    loss = F.cross_entropy(y_hat, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            train_metric.update(y_hat, y)
            acc = train_metric.compute().item()
            progress.set_postfix(
                {
                    "epoch": f"{epoch}",
                    "acc": f"{100 * acc:.2f}%",
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.7f}",
                }
            )

            train_results.append(
                [epoch + ((i + 1) / len(train_dataloader)), loss.item(), acc]
            )
            progress.update(1)

            if break_after_it is not None and i > break_after_it:
                break

        valid_metric.reset()
        model.eval()
        for i, batch in enumerate(valid_dataloader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            y_hat = model(x)

            val_loss = F.cross_entropy(y_hat, y).item()

            valid_metric.update(y_hat, y)
            val_acc = valid_metric.compute().item()

            progress.set_postfix(
                {
                    "epoch": f"{epoch}",
                    "acc": f"{100 * acc:.2f}%",
                    "loss": f"{loss.item():.4f}",
                    "val_acc": f"{100 * val_acc:.2f}%",
                    "val_loss": f"{val_loss:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.7f}",
                }
            )

            valid_results.append(
                [epoch + ((i + 1) / len(valid_dataloader)), val_loss, acc]
            )

            if break_after_it is not None and i > break_after_it:
                break

        if (
            break_after_it is None
            and model_name is not None
            and checkpoint_path is not None
        ):
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": model.state_dict(),
                "loss": loss,
            }

            torch.save(checkpoint, os.path.join(checkpoint_path, model_name))

        scheduler.step()

        early_stopping.check_early_stop(val_loss)

        if early_stopping.stop_training:
            print(f"Early stopping in epoch: {epoch}")
            break

        if break_after_it is not None:
            break

    return train_results, valid_results
