"""
Fine-tuning CvT-13 on CIFAR-10 with professional engineering standards.
Includes monadic error handling, structured logging, and Python 3.12+ modernization.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import torch
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.metrics import accuracy_score, classification_report  # type: ignore
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets  # type: ignore
from transformers import AutoImageProcessor, CvtForImageClassification

# --- Monadic Error Handling ---
T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Result(Generic[T, E]):
    """Generic Result type for monadic error handling."""

    value: T | None = None
    error: E | None = None

    @classmethod
    def ok(cls, value: T) -> "Result[T, E]":
        return cls(value=value)

    @classmethod
    def fail(cls, error: E) -> "Result[T, E]":
        return cls(error=error)

    def is_ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> T:
        if self.error is not None:
            raise ValueError(f"Called unwrap on Failure: {self.error}")
        return cast(T, self.value)


# --- Configuration ---
@dataclass(frozen=True)
class ModelConfig:
    """Configuration for CvT Model training."""

    model_name: str = "microsoft/cvt-13"
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    batch_size: int = 64
    num_epochs: int = 1
    val_size: int = 5000
    lr: float = 5e-5


# --- Logging Setup ---
console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger("cifar-cvt")


# --- Core Logic ---
class CvtTrainer:
    """Functional core for CvT model training and evaluation."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_processor = AutoImageProcessor.from_pretrained(config.model_name)  # type: ignore
        logger.info(f"Initialized CvtTrainer on [bold cyan]{self.device}[/bold cyan]")

    def transform(self, image: Any) -> torch.Tensor:
        """Apply CvT-specific transformations."""
        inputs = self.image_processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0)  # type: ignore

    def prepare_data(
        self,
    ) -> Result[tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], list[str]], str]:
        """Load and split CIFAR-10 data."""
        try:
            logger.info("Loading CIFAR-10 dataset...")
            full_train = datasets.CIFAR10(
                root=str(self.config.data_dir), train=True, download=True, transform=self.transform
            )
            test_dataset = datasets.CIFAR10(
                root=str(self.config.data_dir), train=False, download=True, transform=self.transform
            )

            train_size = len(full_train) - self.config.val_size
            train_dataset, val_dataset = random_split(
                full_train, [train_size, self.config.val_size]
            )

            loaders: tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], list[str]] = (
                DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
                DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False),
                DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False),
                list(full_train.classes),
            )
            logger.info(
                f"Data ready: {train_size} train, {self.config.val_size} val, {len(test_dataset)} test."
            )
            return Result[
                tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], list[str]], str
            ].ok(loaders)
        except Exception as e:  # noqa: BLE001 - boundary handler: the error is reported, not swallowed
            return Result[
                tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], list[str]], str
            ].fail(f"Failed to load data: {e}")

    def initialize_model(self, num_classes: int) -> Result[nn.Module, str]:
        """Load and prepare CvtForImageClassification."""
        try:
            logger.info(f"Loading model {self.config.model_name}...")
            model = CvtForImageClassification.from_pretrained(self.config.model_name)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # type: ignore
            model.to(self.device)  # type: ignore
            return Result[nn.Module, str].ok(model)
        except Exception as e:  # noqa: BLE001 - boundary handler: the error is reported, not swallowed
            return Result[nn.Module, str].fail(f"Failed to initialize model: {e}")

    def train_epoch(self, model: nn.Module, loader: DataLoader[Any]) -> float:
        """Train model for one epoch."""
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Training...", total=len(loader))

            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                progress.update(task, advance=1, description=f"Loss: {loss.item():.4f}")

        return total_loss / len(loader)

    def evaluate(self, model: nn.Module, loader: DataLoader[Any], classes: list[str]) -> float:
        """Evaluate model and print metrics."""
        model.eval()
        all_preds: list[int] = []
        all_true: list[int] = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs.logits, 1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_true.extend(labels.numpy().tolist())

        accuracy = accuracy_score(all_true, all_preds)
        logger.info("\n[bold]Classification Report:[/bold]")
        console.print(classification_report(all_true, all_preds, target_names=classes))
        return float(accuracy)


def main() -> None:
    """Imperative shell for the training pipeline."""
    config = ModelConfig()
    trainer = CvtTrainer(config)

    # Data Path
    data_res = trainer.prepare_data()
    if not data_res.is_ok():
        logger.error(data_res.error)
        return
    train_loader, _val_loader, test_loader, classes = data_res.unwrap()

    # Model Path
    model_res = trainer.initialize_model(len(classes))
    if not model_res.is_ok():
        logger.error(model_res.error)
        return
    model = model_res.unwrap()

    # Architecture Visualization
    logger.info("Model Architecture Overview:")
    console.print(model)

    # Training Path
    for epoch in range(config.num_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{config.num_epochs}")
        avg_loss = trainer.train_epoch(model, train_loader)
        logger.info(
            f"Epoch {epoch + 1} complete. Average Loss: [bold red]{avg_loss:.4f}[/bold red]"
        )

    # Eval Path
    final_acc = trainer.evaluate(model, test_loader, classes)
    logger.info(f"Final Accuracy: [bold green]{final_acc:.4f}[/bold green]")

    # Save Path
    if final_acc > 0.65:
        logger.info("[bold green]SUCCESS: Accuracy is above 65%. Saving model...[/bold green]")
        save_path = Path("models")
        save_path.mkdir(exist_ok=True)
        model_file = save_path / "cvt_cifar10_best.pth"
        torch.save(model.state_dict(), model_file)
        logger.info(f"Model saved to [bold cyan]{model_file}[/bold cyan]")
    else:
        logger.warning("Accuracy is below target (65%). Consider more epochs.")


if __name__ == "__main__":
    main()
