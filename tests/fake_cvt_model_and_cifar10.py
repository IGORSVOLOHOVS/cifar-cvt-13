"""Stand-ins for CvT-13, its image processor and CIFAR-10: nothing is downloaded."""

from types import SimpleNamespace
from typing import Any, ClassVar

import torch
from PIL import Image
from torch import nn


class FakeInputs(dict[str, Any]):
    """A processor's output: a mapping that can be moved to a device."""

    def __init__(self) -> None:
        super().__init__(pixel_values=torch.zeros((1, 3, 8, 8)))
        self.device: torch.device | None = None

    def to(self, device: torch.device) -> "FakeInputs":
        self.device = device
        return self


class FakeProcessor:
    """What AutoImageProcessor.from_pretrained returns: every image becomes 3x8x8."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeInputs:
        self.calls.append(kwargs)
        return FakeInputs()


class TinyCvt(nn.Module):
    """A real, trainable model with CvT's `classifier` head: mean colour, then one linear layer."""

    def __init__(self, num_labels: int = 1000) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.classifier = nn.Linear(3, num_labels)
        self.device: torch.device | None = None

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.classifier(pixel_values.mean(dim=(2, 3))))

    def to(self, *args: Any, **kwargs: Any) -> "TinyCvt":
        self.device = args[0] if args else None
        return self


class FixedLogitsModel(nn.Module):
    """A model that always answers with the same logits, so the winning class is known."""

    def __init__(self, logits: list[float], id2label: dict[int, str] | None = None) -> None:
        super().__init__()
        self.logits = torch.tensor([logits])
        self.config = SimpleNamespace(id2label=id2label)
        self.grad_enabled: list[bool] = []

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        self.grad_enabled.append(torch.is_grad_enabled())
        return SimpleNamespace(logits=self.logits.expand(pixel_values.shape[0], -1))


class FakeCifar10:
    """datasets.CIFAR10 with a handful of 32x32 images and no download."""

    sizes: ClassVar[dict[bool, int]] = {True: 6, False: 4}
    created: ClassVar[list[dict[str, Any]]] = []
    classes: ClassVar[list[str]] = ["airplane", "automobile"]

    def __init__(
        self, root: str, train: bool = True, download: bool = False, transform: Any = None
    ) -> None:
        FakeCifar10.created.append({"root": root, "train": train, "download": download})
        self.transform = transform
        self.length = self.sizes[train]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[Any, int]:
        if index >= self.length:
            raise IndexError(index)
        image = Image.new("RGB", (32, 32), (index * 20, 100, 200))
        return (self.transform(image) if self.transform else image), index % 2
