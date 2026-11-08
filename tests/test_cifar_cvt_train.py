"""cifar_cvt_train.py with CvT-13, its processor and CIFAR-10 replaced."""

import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from fake_cvt_model_and_cifar10 import FakeCifar10, FakeProcessor, FixedLogitsModel, TinyCvt

import cifar_cvt_train
from cifar_cvt_train import CvtTrainer, ModelConfig, Result


def trainer_with(config: ModelConfig) -> CvtTrainer:
    with patch.object(
        cifar_cvt_train.AutoImageProcessor, "from_pretrained", return_value=FakeProcessor()
    ) as load_processor:
        trainer = CvtTrainer(config)
    load_processor.assert_called_once_with(config.model_name)
    return trainer


def small_config(tmp_path: Path) -> ModelConfig:
    return ModelConfig(data_dir=tmp_path, batch_size=2, val_size=2)


def test_an_ok_result_unwraps_to_its_value() -> None:
    result = Result[int, str].ok(7)
    assert result.is_ok()
    assert result.unwrap() == 7


def test_unwrapping_a_failure_raises_with_its_error() -> None:
    result = Result[int, str].fail("no data")
    assert not result.is_ok()
    with pytest.raises(ValueError, match="no data"):
        result.unwrap()


def test_the_default_config_fine_tunes_cvt_13_for_one_epoch() -> None:
    config = ModelConfig()
    assert config.model_name == "microsoft/cvt-13"
    assert config.data_dir == Path("./data")
    assert (config.batch_size, config.num_epochs, config.val_size) == (64, 1, 5000)
    assert config.lr == pytest.approx(5e-5)


def test_transform_drops_the_batch_dimension_of_the_processor_output(tmp_path: Path) -> None:
    trainer = trainer_with(small_config(tmp_path))
    pixels = trainer.transform("an image")
    assert pixels.shape == (3, 8, 8)
    assert trainer.image_processor.calls == [{"images": "an image", "return_tensors": "pt"}]


def test_prepare_data_splits_the_training_set_and_keeps_the_class_names(tmp_path: Path) -> None:
    trainer = trainer_with(small_config(tmp_path))
    FakeCifar10.created.clear()
    with patch.object(cifar_cvt_train.datasets, "CIFAR10", FakeCifar10):
        result = trainer.prepare_data()
    train, val, test, classes = result.unwrap()
    assert (len(train.dataset), len(val.dataset), len(test.dataset)) == (4, 2, 4)  # type: ignore[arg-type]
    assert classes == ["airplane", "automobile"]
    assert FakeCifar10.created == [
        {"root": str(tmp_path), "train": True, "download": True},
        {"root": str(tmp_path), "train": False, "download": True},
    ]
    images, labels = next(iter(test))
    assert images.shape == (2, 3, 8, 8)
    assert labels.tolist() == [0, 1]


def test_prepare_data_reports_a_failed_download_instead_of_raising(tmp_path: Path) -> None:
    trainer = trainer_with(small_config(tmp_path))
    with patch.object(cifar_cvt_train.datasets, "CIFAR10", side_effect=OSError("offline")):
        result = trainer.prepare_data()
    assert not result.is_ok()
    assert result.error == "Failed to load data: offline"


def test_initialize_model_replaces_the_imagenet_head_with_one_for_each_class(
    tmp_path: Path,
) -> None:
    trainer = trainer_with(small_config(tmp_path))
    model = TinyCvt(num_labels=1000)
    with patch.object(
        cifar_cvt_train.CvtForImageClassification, "from_pretrained", return_value=model
    ) as load_model:
        result = trainer.initialize_model(10)
    load_model.assert_called_once_with("microsoft/cvt-13")
    assert result.unwrap() is model
    assert (model.classifier.in_features, model.classifier.out_features) == (3, 10)
    assert model.device == trainer.device


def test_initialize_model_reports_missing_weights_instead_of_raising(tmp_path: Path) -> None:
    trainer = trainer_with(small_config(tmp_path))
    with patch.object(
        cifar_cvt_train.CvtForImageClassification,
        "from_pretrained",
        side_effect=OSError("no weights"),
    ):
        result = trainer.initialize_model(10)
    assert result.error == "Failed to initialize model: no weights"


def test_train_epoch_returns_the_mean_loss_and_moves_the_weights(tmp_path: Path) -> None:
    trainer = trainer_with(small_config(tmp_path))
    model = TinyCvt(num_labels=2)
    before = model.classifier.weight.detach().clone()
    torch.manual_seed(1)
    loader = [
        (torch.rand(2, 3, 8, 8), torch.tensor([0, 1])),
        (torch.rand(2, 3, 8, 8), torch.tensor([1, 0])),
    ]
    loss = trainer.train_epoch(model, loader)  # type: ignore[arg-type]
    assert model.training
    assert loss > 0
    assert not torch.equal(before, model.classifier.weight)


def test_evaluate_returns_the_share_of_images_whose_class_the_model_names(
    tmp_path: Path,
) -> None:
    trainer = trainer_with(small_config(tmp_path))
    model = FixedLogitsModel([0.0, 2.0])
    loader = [
        (torch.zeros(2, 3, 8, 8), torch.tensor([1, 1])),
        (torch.zeros(2, 3, 8, 8), torch.tensor([0, 1])),
    ]
    accuracy = trainer.evaluate(model, loader, ["airplane", "automobile"])  # type: ignore[arg-type]
    assert accuracy == pytest.approx(0.75)
    assert not model.training
    assert model.grad_enabled == [False, False]


@pytest.fixture
def run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> Callable[[float], list[logging.LogRecord]]:
    """main() on two fake CIFAR images, with the accuracy evaluate() reports fixed."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cifar_cvt_train, "ModelConfig", lambda: small_config(tmp_path))
    # Under pytest the script's basicConfig finds the root logger taken, so INFO is set here.
    caplog.set_level(logging.INFO, logger="cifar-cvt")

    def run(accuracy: float) -> list[logging.LogRecord]:
        with (
            patch.object(
                cifar_cvt_train.AutoImageProcessor,
                "from_pretrained",
                return_value=FakeProcessor(),
            ),
            patch.object(
                cifar_cvt_train.CvtForImageClassification,
                "from_pretrained",
                return_value=TinyCvt(),
            ),
            patch.object(cifar_cvt_train.datasets, "CIFAR10", FakeCifar10),
            patch.object(CvtTrainer, "evaluate", return_value=accuracy),
        ):
            cifar_cvt_train.main()
        return [record for record in caplog.records if record.name == "cifar-cvt"]

    return run


def test_main_saves_the_model_when_accuracy_is_above_65_percent(
    tmp_path: Path, run_main: Callable[[float], list[logging.LogRecord]]
) -> None:
    records = run_main(0.9)
    saved = tmp_path / "models" / "cvt_cifar10_best.pth"
    assert saved.is_file()
    state = torch.load(saved)
    assert tuple(state["classifier.weight"].shape) == (2, 3)
    assert records[0].getMessage().startswith("Initialized CvtTrainer on")
    assert "Model saved to" in records[-1].getMessage()


def test_main_saves_nothing_below_65_percent(
    tmp_path: Path, run_main: Callable[[float], list[logging.LogRecord]]
) -> None:
    records = run_main(0.5)
    assert not (tmp_path / "models").exists()
    assert records[-1].levelno == logging.WARNING
    assert "below target" in records[-1].getMessage()


def test_main_stops_before_the_model_when_the_data_fails(
    monkeypatch: pytest.MonkeyPatch, run_main: Callable[[float], list[logging.LogRecord]]
) -> None:
    monkeypatch.setattr(
        CvtTrainer, "prepare_data", lambda self: Result[tuple[()], str].fail("no data")
    )
    monkeypatch.setattr(CvtTrainer, "initialize_model", lambda *args: pytest.fail("loaded"))
    records = run_main(0.9)
    assert records[-1].levelno == logging.ERROR
    assert records[-1].getMessage() == "no data"


def test_main_stops_before_training_when_the_model_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    run_main: Callable[[float], list[logging.LogRecord]],
) -> None:
    monkeypatch.setattr(
        CvtTrainer,
        "initialize_model",
        lambda self, num_classes: Result[torch.nn.Module, str].fail("no weights"),
    )
    monkeypatch.setattr(CvtTrainer, "train_epoch", lambda *args: pytest.fail("trained"))
    records = run_main(0.9)
    assert records[-1].getMessage() == "no weights"
    assert not (tmp_path / "models").exists()
