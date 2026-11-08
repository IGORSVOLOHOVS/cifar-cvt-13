"""inference.py with CvT-13 and its processor replaced."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fake_cvt_model_and_cifar10 import FakeProcessor, FixedLogitsModel
from PIL import Image

import inference


def winner_at(index: int, size: int) -> list[float]:
    logits = [0.0] * size
    logits[index] = 5.0
    return logits


def predict(
    tmp_path: Path, model: FixedLogitsModel, capsys: pytest.CaptureFixture[str]
) -> tuple[str, FakeProcessor]:
    image = tmp_path / "plane.png"
    Image.new("L", (32, 32), 128).save(image)
    processor = FakeProcessor()
    with (
        patch.object(
            inference.AutoImageProcessor, "from_pretrained", return_value=processor
        ) as load_processor,
        patch.object(
            inference.CvtForImageClassification, "from_pretrained", return_value=model
        ) as load_model,
    ):
        inference.run_inference(str(image), "some/checkpoint")
    load_processor.assert_called_once_with("some/checkpoint")
    load_model.assert_called_once_with("some/checkpoint")
    return capsys.readouterr().out, processor


def test_an_imagenet_checkpoint_names_the_class_from_its_own_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = {index: f"imagenet-{index}" for index in range(1000)}
    model = FixedLogitsModel(winner_at(404, 1000), id2label=labels)
    out, processor = predict(tmp_path, model, capsys)
    assert "Prediction: imagenet-404" in out
    assert "Class Index: 404" in out
    assert model.grad_enabled == [False]
    assert not model.training
    assert processor.calls[0]["images"].mode == "RGB"


def test_a_ten_class_fine_tune_without_labels_gets_the_cifar_names(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = {index: f"LABEL_{index}" for index in range(10)}
    out, _ = predict(tmp_path, FixedLogitsModel(winner_at(3, 10), id2label=labels), capsys)
    assert "Prediction: cat" in out


def test_an_index_past_the_cifar_names_is_shown_as_a_number(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, _ = predict(tmp_path, FixedLogitsModel(winner_at(42, 50)), capsys)
    assert "Prediction: 42" in out


def test_main_passes_the_image_and_the_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["inference.py", "--image", "x.png"])
    with patch.object(inference, "run_inference") as run:
        inference.main()
    run.assert_called_once_with("x.png", "microsoft/cvt-13")


def test_main_requires_an_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["inference.py"])
    with pytest.raises(SystemExit):
        inference.main()
