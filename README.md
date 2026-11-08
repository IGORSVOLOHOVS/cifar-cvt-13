# cifar-cvt-13

Two console scripts around Microsoft's Convolutional Vision Transformer CvT-13 (`microsoft/cvt-13`) and CIFAR-10.

- `cifar_cvt_train.py` fine-tunes CvT-13 on CIFAR-10 for one epoch: it holds 5,000 training images out as a validation split, replaces the ImageNet head with one of ten classes, trains with AdamW at `5e-5`, batch size 64, and prints scikit-learn's classification report on the CIFAR-10 test set. When the test accuracy is above 65% it saves the weights to `models/cvt_cifar10_best.pth`.
- `inference.py` classifies one image with a Hugging Face checkpoint, `microsoft/cvt-13` by default, and prints the predicted label and class index.

Loading the data and the model returns a small `Result` value instead of raising, so `main()` reports either failure and stops.

## Use

```powershell
.\recipe.ps1 run                                        # cifar_cvt_train.py
.\recipe.ps1 run inference.py --image docs\assets\sample_airplane.png
```

`docs/install.md` says what to install and `docs/manual.md` what each script prints. `.\recipe.ps1 test` runs the tests, mypy and ruff with the model, the image processor and CIFAR-10 replaced: no test downloads weights or data.

## Licence

MIT, see `LICENSE`.

## Lab requirements

This repository complies with [revision 4.0.0](http://127.0.0.1:3000/igors-lab/software-requirements/src/tag/v4.0.0/BEST_REQUIREMENTS.md) of the lab requirements, except R-02 (#8), R-09 (#9), R-41 (#10), R-42 (#11).
