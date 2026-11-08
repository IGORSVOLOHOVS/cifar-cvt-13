# Manual

Run each script from the root of the repository, through `.\recipe.ps1 run <script>`: the paths `data/` and `models/` are relative.

## Train

`cifar_cvt_train.py` (the default of `.\recipe.ps1 run`) takes no options. Its settings are the fields of `ModelConfig` in the code: model `microsoft/cvt-13`, data in `./data`, batch size 64, one epoch, 5,000 validation images, learning rate `5e-5`.

It logs the device it runs on, loads CIFAR-10 and prints how many images went to training, validation and test, loads the model, prints its architecture, and trains with a progress bar that shows the current loss. After each epoch it logs the average loss. It then prints scikit-learn's classification report for the test set and `Final Accuracy: <value>`.

Above an accuracy of 0.65 it saves the model's `state_dict` to `models/cvt_cifar10_best.pth` and says where. Otherwise it warns `Accuracy is below target (65%). Consider more epochs.` and saves nothing. When the data or the model cannot be loaded it logs `Failed to load data: <reason>` or `Failed to initialize model: <reason>` and stops. It exits 0 in every one of these cases.

The validation split is held out from training but not evaluated.

## Infer

```powershell
.\recipe.ps1 run inference.py --image <path> [--model <checkpoint>]
```

`inference.py` loads the processor and the model named by `--model`, `microsoft/cvt-13` by default, converts the image to RGB and prints a panel with `Prediction` and `Class Index`. The label comes from the checkpoint's own `id2label`; for the default ImageNet checkpoint that is an ImageNet class, not a CIFAR-10 one. Only a ten-class checkpoint whose labels are the placeholders `LABEL_0` to `LABEL_9` gets the CIFAR-10 names. An index with no label is printed as the number.

`--model` takes a Hugging Face checkpoint name or folder. The `.pth` file the training script saves holds only weights, so `inference.py` cannot load it as it stands.
