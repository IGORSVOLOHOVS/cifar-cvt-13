# cifar-cvt-13 - software requirements specification

Reverse-engineered on 2026-11-18 from commit `a8b0dd1` (tag `v1.0.0`). Every
requirement describes behaviour present in the code at that commit; sources are
`path:line`. Documentation hub: [`docs/README.md`](docs/README.md).

## 1. Purpose and scope

The system fine-tunes the pretrained CvT-13 classifier (`microsoft/cvt-13`) on
CIFAR-10, reports its accuracy on the CIFAR-10 test set, saves the weights above
a fixed accuracy, and classifies a single image with a Hugging Face checkpoint
(`README.md:3-6`, `pyproject.toml:17`).

In scope: `cifar_cvt_train.py`, `inference.py`, their tests under `tests/`, and
the `recipe.ps1` verbs as they are configured for this project.

Out of scope: the CvT architecture and its weights (supplied by `transformers`
and the Hugging Face Hub), the CIFAR-10 files (supplied by `torchvision`), the
lab tooling shared with repo-quality-template (hooks, Gitea workflows, branch
policy), and the files under `scratch/` and `workflow_prompts/`, which are not
imported by either script.

## 2. Context

| Element | Role | Source |
| --- | --- | --- |
| Operator | Runs the scripts from the repository root in a shell; not otherwise described | `docs/manual.md:13` |
| Developer | Runs `recipe.ps1 test` before proposing a change | `CLAUDE.md` rule 3 |
| Hugging Face Hub | Supplies the image processor and model through `from_pretrained` | `cifar_cvt_train.py:78`, `:126`; `inference.py:26-27` |
| torchvision CIFAR-10 download | Supplies the training and test sets (`download=True`) | `cifar_cvt_train.py:92-97` |
| Local file system | `./data`, `models/`, the Hugging Face cache, `dist/` | `cifar_cvt_train.py:53`, `:222-225`; `docs/install.md:8`, `:10` |
| GPU (optional) | Used when `torch.cuda.is_available()` | `cifar_cvt_train.py:77`, `inference.py:22` |

Runtime: CPython 3.13 (`pyproject.toml:21`), dependencies from `uv.lock` through
`uv run --frozen` (`recipe.ps1:63`), PowerShell for `recipe.ps1`. On Windows the
locked `torch` is the CPU build (`pyproject.toml:23-25`).

## 3. Functional requirements

### 3.1 Result type and configuration

- **FR-1** The system shall provide a frozen `Result[T, E]` with `ok`, `fail`, `is_ok` and `unwrap`, where `unwrap` on a failure raises `ValueError` carrying the error. Source: `cifar_cvt_train.py:23-44`. Verification: `test_an_ok_result_unwraps_to_its_value`, `test_unwrapping_a_failure_raises_with_its_error`.
- **FR-2** The system shall default training to model `microsoft/cvt-13`, data directory `./data`, batch size 64, 1 epoch, 5000 validation images and learning rate `5e-5`. Source: `cifar_cvt_train.py:48-57`. Verification: `test_the_default_config_fine_tunes_cvt_13_for_one_epoch`.
- **FR-3** The system shall run on CUDA when available and on the CPU otherwise. Source: `cifar_cvt_train.py:77`, `inference.py:22`. Verification: no test.

### 3.2 Training data

- **FR-4** The system shall load the image processor of `model_name` when the trainer is created and turn each image into the processor's `pixel_values` without the batch dimension. Source: `cifar_cvt_train.py:78`, `:81-84`. Verification: `test_transform_drops_the_batch_dimension_of_the_processor_output`.
- **FR-5** The system shall load the CIFAR-10 training and test sets into `data_dir` with download enabled. Source: `cifar_cvt_train.py:92-97`. Verification: `test_prepare_data_splits_the_training_set_and_keeps_the_class_names`.
- **FR-6** The system shall split the training set at random into `len - val_size` training and `val_size` validation images. Source: `cifar_cvt_train.py:99-102`. Verification: `test_prepare_data_splits_the_training_set_and_keeps_the_class_names`.
- **FR-7** The system shall return training, validation and test loaders of `batch_size` (only the training loader shuffled) together with the dataset's class names. Source: `cifar_cvt_train.py:104-109`. Verification: `test_prepare_data_splits_the_training_set_and_keeps_the_class_names`.
- **FR-8** The system shall return a failed `Result` with the text `Failed to load data: <reason>` when loading the data raises. Source: `cifar_cvt_train.py:117-120`. Verification: `test_prepare_data_reports_a_failed_download_instead_of_raising`.

### 3.3 Model, training and evaluation

- **FR-9** The system shall load `CvtForImageClassification` from `model_name`, replace its `classifier` with a linear layer of the same input width and one output per class, and move it to the device. Source: `cifar_cvt_train.py:126-129`. Verification: `test_initialize_model_replaces_the_imagenet_head_with_one_for_each_class`.
- **FR-10** The system shall return a failed `Result` with the text `Failed to initialize model: <reason>` when loading the model raises. Source: `cifar_cvt_train.py:130-131`. Verification: `test_initialize_model_reports_missing_weights_instead_of_raising`.
- **FR-11** The system shall train one epoch in training mode with AdamW at `lr` and cross-entropy on the logits, one optimizer step per batch, and return the mean batch loss. Source: `cifar_cvt_train.py:133-162`. Verification: `test_train_epoch_returns_the_mean_loss_and_moves_the_weights`.
- **FR-12** The system shall show a progress bar with the current batch loss while training. Source: `cifar_cvt_train.py:140-160`. Verification: no test.
- **FR-13** The system shall evaluate in eval mode without gradients, predict the arg-max of the logits, print scikit-learn's classification report with the class names, and return the accuracy. Source: `cifar_cvt_train.py:164-181`. Verification: `test_evaluate_returns_the_share_of_images_whose_class_the_model_names`.

### 3.4 Training pipeline

- **FR-14** The system shall, in `main()`, load data, then the model, print the model architecture, train for `num_epochs` epochs logging each average loss, evaluate on the test set and log `Final Accuracy`. Source: `cifar_cvt_train.py:184-217`. Verification: `test_main_saves_the_model_when_accuracy_is_above_65_percent`.
- **FR-15** The system shall stop after logging the error when the data fails, before loading the model, and when the model fails, before training. Source: `cifar_cvt_train.py:190-201`. Verification: `test_main_stops_before_the_model_when_the_data_fails`, `test_main_stops_before_training_when_the_model_fails`.
- **FR-16** The system shall save the model's `state_dict` to `models/cvt_cifar10_best.pth`, creating `models/`, when the test accuracy is above 0.65. Source: `cifar_cvt_train.py:220-226`. Verification: `test_main_saves_the_model_when_accuracy_is_above_65_percent`.
- **FR-17** The system shall log the warning `Accuracy is below target (65%). Consider more epochs.` and save nothing otherwise. Source: `cifar_cvt_train.py:227-228`. Verification: `test_main_saves_nothing_below_65_percent`.

### 3.5 Inference

- **FR-18** The system shall accept a required `--image` path and an optional `--model` (default `microsoft/cvt-13`) and pass both to `run_inference`. Source: `inference.py:83-88`. Verification: `test_main_passes_the_image_and_the_default_model`, `test_main_requires_an_image`.
- **FR-19** The system shall load the processor and model of the given checkpoint, convert the image to RGB, and take the arg-max of the logits in eval mode without gradients. Source: `inference.py:26-39`. Verification: `test_an_imagenet_checkpoint_names_the_class_from_its_own_labels`.
- **FR-20** The system shall name the predicted class from the checkpoint's `id2label`; when the label is missing, or the checkpoint has 10 labels of the form `LABEL_<n>`, it shall use the CIFAR-10 name for indices below 10 and the index as text otherwise. Source: `inference.py:42-68`. Verification: `test_an_imagenet_checkpoint_names_the_class_from_its_own_labels`, `test_a_ten_class_fine_tune_without_labels_gets_the_cifar_names`, `test_an_index_past_the_cifar_names_is_shown_as_a_number`.
- **FR-21** The system shall print a panel titled `Inference Result` with `Prediction: <label>` and `Class Index: <index>`. Source: `inference.py:71-78`. Verification: `test_an_imagenet_checkpoint_names_the_class_from_its_own_labels` checks both lines; the other two label tests check `Prediction:` only.

### 3.6 Repository tooling

- **FR-22** The repository shall provide the `recipe.ps1` verbs of repo-quality-template (lab requirements revision 4.0.0, exceptions R-02, R-09, R-41, R-42), with `run` defaulting to `cifar_cvt_train.py` and `test` running the version check, pytest, mypy, `ruff check` and `ruff format`. Source: `recipe.ps1:47-111`, `README.md:25`. Verification: `.gitea/workflows/test.yaml:56` runs `.\recipe.ps1 test`.

### 3.7 Declared but not implemented

| Claim | Where | What the code does |
| --- | --- | --- |
| "structured logging" | `cifar_cvt_train.py:3` | Free-text messages through `RichHandler` (`cifar_cvt_train.py:62-67`) |
| "Inference script for CvT-13 model fine-tuned on CIFAR-10" | `inference.py:2` | No loader for the saved `.pth`; the default checkpoint is ImageNet-1k (`inference.py:55-59`, `docs/manual.md:33`) |
| `data/` and `models/` are gitignored | `docs/install.md:8` | `.gitignore` lists neither since `ead7824` |
| A validation split | `README.md:5`, FR-6 | Built, then discarded (`cifar_cvt_train.py:194`, `docs/manual.md:23`) |
| A multi-stage Dockerfile, performance baselines and similar steps | `workflow_prompts/07-docker-ci.md`, `14-perf-baselines.md` and others | Prompt texts only; the repository has no Dockerfile and no benchmarks (`git ls-files`). CI (`.gitea/workflows/`) and a wheel (`recipe.ps1 build`) do exist |

## 4. Interfaces

| Command | Arguments | Effect | Exit | Source |
| --- | --- | --- | --- | --- |
| `python cifar_cvt_train.py` | none | FR-14 to FR-17 | 0, including after a reported failure | `cifar_cvt_train.py:231-232`, `docs/manual.md:21` |
| `python inference.py` | `--image PATH` (required), `--model NAME` | FR-18 to FR-21 | argparse error without `--image`; uncaught exception on an unreadable image | `inference.py:83-88`, `:32` |
| `.\recipe.ps1 run [script args]` | script and its arguments; none means `cifar_cvt_train.py` | `uv run --frozen python ...` | the script's code | `recipe.ps1:60-65` |
| `.\recipe.ps1 test` | none | version check, hooks path, pytest, mypy, ruff check, ruff format | 1 at the first failing step, else 0 | `recipe.ps1:84-111` |
| `.\recipe.ps1 build` / `publish` / `info` / `clean` | none | wheel in `dist\` / wheel copied under `LAB_SHARE` / print identity / delete build outputs | uv's code / 1 without `LAB_SHARE` / 0 / 0 | `recipe.ps1:48-83` |

| Python API | Signature | Source |
| --- | --- | --- |
| `CvtTrainer` | `__init__(config)`, `transform(image)`, `prepare_data()`, `initialize_model(num_classes)`, `train_epoch(model, loader)`, `evaluate(model, loader, classes)` | `cifar_cvt_train.py:72-181` |
| `run_inference` | `run_inference(image_path: str, model_name: str = "microsoft/cvt-13") -> None` | `inference.py:20` |

## 5. Data

| Entity | Storage and format | Source |
| --- | --- | --- |
| CIFAR-10 | torchvision's layout under `./data`, downloaded on first use (about 170 MB) | `cifar_cvt_train.py:92-97`, `docs/install.md:8` |
| Class names | Training: the dataset's `classes`; inference fallback: the fixed 10-name list | `cifar_cvt_train.py:108`, `inference.py:42-53` |
| Fine-tuned weights | `models/cvt_cifar10_best.pth`, `torch.save` of the `state_dict` only, no config or labels | `cifar_cvt_train.py:222-225`, `docs/manual.md:33` |
| Pretrained weights and processor | Hugging Face cache | `docs/install.md:8` |
| Manual test list | `tests/manual/tests.json`, `version: 1`, empty `tests` list | `tests/manual/tests.json`, `tests/test_manual_list.py:30-51` |
| Wheel | `dist\*.whl` with modules `cifar_cvt_train` and `inference` | `recipe.ps1:49`, `pyproject.toml:46` |

## 6. Non-functional requirements

- **NFR-1** Python 3.13 only. Source: `pyproject.toml:21`, `uv.lock:3`.
- **NFR-2** Every direct dependency is pinned to an exact version and runs from the frozen lock. Source: `pyproject.toml:26-31`, `:38-42`, `recipe.ps1:63`.
- **NFR-3** No test downloads weights or data. Source: `tests/conftest.py:6`, `tests/fake_cvt_model_and_cifar10.py`. Verification: `test_the_hub_is_offline_for_the_whole_run`.
- **NFR-4** Coverage of the project's code, branches included, is at least 97%. Source: `pyproject.toml:80`, `:85`.
- **NFR-5** Both scripts pass mypy in strict mode. Source: `pyproject.toml:91-94`.
- **NFR-6** The code passes ruff (rules E, W, F, I, UP, B, BLE, SIM, C90, S, RUF; line length 100) and is formatted with LF line endings. Source: `pyproject.toml:52-71`.
- **NFR-7** No key is committed; `.claude/.env` is gitignored. Source: `.gitignore`, `CLAUDE.md` rule 4.

## 7. Constraints and dependencies

| Package | Version | Kind | Source |
| --- | --- | --- | --- |
| torch | 2.11.0 | runtime | `pyproject.toml:26` |
| torchvision | 0.26.0 | runtime | `pyproject.toml:27` |
| transformers | 5.5.4 | runtime | `pyproject.toml:28` |
| pillow | 12.2.0 | runtime | `pyproject.toml:29` |
| rich | 15.0.0 | runtime | `pyproject.toml:30` |
| scikit-learn | 1.7.2 | runtime | `pyproject.toml:31` |
| huggingface-hub | 1.33.0 | transitive, used directly by a test | `uv.lock:307-308`, `tests/test_no_weights_are_downloaded.py:3` |
| ruff, mypy, pytest, pytest-cov, coverage | 0.16.1, 2.3.0, 8.4.2, 5.0.0, 7.15.2 | dev | `pyproject.toml:38-42` |
| setuptools | 80.9.0 | build backend | `pyproject.toml:10` |

`uv.lock` holds 73 packages. `class Result[T, E]` uses the PEP 695 syntax, which
needs Python 3.12 or later (`cifar_cvt_train.py:24`).

## 8. Acceptance

1. `.\recipe.ps1 test` exits 0: checks FR-1 to FR-21 except FR-3 and FR-12
   (no test) through the tests named above, and NFR-3 to NFR-6. Formatting is
   applied in place, not checked; `git diff --exit-code` after the verb checks it
   (`.gitea/workflows/test.yaml:56-58`).
2. With network access, `.\recipe.ps1 run` prints the sequence in
   `docs/manual.md:19-21` and either writes `models/cvt_cifar10_best.pth` or logs
   the below-target warning (FR-14 to FR-17).
3. `.\recipe.ps1 run inference.py --image docs\assets\sample_airplane.png` prints
   the `Inference Result` panel (FR-18 to FR-21).
4. `.\recipe.ps1 build` writes a wheel to `dist\` that contains the two modules.

No manual test is recorded: `tests/manual/tests.json` is empty.

## 9. Open questions

1. Who uses the scripts. Partly answered: the course brief that commit `3dfb47e`
   removed (`docs/task.md`, a paid course's lesson per the commit message) asked
   for this fine-tune, so the scripts began as a course assignment; no user is
   named.
2. What accuracy a real run reaches; no result is recorded in the repository.
   The 65% threshold is answered: the removed brief said a correct solution
   exceeds 65% test accuracy after one epoch (`git show 3dfb47e`).
3. Whether the validation split is meant to be evaluated or used for model
   selection. The removed brief creates it with `random_split` (5000 of 50000)
   and does not say what it is for.
4. Whether the saved `.pth` is meant to be loaded by `inference.py` or elsewhere.
5. Expected training time; the Windows lock is CPU-only. Target hardware is
   partly answered: the removed brief asked for training on a GPU.
6. Whether `data/` and `models/` should be gitignored again. They were, together
   with `*.pth`, from `07844a3` until `ead7824` (2026-11-08) replaced
   `.gitignore`; `docs/install.md:8` still describes that earlier state.
7. Whether a Gitea runner serves this repository, and whether the GitHub copy
   should run any CI.
8. Whether the pdoc pages in `docs/` are meant to be regenerated and to cover
   `inference.py`.
9. What role `workflow_prompts/` and `scratch/` play; neither is referenced by the
   scripts, the tests or the README.
10. Whether a fixed random seed for the split is intended; the removed brief's
    example passes none either.
