# Install

1. Install [uv](https://docs.astral.sh/uv/) and Python 3.13. The pins are locked for 3.13 only.
2. Clone this repository and run `uv sync --frozen` at its root.

On Windows the locked `torch` is the CPU build from PyPI, so the scripts run on the CPU there. They pick CUDA by themselves where a CUDA build of torch finds a GPU.

The first run of `cifar_cvt_train.py` downloads CIFAR-10, about 170 MB, into `data/`, and the first run of either script downloads the CvT-13 weights into the Hugging Face cache. Both need a network connection and disk space. `data/` and `models/` are gitignored. `.\recipe.ps1 test` needs none of that: `tests/conftest.py` sets `HF_HUB_OFFLINE`, the tests replace the model, the processor and CIFAR-10, and a test that reached for the weights would fail instead of fetching them.

`.\recipe.ps1 build` packs the two scripts into a wheel in `dist\`.
