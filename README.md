# Frequency Subspace Enhancement

Python 3.11 + PyTorch 2.3.1 project for exploring model sensitivity to controlled frequency-domain perturbations on CIFAR-10. The pipeline builds a PCA-based tail subspace of DCT mid-frequency coefficients, constructs a common research direction, and applies lightweight, reversible adjustments on the Y channel of YUV images.

## Project layout
- `configs/config.yaml` — experiment parameters (data paths, PCA settings, training hyperparams).
- `src/` — modular code for DCT/IDCT, color conversion, frequency utilities, datasets, and training helpers.
- `main.py` — end-to-end script to build PCA stats, train ResNet-18, and evaluate clean vs enhanced data.
- `requirements.txt` — Python dependencies.

## Quick start
1. Install dependencies: `pip install -r requirements.txt`
2. Run the experiment: `python main.py`

The script will download CIFAR-10, compute frequency statistics (saved to `./artifacts/freq_stats.pkl`), train a CIFAR-adapted ResNet-18 with a configurable fraction of enhanced samples, and report both clean accuracy and representation shifts on fully enhanced test data.
