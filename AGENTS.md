# AGENTS.md

## Quick Start

```bash
cd /home/doki/main_lab/KAMemory
pip install -r requirements.txt  # torch, numpy, matplotlib, tqdm, wandb
python src/main.py              # runs default BTSP model
python src/main.py --load --idx 2  # load saved session
```

## Project Structure

- `src/main.py` - main entrypoint for BTSP model
- `src/models.py` - Autoencoder, MTL (Multiple Timescale Learning) models
- `src/training.py` - training logic
- `src/utils.py` - data generation (sparse_stimulus_generator, stimulus_generator)
- `src/visualization.py` - plotting utilities
- `src/configs/` - JSON config files (base_configs.json, lap_configs.json)
- `notebooks/lab_1.py` - Jupyter-style exploration notebook

## Key Commands

```bash
# Train autoencoder + MTL model
python src/main.py --num 1

# Load a saved session
python src/main.py --load --idx <session_id>

# Run parameter search with wandb
python src/optim_wb/param_search.py
```

## Architecture

- Architecture: EI (entorhinal cortex) → CA3 → CA1 → EC (output)
- Uses sparsemoid activation (softmax top-K with beta temperature)
- MTL model with learnable CA3-CA1 weights (Hebbian-like learning)
- Autoencoder trained first, then weights transferred to MTL

## Important Notes

- Imports use custom path hack: `sys.path.append(os.path.abspath(__file__).split("src")[0] + "src")`
- Notebooks use: `sys.path.append(os.path.expanduser('~/Research/lab/KAMemory/src'))`
- No test framework - verify by running main.py
- wandb for experiment tracking (in optim_wb/)