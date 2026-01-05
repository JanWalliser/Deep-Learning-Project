# Deep Learning Project: Class Imbalance on MNIST (FC vs. CNN)

This repository contains an empirical study of **class imbalance effects** on training dynamics and performance for:
- a **Fully-Connected Network (FC / MLP)** baseline, and
- a **Convolutional Neural Network (CNN)** baseline,

evaluated on MNIST in both:
- **Binary** settings (2 classes), and
- **Multiclass** settings (10 classes).

The code automatically selects the device as **CUDA if available, else CPU**.

---

## Setup

### 1) Create and activate a virtual environment
**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2) Install dependencies
```bash
pip install torch torchvision torchaudio
pip install wandb numpy matplotlib pyyaml tqdm scikit-learn
```

### 3) (Optional) Weights & Biases login
If `logging.use_wandb: true`, you should log in once:
```bash
wandb login
```

To run without W&B, set in `configs/base.yaml`:
```yaml
logging:
  use_wandb: false
```

---

## Running Experiments (Binary vs. 10-class)

Experiments are defined by **merging multiple YAML config files**. The final configuration is created by applying configs
**left → right** exactly as passed to `--config`.

### Config merge rule (override order)
- Configs are merged **left → right**.
- **Later files override earlier files** for overlapping keys.

Recommended order:
1. `configs/base.yaml` (shared defaults)
2. `configs/task_*.yaml` (task mode / number of classes)
3. `configs/model_*.yaml` (model selection + architecture params)
4. `configs/opt_*.yaml` (optimizer selection + hyperparameters)
5. `configs/imbalance_*.yaml` (imbalance setting)

---

## Binary classification runs (2 classes)

### FC + no imbalance
```bash
python -m src.main \
  --config configs/base.yaml \
  --config configs/model_fc.yaml \
  --config configs/imbalance_none.yaml
```

### CNN + severe imbalance
```bash
python -m src.main \
  --config configs/base.yaml \
  --config configs/model_cnn.yaml \
  --config configs/imbalance_severe.yaml
```

---

## 10-class (multiclass) runs (MNIST 0–9)

Enable multiclass mode by adding the task config (it overrides `model.num_classes` to 10).

### FC + moderate imbalance
```bash
python -m src.main \
  --config configs/base.yaml \
  --config configs/task_multiclass.yaml \
  --config configs/model_fc.yaml \
  --config configs/imbalance_moderate.yaml
```

### CNN + balanced / no imbalance
```bash
python -m src.main \
  --config configs/base.yaml \
  --config configs/task_multiclass.yaml \
  --config configs/model_cnn.yaml \
  --config configs/imbalance_balanced.yaml
```

---

## Selecting the model (FC vs. CNN)

Model selection is controlled by `model.name`:
- `model_fc.yaml` sets `model.name: fc` and FC-specific parameters.
- `model_cnn.yaml` sets `model.name: cnn` and CNN-specific parameters.

The code calls `build_model(cfg)` (model factory), which reads `cfg["model"]["name"]` and instantiates the corresponding model.

---

## Selecting the optimizer (e.g., AdamW)

Optimizer selection is controlled by `training.optimizer.*`. For example, adding `opt_adamw.yaml` typically overrides:
- `training.optimizer.name` (e.g., `adamw`)
- `training.lr`
- `training.weight_decay`
- and any optimizer-specific parameters (e.g., betas)

Example (CNN + AdamW + multiclass):
```bash
python -m src.main \
  --config configs/base.yaml \
  --config configs/task_multiclass.yaml \
  --config configs/model_cnn.yaml \
  --config configs/opt_adamw.yaml \
  --config configs/imbalance_none.yaml
```

In this command:
- `task_multiclass.yaml` overrides the number of classes (10),
- `model_cnn.yaml` selects the CNN architecture,
- `opt_adamw.yaml` overrides optimizer hyperparameters from `base.yaml`,
- `imbalance_none.yaml` defines the (no-)imbalance setting.

---

## Notes
- MNIST will be downloaded automatically on first run (via `torchvision`).
- Outputs such as checkpoints/logs (if enabled) are written under `outputs/`.
