# CholMA

## Installation

```bash
conda create -n cholma python=3.10 -y
conda activate cholma

pip install torch torchvision
pip install -r requirements.txt
```

Additional dependencies:

```bash
pip install timm pandas scikit-learn albumentations
```

## Dataset

Download CholecT50 from the official source:

🔗 https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md

After unpacking, set its absolute path in `config.yaml` under `parent_path`. The expected layout is:

```
CholecT50/
├── data/   # 50 video frame folders (VID01, VID02, ..., VID111)
│   ├── VID01/
│   ├── VID02/
│   ├── ...
│   └── VID111/
├── dataframes/
└── labels/
```

## Annotations

CholMA annotation CSVs live under `./annotation/` (e.g. `CholMA_CA.csv`, `CholMA_MA.csv`, `CholMA_Soft.csv`).

## Training

A single entry point (`main.py`) selects the training / validation strategy via the `train_strategy` field of `config.yaml`. Override it from the CLI when needed.

| Strategy | Train label        | Val label          |
|----------|--------------------|--------------------|
| `CA`     | Complete-Agreement | Complete-Agreement |
| `MA`     | Majority-Agreement | Majority-Agreement |
| `Soft`   | SoftLabel          | SoftLabel          |

```bash
# Local
python main.py                                # use config.yaml defaults

# SLURM
sbatch run_train.sh
```

## Evaluation

```bash
python evaluate.py
```

Reported metrics:

| Metric    | Description |
|-----------|-------------|
| **CoAP**  | Consensus-aware mAP — average of mAP values computed after binarizing soft GT at τ ∈ {0.33, 0.66, 1.00}. |
| **HardAP** | Conventional mAP at a single binarization threshold (default `0.5`, i.e. majority vote). |
| **MAE**   | Mean Absolute Error on positive entries (y > 1e-4). |
