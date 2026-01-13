# Object Detection on Aerial Imagery for Coconut Trees Detection

Coconut tree detection from drone imagery using YOLOv8, YOLOv12, and RT-DETR models with OpenStreetMap labels and OpenAerialMap imagery.

## Overview

- OpenStreetMap point data: bounding boxes with buffer zones
- Tile large aerial imagery (256×256 at 9cm/pixel): [Source](https://map.openaerialmap.org/#/-175.34221936224426,-21.095929709180027,15/square/20002233030/5a28640ebac48e5b1c58a81d?_k=4yyxj6) 
- Convert geographic coordinates to YOLO format
- Train multiple models (YOLOv8, YOLOv12, RT-DETR) on coconut trees from Kolovai, Tonga
- Hyperparameter optimization using Optuna

**Source**: World Bank - Automated Feature Detection of Aerial Imagery from the South Pacific

## Data

**Original Statistics**:
- Total: 10,631 trees (Coconut: 10,092 | Mango: 261 | Banana: 181 | Papaya: 97)
- Target: Coconut trees only

**Processed Dataset**:
- Total trees: 10,092
- Total tiles: 551
- Processed tiles: 448
- Skipped non-labeled tiles: 103
- Total input trees: 11,726

**Split Configuration** (70/20/10):

| Split | Tiles | Labels |
|-------|-------|--------|
| Train | 313   | 8,770  |
| Val   | 89    | 2,008  |
| Test  | 46    | 948    |

<img width="801" height="642" alt="image" src="https://github.com/user-attachments/assets/cb51547a-64a9-4f4d-a860-96b38b97725b" />


**Splitting Strategy**: Spatial splitting based on tiles along the Y-axis to prevent data leakage and ensure geographic separation between train/val/test sets.

**Image Specifications**:
- Resolution: 256×256px
- Zoom level: 19
- Coordinate system: EPSG:4326
- Ground sampling distance: 9cm/pixel

## Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU |
| GPU Memory | 15.57 GB |
| CPU | Intel Core i9-14900HX (32 cores) |
| RAM | 64 GB |
| Swap | 72 GB |


## Structure

```text
data/
├── raw/                # OAM imagery + OSM points
├── chips/              # 256×256 tiles (.tif)
├── labels/             # Per-tile annotations (.geojson)
└── yolo/
    ├── train/          # Training data (.png + .txt)
    ├── val/            # Validation data
    └── config.yaml     # YOLO config

notebooks/
├── pipeline.ipynb      # Full pipeline with dl4cv-oda package
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/kshitijrajsharma/dl4cv-oda
cd dl4cv-oda
uv sync
```

## Development (version bump)

```bash
uv sync --extra dev
cz bump
git push --tags
```

## Workflow

<img width="2833" height="1411" alt="image" src="https://github.com/user-attachments/assets/523f03b8-ff87-4c02-8c14-12c70e20e69f" />

**1. Clean OSM Data**: Filter coconut trees, generate buffered bounding boxes

**2. Tile Imagery**: Create 256×256 tiles, clip labels to tile extents

**3. YOLO Conversion**: Transform coordinates (EPSG:4326 → pixels → normalized [0,1])

```python
row, col = src.index(lon, lat)
x_norm = col / img_width
y_norm = row / img_height
```

**4. Hyperparameter Optimization**: Optuna-based tuning

**5. Train**: Models with optimized hyperparameters

## Training Configuration

Two training approaches were implemented:

1. **Baseline**: Default hyperparameters from YOLO and RT-DETR
2. **Optimized**: Optuna-based hyperparameter tuning followed by training

### Base Training Parameters

| Parameter | Value |
|-----------|-------|
| Image Size | 256×256 |
| Epochs | 200 |
| Early Stopping Patience | 30 |
| Batch Size | 16 |
| Optimizer | auto |

### Hyperparameter Tuning Configuration

Optuna was used to optimize:

- **Learning Rate**: Step size for gradient updates
- **Weight Decay**: Regularization to prevent overfitting
- **Batch Size**: Balance between gradient stability and GPU memory

| Parameter | Range | Iterations | Tune Epochs | Patience |
|-----------|-------|------------|-------------|----------|
| Learning Rate | [1e-5, 1e-2] | 6 | 60 | 10 |
| Weight Decay | [1e-6, 1e-3] | 6 | 60 | 10 |
| Batch Size | [8, 16, 32] | 6 | 60 | 10 |

## Model Comparison

### Architecture Overview

| Model | Type | Layers | Parameters | GFLOPs | Size (MB) |
|-------|------|--------|------------|--------|-----------|
| YOLOv8l | CNN-based | 209 | 43.6M | 165.4 | 87.6 |
| YOLOv12l | CNN + Attention | 488 | 26.4M | 89.4 | 53.5 |
| RT-DETR-l | Transformer | 465 | 32.8M | 108.0 | 66.2 |

RT-DETR-l Architecture : 
<img width="1073" height="374" alt="image" src="https://github.com/user-attachments/assets/595f699e-3039-45b5-9b00-7a34e1ec8618" />


**Model Selection**: Large (L) variants were selected for all models to enable fair comparison, as RT-DETR is only available in large and extra-large configurations.

### Performance Results

| Model | Val F1 | Test F1 | Test mAP50 | Epochs | Training Time |
|-------|--------|---------|------------|--------|---------------|
| **RT-DETR-l** | 0.7478 | **0.7876** | 0.7369 | 95* | 6.96 min |
| **YOLOv12l** | 0.7348 | 0.7668 | 0.7085 | 113* | 3.84 min |
| **YOLOv8l** | 0.7164 | 0.7630 | **0.7283** | 48* | 1.38 min |

*Stopped at epoch N due to early stopping (patience=30)

### Key Findings

- **RT-DETR-l** achieved the highest test F1 score (0.7876) with transformer-based architecture
- **YOLOv12l** offers the best model efficiency (53.5 MB) with attention-aware mechanisms
- **YOLOv8l** provides fastest training time (0.023 hrs) with traditional CNN approach
- All models demonstrate strong generalization (F1 > 0.76 on test set)

## Training Example

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='data/yolo/config.yaml',
    epochs=200,
    imgsz=256,
    batch=16,
    patience=30
)
```

## References

- [OpenAerialMap](https://openaerialmap.org/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [Optuna](https://optuna.org/)

