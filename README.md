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
├── tiles.geojson       # Tile grid metadata
├── trees_box.geojson   # Buffered bounding boxes
└── yolo/
    ├── train/          # Training data (.png + .txt)
    ├── val/            # Validation data (.png + .txt)
    ├── test/           # Test data (.png + .txt)
    └── config.yaml     # YOLO config

notebooks/
├── pipeline.ipynb                  # Full pipeline with dl4cv-oda package
├── results/                        # Experiment results
│   └── full_pipeline_20260113_225000/  # Latest experiment
src/
└── dl4cv_oda/          # Package source code
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
uv sync --all-extras
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

**4. Spatial Split**: Split tiles into train/val/test (70/20/10) along Y-axis to ensure geographic separation and prevent data leakage

**5. Hyperparameter Optimization**: Optuna-based tuning (learning rate, weight decay, batch size)

**6. Train**: Models with both default and optimized hyperparameters

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

| Model | Type | Layers | Parameters | Size (MB) |
|-------|------|--------|------------|-----------|
| YOLOv8l | CNN-based | 209 | 43.7M | 83.5 |
| YOLOv12l | CNN + Attention | 488 | 26.5M | 51.0 |
| RT-DETR-l | Transformer | 465 | 33.0M | 63.1 |

RT-DETR-l Architecture:

<img width="1073" height="374" alt="image" src="https://github.com/user-attachments/assets/595f699e-3039-45b5-9b00-7a34e1ec8618" />

**Model Selection**: Large (L) variants were selected for all models to enable fair comparison, as RT-DETR is only available in large and extra-large configurations.

### Performance Results

#### Base Models (Default Hyperparameters)

| Model | Val F1 | Test F1 | Test mAP50 | Test Precision | Test Recall | Epochs | Training Time |
|-------|--------|---------|------------|----------------|-------------|--------|---------------|
| **RT-DETR-l** | **0.7478** | **0.7876** | **0.7369** | 0.7760 | **0.7996** | 95* | 7.10 min |
| **YOLOv12l** | 0.7348 | 0.7668 | 0.7085 | 0.7708 | 0.7628 | 113* | 3.99 min |
| **YOLOv8l** | 0.7164 | 0.7630 | 0.7283 | **0.7791** | 0.7476 | 48* | **1.50 min** |

*All models stopped early due to patience=30 (no improvement for 30 consecutive epochs)

**Training Configuration**: Max epochs=200, Early stopping patience=30, Batch size=16, Image size=256×256

#### Optuna-Tuned Models

| Model | Val F1 | Test F1 | Test mAP50 | Test Precision | Test Recall |
|-------|--------|---------|------------|----------------|-------------|
| **RT-DETR-l** | **0.7468** | **0.7747** | **0.7355** | 0.7729 | **0.7764** |
| **YOLOv12l** | 0.7193 | 0.7614 | 0.7006 | 0.7677 | 0.7553 |
| **YOLOv8l** | 0.6888 | 0.7417 | 0.6801 | **0.7800** | 0.7069 |


### Inference Performance

| Model | Val Inference | Test Inference |
|-------|---------------|----------------|
| YOLOv8l | 2.84 sec | **2.17 sec** |
| YOLOv12l | 3.08 sec | 2.39 sec |
| RT-DETR-l | 3.69 sec | 2.64 sec |

### Key Findings

#### Model Performance Summary

**Best Overall Performance**: RT-DETR-l (Base)
- Highest Test F1: **0.7876** (best balance of precision and recall)
- Highest Test mAP50: **0.7369** (best detection accuracy)
- Best Recall: **0.7996** (superior at finding coconut trees, minimal false negatives)
- Trade-off: Slowest training (7.10 min) and inference times

**Best Efficiency**: YOLOv12l
- Smallest model size: **51.0 MB** (40% smaller than YOLOv8l)
- Competitive performance: Test F1 of 0.7668 (base) and 0.7614 (optuna)
- Moderate training time: 3.99 min (base), 2.39 min (optuna)
- Best choice for resource-constrained deployments

**Fastest Training & Inference**: YOLOv8l
- Training time: **1.50 min** (5× faster than RT-DETR-l)
- Fastest inference: 2.17 sec on test set
- Solid performance: Test F1 of 0.7630 (base)
- Best for rapid prototyping and real-time applications

### Experiment Results

Detailed results including training curves, validation predictions, and hyperparameters are available [here](notebooks/results/full_pipeline_20260113_225000/):
```
notebooks/results/full_pipeline_20260113_225000/
```

Each model folder contains:
- `results.csv`: Per-epoch metrics
- `results.png`: Training curves
- `val_batch1_labels.jpg` & `val_batch1_pred.jpg`: Validation predictions
- `args.yaml`: Training configuration
- Best hyperparameters from Optuna tuning

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

