# Object Detection on Aerial Imagery for Coconut Trees Detection

Coconut tree detection from drone imagery using YOLOv8, Yolov12, and RT_DERT models with OpenStreetMap labels and OpenAerialMap imagery.

## Overview

- OpenStreetMap point data: bounding boxes with buffer zones
- Tile large aerial imagery (256×256 at 9cm/pixel): [Source](https://map.openaerialmap.org/#/-175.34221936224426,-21.095929709180027,15/square/20002233030/5a28640ebac48e5b1c58a81d?_k=4yyxj6) 
- Convert geographic coordinates to YOLO format
- Train multiple models of YOLOv8 (nano, small, medium) on coconut trees from Kolovai, Tonga
- Train Yolov12 and also RT-DERT.

**Source**: World Bank - Automated Feature Detection of Aerial Imagery from the South Pacific

## Data

**Statistics**:
- **Original**: 10,631 trees (Coconut: 10,092 | Mango: 261 | Banana: 181 | Papaya: 97)
- **Target**: Coconut trees only
Clipped labels to tiles: {'total_trees': 10092, 'total_tiles': 551, 'processed_input_tiles': 448, 'skipped_nonlabeled_tiles': 103, 'total_input_trees': 11726}
Converted to YOLO format
Split:  train=313, val=89, test=46

70 / 20 / 10 for train/val/test

- **Tiles**: 256×256px at zoom 19, EPSG:4326


## Structure

```
data/
├── raw/                # OAM imagery + OSM points
├── chips/              # 256×256 tiles (.tif)
├── labels/             # Per-tile annotations (.geojson)
└── yolo/
    ├── train/          # Training data (.png + .txt)
    ├── val/            # Validation data
    └── config.yaml     # YOLO config

notebooks/
├── experiment.ipynb         # including the dl4cv-oda package and all functions
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/kshitijrajsharma/dl4cv-oda
cd dl4cv-oda
uv sync
```

## Development ( version bump)

```bash
uv sync --extra dev
cz bump
git push --tags
```

## Workflow
<img width="2833" height="1411" alt="image" src="https://github.com/user-attachments/assets/523f03b8-ff87-4c02-8c14-12c70e20e69f" />


**1. Clean OSM Data** : Filter coconut trees, generate buffered bounding boxes

**2. Tile Imagery** : Create 256×256 tiles, clip labels to tile extents

**3. YOLO Conversion** : Transform coordinates (EPSG:4326 : pixels : normalized [0,1])

```python
row, col = src.index(lon, lat)  # rasterio
x_norm = col / img_width
y_norm = row / img_height
```

**4. Train** : YOLOv8n, 100 epochs, batch 16

```python snapshot
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data/yolo/config.yaml', epochs=100, imgsz=256, batch=16)
```

## References

- [OpenAerialMap](https://openaerialmap.org/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [uv Package Manager](https://github.com/astral-sh/uv)
