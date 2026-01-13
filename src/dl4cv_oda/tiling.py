import os
import re
from pathlib import Path

import geopandas as gpd
from geomltoolkits.downloader import tms


async def download_tiles(bbox, zoom, tms_url, output_dir, prefix="OAM"):
    os.makedirs(output_dir, exist_ok=True)
    await tms.download_tiles(
        tms=tms_url,
        zoom=zoom,
        out=output_dir,
        bbox=bbox,
        georeference=True,
        dump_tile_geometries_as_geojson=True,
        prefix=prefix,
    )


def parse_tile_id(tile_id_str):
    match = re.match(r"Tile\(x=(\d+), y=(\d+), z=(\d+)\)", tile_id_str)
    if not match:
        raise ValueError(f"Cannot parse tile ID: {tile_id_str}")
    return match.groups()


def clip_labels_to_tiles(trees_path, tiles_path, output_dir, prefix="OAM"):
    trees = gpd.read_file(trees_path)
    tiles = gpd.read_file(tiles_path)

    trees.to_crs(epsg=4326, inplace=True)
    tiles.to_crs(epsg=4326, inplace=True)

    os.makedirs(output_dir, exist_ok=True)

    stats = {
        "total_trees": len(trees),
        "total_tiles": len(tiles),
        "processed_input_tiles": 0,
        "skipped_nonlabeled_tiles": 0,
        "total_input_trees": 0,
    }

    for _, tile in tiles.iterrows():
        x, y, z = parse_tile_id(tile["id"])
        filename = f"{prefix}-{x}-{y}-{z}.geojson"

        intersecting = trees[trees.intersects(tile.geometry)].copy()
        if intersecting.empty:
            stats["skipped_nonlabeled_tiles"] += 1
            continue

        clipped = gpd.clip(intersecting, tile.geometry)
        clipped.to_file(Path(output_dir) / filename, driver="GeoJSON")

        stats["processed_input_tiles"] += 1
        stats["total_input_trees"] += len(clipped)

    return stats
