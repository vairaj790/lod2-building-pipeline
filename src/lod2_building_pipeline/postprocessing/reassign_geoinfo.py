# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from shapely.geometry import LineString


def load_heat_result(npy_path: Path) -> Dict:
    data = np.load(npy_path, allow_pickle=True)

    if isinstance(data, np.ndarray) and data.shape == ():
        data = data.item()

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict-like HEAT result in {npy_path}, got {type(data)}")

    return data


def resolve_heat_output_dirs(heat_output_dir: Path, result_name: str) -> Tuple[Path, Path]:
    result_name = str(result_name).strip()

    if result_name:
        npy_dir = heat_output_dir / f"npy_{result_name}"
        overlay_dir = heat_output_dir / f"viz_{result_name}"
    else:
        npy_dir = heat_output_dir / "npy"
        overlay_dir = heat_output_dir / "viz"

    if not npy_dir.is_dir():
        raise FileNotFoundError(f"NPY output directory not found: {npy_dir}")

    if not overlay_dir.is_dir():
        raise FileNotFoundError(f"Overlay output directory not found: {overlay_dir}")

    return npy_dir, overlay_dir


def run_postprocessing(
    heat_output_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    patch_size: int,
    result_name: str,
) -> None:
    if patch_size not in (256, 512):
        raise ValueError("patch_size must be 256 or 512")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    npy_dir, overlay_dir = resolve_heat_output_dirs(heat_output_dir, result_name)

    geojson_output_dir = output_dir / "geojson_files"
    geotiff_output_dir = output_dir / "geotiff_files"

    geojson_output_dir.mkdir(parents=True, exist_ok=True)
    geotiff_output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    matching_keys = [key for key in metadata.keys() if key.endswith(f"_{patch_size}")]

    print("=" * 80)
    print("Running HEAT georeferencing postprocessing")
    print("=" * 80)
    print("HEAT output dir :", heat_output_dir)
    print("Metadata        :", metadata_path)
    print("Output dir      :", output_dir)
    print("Patch size      :", patch_size)
    print("Result name     :", result_name)
    print("Metadata entries:", len(metadata))
    print("Matching entries:", len(matching_keys))
    print("=" * 80)

    if not matching_keys:
        raise RuntimeError(f"No metadata keys found ending with _{patch_size}")

    processed_count = 0
    missing_npy_count = 0
    missing_overlay_count = 0
    empty_content_count = 0
    load_error_count = 0
    processing_error_count = 0

    for metadata_key in matching_keys:
        meta = metadata[metadata_key]
        base_name = metadata_key.rsplit("_", 1)[0]

        npy_path = npy_dir / f"{base_name}_results.npy"
        overlay_path = overlay_dir / f"{base_name}_pred_edge.png"

        if not npy_path.exists():
            missing_npy_count += 1
            continue

        if not overlay_path.exists():
            missing_overlay_count += 1
            continue

        try:
            data = load_heat_result(npy_path)
        except Exception as ex:
            load_error_count += 1
            print(f"SKIP load error: {base_name} -> {type(ex).__name__}: {ex}")
            continue

        corners = data.get("corners", [])
        edges = data.get("edges", [])

        if len(corners) == 0 or len(edges) == 0:
            empty_content_count += 1
            continue

        try:
            transform = Affine(*meta["transform"])
            scale_x, scale_y = meta["scale_factors"]
            original_width, original_height = meta["original_size"]
            crs = meta.get("crs", "EPSG:25832")

            rescaled_transform = transform * Affine.scale(scale_x, scale_y)
            corners_orig = [rescaled_transform * (float(x), float(y)) for x, y in corners]

            geometries = []
            properties = []

            for edge_id, (source_idx, target_idx) in enumerate(edges):
                source_idx = int(source_idx)
                target_idx = int(target_idx)

                if source_idx >= len(corners_orig) or target_idx >= len(corners_orig):
                    continue

                line = LineString([corners_orig[source_idx], corners_orig[target_idx]])
                geometries.append(line)
                properties.append(
                    {
                        "edge_id": int(edge_id),
                        "source": source_idx,
                        "target": target_idx,
                        "image_id": base_name,
                    }
                )

            if not geometries:
                empty_content_count += 1
                continue

            gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
            geojson_out = geojson_output_dir / f"{base_name}.geojson"
            gdf.to_file(geojson_out, driver="GeoJSON")

            overlay_img = Image.open(overlay_path).convert("RGB")
            resized_img = overlay_img.resize(
                (int(original_width), int(original_height)),
                resample=Image.BILINEAR,
            )
            overlay_arr = np.array(resized_img)
            overlay_arr = np.moveaxis(overlay_arr, -1, 0)

            tif_out = geotiff_output_dir / f"{base_name}.tif"
            with rasterio.open(
                tif_out,
                "w",
                driver="GTiff",
                height=int(original_height),
                width=int(original_width),
                count=3,
                dtype=overlay_arr.dtype,
                crs=crs,
                transform=transform,
            ) as dst:
                dst.write(overlay_arr)

            processed_count += 1

            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")

        except Exception as ex:
            processing_error_count += 1
            print(f"SKIP processing error: {base_name} -> {type(ex).__name__}: {ex}")
            continue

    print("\n" + "=" * 80)
    print("POSTPROCESSING SUMMARY")
    print("=" * 80)
    print(f"Processed successfully : {processed_count}")
    print(f"Missing NPY files      : {missing_npy_count}")
    print(f"Missing overlay files  : {missing_overlay_count}")
    print(f"Empty corners/edges    : {empty_content_count}")
    print(f"NPY load errors        : {load_error_count}")
    print(f"Processing errors      : {processing_error_count}")
    print("=" * 80)
    print(f"GeoJSON folder         : {geojson_output_dir}")
    print(f"GeoTIFF folder         : {geotiff_output_dir}")
    print("=" * 80)
