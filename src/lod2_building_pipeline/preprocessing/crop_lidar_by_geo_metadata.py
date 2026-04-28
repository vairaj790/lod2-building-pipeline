# -*- coding: utf-8 -*-

"""
Crop one main LiDAR file into per-building LAZ files using geo_metadata.json.

This module is config-driven through scripts/run_lidar_crop.py.

Main idea:
- use crop extents from work/heat_input/geo_metadata.json
- crop MAIN_LIDAR_PATH into per-building LAZ files
- save outputs to work/Lidar_input/
- preserve building names so later 3D fusion can match:

    georeferenced_output/geojson_files/building_x.geojson
    Lidar_input/building_x.laz
"""

import json
from pathlib import Path

import laspy
import numpy as np
from pyproj import CRS, Transformer
from rasterio.transform import Affine, array_bounds


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_unique_buildings(metadata_path: Path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    unique = {}
    conflicts = []

    for _, entry in metadata.items():
        name = entry["name"]
        sig = (
            tuple(entry["transform"]),
            tuple(entry["original_size"]),
            entry["crs"],
        )

        if name not in unique:
            unique[name] = entry
        else:
            old = unique[name]
            old_sig = (
                tuple(old["transform"]),
                tuple(old["original_size"]),
                old["crs"],
            )
            if sig != old_sig:
                conflicts.append(name)

    if not unique:
        raise RuntimeError("No valid building entries found in geo_metadata.json")

    crs_values = sorted({entry["crs"] for entry in unique.values()})
    if len(crs_values) != 1:
        raise RuntimeError(f"Expected one metadata CRS, found: {crs_values}")

    return unique, crs_values[0], conflicts


def metadata_entry_bbox(entry, extra_buffer_m=0.0):
    a, b, c, d, e, f = entry["transform"]
    transform = Affine(a, b, c, d, e, f)
    original_width, original_height = entry["original_size"]

    west, south, east, north = array_bounds(
        original_height,
        original_width,
        transform,
    )

    minx = min(west, east) - float(extra_buffer_m)
    maxx = max(west, east) + float(extra_buffer_m)
    miny = min(south, north) - float(extra_buffer_m)
    maxy = max(south, north) + float(extra_buffer_m)

    return minx, miny, maxx, maxy


def read_lidar_crs(lidar_path: Path):
    with laspy.open(str(lidar_path)) as reader:
        parsed = reader.header.parse_crs()
        if parsed is None:
            raise RuntimeError(
                "The main LiDAR file has no CRS in its header. "
                "Please assign the correct CRS first, otherwise the crop extents "
                "cannot be matched safely."
            )
        return parsed


def build_building_index(unique_entries, extra_buffer_m=0.0):
    names = []
    osm_ids = []
    minx = []
    miny = []
    maxx = []
    maxy = []

    for name, entry in sorted(unique_entries.items()):
        bx0, by0, bx1, by1 = metadata_entry_bbox(
            entry,
            extra_buffer_m=extra_buffer_m,
        )
        names.append(name)
        osm_ids.append(entry.get("osm_id"))
        minx.append(bx0)
        miny.append(by0)
        maxx.append(bx1)
        maxy.append(by1)

    return {
        "names": np.array(names, dtype=object),
        "osm_ids": np.array(osm_ids, dtype=object),
        "minx": np.array(minx, dtype=np.float64),
        "miny": np.array(miny, dtype=np.float64),
        "maxx": np.array(maxx, dtype=np.float64),
        "maxy": np.array(maxy, dtype=np.float64),
    }


def chunk_overlapping_buildings(
    xmin_arr,
    ymin_arr,
    xmax_arr,
    ymax_arr,
    chunk_minx,
    chunk_miny,
    chunk_maxx,
    chunk_maxy,
):
    return np.where(
        (xmax_arr >= chunk_minx)
        & (xmin_arr <= chunk_maxx)
        & (ymax_arr >= chunk_miny)
        & (ymin_arr <= chunk_maxy)
    )[0]


def append_points(store, name, x, y, z, classification=None):
    if name not in store:
        store[name] = {
            "x": [],
            "y": [],
            "z": [],
            "classification": [] if classification is not None else None,
        }

    store[name]["x"].append(np.asarray(x, dtype=np.float64))
    store[name]["y"].append(np.asarray(y, dtype=np.float64))
    store[name]["z"].append(np.asarray(z, dtype=np.float64))

    if classification is not None:
        if store[name]["classification"] is None:
            store[name]["classification"] = []
        store[name]["classification"].append(np.asarray(classification))


def write_laz(output_path, x, y, z, classification, reference_header, output_crs):
    header = laspy.LasHeader(
        point_format=reference_header.point_format,
        version=reference_header.version,
    )
    header.scales = reference_header.scales

    header.offsets = np.array(
        [
            float(np.floor(np.min(x))),
            float(np.floor(np.min(y))),
            float(np.floor(np.min(z))),
        ]
    )

    try:
        header.add_crs(output_crs)
    except Exception:
        pass

    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z

    if classification is not None:
        try:
            las.classification = classification
        except Exception:
            pass

    las.write(str(output_path))


def crop_lidar_by_geo_metadata(
    main_lidar_path: Path,
    geo_metadata_json: Path,
    output_dir: Path,
    extra_buffer_m: float = 0.0,
    points_per_chunk: int = 2_000_000,
    save_merged_debug_file: bool = True,
    merged_debug_name: str = "merged_cropped_buildings_from_metadata.laz",
):
    main_lidar_path = Path(main_lidar_path)
    geo_metadata_json = Path(geo_metadata_json)
    output_dir = Path(output_dir)

    if not main_lidar_path.is_file():
        raise FileNotFoundError(f"Main LiDAR file not found: {main_lidar_path}")

    if not geo_metadata_json.is_file():
        raise FileNotFoundError(f"Geo metadata JSON not found: {geo_metadata_json}")

    ensure_dir(output_dir)

    unique_entries, metadata_crs_text, conflicts = load_unique_buildings(geo_metadata_json)
    metadata_crs = CRS.from_user_input(metadata_crs_text)
    lidar_crs = read_lidar_crs(main_lidar_path)

    print("=" * 80)
    print("Cropping main LiDAR using geo_metadata.json")
    print("=" * 80)
    print("Main LiDAR       :", main_lidar_path)
    print("Geo metadata     :", geo_metadata_json)
    print("Output directory :", output_dir)
    print("Metadata CRS     :", metadata_crs.to_string())
    print("LiDAR CRS        :", lidar_crs.to_string())
    print("Unique buildings :", len(unique_entries))
    print("Extra buffer (m) :", extra_buffer_m)
    print("Points per chunk :", points_per_chunk)
    print("=" * 80)

    needs_transform = metadata_crs != lidar_crs
    if needs_transform:
        print(
            "LiDAR CRS differs from metadata CRS. "
            "XY coordinates will be transformed to metadata CRS before cropping."
        )
        transformer = Transformer.from_crs(lidar_crs, metadata_crs, always_xy=True)
    else:
        print("LiDAR CRS already matches metadata CRS.")
        transformer = None

    building_index = build_building_index(
        unique_entries,
        extra_buffer_m=extra_buffer_m,
    )

    all_minx = float(np.min(building_index["minx"]))
    all_miny = float(np.min(building_index["miny"]))
    all_maxx = float(np.max(building_index["maxx"]))
    all_maxy = float(np.max(building_index["maxy"]))

    point_store = {}
    manifest = []

    chunk_counter = 0
    total_input_points = 0
    total_selected_points = 0

    with laspy.open(str(main_lidar_path)) as reader:
        source_header = reader.header

        for points_chunk in reader.chunk_iterator(int(points_per_chunk)):
            chunk_counter += 1

            x_raw = np.asarray(points_chunk.x)
            y_raw = np.asarray(points_chunk.y)
            z_raw = np.asarray(points_chunk.z)

            total_input_points += len(x_raw)

            if transformer is not None:
                x, y = transformer.transform(x_raw, y_raw)
            else:
                x, y = x_raw, y_raw

            chunk_minx = float(np.min(x))
            chunk_miny = float(np.min(y))
            chunk_maxx = float(np.max(x))
            chunk_maxy = float(np.max(y))

            if (
                chunk_maxx < all_minx
                or chunk_minx > all_maxx
                or chunk_maxy < all_miny
                or chunk_miny > all_maxy
            ):
                print(f"Chunk {chunk_counter}: no overlap with global metadata extent -> skipped")
                continue

            candidate_ids = chunk_overlapping_buildings(
                building_index["minx"],
                building_index["miny"],
                building_index["maxx"],
                building_index["maxy"],
                chunk_minx,
                chunk_miny,
                chunk_maxx,
                chunk_maxy,
            )

            if len(candidate_ids) == 0:
                print(f"Chunk {chunk_counter}: no candidate buildings -> skipped")
                continue

            try:
                classification = np.asarray(points_chunk.classification)
            except Exception:
                classification = None

            print(
                f"Chunk {chunk_counter}: {len(x):,} points | "
                f"candidate buildings: {len(candidate_ids)}"
            )

            for idx in candidate_ids:
                name = str(building_index["names"][idx])
                bx0 = float(building_index["minx"][idx])
                by0 = float(building_index["miny"][idx])
                bx1 = float(building_index["maxx"][idx])
                by1 = float(building_index["maxy"][idx])

                mask = (x >= bx0) & (x <= bx1) & (y >= by0) & (y <= by1)

                if not np.any(mask):
                    continue

                append_points(
                    point_store,
                    name,
                    x[mask],
                    y[mask],
                    z_raw[mask],
                    classification=(classification[mask] if classification is not None else None),
                )

                total_selected_points += int(np.count_nonzero(mask))

    saved_count = 0
    merged_x = []
    merged_y = []
    merged_z = []
    merged_cls = []
    any_cls_available = False

    for name in sorted(point_store.keys()):
        x = np.concatenate(point_store[name]["x"])
        y = np.concatenate(point_store[name]["y"])
        z = np.concatenate(point_store[name]["z"])

        classification = None
        if (
            point_store[name]["classification"] is not None
            and len(point_store[name]["classification"]) > 0
        ):
            classification = np.concatenate(point_store[name]["classification"])
            any_cls_available = True

        if len(x) == 0:
            continue

        output_path = output_dir / f"{name}.laz"
        write_laz(output_path, x, y, z, classification, source_header, metadata_crs)

        saved_count += 1

        manifest.append(
            {
                "name": name,
                "point_count": int(len(x)),
                "output_laz": str(output_path),
            }
        )

        if save_merged_debug_file:
            merged_x.append(x)
            merged_y.append(y)
            merged_z.append(z)
            if classification is not None:
                merged_cls.append(classification)


    if save_merged_debug_file and len(merged_x) > 0:
        merged_x_all = np.concatenate(merged_x)
        merged_y_all = np.concatenate(merged_y)
        merged_z_all = np.concatenate(merged_z)

        if any_cls_available and len(merged_cls) > 0:
            merged_cls_all = np.concatenate(merged_cls)
        else:
            merged_cls_all = None

        merged_out = output_dir / merged_debug_name
        write_laz(
            merged_out,
            merged_x_all,
            merged_y_all,
            merged_z_all,
            merged_cls_all,
            source_header,
            metadata_crs,
        )
        print(f"Saved merged debug LAZ: {merged_out}")

    manifest_out = output_dir / "crop_manifest_from_geo_metadata.json"
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "main_lidar_path": str(main_lidar_path),
                "geo_metadata_json": str(geo_metadata_json),
                "metadata_crs": metadata_crs.to_string(),
                "lidar_crs": lidar_crs.to_string(),
                "reprojected_xy_to_metadata_crs": bool(needs_transform),
                "extra_buffer_m": float(extra_buffer_m),
                "points_per_chunk": int(points_per_chunk),
                "unique_buildings_requested": int(len(unique_entries)),
                "files_saved": int(saved_count),
                "total_input_points_seen": int(total_input_points),
                "total_selected_points_before_per_building_save": int(total_selected_points),
                "conflicting_duplicate_names": sorted(conflicts),
                "outputs": manifest,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("LiDAR crop finished.")
    print("=" * 80)
    print(f"Files saved : {saved_count}")
    print(f"Manifest    : {manifest_out}")
    print("=" * 80)
