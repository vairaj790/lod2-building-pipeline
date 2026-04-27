# -*- coding: utf-8 -*-
"""
combined_sam3_osm_heat_dataset_creator_v4.py

RUN WHERE SAM3 IS INSTALLED.

Purpose
-------
This is the combined version of:
1) hpc_make_sam3_mask_crops.py
2) pc_crop_sat_to_jpg_with_osm_naming.py
3) the older combined heat_data_creater.py

It does NOT write per-building SAM3 mask crops anymore.
It uses SAM3 masks internally to find building/roof regions, uses OSM only for naming,
and writes the final HEAT-style outputs directly.

Final outputs
-------------
OUTPUT_BASE/
├── rgb_jpg_256/
│   ├── building_<osm_id>.jpg
│   └── ...
├── rgb_jpg_512/
│   ├── building_<osm_id>.jpg
│   └── ...
├── rgb_tif_original/
│   ├── building_<osm_id>.tif
│   └── ...
├── geo_metadata.json
├── lists/
│   ├── all_list.txt
│   ├── train_list.txt
│   └── valid_list.txt
├── common/
│   └── sam3_mask_satgrid.tif                 # optional debug/intermediate full mask
└── misc/
    ├── osm_buildings_cache.geojson           # OSM cache, if enabled
    ├── fetched_buildings_sat_crs.geojson     # debug copy in satellite CRS
    └── sam3_regions_satcrs.geojson           # debug regions from SAM3 mask

Important logic
---------------
- SAM3 decides the crop geometry.
- OSM is used only to assign building_<osm_id> names.
- If HPC cannot call Overpass, fetch/save the OSM cache once on a machine with internet,
  then copy misc/osm_buildings_cache.geojson to OUTPUT_BASE/misc/ on the HPC and rerun.
- If OSM is unavailable and fallback is enabled, the script still writes outputs using
  sequential names: building_0, building_1, building_2, ...
"""

from __future__ import annotations

import os
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import rasterio
from rasterio.windows import Window
from rasterio.mask import mask as rio_mask
from rasterio.features import shapes as rio_shapes
from rasterio.crs import CRS as RioCRS

import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pyproj import CRS, Transformer

import torch
import urllib.parse
import urllib.request
import urllib.error


from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ======================================================================================
# CONFIG - EDIT THESE PATHS ONLY
# ======================================================================================

SATELLITE_TIF = r"/path/to/input/satellite.tif"
OUTPUT_BASE = r"./work/preprocessing_output"

# Optional SAM3 checkpoint. Leave empty if your SAM3 installation loads its default model.
SAM3_CHECKPOINT_PATH = r""


# ======================================================================================
# SAM3 SETTINGS
# ======================================================================================

PROMPT = "building roof"
SCORE_THRESH = 0.20
MIN_MASK_AREA_PX = 200

TILE_SIZE_PX = 1024
USE_OVERLAP = False
OVERLAP_PX = 256
SAT_RGB_BANDS = (1, 2, 3)

# Region filtering and crop buffer.
# If SATELLITE_TIF is EPSG:25832, these are meters.
CROP_BUFFER_M = 5.0
MIN_REGION_AREA_M2 = 20.0
MAX_PATCHES: Optional[int] = None


# ======================================================================================
# OSM / OVERPASS SETTINGS
# ======================================================================================

# OSM is used only for assigning building_<osm_id> names.
USE_OSM_CACHE = True
OSM_CACHE_GEOJSON = "osm_buildings_cache.geojson"

# If True, the script will not call Overpass. It will only use the cache file.
# Useful on HPC compute nodes without internet.
OSM_CACHE_ONLY = False

OSM_TILE_SIZE_DEG = 0.005
OSM_MIN_TILE_SIZE_DEG = 0.0025
OVERPASS_TIMEOUT_S = 180
OVERPASS_SLEEP_S = 1.0
MAX_RETRIES_PER_TILE = 6
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

MIN_INTERSECTION_AREA = 0.01

# If OSM cannot be fetched/loaded at all, still create outputs from SAM3 regions.
# In that mode, names become building_0, building_1, building_2, ...
# Metadata will store osm_id=None and naming_mode="sequential_fallback".
FALLBACK_TO_SEQUENTIAL_NAMING_IF_OSM_UNAVAILABLE = True

# If OSM is available but an individual SAM3 region has no OSM match, keep the
# original strict behavior by skipping it. Set True only if you also want unmatched
# individual regions saved with sequential fallback names.
FALLBACK_TO_SEQUENTIAL_NAMING_FOR_UNMATCHED_REGIONS = False


# ======================================================================================
# OUTPUT SETTINGS
# ======================================================================================

JPG_SIZES = [256, 512]
JPG_QUALITY = 95
VALID_LAST_N = 300

WRITE_INTERMEDIATE_MASK_TIF = True
WRITE_REGIONS_GEOJSON = True
WRITE_OSM_DEBUG_GEOJSON = True

# Keep all original satellite bands in rgb_tif_original if available.
# This preserves your previous script-2 behavior.
# Set to "rgb_only" if you want exactly 3-band GeoTIFF crops.
GEOTIFF_OUTPUT_MODE = "all_bands"  # options: "all_bands", "rgb_only"


# ======================================================================================
# BASIC HELPERS
# ======================================================================================


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def normalize_crs_for_pyproj(ds_crs) -> CRS:
    """Normalize raster CRS for pyproj; fallback to EPSG:25832 for local/engineering CRS."""
    c = CRS.from_user_input(ds_crs)
    wkt = c.to_wkt() if c is not None else ""
    if (c is None) or ("LOCAL_CS" in wkt) or c.is_engineering:
        return CRS.from_epsg(25832).to_2d()
    return c.to_2d()


def windows_for_size(width: int, height: int, tile_size: int, overlap: int = 0) -> List[Window]:
    stride = tile_size if overlap <= 0 else (tile_size - overlap)
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile_size")

    windows: List[Window] = []
    row = 0
    while row < height:
        col = 0
        win_h = min(tile_size, height - row)
        while col < width:
            win_w = min(tile_size, width - col)
            windows.append(Window(col_off=col, row_off=row, width=win_w, height=win_h))
            col += stride
        row += stride
    return windows


def rgb_to_uint8(rgb_hwc: np.ndarray) -> np.ndarray:
    """Convert a 3-band HWC array to uint8 using robust 1-99 percentile stretching."""
    if rgb_hwc.dtype == np.uint8:
        return rgb_hwc

    x = rgb_hwc.astype(np.float32)
    out = np.zeros_like(x, dtype=np.uint8)

    for k in range(3):
        ch = x[:, :, k]
        finite = np.isfinite(ch)
        if not finite.any():
            continue
        lo = float(np.percentile(ch[finite], 1))
        hi = float(np.percentile(ch[finite], 99))
        if hi <= lo:
            hi = lo + 1.0
        y = (ch - lo) * (255.0 / (hi - lo))
        out[:, :, k] = np.clip(y, 0, 255).astype(np.uint8)

    return out


def safe_geodataframe_to_file(gdf: gpd.GeoDataFrame, out_path: str) -> None:
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
        gdf.to_file(out_path, driver="GeoJSON")
    except Exception as ex:
        print(f"⚠️ Could not write GeoJSON: {out_path} -> {type(ex).__name__}: {ex}")


# ======================================================================================
# SAM3 HELPERS
# ======================================================================================


def _ensure_hw_mask(mask_array: np.ndarray) -> np.ndarray:
    m = np.asarray(mask_array)
    m = np.squeeze(m)
    if m.ndim != 2:
        raise ValueError(f"Mask must be 2D after squeeze, got {m.shape}")
    return m.astype(np.float32)


def _masks_to_list_np(masks) -> List[np.ndarray]:
    out: List[np.ndarray] = []

    if isinstance(masks, torch.Tensor):
        t = masks.detach().float().cpu()
        if t.ndim == 2:
            out.append(_ensure_hw_mask(t.numpy()))
            return out
        if t.ndim == 3:
            for i in range(t.shape[0]):
                out.append(_ensure_hw_mask(t[i].numpy()))
            return out
        if t.ndim == 4:
            for i in range(t.shape[0]):
                mi = t[i]
                if mi.ndim == 3:
                    mi = mi[0]
                out.append(_ensure_hw_mask(mi.numpy()))
            return out
        raise ValueError(f"Unsupported tensor mask shape: {tuple(t.shape)}")

    for m in masks:
        if isinstance(m, torch.Tensor):
            out.append(_ensure_hw_mask(m.detach().float().cpu().numpy()))
        else:
            out.append(_ensure_hw_mask(np.asarray(m, dtype=np.float32)))

    return out


def _scores_to_list(scores) -> Optional[List[float]]:
    if scores is None:
        return None
    if isinstance(scores, torch.Tensor):
        return scores.detach().float().cpu().numpy().tolist()
    return [float(s) for s in list(scores)]


def build_sam3_processor(device: str) -> Sam3Processor:
    if SAM3_CHECKPOINT_PATH and os.path.exists(SAM3_CHECKPOINT_PATH):
        try:
            model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT_PATH)
        except TypeError:
            model = build_sam3_image_model()
    else:
        model = build_sam3_image_model()

    model.to(device)
    model.eval()
    return Sam3Processor(model)


def sam3_predict_union_mask(processor: Sam3Processor, rgb_u8_hwc: np.ndarray) -> np.ndarray:
    img = Image.fromarray(rgb_u8_hwc, mode="RGB")
    height, width = rgb_u8_hwc.shape[:2]

    with torch.no_grad():
        state = processor.set_image(img)
        out = processor.set_text_prompt(prompt=PROMPT, state=state)

    masks_raw = out.get("masks", None)
    if masks_raw is None:
        return np.zeros((height, width), dtype=np.uint8)

    scores_raw = out.get("scores", None)
    masks_np = _masks_to_list_np(masks_raw)
    scores_list = _scores_to_list(scores_raw)

    if scores_list is not None and len(scores_list) == len(masks_np):
        keep = [i for i, s in enumerate(scores_list) if float(s) >= SCORE_THRESH]
        masks_np = [masks_np[i] for i in keep]

    if not masks_np:
        return np.zeros((height, width), dtype=np.uint8)

    combined = np.zeros((height, width), dtype=np.float32)
    for m in masks_np:
        if m.shape != (height, width):
            print(f"⚠️ SAM3 mask shape mismatch: expected {(height, width)}, got {m.shape}. Ignoring this tile.")
            return np.zeros((height, width), dtype=np.uint8)
        if int((m > 0.5).sum()) < int(MIN_MASK_AREA_PX):
            continue
        combined = np.maximum(combined, m)

    return (combined > 0.5).astype(np.uint8)


def create_full_sam3_mask(sat_tif: str, out_mask_tif: str, processor: Sam3Processor) -> None:
    """Run SAM3 tile-by-tile and write one full-scene georeferenced mask."""
    with rasterio.open(sat_tif) as sat_ds:
        sat_crs_pyproj = normalize_crs_for_pyproj(sat_ds.crs)
        rio_sat_crs = RioCRS.from_user_input(sat_crs_pyproj.to_wkt())
        width, height = sat_ds.width, sat_ds.height

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "uint8",
            "nodata": 0,
            "transform": sat_ds.transform,
            "crs": rio_sat_crs,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        overlap = OVERLAP_PX if USE_OVERLAP else 0
        wins = windows_for_size(width, height, TILE_SIZE_PX, overlap=overlap)

        tmp_path = out_mask_tif + ".tmp.tif"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        print("Creating full georeferenced SAM3 mask GeoTIFF on satellite grid...")
        with rasterio.open(tmp_path, "w", **profile) as mask_out:
            for win in tqdm(wins, desc="SAM3 mask tiles", unit="tile"):
                sat_rgb = sat_ds.read(indexes=list(SAT_RGB_BANDS), window=win)
                sat_rgb = np.moveaxis(sat_rgb, 0, -1)
                rgb_u8 = rgb_to_uint8(sat_rgb)

                mask01 = sam3_predict_union_mask(processor, rgb_u8)
                mask_out.write(mask01.astype(np.uint8), 1, window=win)

        os.replace(tmp_path, out_mask_tif)
        print("✅ Wrote SAM3 mask:", out_mask_tif)


def vectorize_sam3_mask(mask_tif: str, out_regions_geojson: Optional[str] = None) -> List[object]:
    """Convert full-scene SAM3 mask into dissolved polygon regions in satellite CRS."""
    with rasterio.open(mask_tif) as mask_ds:
        mask_arr = mask_ds.read(1)
        mask_bool = mask_arr > 0
        if not mask_bool.any():
            raise RuntimeError("SAM3 full-scene mask is empty. Try PROMPT='building' or lower SCORE_THRESH.")

        geoms = []
        for geom, val in rio_shapes(mask_arr.astype(np.uint8), mask=mask_bool, transform=mask_ds.transform):
            if int(val) != 1:
                continue
            g = shape(geom)
            if not g.is_valid:
                g = g.buffer(0)
            if g.is_empty:
                continue
            if float(g.area) < float(MIN_REGION_AREA_M2):
                continue
            geoms.append(g)

        if not geoms:
            raise RuntimeError("No SAM3 regions after filtering. Lower MIN_REGION_AREA_M2 or tune PROMPT.")

        dissolved = unary_union(geoms)
        if dissolved.geom_type == "Polygon":
            regions = [dissolved]
        else:
            regions = list(dissolved.geoms)

        regions = [r for r in regions if (not r.is_empty and float(r.area) >= float(MIN_REGION_AREA_M2))]

        if MAX_PATCHES is not None:
            regions = regions[: int(MAX_PATCHES)]
            print(f"⚠️ MAX_PATCHES active -> using first {len(regions)} regions")

        print(f"✅ SAM3 regions detected: {len(regions)}")

        if out_regions_geojson is not None and WRITE_REGIONS_GEOJSON:
            features = []
            for i, r in enumerate(regions, start=1):
                features.append({
                    "type": "Feature",
                    "geometry": mapping(r),
                    "properties": {
                        "rid": int(i),
                        "area": float(r.area),
                    },
                })
            with open(out_regions_geojson, "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
            print("🗂️ Wrote SAM3 regions GeoJSON:", out_regions_geojson)

        return regions


# ======================================================================================
# OSM HELPERS
# ======================================================================================


def transform_bounds_to_wgs84(bounds, src_crs) -> Tuple[float, float, float, float]:
    transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    west, south = transformer.transform(bounds.left, bounds.bottom)
    east, north = transformer.transform(bounds.right, bounds.top)
    return west, south, east, north


def tile_bbox(west: float, south: float, east: float, north: float, tile_size: float) -> List[Tuple[float, float, float, float]]:
    lat_steps = max(1, math.ceil((north - south) / tile_size))
    lon_steps = max(1, math.ceil((east - west) / tile_size))

    tiles: List[Tuple[float, float, float, float]] = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            s = south + i * tile_size
            n = min(south + (i + 1) * tile_size, north)
            w = west + j * tile_size
            e = min(west + (j + 1) * tile_size, east)
            tiles.append((w, s, e, n))
    return tiles


def split_tile(tb: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
    w, s, e, n = tb
    mx = (w + e) / 2.0
    my = (s + n) / 2.0
    return [
        (w, s, mx, my),
        (mx, s, e, my),
        (w, my, mx, n),
        (mx, my, e, n),
    ]


def tile_size_of(tb: Tuple[float, float, float, float]) -> float:
    w, s, e, n = tb
    return max(abs(e - w), abs(n - s))


def fetch_overpass_json(endpoint: str, query: str, timeout_s: int) -> dict:
    """
    Send a raw Overpass POST request and parse JSON directly.

    This avoids overpy's node-resolution behaviour, which can raise
    DataIncomplete even when `out geom;` already returned the geometry.
    """
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "sam3-osm-heat-dataset-creator/1.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=int(timeout_s)) as response:
        raw = response.read().decode("utf-8", errors="replace")

    return json.loads(raw)


def fetch_buildings_for_tile(
    tb: Tuple[float, float, float, float],
    endpoints: List[str],
    timeout_s: int,
    sleep_s: float,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
) -> Tuple[List[dict], bool]:
    """
    Fetch building ways for one WGS84 tile.

    Returns:
        features, request_succeeded

    request_succeeded=True means Overpass responded successfully, even if there are zero buildings.
    request_succeeded=False means all endpoints/retries failed.
    """
    w, s, e, n = tb

    # `out geom;` returns geometry directly in JSON. We parse the JSON ourselves
    # instead of using overpy, because overpy may still try to resolve nodes.
    query = f"""
    [out:json][timeout:{int(timeout_s)}];
    way["building"]({s},{w},{n},{e});
    out geom;
    """

    last_error = None

    for endpoint in endpoints:
        for attempt in range(1, max_retries + 1):
            try:
                data = fetch_overpass_json(endpoint, query, timeout_s=int(timeout_s) + 30)
                time.sleep(float(sleep_s))

                features: List[dict] = []
                for element in data.get("elements", []):
                    if element.get("type") != "way":
                        continue

                    geom = element.get("geometry", [])
                    coords = []
                    for pt in geom:
                        lon = pt.get("lon", None)
                        lat = pt.get("lat", None)
                        if lon is not None and lat is not None:
                            coords.append((float(lon), float(lat)))

                    if len(coords) >= 3:
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        features.append({
                            "type": "Feature",
                            "geometry": {"type": "Polygon", "coordinates": [coords]},
                            "properties": {"id": int(element.get("id"))},
                        })

                return features, True

            except urllib.error.HTTPError as ex:
                last_error = ex
                status = getattr(ex, "code", None)
                print(
                    f"❌ Overpass HTTP failed | endpoint={endpoint} | attempt={attempt}/{max_retries} | "
                    f"tile={tb} | HTTP {status}: {ex}"
                )

                # 400/406 usually means this endpoint rejects the query/request format.
                # Trying the same endpoint repeatedly is usually wasted time.
                if status in (400, 403, 405, 406, 414):
                    print("   ↪ Endpoint rejected the request. Trying the next endpoint.")
                    break

                wait_s = min(float(backoff_max), float(backoff_base) * (2 ** (attempt - 1)))
                print(f"   ⏳ Waiting {wait_s:.1f}s before retry...")
                time.sleep(wait_s)

            except Exception as ex:
                last_error = ex
                err_name = type(ex).__name__
                print(
                    f"❌ Overpass failed | endpoint={endpoint} | attempt={attempt}/{max_retries} | "
                    f"tile={tb} | {err_name}: {ex}"
                )

                wait_s = min(float(backoff_max), float(backoff_base) * (2 ** (attempt - 1)))
                print(f"   ⏳ Waiting {wait_s:.1f}s before retry...")
                time.sleep(wait_s)

    print(f"🚫 Tile request failed after all endpoints/retries: {tb} | last_error={last_error}")
    return [], False


def fetch_osm_buildings_for_satellite(sat_ds, osm_cache_path: str) -> gpd.GeoDataFrame:
    """Fetch/load OSM building footprints and return them in the satellite CRS."""
    sat_crs_pyproj = normalize_crs_for_pyproj(sat_ds.crs)

    if USE_OSM_CACHE and os.path.exists(osm_cache_path):
        print("🧠 Using cached OSM footprints:", osm_cache_path)
        gdf = gpd.read_file(osm_cache_path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if "id" in gdf.columns:
            gdf["id"] = gdf["id"].astype(int)
        gdf = gdf.to_crs(sat_crs_pyproj)
        print(f"✅ Loaded {len(gdf)} OSM buildings from cache.")
        return gdf

    if OSM_CACHE_ONLY:
        raise RuntimeError(
            "OSM_CACHE_ONLY=True, but no cache exists at: "
            f"{osm_cache_path}\n"
            "Create/copy osm_buildings_cache.geojson first, or set OSM_CACHE_ONLY=False."
        )

    west, south, east, north = transform_bounds_to_wgs84(sat_ds.bounds, sat_crs_pyproj)
    print(f"🔍 Querying Overpass for satellite extent: S={south}, W={west}, N={north}, E={east}")

    queue = tile_bbox(west, south, east, north, float(OSM_TILE_SIZE_DEG))
    print(f"📌 Initial OSM tiles: {len(queue)} with tile size {OSM_TILE_SIZE_DEG}")

    all_features: List[dict] = []
    pbar = tqdm(total=len(queue), desc="Querying OSM tiles", unit="tile")

    i = 0
    while i < len(queue):
        tb = queue[i]
        features, ok = fetch_buildings_for_tile(
            tb=tb,
            endpoints=OVERPASS_ENDPOINTS,
            timeout_s=int(OVERPASS_TIMEOUT_S),
            sleep_s=float(OVERPASS_SLEEP_S),
            max_retries=int(MAX_RETRIES_PER_TILE),
            backoff_base=float(BACKOFF_BASE_S),
            backoff_max=float(BACKOFF_MAX_S),
        )

        if ok:
            all_features.extend(features)
            pbar.update(1)
            i += 1
            continue

        # Only split failed request tiles, not valid empty tiles.
        ts = tile_size_of(tb)
        if ts > float(OSM_MIN_TILE_SIZE_DEG):
            subtiles = split_tile(tb)
            queue.pop(i)
            queue[i:i] = subtiles
            pbar.total = len(queue)
            pbar.refresh()
            print(
                f"🔁 Split failed tile into 4: {ts:.6f} deg -> "
                f"{tile_size_of(subtiles[0]):.6f} deg. Queue size={len(queue)}"
            )
            continue

        print(f"⏭️ Skipping failed tile at minimum tile size: {tb}")
        pbar.update(1)
        i += 1

    pbar.close()

    if not all_features:
        raise RuntimeError(
            "OSM fetch returned zero buildings. This may mean internet/Overpass is blocked, "
            "the query extent is wrong, or the area has no OSM building footprints."
        )

    gdf_wgs84 = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    if "id" not in gdf_wgs84.columns:
        raise RuntimeError("OSM features were fetched, but no 'id' column exists.")

    gdf_wgs84["id"] = gdf_wgs84["id"].astype(int)
    gdf_wgs84 = gdf_wgs84.drop_duplicates(subset=["id"]).reset_index(drop=True)

    if USE_OSM_CACHE:
        ensure_dir(os.path.dirname(osm_cache_path))
        safe_geodataframe_to_file(gdf_wgs84, osm_cache_path)
        print("💾 Cached OSM footprints in WGS84:", osm_cache_path)

    gdf_sat = gdf_wgs84.to_crs(sat_crs_pyproj)
    print(f"✅ Fetched {len(gdf_sat)} unique OSM buildings.")
    return gdf_sat


def compute_best_osm_id(region_geom, osm_gdf_satcrs: gpd.GeoDataFrame) -> Optional[int]:
    """Match one SAM3 region to the OSM building with max intersection area; IoU breaks ties."""
    if osm_gdf_satcrs is None or len(osm_gdf_satcrs) == 0:
        return None

    minx, miny, maxx, maxy = region_geom.bounds
    candidates = osm_gdf_satcrs.cx[minx:maxx, miny:maxy]
    if candidates is None or len(candidates) == 0:
        return None

    best_id = None
    best_inter = 0.0
    best_iou = 0.0

    for _, row in candidates.iterrows():
        osm_geom = row.geometry
        if osm_geom is None or osm_geom.is_empty:
            continue
        if not osm_geom.intersects(region_geom):
            continue

        inter = float(osm_geom.intersection(region_geom).area)
        if inter <= 0:
            continue

        union = float(osm_geom.union(region_geom).area)
        iou = inter / union if union > 0 else 0.0

        if inter > best_inter:
            best_inter = inter
            best_iou = iou
            best_id = int(row["id"])
        elif abs(inter - best_inter) < 1e-9 and iou > best_iou:
            best_iou = iou
            best_id = int(row["id"])

    if best_id is None:
        return None

    if best_inter < float(MIN_INTERSECTION_AREA):
        return None

    return best_id


# ======================================================================================
# CROP / SAVE HELPERS
# ======================================================================================


def crop_satellite_by_region(sat_ds, region_geom, buffer_m: float):
    """
    Crop the original satellite GeoTIFF using the SAM3 region polygon.

    Returns:
        crop_to_save: array CxHxW, original dtype, either all bands or RGB only
        rgb_u8: HxWx3 uint8 for JPG export
        crop_transform: rasterio affine transform
        width, height
    """
    crop_geom = region_geom.buffer(float(buffer_m))
    if crop_geom.is_empty:
        return None, None, None, None, None

    try:
        crop_all, crop_transform = rio_mask(sat_ds, [mapping(crop_geom)], crop=True, filled=True, nodata=0)
    except Exception as ex:
        print(f"⚠️ Satellite crop failed: {type(ex).__name__}: {ex}")
        return None, None, None, None, None

    if crop_all.ndim != 3 or crop_all.shape[0] < 3:
        return None, None, None, None, None

    height, width = int(crop_all.shape[1]), int(crop_all.shape[2])
    if height < 16 or width < 16:
        return None, None, None, None, None

    rgb_hwc = np.moveaxis(crop_all[:3], 0, -1)
    rgb_u8 = rgb_to_uint8(rgb_hwc)

    if GEOTIFF_OUTPUT_MODE == "rgb_only":
        crop_to_save = crop_all[:3]
    elif GEOTIFF_OUTPUT_MODE == "all_bands":
        crop_to_save = crop_all
    else:
        raise ValueError("GEOTIFF_OUTPUT_MODE must be either 'all_bands' or 'rgb_only'")

    return crop_to_save, rgb_u8, crop_transform, width, height


def save_geotiff_crop(crop_array: np.ndarray, crop_transform, crop_crs, out_path: str) -> None:
    profile = {
        "driver": "GTiff",
        "height": int(crop_array.shape[1]),
        "width": int(crop_array.shape[2]),
        "count": int(crop_array.shape[0]),
        "dtype": crop_array.dtype,
        "crs": crop_crs,
        "transform": crop_transform,
        "nodata": 0,
        "compress": "lzw",
    }

    tmp_path = out_path + ".tmp.tif"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(crop_array)

    os.replace(tmp_path, out_path)


def resize_and_save_jpg(rgb_hwc_u8: np.ndarray, out_path: str, size: int) -> None:
    img = Image.fromarray(rgb_hwc_u8, mode="RGB")
    resized = img.resize((int(size), int(size)), Image.BILINEAR)
    resized.save(out_path, format="JPEG", quality=int(JPG_QUALITY))


def write_name_lists(all_names: List[str], out_lists: str) -> None:
    all_names_sorted = sorted(set(all_names))

    all_list_path = os.path.join(out_lists, "all_list.txt")
    train_list_path = os.path.join(out_lists, "train_list.txt")
    valid_list_path = os.path.join(out_lists, "valid_list.txt")

    with open(all_list_path, "w", encoding="utf-8") as f:
        for name in all_names_sorted:
            f.write(name + "\n")

    if len(all_names_sorted) >= int(VALID_LAST_N):
        valid_names = all_names_sorted[-int(VALID_LAST_N):]
        train_names = all_names_sorted[: -int(VALID_LAST_N)]
    else:
        valid_names = all_names_sorted
        train_names = []

    with open(valid_list_path, "w", encoding="utf-8") as f:
        for name in valid_names:
            f.write(name + "\n")

    with open(train_list_path, "w", encoding="utf-8") as f:
        for name in train_names:
            f.write(name + "\n")


# ======================================================================================
# MAIN PIPELINE
# ======================================================================================


def run() -> None:
    out_root = ensure_dir(OUTPUT_BASE)
    out_jpg_256 = ensure_dir(os.path.join(out_root, "rgb_jpg_256"))
    out_jpg_512 = ensure_dir(os.path.join(out_root, "rgb_jpg_512"))
    out_tif_original = ensure_dir(os.path.join(out_root, "rgb_tif_original"))
    out_lists = ensure_dir(os.path.join(out_root, "lists"))
    out_common = ensure_dir(os.path.join(out_root, "common"))
    out_misc = ensure_dir(os.path.join(out_root, "misc"))

    mask_tif_path = os.path.join(out_common, "sam3_mask_satgrid.tif")
    regions_geojson_path = os.path.join(out_misc, "sam3_regions_satcrs.geojson")
    osm_cache_path = os.path.join(out_misc, OSM_CACHE_GEOJSON)
    osm_sat_debug_path = os.path.join(out_misc, "fetched_buildings_sat_crs.geojson")
    metadata_path = os.path.join(out_root, "geo_metadata.json")

    print("=" * 90)
    print("Combined SAM3 + OSM HEAT dataset creator")
    print("SATELLITE_TIF:", SATELLITE_TIF)
    print("OUTPUT_BASE  :", OUTPUT_BASE)
    print("=" * 90)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    print("Loading SAM3 once...")
    processor = build_sam3_processor(device)

    # 1) SAM3 mask creation
    if WRITE_INTERMEDIATE_MASK_TIF:
        create_full_sam3_mask(SATELLITE_TIF, mask_tif_path, processor)
    else:
        raise RuntimeError(
            "WRITE_INTERMEDIATE_MASK_TIF=False is not supported in this version. "
            "Keeping the full mask on disk makes the pipeline safer and easier to debug."
        )

    # 2) Vectorize SAM3 regions
    print("Vectorizing SAM3 mask into regions...")
    regions = vectorize_sam3_mask(mask_tif_path, out_regions_geojson=regions_geojson_path)

    # 3) Fetch/load OSM footprints and process crops
    with rasterio.open(SATELLITE_TIF) as sat_ds:
        sat_crs_pyproj = normalize_crs_for_pyproj(sat_ds.crs)
        sat_crs_for_output = sat_ds.crs if sat_ds.crs is not None else RioCRS.from_user_input(sat_crs_pyproj.to_wkt())

        print("SAT CRS:", sat_crs_pyproj.to_string())
        print("SAT size:", sat_ds.width, sat_ds.height)
        print("SAT bounds:", sat_ds.bounds)

        print("Fetching/loading OSM buildings for naming...")
        osm_gdf = None
        osm_available = False
        use_sequential_fallback_for_all = False

        try:
            osm_gdf = fetch_osm_buildings_for_satellite(sat_ds, osm_cache_path)
            osm_available = osm_gdf is not None and len(osm_gdf) > 0
        except Exception as ex:
            print("⚠️ OSM fetch/load failed:", type(ex).__name__, str(ex))
            if FALLBACK_TO_SEQUENTIAL_NAMING_IF_OSM_UNAVAILABLE:
                print("⚠️ Continuing without OSM. Outputs will be named building_0, building_1, building_2, ...")
                use_sequential_fallback_for_all = True
            else:
                raise

        if osm_available:
            if WRITE_OSM_DEBUG_GEOJSON:
                safe_geodataframe_to_file(osm_gdf, osm_sat_debug_path)
                print("🗂️ Saved OSM footprints in satellite CRS:", osm_sat_debug_path)
        elif use_sequential_fallback_for_all:
            print("ℹ️ OSM debug GeoJSON will not be written because no OSM footprints are available.")
        else:
            raise RuntimeError("OSM footprints are unavailable and sequential fallback is disabled.")

        metadata: Dict[str, dict] = {}
        all_names: List[str] = []
        used_names = set()

        skip = {
            "crop_empty": 0,
            "sat_crop_fail": 0,
            "too_small": 0,
            "no_osm_match": 0,
            "sequential_fallback_named": 0,
            "name_collision": 0,
            "exception": 0,
        }

        pbar = tqdm(enumerate(regions, start=1), total=len(regions), desc="Saving final patches", unit="region")

        for region_idx, region in pbar:
            try:
                if region is None or region.is_empty:
                    skip["crop_empty"] += 1
                    continue

                osm_id: Optional[int] = None
                naming_mode = "osm_named"

                if use_sequential_fallback_for_all:
                    sequential_id = len(all_names)
                    base_name = f"building_{sequential_id}"
                    naming_mode = "sequential_fallback"
                    skip["sequential_fallback_named"] += 1
                else:
                    osm_id = compute_best_osm_id(region, osm_gdf)
                    if osm_id is None:
                        skip["no_osm_match"] += 1
                        if FALLBACK_TO_SEQUENTIAL_NAMING_FOR_UNMATCHED_REGIONS:
                            sequential_id = len(all_names)
                            base_name = f"building_{sequential_id}"
                            naming_mode = "sequential_fallback_unmatched_region"
                            skip["sequential_fallback_named"] += 1
                        else:
                            continue
                    else:
                        base_name = f"building_{int(osm_id)}"

                crop_to_save, rgb_u8, crop_transform, orig_w, orig_h = crop_satellite_by_region(
                    sat_ds=sat_ds,
                    region_geom=region,
                    buffer_m=float(CROP_BUFFER_M),
                )

                if crop_to_save is None or rgb_u8 is None or crop_transform is None:
                    skip["sat_crop_fail"] += 1
                    continue

                if int(orig_w) < 16 or int(orig_h) < 16:
                    skip["too_small"] += 1
                    continue

                final_name = base_name

                if final_name in used_names:
                    skip["name_collision"] += 1
                    suffix = 2
                    while f"{base_name}_r{suffix}" in used_names:
                        suffix += 1
                    final_name = f"{base_name}_r{suffix}"

                used_names.add(final_name)

                # Save original georeferenced GeoTIFF crop.
                tif_path = os.path.join(out_tif_original, f"{final_name}.tif")
                save_geotiff_crop(crop_to_save, crop_transform, sat_crs_for_output, tif_path)

                metadata[f"{final_name}_original"] = {
                    "name": final_name,
                    "osm_id": int(osm_id) if osm_id is not None else None,
                    "region_index": int(region_idx),
                    "naming_mode": naming_mode,
                    "crs": sat_crs_pyproj.to_string(),
                    "transform": [
                        crop_transform.a,
                        crop_transform.b,
                        crop_transform.c,
                        crop_transform.d,
                        crop_transform.e,
                        crop_transform.f,
                    ],
                    "original_size": [int(orig_w), int(orig_h)],
                    "geotiff_path": tif_path,
                    "band_count": int(crop_to_save.shape[0]),
                    "dtype": str(crop_to_save.dtype),
                    "geotiff_output_mode": GEOTIFF_OUTPUT_MODE,
                    "source": "sam3_region_crop_" + naming_mode,
                }

                # Save JPG versions and metadata.
                for size in JPG_SIZES:
                    size = int(size)
                    out_dir = out_jpg_256 if size == 256 else out_jpg_512
                    jpg_path = os.path.join(out_dir, f"{final_name}.jpg")
                    resize_and_save_jpg(rgb_u8, jpg_path, size)

                    scale_x = float(orig_w) / float(size)
                    scale_y = float(orig_h) / float(size)

                    metadata[f"{final_name}_{size}"] = {
                        "name": final_name,
                        "osm_id": int(osm_id) if osm_id is not None else None,
                        "region_index": int(region_idx),
                        "naming_mode": naming_mode,
                        "crs": sat_crs_pyproj.to_string(),
                        "transform": [
                            crop_transform.a,
                            crop_transform.b,
                            crop_transform.c,
                            crop_transform.d,
                            crop_transform.e,
                            crop_transform.f,
                        ],
                        "original_size": [int(orig_w), int(orig_h)],
                        "resized_size": [size, size],
                        "scale_factors": [float(scale_x), float(scale_y)],
                        "jpg_path": jpg_path,
                        "geotiff_path": tif_path,
                        "source": "sam3_region_crop_" + naming_mode,
                    }

                all_names.append(final_name)
                pbar.set_postfix(saved=len(all_names), no_osm=skip["no_osm_match"])

            except Exception as ex:
                skip["exception"] += 1
                tqdm.write(f"⚠️ Skipping region_{region_idx:06d}: {type(ex).__name__}: {ex}")
                continue

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    write_name_lists(all_names, out_lists)

    print("\n✅ Done.")
    print("Saved patches:", len(sorted(set(all_names))))
    print("Skip breakdown:", skip)
    print("Metadata:", metadata_path)
    print("GeoTIFF original crops:", out_tif_original)
    print("JPG256:", out_jpg_256)
    print("JPG512:", out_jpg_512)
    print("Lists:", out_lists)
    print("SAM3 full mask:", mask_tif_path)
    print("SAM3 regions:", regions_geojson_path)
    print("OSM cache:", osm_cache_path)


if __name__ == "__main__":
    if not os.path.exists(SATELLITE_TIF):
        raise FileNotFoundError(f"SATELLITE_TIF not found: {SATELLITE_TIF}")
    if SAM3_CHECKPOINT_PATH and (not os.path.exists(SAM3_CHECKPOINT_PATH)):
        raise FileNotFoundError(f"SAM3_CHECKPOINT_PATH not found: {SAM3_CHECKPOINT_PATH}")
    run()
