# -*- coding: utf-8 -*-
"""
Batch roof reconstruction with 2D editing/deletion + 3D validation (Open3D),
adaptive base-Z from LiDAR (per-vertex up-shoot with adaptive radius),
GeoTIFF overlay toggle, connected-component tagging, and plane-based
roof reconstruction without interactive Stage-I height picking.

This version:
  • Extracts ALL closed loops per connected component
  • Builds base + walls per loop
  • Tags generated edges with component_id, loop_id, ring_order
  • Supports GeoTIFF overlay toggle with T
  • Supports real undo in DELETE mode with U
  • Processes ONLY files whose basenames exist inside PROCESS_ONLY_FROM_DIR
"""

import os
import glob
import json
import numpy as np
import laspy
import csv
import geopandas as gpd
import networkx as nx
import matplotlib
import rasterio
import matplotlib.pyplot as plt
from matplotlib.path import Path

from collections import Counter, deque, defaultdict
from shapely.geometry import LineString, MultiPoint, Point, Polygon, mapping, box
from shapely.ops import polygonize, unary_union, triangulate
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import NearestNeighbors

import open3d as o3d
from open3d import geometry, utility


# --- reduce mpl default keymap interference
matplotlib.rcParams["keymap.quit"] = []
matplotlib.rcParams["keymap.back"] = []
matplotlib.rcParams["keymap.forward"] = []
matplotlib.rcParams["keymap.save"] = []


# ==========================
# === USER CONFIG (edit) ===
# ==========================
GEOJSON_DIR = "./work/georeferenced_output/geojson_files"
LAZ_DIR = "./work/Lidar_input"
GEOTIFF_DIR = "./work/heat_input/rgb_tif_original"
OUTPUT_DIR = "./work/3D_output"
RMSE_CSV_PATH = "./work/3D_output/roof_rmse_results.csv"
SNAPSHOT_DIR = "./work/3D_output/snapshots" 

LIDAR_TO_MESH_MAX_DIST = 2.0          # optional outlier cap for evaluation points
ROOF_CLASS_CODE = 6                   # ASPRS building class, if available
GROUND_CLASS_CODE = 2                 # ASPRS ground class
ROOF_MIN_HEIGHT_ABOVE_GROUND = 1.5    # fallback roof filtering when building class is absent
LIDAR_PREVIEW_COLOR = [0.55, 0.55, 0.55]


# Only files whose basename exists here will be processed.
# Example: if "building_001.geojson" exists here, then the script will process:
#   GEOJSON_DIR\building_001.geojson
#   LAZ_DIR\building_001.laz
#   GEOTIFF_DIR\building_001.tif/.tiff
PROCESS_ONLY_FROM_DIR = None

# Hard defaults (used for every edge unless temporarily overridden in fixer)
THRESHOLD_HARD_DEFAULT          = 0.2   # corridor width (m)
BIN_WIDTH_HARD_DEFAULT          = 0.2   # Z binning (m)
PRIOR_BIAS_HARD_DEFAULT         = True  # bias Z-bin toward prior heights (per-component if tagged)
RANSAC_RESIDUAL_HARD_DEFAULT    = 0.3   # meters
RANSAC_MIN_SAMPLES_HARD_DEFAULT = 3
USE_ICP_HARD_DEFAULT            = False
ICP_THRESHOLD                   = 50.0

# Vertex roof-pick radius (m) and selection radii
RADIUS_VERTEX_ROOF = 0.30
VERTEX_PICK_RADIUS = 0.20   # in HEIGHT mode, click within this to pick a vertex
EDGE_PICK_DIST     = 1.2    # in HEIGHT mode, click within this to pick an edge

# Adaptive base radius clamps
MIN_RADIUS_M = 0.30
MAX_RADIUS_M = 12.0

# 3D yellow highlight (chosen bin) rendering controls
YELLOW_VOXEL_DOWNSAMPLE = 0.15
YELLOW_SPHERE_RADIUS    = 0.07# Vertex assignment / splitting from fitted face planes
YELLOW_SPHERE_MAX       = 2000

# Plane fitting from closed roof face loops
PLANE_FACE_BUFFER              = THRESHOLD_HARD_DEFAULT
PLANE_RANSAC_RESIDUAL          = 0.20
PLANE_RANSAC_MIN_SAMPLES       = 3
PLANE_MIN_POINTS               = 30
PLANE_MIN_INLIERS              = 20
PLANE_MESH_COLOR               = [0.2, 0.8, 0.8]
PLANE_OUTLINE_COLOR            = [0.8, 0.0, 0.8]
PREVIEW_COLUMN_GAP_FACTOR      = 1.7   # center-to-center spacing between GT LiDAR and model preview
PREVIEW_MIN_COLUMN_GAP         = 12.0
ROOF_SEAM_MIN_VERTICAL_GAP     = 0.60  # seam is kept only when duplicated roof-edge Z gap exceeds this

# Vertex assignment / splitting from fitted face planes
ENABLE_PLANE_VERTEX_SPLIT_CORRECTION = True
PLANE_VERTEX_SPLIT_GAP_FACTOR        = 1.0   # vertical split gap = factor * estimated XY spacing
PLANE_CORRECTED_VERTEX_COLOR         = [1.0, 0.0, 1.0]

# Base-ring regularization
BASE_DOMINANT_FRAC            = 0.70   # at least 70% must support the dominant low group
BASE_Z_CLUSTER_GAP            = 0.60   # max z-gap inside one ground cluster (meters)
BASE_TOO_HIGH_ABOVE_DOMINANT  = 0.60   # if a base vertex is this much above dominant cluster median -> abnormal
BASE_NEAR_ROOF_EPS            = 0.80   # if base is within this distance of roof z -> suspicious
BASE_DOMINANT_MIN_ROOF_DROP   = 1.50   # dominant group must be clearly below roof to trigger near-roof correction
BASE_SPIKE_MIN_RESIDUAL       = 0.80   # isolated base-loop bump must exceed this vertical error (m)
BASE_SPIKE_LOCAL_MARGIN       = 0.20   # bump must be this far above/below both neighbouring base heights
BASE_SPIKE_MAX_RUN            = 2      # correct isolated one- or two-vertex base spikes
BASE_SPIKE_MAX_PASSES         = 2      # repeat once after correcting the strongest local spikes

# Edge colors in 3D viewer (same for both variants)
ROOF_EDGE_COLOR = [0.10, 0.35, 1.00]   # blue = adjusted/final roof edges
PRE_ADJUSTMENT_ROOF_EDGE_COLOR = [1.0, 0.0, 0.0]   # red = previous roof edges
WALL_EDGE_COLOR = [0.0, 0.6, 0.0]   # green
BASE_EDGE_COLOR = [0.0, 0.0, 1.0]   # blue

ROOF_FACE_COLOR      = [0.95, 0.75, 0.75]
WALL_FACE_COLOR      = [0.80, 0.95, 0.80]
BASE_FACE_COLOR      = [0.80, 0.85, 0.98]
ROOF_SEAM_FACE_COLOR = [1.00, 0.75, 0.30]

KEY_HELP = r"""
====================== KEY / COMMAND CHEAT-SHEET ======================

[2D DELETE mode] (default)
  Left click near geom    -> Select nearest EDGE/vertex
  Left drag               -> Rectangle-select vertices / edges
  E                       -> Delete SELECTED EDGE(S)
  V                       -> Delete SELECTED VERTEX/VERTICES (and incident edges)
  U                       -> Undo last delete
  A                       -> Accept/save updated 2D skeleton to GeoJSON
  D                       -> Switch to DELETE mode
  T                       -> Toggle GeoTIFF overlay ON/OFF
  C                       -> Tag connected components
  3                       -> Build 3D result + open 3D validator
  Q                       -> Quit batch (no save)
  (Close window)          -> Quit batch (no save)

[2D EDIT mode]
  Left click              -> Add new vertex
  Right click             -> Select point / line
  Right double click      -> Connect 2 selected points
  X                       -> Delete selected point / line
  U or Z                  -> Undo last edit action
  A                       -> Accept/save edited 2D skeleton to GeoJSON
  D / E                   -> Switch mode

[3D PREVIEW]
  N                       -> Save & Next
  P                       -> Save & Prev
  R                       -> Redo this file (no save; returns to 2D)
  Q                       -> Quit batch (no save)
  L                       -> Toggle LiDAR/debug overlays ON/OFF
  O                       -> Toggle previous red roof edges ON/OFF
  F                       -> Toggle fitted blue roof plane overlays ON/OFF
  G                       -> Toggle generated light roof surfaces ON/OFF
  W                       -> Toggle wall surfaces ON/OFF
  S                       -> Save PNG snapshot of current 3D view
  (Close window)          -> Quit (no save)
=======================================================================
"""


# ----------------- Geometry / helpers -----------------
def point_segment_distance(pt, p1, p2):
    v = p2 - p1
    w = pt - p1
    L2 = v.dot(v)
    if L2 == 0:
        return np.linalg.norm(w)
    t = np.clip(w.dot(v) / L2, 0.0, 1.0)
    proj = p1 + t * v
    return np.linalg.norm(pt - proj)


def load_geotiff_for_display(tif_path):
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"GeoTIFF file not found: {tif_path}")

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        count = src.count

        def _normalize(arr):
            arr = arr.astype(np.float32)
            finite = np.isfinite(arr)
            if not np.any(finite):
                return np.zeros_like(arr, dtype=np.float32)

            vals = arr[finite]
            lo = np.percentile(vals, 2)
            hi = np.percentile(vals, 98)
            if hi <= lo:
                hi = lo + 1e-6

            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
            return arr

        if count >= 3:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
            img = _normalize(img)
            mode = "rgb"
        else:
            img = src.read(1)
            img = _normalize(img)
            mode = "gray"

        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    return img, extent, mode


def compute_avg_z_with_z_filtering(
    pts,
    p1,
    p2,
    thresh,
    prior_heights_list,
    bin_width,
    ransac_residual,
    ransac_min_samples,
    prior_bias=True
):
    dists = np.array([point_segment_distance(pt[:2], p1, p2) for pt in pts])
    sel = pts[dists < thresh]

    if sel.shape[0] < 3:
        return None, np.empty((0, 3)), None, "few_pts", None

    z_vals = sel[:, 2]
    z_bins = np.round(z_vals / bin_width) * bin_width
    uniq = np.unique(z_bins)

    bin_groups = {
        bz: sel[np.abs(z_vals - bz) < (bin_width / 2)]
        for bz in uniq
    }

    eligible = [
        (bz, grp)
        for bz, grp in bin_groups.items()
        if grp.shape[0] >= ransac_min_samples
    ]

    if not eligible:
        return None, np.empty((0, 3)), None, "bin_empty", None

    if prior_bias and len(prior_heights_list) > 0:
        prior_mode = Counter(np.round(np.array(prior_heights_list), 1)).most_common(1)[0][0]
        chosen_bin = min(eligible, key=lambda t: abs(t[0] - prior_mode))[1]
    else:
        chosen_bin = max(eligible, key=lambda t: t[0])[1]

    X = chosen_bin[:, :2]
    y = chosen_bin[:, 2]

    model = RANSACRegressor(
        residual_threshold=ransac_residual,
        min_samples=ransac_min_samples
    )
    model.fit(X, y)
    inliers = chosen_bin[model.inlier_mask_]

    if inliers.shape[0] == 0:
        return None, np.empty((0, 3)), None, "ransac_zero", chosen_bin

    line_coords_2d = np.array([p1, p2])
    z_line = model.predict(line_coords_2d)
    ransac_line = np.column_stack([line_coords_2d, z_line])

    return inliers[:, 2].mean(), inliers, ransac_line, None, chosen_bin


def icp_align(lidar_pts, skeleton_pts2d):
    hull = MultiPoint(lidar_pts[:, :2]).convex_hull
    hull_xy = np.array(hull.exterior.coords)

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(
        np.hstack([hull_xy, np.zeros((len(hull_xy), 1))])
    )

    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(
        np.hstack([skeleton_pts2d, np.zeros((len(skeleton_pts2d), 1))])
    )

    reg = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        ICP_THRESHOLD,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print("\n🧠 ICP RMSE:", reg.inlier_rmse)

    H = reg.transformation
    ones = np.ones((len(lidar_pts), 1))
    pts_hom = np.hstack([lidar_pts[:, :2], np.zeros((len(lidar_pts), 1)), ones])
    transformed = (H @ pts_hom.T).T
    lidar_pts[:, 0], lidar_pts[:, 1] = transformed[:, 0], transformed[:, 1]

    return lidar_pts


def build_vertex_incident_edge_lengths(corners_xy, edges_idx):
    lengths = {i: [] for i in range(len(corners_xy))}
    for i, j in edges_idx:
        p1 = corners_xy[i]
        p2 = corners_xy[j]
        L = float(np.linalg.norm(p2 - p1))
        lengths[i].append(L)
        lengths[j].append(L)
    return lengths


def radius_for_vertex(v_idx, lengths_dict):
    inc = lengths_dict.get(v_idx, [])
    r = 0.5 * min(inc) if len(inc) > 0 else MIN_RADIUS_M
    return max(MIN_RADIUS_M, min(MAX_RADIUS_M, r))


def base_z_at_vertex_upshoot_adaptive(vertex_xy, lidar_pts, lidar_classes, radius_m):
    dx = lidar_pts[:, 0] - vertex_xy[0]
    dy = lidar_pts[:, 1] - vertex_xy[1]
    rr = np.hypot(dx, dy)

    mask = rr <= radius_m
    cand = lidar_pts[mask]

    if cand.shape[0] == 0:
        return None

    if lidar_classes is not None:
        cls = lidar_classes[mask]
        ground_mask = (cls == 2)
        if np.any(ground_mask):
            return float(np.min(cand[ground_mask, 2]))

    return float(np.min(cand[:, 2]))


def load_skeleton_from_geojson(geojson_path):
    gdf = gpd.read_file(geojson_path)

    node_dict = {}
    edges_roof = []
    edge_props_roof = []

    for idx_row, row in gdf.iterrows():
        s = int(row["source"])
        t = int(row["target"])
        coords = list(row.geometry.coords)

        if s not in node_dict:
            node_dict[s] = coords[0]
        if t not in node_dict:
            node_dict[t] = coords[1]

        edges_roof.append((s, t))
        edge_props_roof.append({
            "edge_id": int(row.get("edge_id", idx_row)),
            "image_id": str(row.get("image_id", ""))
        })

    node_ids_sorted = sorted(node_dict)
    id_to_idx = {nid: idx for idx, nid in enumerate(node_ids_sorted)}
    idx_to_id = {v: k for k, v in id_to_idx.items()}

    corners = np.array([[*node_dict[nid], 0.0] for nid in node_ids_sorted], dtype=float)
    edges = [[id_to_idx[s], id_to_idx[t]] for s, t in edges_roof]
    edge_types = ["roof"] * len(edges)
    edge_props = edge_props_roof[:]

    return corners, edges, edge_types, edge_props, idx_to_id


def load_lidar_from_laz(laz_path):
    if not os.path.exists(laz_path):
        raise FileNotFoundError(f"LiDAR file not found: {laz_path}")

    size = os.path.getsize(laz_path)
    if size == 0:
        raise ValueError(f"EMPTY_LAZ: {laz_path} is 0 bytes")

    try:
        las = laspy.read(laz_path)
    except Exception as e:
        raise RuntimeError(f"LASPY_READ_FAILED for {laz_path}: {e}")

    pts = np.vstack([las.x, las.y, las.z]).T

    try:
        classes = np.array(las.classification)
    except Exception:
        classes = None

    return pts, classes


def remap_after_vertex_deletions(corners, edges, edge_types, edge_props, deleted_vertices_set, idx_to_id=None):
    keep = np.ones(len(corners), dtype=bool)
    for vi in deleted_vertices_set:
        if 0 <= vi < len(keep):
            keep[vi] = False

    old_to_new = {}
    new_idx = 0
    for old_idx, k in enumerate(keep):
        if k:
            old_to_new[old_idx] = new_idx
            new_idx += 1

    new_corners = corners[keep].copy()
    new_edges = []
    new_types = []
    new_props = []

    for e, t, p in zip(edges, edge_types, edge_props):
        s, v = e
        if s in old_to_new and v in old_to_new:
            new_edges.append([old_to_new[s], old_to_new[v]])
            new_types.append(t)
            new_props.append(p)

    new_idx_to_id = None
    if idx_to_id is not None:
        new_idx_to_id = {}
        for old_idx, nn in old_to_new.items():
            if old_idx in idx_to_id:
                new_idx_to_id[nn] = idx_to_id[old_idx]

    return new_corners, new_edges, new_types, new_props, old_to_new, new_idx_to_id


def save_3d_geojson(corners, edges, edge_types, edge_props, idx_to_id, output_path, surface_faces=None, face_centroids=None):
    idx_to_id = dict(idx_to_id or {})
    used_ids = set(idx_to_id.values()) if idx_to_id else set()
    next_id = (max(used_ids) + 1) if used_ids else 0

    for i in range(len(corners)):
        if i not in idx_to_id:
            idx_to_id[i] = next_id
            next_id += 1

    def _polygon_coords_from_indices(indices):
        coords = []
        for vi in indices:
            p = corners[int(vi)]
            coords.append([float(p[0]), float(p[1]), float(p[2])])

        if len(coords) >= 3 and coords[0] != coords[-1]:
            coords.append(coords[0][:])

        return coords

    features = []

    for i_e, (e, t, props) in enumerate(zip(edges, edge_types, edge_props)):
        s_idx, t_idx = int(e[0]), int(e[1])
        p1, p2 = corners[s_idx], corners[t_idx]

        line = LineString([
            (float(p1[0]), float(p1[1]), float(p1[2])),
            (float(p2[0]), float(p2[1]), float(p2[2]))
        ])

        out_props = dict(props)
        out_props.update({
            "edge_id": int(props.get("edge_id", i_e)),
            "source": int(idx_to_id[s_idx]),
            "target": int(idx_to_id[t_idx]),
            "type": t,
        })

        features.append({
            "type": "Feature",
            "geometry": mapping(line),
            "properties": out_props
        })

    for i_f, sf in enumerate(surface_faces or []):
        ring = [int(v) for v in sf.get("vertex_indices", [])]
        if len(ring) < 3:
            continue

        poly_coords = _polygon_coords_from_indices(ring)
        if len(poly_coords) < 4:
            continue

        out_props = {k: v for k, v in sf.items() if k != "vertex_indices"}
        out_props.update({
            "feature_kind": "surface",
            "surface_id": int(i_f),
            "vertex_indices": [int(v) for v in ring],
            "vertex_ids": [int(idx_to_id[int(v)]) for v in ring],
        })

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [poly_coords]
            },
            "properties": out_props
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::25832"}
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tmp_path = output_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(fc, f)
        os.replace(tmp_path, output_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    print(f"✅ Saved updated 3D GeoJSON to {output_path}")

def _surface_type_is_roof(surface_type_value):
    s = str(surface_type_value).lower().strip()
    return s.startswith("roof")


def fit_plane_from_3d_ring(coords3d):
    arr = np.asarray(coords3d, dtype=float)

    if arr.shape[0] < 3:
        return 0.0, 0.0, float(arr[:, 2].mean()) if arr.shape[0] > 0 else 0.0

    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]

    xy = arr[:, :2]
    z = arr[:, 2]

    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(arr), dtype=float)])

    if np.linalg.matrix_rank(A) < 3:
        return 0.0, 0.0, float(np.mean(z))

    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    return float(a), float(b), float(c)


def triangulate_3d_ring(coords3d):
    coords3d = [tuple(map(float, c)) for c in coords3d]
    if len(coords3d) < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    if coords3d[0] == coords3d[-1]:
        coords3d = coords3d[:-1]

    if len(coords3d) < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    xy_poly = Polygon([(x, y) for x, y, _ in coords3d])
    if not xy_poly.is_valid:
        xy_poly = xy_poly.buffer(0)

    if xy_poly.is_empty or xy_poly.area <= 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    a, b, c = fit_plane_from_3d_ring(coords3d)

    vertices = []
    triangles_idx = []

    tri_polys = triangulate(xy_poly)
    for tri in tri_polys:
        rep = tri.representative_point()
        if not xy_poly.buffer(1e-9).covers(rep):
            continue

        tri_xy = list(tri.exterior.coords)[:-1]
        if len(tri_xy) != 3:
            continue

        base_idx = len(vertices)
        for x, y in tri_xy:
            z = a * x + b * y + c
            vertices.append([float(x), float(y), float(z)])

        triangles_idx.append([base_idx, base_idx + 1, base_idx + 2])

    if len(triangles_idx) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    return np.asarray(vertices, dtype=float), np.asarray(triangles_idx, dtype=np.int32)


def build_mesh_from_surface_faces(corners, surface_faces):
    all_vertices = []
    all_triangles = []

    for sf in surface_faces or []:
        if not _surface_type_is_roof(sf.get("surface_type", sf.get("type", ""))):
            continue

        ring = [int(v) for v in sf.get("vertex_indices", [])]
        if len(ring) < 3:
            continue

        coords3d = [tuple(corners[v]) for v in ring]
        verts_i, tris_i = triangulate_3d_ring(coords3d)

        if tris_i.shape[0] == 0:
            continue

        offset = len(all_vertices)
        all_vertices.extend(verts_i.tolist())
        all_triangles.extend((tris_i + offset).tolist())

    if len(all_triangles) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    return np.asarray(all_vertices, dtype=float), np.asarray(all_triangles, dtype=np.int32)


def build_raycast_scene(vertices, triangles):
    mesh_legacy = o3d.geometry.TriangleMesh()
    mesh_legacy.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh_legacy.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return scene

def extract_roof_reference_points(lidar_pts, lidar_classes=None):
    """
    Return LiDAR points used as roof reference for evaluation.

    Priority:
    1) If ASPRS building class (6) exists, use it directly.
    2) Else, remove ground class (2) if available.
    3) Fallback: keep points sufficiently above the local minimum z.
    """
    if lidar_pts is None or len(lidar_pts) == 0:
        return np.empty((0, 3), dtype=float)

    # Case 1: building class available
    if lidar_classes is not None:
        building_mask = (lidar_classes == ROOF_CLASS_CODE)
        if np.any(building_mask):
            return lidar_pts[building_mask]

    # Case 2: remove ground class if available
    if lidar_classes is not None:
        non_ground = lidar_pts[lidar_classes != GROUND_CLASS_CODE]
        if len(non_ground) > 0:
            z0 = np.min(lidar_pts[:, 2])
            roof_mask = non_ground[:, 2] >= (z0 + ROOF_MIN_HEIGHT_ABOVE_GROUND)
            roof_pts = non_ground[roof_mask]
            if len(roof_pts) > 0:
                return roof_pts

    # Case 3: fallback based only on height above local minimum
    z0 = np.min(lidar_pts[:, 2])
    roof_mask = lidar_pts[:, 2] >= (z0 + ROOF_MIN_HEIGHT_ABOVE_GROUND)
    return lidar_pts[roof_mask]


def compute_point_to_mesh_rmse(points_xyz, vertices, triangles, max_dist=None):
    """
    Compute LiDAR point-to-roof-surface RMSE.
    """
    if points_xyz is None or len(points_xyz) == 0:
        return None
    if triangles.shape[0] == 0:
        return None

    scene = build_raycast_scene(vertices, triangles)

    d = scene.compute_distance(
        o3d.core.Tensor(points_xyz.astype(np.float32), dtype=o3d.core.Dtype.Float32)
    ).numpy().astype(float)

    if max_dist is not None:
        d = d[np.isfinite(d) & (d <= float(max_dist))]
    else:
        d = d[np.isfinite(d)]

    if d.size == 0:
        return None

    return float(np.sqrt(np.mean(np.square(d))))


def build_centered_lidar_preview_cloud(lidar_pts, x_offset=0.0, z_origin=None):
    if lidar_pts is None or len(lidar_pts) == 0:
        return None

    pts = lidar_pts.copy()
    ctr = pts.mean(axis=0)
    if z_origin is None:
        z_origin = ctr[2]

    pts[:, 0] = pts[:, 0] - ctr[0] + x_offset
    pts[:, 1] = pts[:, 1] - ctr[1]
    pts[:, 2] = pts[:, 2] - z_origin

    zs = lidar_pts[:, 2]
    zmin, zmax = zs.min(), zs.max()
    norm_z = (zs - zmin) / (zmax - zmin + 1e-8)
    cmap = plt.get_cmap("viridis")
    colors = cmap(norm_z)[:, :3]

    pcd = geometry.PointCloud(utility.Vector3dVector(pts))
    pcd.colors = utility.Vector3dVector(colors)
    return pcd

def build_legacy_mesh(vertices, triangles, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def upsert_rmse_row(csv_path, row_dict):
    fieldnames = [
        "basename",
        "n_roof_ref_points",
        "rmse_stage1",
        "rmse_full",
        "better_method",
    ]

    rows = []
    found = False

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("basename", "") == row_dict["basename"]:
                    rows.append(row_dict)
                    found = True
                else:
                    rows.append(row)

    if not found:
        rows.append(row_dict)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_2d_roof_geojson(corners, edges, edge_types, edge_props, idx_to_id, geojson_path):
    valid = ~np.isnan(corners).any(axis=1)

    idx_to_id = dict(idx_to_id or {})
    used_ids = set(idx_to_id.values()) if idx_to_id else set()
    next_id = (max(used_ids) + 1) if used_ids else 0

    for i in range(len(corners)):
        if valid[i] and i not in idx_to_id:
            idx_to_id[i] = next_id
            next_id += 1

    features = []

    for i_e, (e, t, props) in enumerate(zip(edges, edge_types, edge_props)):
        if t != "roof":
            continue

        s_idx, t_idx = int(e[0]), int(e[1])
        if not valid[s_idx] or not valid[t_idx]:
            continue

        p1 = corners[s_idx]
        p2 = corners[t_idx]

        line = LineString([
            (float(p1[0]), float(p1[1])),
            (float(p2[0]), float(p2[1]))
        ])

        out_props = {
            "edge_id": int(props.get("edge_id", i_e)),
            "image_id": str(props.get("image_id", "edited_roof")),
            "source": int(idx_to_id[s_idx]),
            "target": int(idx_to_id[t_idx]),
        }

        features.append({
            "type": "Feature",
            "geometry": mapping(line),
            "properties": out_props
        })

    fc = {
        "type": "FeatureCollection",
        "features": features,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::25832"}
        }
    }

    os.makedirs(os.path.dirname(geojson_path), exist_ok=True)

    tmp_path = geojson_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(fc, f)
        os.replace(tmp_path, geojson_path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    print(f"✅ Updated 2D roof GeoJSON: {geojson_path}")
    return idx_to_id


def _coords_to_ring_indices(coords_xy, comp_nodes, corners):
    if len(coords_xy) >= 2 and coords_xy[0] == coords_xy[-1]:
        coords_xy = coords_xy[:-1]

    comp_nodes = list(sorted(set(int(n) for n in comp_nodes)))
    comp_xy = np.asarray([corners[i, :2] for i in comp_nodes])

    ring = []
    last = None

    for x, y in coords_xy:
        d = np.hypot(comp_xy[:, 0] - x, comp_xy[:, 1] - y)
        k_local = int(np.argmin(d))
        v_idx = comp_nodes[k_local]
        if (last is None) or (v_idx != last):
            ring.append(v_idx)
            last = v_idx

    if len(ring) >= 4 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(ring) >= 2 and ring[0] == ring[-1]:
        ring = ring[:-1]
    if len(set(ring)) < 3:
        return []

    xy = corners[ring, :2]
    s = float(
        np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) -
        np.dot(xy[:, 1], np.roll(xy[:, 0], -1))
    )
    if s < 0.0:
        ring = ring[::-1]

    return ring


def extract_component_loops(corners, edges, comp_edge_idxs, comp_nodes=None):
    if comp_nodes is None:
        comp_nodes = sorted(set([n for ei in comp_edge_idxs for n in edges[ei]]))

    comp_nodes = list(sorted(set(int(n) for n in comp_nodes)))
    if len(comp_nodes) < 3 or len(comp_edge_idxs) == 0:
        return {"exteriors": [], "holes": []}

    segs = []
    for ei in comp_edge_idxs:
        u, v = edges[ei]
        p1 = tuple(corners[u, :2])
        p2 = tuple(corners[v, :2])
        segs.append(LineString([p1, p2]))

    polys = list(polygonize(segs))
    if not polys:
        pts = [tuple(corners[i, :2]) for i in comp_nodes]
        hull = MultiPoint(pts).convex_hull
        if hull.is_empty or hull.geom_type != "Polygon":
            return {"exteriors": [], "holes": []}
        polys = [hull]

    merged = unary_union(polys)
    polys_list = [merged] if merged.geom_type == "Polygon" else (
        list(merged.geoms) if merged.geom_type == "MultiPolygon" else []
    )

    if not polys_list:
        return {"exteriors": [], "holes": []}

    exteriors = []
    holes_all = []

    for poly in polys_list:
        ext_coords = list(poly.exterior.coords)
        ring = _coords_to_ring_indices(ext_coords, comp_nodes, corners)
        if len(ring) < 3:
            continue

        hole_rings = []
        for interior in poly.interiors:
            h_coords = list(interior.coords)
            h_ring = _coords_to_ring_indices(h_coords, comp_nodes, corners)
            if len(h_ring) >= 3:
                xyh = corners[h_ring, :2]
                sh = float(
                    np.dot(xyh[:, 0], np.roll(xyh[:, 1], -1)) -
                    np.dot(xyh[:, 1], np.roll(xyh[:, 0], -1))
                )
                if sh > 0.0:
                    h_ring = h_ring[::-1]
                hole_rings.append(h_ring)

        exteriors.append(ring)
        holes_all.append(hole_rings)

    return {"exteriors": exteriors, "holes": holes_all}


def extract_component_face_polygons(corners, edges, comp_edge_idxs, comp_nodes=None):
    if comp_nodes is None:
        comp_nodes = sorted(set([n for ei in comp_edge_idxs for n in edges[ei]]))

    comp_nodes = list(sorted(set(int(n) for n in comp_nodes)))
    if len(comp_nodes) < 3 or len(comp_edge_idxs) == 0:
        return []

    segs = []
    for ei in comp_edge_idxs:
        u, v = edges[ei]
        p1 = tuple(corners[u, :2])
        p2 = tuple(corners[v, :2])
        segs.append(LineString([p1, p2]))

    polys = list(polygonize(segs))
    if not polys:
        return []

    faces = []
    for poly in polys:
        if poly.is_empty or poly.area <= 0:
            continue

        ring = _coords_to_ring_indices(list(poly.exterior.coords), comp_nodes, corners)
        if len(ring) < 3:
            continue

        faces.append({
            "polygon": poly,
            "ring": ring,
            "area": float(poly.area)
        })

    faces.sort(key=lambda d: d["area"], reverse=True)
    return faces


def mask_points_in_polygon_geom(xy, geom):
    if len(xy) == 0 or geom.is_empty:
        return np.zeros(len(xy), dtype=bool)

    if geom.geom_type == "Polygon":
        ext = np.asarray(geom.exterior.coords)
        mask = Path(ext).contains_points(xy, radius=1e-9)

        for hole in geom.interiors:
            hxy = np.asarray(hole.coords)
            mask &= ~Path(hxy).contains_points(xy, radius=1e-9)

        return mask

    if geom.geom_type == "MultiPolygon":
        out = np.zeros(len(xy), dtype=bool)
        for g in geom.geoms:
            out |= mask_points_in_polygon_geom(xy, g)
        return out

    return np.zeros(len(xy), dtype=bool)


def select_lidar_points_in_buffered_polygon(lidar_pts, poly, buffer_dist):
    geom = poly.buffer(buffer_dist)
    if geom.is_empty:
        return np.empty((0, 3)), geom

    minx, miny, maxx, maxy = geom.bounds
    bbox_mask = (
        (lidar_pts[:, 0] >= minx) & (lidar_pts[:, 0] <= maxx) &
        (lidar_pts[:, 1] >= miny) & (lidar_pts[:, 1] <= maxy)
    )

    cand = lidar_pts[bbox_mask]
    if cand.shape[0] == 0:
        return np.empty((0, 3)), geom

    inside_mask = mask_points_in_polygon_geom(cand[:, :2], geom)
    return cand[inside_mask], geom


def fit_plane_ransac(points_xyz, residual_threshold, min_samples, min_inliers):
    if points_xyz.shape[0] < max(min_samples, 3):
        return None

    X = points_xyz[:, :2]
    y = points_xyz[:, 2]

    model = RANSACRegressor(
        residual_threshold=residual_threshold,
        min_samples=min_samples
    )
    model.fit(X, y)

    if model.inlier_mask_ is None:
        return None

    inliers = points_xyz[model.inlier_mask_]
    if inliers.shape[0] < min_inliers:
        return None

    est = model.estimator_
    a = float(est.coef_[0])
    b = float(est.coef_[1])
    c = float(est.intercept_)

    pred_in = model.predict(inliers[:, :2])
    rmse = float(np.sqrt(np.mean((inliers[:, 2] - pred_in) ** 2)))

    return {
        "a": a,
        "b": b,
        "c": c,
        "inliers": inliers,
        "rmse": rmse,
        "inlier_count": int(inliers.shape[0]),
    }


def fit_planes_for_component_faces(
    corners,
    edges,
    comp_edge_idxs,
    comp_nodes,
    lidar_pts,
    component_id,
    buffer_dist=PLANE_FACE_BUFFER,
    residual_threshold=PLANE_RANSAC_RESIDUAL,
    min_samples=PLANE_RANSAC_MIN_SAMPLES,
    min_points=PLANE_MIN_POINTS,
    min_inliers=PLANE_MIN_INLIERS
):
    face_defs = extract_component_face_polygons(
        corners, edges, comp_edge_idxs, comp_nodes=comp_nodes
    )

    fitted = []

    for face_id, face in enumerate(face_defs):
        poly = face["polygon"]
        ring = face["ring"]

        pts_sel, buffered_geom = select_lidar_points_in_buffered_polygon(
            lidar_pts, poly, buffer_dist
        )

        if pts_sel.shape[0] < min_points:
            print(
                f"⚠️ Comp {component_id} face {face_id}: only {pts_sel.shape[0]} pts "
                f"in buffered face region; skipping plane fit."
            )
            continue

        plane = fit_plane_ransac(
            pts_sel,
            residual_threshold=residual_threshold,
            min_samples=min_samples,
            min_inliers=min_inliers
        )

        if plane is None:
            print(
                f"⚠️ Comp {component_id} face {face_id}: no stable dominant plane found."
            )
            continue

        fitted.append({
            "component_id": int(component_id),
            "face_id": int(face_id),
            "ring": ring[:],
            "polygon": poly,
            "buffered_polygon": buffered_geom,
            "area": float(poly.area),
            "a": plane["a"],
            "b": plane["b"],
            "c": plane["c"],
            "rmse": plane["rmse"],
            "inlier_count": plane["inlier_count"],
            "inlier_points": plane["inliers"],
        })

        print(
            f"✅ Comp {component_id} face {face_id}: plane fit with "
            f"{plane['inlier_count']} inliers, RMSE={plane['rmse']:.3f}"
        )

    return fitted


def plane_z_at_xy(a, b, c, x, y):
    return a * x + b * y + c

def cluster_1d_indices_by_gap(values, gap):
    """
    Cluster 1D values by sorted vertical gap.
    Returns clusters as lists of original indices.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []

    order = np.argsort(values)
    clusters = [[int(order[0])]]

    for pos in range(1, len(order)):
        i_prev = int(order[pos - 1])
        i_curr = int(order[pos])

        if abs(values[i_curr] - values[i_prev]) > gap:
            clusters.append([i_curr])
        else:
            clusters[-1].append(i_curr)

    return clusters


def fit_base_plane_least_squares(xy, z):
    """
    Fit z = ax + by + c from a small set of already-filtered base vertices.
    Returns (a, b, c) or None if not stable enough.
    """
    xy = np.asarray(xy, dtype=float)
    z = np.asarray(z, dtype=float)

    if xy.shape[0] < 3:
        return None

    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(xy.shape[0], dtype=float)])

    # Need full rank to solve a proper plane
    if np.linalg.matrix_rank(A) < 3:
        return None

    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    return float(a), float(b), float(c)


def correct_base_ring_slope_spikes(base_verts, roof_ring_xyz=None):
    """
    Correct isolated base-loop height spikes after the point-cloud based base
    assignment. This preserves gradual ground slope by interpolating a bad
    one- or two-vertex run from its neighbouring base vertices.
    """
    base_verts = np.asarray(base_verts, dtype=float).copy()
    roof_ring_xyz = None if roof_ring_xyz is None else np.asarray(roof_ring_xyz, dtype=float)

    n = base_verts.shape[0]
    if n < 4:
        return base_verts, []

    max_run = int(max(1, min(BASE_SPIKE_MAX_RUN, n - 2)))
    logs = []

    for pass_idx in range(int(max(1, BASE_SPIKE_MAX_PASSES))):
        z = base_verts[:, 2].copy()
        xy = base_verts[:, :2]

        edge_lengths = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
        if not np.all(np.isfinite(edge_lengths)):
            break

        candidates = []

        for run_len in range(max_run, 0, -1):
            for start in range(n):
                run = [int((start + k) % n) for k in range(run_len)]
                before = int((start - 1) % n)
                after = int((start + run_len) % n)

                if before in run or after in run or before == after:
                    continue

                z_before = float(z[before])
                z_after = float(z[after])
                run_z = np.asarray([z[i] for i in run], dtype=float)

                if not np.all(np.isfinite(run_z)) or not np.isfinite(z_before) or not np.isfinite(z_after):
                    continue

                high_spike = np.all(run_z > (max(z_before, z_after) + BASE_SPIKE_LOCAL_MARGIN))
                low_spike = np.all(run_z < (min(z_before, z_after) - BASE_SPIKE_LOCAL_MARGIN))
                if not (high_spike or low_spike):
                    continue

                path_lengths = [
                    float(edge_lengths[(before + k) % n])
                    for k in range(run_len + 1)
                ]
                if not np.all(np.isfinite(path_lengths)):
                    continue

                total_len = float(np.sum(path_lengths))
                if total_len <= 1e-9:
                    continue

                expected = []
                dist_from_before = 0.0
                for k in range(run_len):
                    dist_from_before += path_lengths[k]
                    t = dist_from_before / total_len
                    expected.append(z_before + t * (z_after - z_before))

                expected = np.asarray(expected, dtype=float)
                residuals = run_z - expected

                if high_spike:
                    if float(np.min(residuals)) < BASE_SPIKE_MIN_RESIDUAL:
                        continue
                else:
                    if float(np.min(-residuals)) < BASE_SPIKE_MIN_RESIDUAL:
                        continue

                candidates.append({
                    "run": run,
                    "expected": expected,
                    "residuals": residuals,
                    "score": float(np.min(np.abs(residuals))),
                    "run_len": int(run_len),
                    "pass_idx": int(pass_idx),
                    "spike_type": "high" if high_spike else "low",
                })

        if len(candidates) == 0:
            break

        candidates.sort(key=lambda d: (d["run_len"], d["score"]), reverse=True)
        changed_this_pass = set()

        for cand in candidates:
            if any(i in changed_this_pass for i in cand["run"]):
                continue

            for local_idx, ring_idx in enumerate(cand["run"]):
                old_z = float(base_verts[ring_idx, 2])
                new_z = float(cand["expected"][local_idx])
                if not np.isfinite(new_z) or abs(old_z - new_z) < 1e-9:
                    continue

                roof_z = (
                    float(roof_ring_xyz[ring_idx, 2])
                    if roof_ring_xyz is not None and ring_idx < len(roof_ring_xyz)
                    else float("nan")
                )

                base_verts[ring_idx, 2] = new_z
                changed_this_pass.add(int(ring_idx))

                logs.append({
                    "ring_vertex_order": int(ring_idx),
                    "old_z": old_z,
                    "new_z": new_z,
                    "roof_z": roof_z,
                    "roof_drop_old": float(roof_z - old_z) if np.isfinite(roof_z) else float("nan"),
                    "method": "local_base_slope_interpolation",
                    "reason": {
                        "local_slope_spike": True,
                        "spike_type": cand["spike_type"],
                        "run_len": int(cand["run_len"]),
                        "pass_idx": int(cand["pass_idx"]),
                        "residual": float(cand["residuals"][local_idx]),
                    }
                })

        if len(changed_this_pass) == 0:
            break

    return base_verts, logs


def regularize_base_ring_vertices(base_verts, roof_ring_xyz):
    """
    For one exterior ring:
      - find the dominant low base-height cluster
      - require at least BASE_DOMINANT_FRAC support
      - fit a small base plane from that cluster
      - correct abnormal vertices vertically onto that plane
      - then remove isolated local height spikes along the base loop
    """
    base_verts = np.asarray(base_verts, dtype=float).copy()
    roof_ring_xyz = np.asarray(roof_ring_xyz, dtype=float)

    n = base_verts.shape[0]
    if n < 3:
        return base_verts, []

    base_z = base_verts[:, 2]
    roof_z = roof_ring_xyz[:, 2]

    clusters = cluster_1d_indices_by_gap(base_z, BASE_Z_CLUSTER_GAP)
    if not clusters:
        return correct_base_ring_slope_spikes(base_verts, roof_ring_xyz)

    # choose largest cluster; if tied, prefer the lower one
    def cluster_key(idx_list):
        return (len(idx_list), -float(np.median(base_z[idx_list])))

    dominant = max(clusters, key=cluster_key)
    dominant = sorted(dominant)

    min_needed = int(np.ceil(BASE_DOMINANT_FRAC * n))
    if len(dominant) < min_needed:
        print(
            f"ℹ️ Base regularization skipped: dominant group has only "
            f"{len(dominant)}/{n} vertices (< {min_needed})."
        )
        return correct_base_ring_slope_spikes(base_verts, roof_ring_xyz)

    dom_base_z = base_z[dominant]
    dom_roof_drop = roof_z[dominant] - dom_base_z

    dom_base_median = float(np.median(dom_base_z))
    dom_roof_drop_median = float(np.median(dom_roof_drop))

    plane = fit_base_plane_least_squares(base_verts[dominant, :2], base_verts[dominant, 2])

    correction_logs = []

    dominant_set = set(dominant)

    for i in range(n):
        this_base_z = float(base_verts[i, 2])
        this_roof_z = float(roof_z[i])
        this_roof_drop = this_roof_z - this_base_z

        too_high_vs_dominant = (this_base_z - dom_base_median) > BASE_TOO_HIGH_ABOVE_DOMINANT

        suspicious_near_roof = (
            (this_roof_drop < BASE_NEAR_ROOF_EPS) and
            (dom_roof_drop_median > BASE_DOMINANT_MIN_ROOF_DROP)
        )

        outside_dominant_cluster = (i not in dominant_set)

        if too_high_vs_dominant or suspicious_near_roof or outside_dominant_cluster:
            old_z = float(base_verts[i, 2])

            if plane is not None:
                a, b, c = plane
                new_z = plane_z_at_xy(a, b, c, base_verts[i, 0], base_verts[i, 1])
                method = "dominant_base_plane"
            else:
                new_z = dom_base_median
                method = "dominant_base_median"

            base_verts[i, 2] = float(new_z)

            correction_logs.append({
                "ring_vertex_order": int(i),
                "old_z": old_z,
                "new_z": float(new_z),
                "roof_z": this_roof_z,
                "roof_drop_old": float(this_roof_drop),
                "dominant_group_size": int(len(dominant)),
                "ring_size": int(n),
                "method": method,
                "reason": {
                    "outside_dominant_cluster": bool(outside_dominant_cluster),
                    "too_high_vs_dominant": bool(too_high_vs_dominant),
                    "suspicious_near_roof": bool(suspicious_near_roof),
                }
            })

    base_verts, spike_logs = correct_base_ring_slope_spikes(base_verts, roof_ring_xyz)
    correction_logs.extend(spike_logs)

    return base_verts, correction_logs

def build_plane_mesh_from_polygon(poly, a, b, c, origin):
    tri_polys = triangulate(poly)
    vertices = []
    triangles = []

    for tri in tri_polys:
        rep = tri.representative_point()
        if not poly.buffer(1e-9).covers(rep):
            continue

        coords = list(tri.exterior.coords)[:-1]
        if len(coords) != 3:
            continue

        base_idx = len(vertices)
        for x, y in coords:
            z = plane_z_at_xy(a, b, c, x, y)
            vertices.append([x - origin[0], y - origin[1], z - origin[2]])

        triangles.append([base_idx, base_idx + 1, base_idx + 2])

    if len(triangles) == 0:
        return None

    mesh = geometry.TriangleMesh()
    mesh.vertices = utility.Vector3dVector(np.asarray(vertices, dtype=float))
    mesh.triangles = utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.paint_uniform_color(PLANE_MESH_COLOR)
    mesh.compute_vertex_normals()
    return mesh


def build_plane_outline_lineset(poly, a, b, c, origin):
    xy = np.asarray(poly.exterior.coords[:-1], dtype=float)
    if xy.shape[0] < 3:
        return None

    pts3d = []
    for x, y in xy:
        z = plane_z_at_xy(a, b, c, x, y)
        pts3d.append([x - origin[0], y - origin[1], z - origin[2]])

    idxs = [[i, (i + 1) % len(pts3d)] for i in range(len(pts3d))]

    ls = geometry.LineSet(
        points=utility.Vector3dVector(np.asarray(pts3d, dtype=float)),
        lines=utility.Vector2iVector(np.asarray(idxs, dtype=np.int32))
    )
    ls.colors = utility.Vector3dVector([PLANE_OUTLINE_COLOR] * len(idxs))
    return ls

def surface_color_from_type(surface_type):
    if surface_type == "roof":
        return ROOF_FACE_COLOR
    if surface_type == "wall":
        return WALL_FACE_COLOR
    if surface_type == "base":
        return BASE_FACE_COLOR
    if surface_type == "roof_seam":
        return ROOF_SEAM_FACE_COLOR
    return [0.85, 0.85, 0.85]


def build_surface_mesh_from_ring(corners, vertex_indices, origin, color, x_shift=None):
    """
    Build a mesh from an ordered polygon ring using polygon-respecting
    triangulation. This avoids fan-triangulation artifacts for concave faces.
    """
    ring = dedupe_ring_vertex_indices(vertex_indices)
    if len(ring) < 3:
        return None

    verts = np.asarray([corners[v] for v in ring], dtype=float)

    # drop repeated closing vertex if present
    if len(verts) >= 2 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]

    if verts.shape[0] < 3:
        return None

    if x_shift is None:
        x_shift = np.zeros(3, dtype=float)

    verts_tri, tris = triangulate_planar_ring_3d(verts)
    if tris.shape[0] == 0:
        return None

    verts_disp = (verts_tri - origin) + x_shift
    mesh = geometry.TriangleMesh()
    mesh.vertices = utility.Vector3dVector(np.asarray(verts_disp, dtype=float))
    mesh.triangles = utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def build_wall_quad_mesh_from_ring(corners, vertex_indices, origin, color, x_shift=None):
    """
    Build a wall mesh directly from the 4 boundary vertices.

    Wall faces are already created as:
        [r1, r2, b2, b1]

    Do not use Shapely triangulation for walls, because vertical / slightly
    non-planar wall quads can project badly and produce triangles outside
    the intended edge boundary.
    """
    ring = dedupe_ring_vertex_indices(vertex_indices)

    if len(ring) != 4:
        return build_surface_mesh_from_ring(
            corners=corners,
            vertex_indices=ring,
            origin=origin,
            color=color,
            x_shift=x_shift
        )

    verts = np.asarray([corners[int(v)] for v in ring], dtype=float)

    if not np.all(np.isfinite(verts)):
        return None

    if x_shift is None:
        x_shift = np.zeros(3, dtype=float)

    verts_disp = (verts - origin) + x_shift

    triangles = np.asarray([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    mesh = geometry.TriangleMesh()
    mesh.vertices = utility.Vector3dVector(verts_disp.astype(float))
    mesh.triangles = utility.Vector3iVector(triangles)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return mesh

def dedupe_ring_vertex_indices(vertex_indices):
    ring = [int(v) for v in vertex_indices]
    if len(ring) == 0:
        return []

    deduped = [ring[0]]
    for vi in ring[1:]:
        if vi != deduped[-1]:
            deduped.append(vi)

    if len(deduped) >= 2 and deduped[0] == deduped[-1]:
        deduped = deduped[:-1]

    return deduped

def remap_exterior_ring_to_lifted_roof_vertices(
    ring,
    corners,
    edges,
    edge_types,
    xy_tol=1e-7
):
    """
    When Stage-II vertex splitting creates multiple roof vertices at the same XY,
    exterior loop extraction may return the old/lower vertex.

    This remaps each exterior-ring vertex to the best matching final roof vertex:
      1) same XY location,
      2) connected to neighbouring ring vertices by roof edges if possible,
      3) highest Z among valid candidates.

    This makes wall surfaces use the lifted roof edge, without creating roof seams
    on exterior/base loops.
    """
    ring = [int(v) for v in ring]
    if len(ring) < 3:
        return ring

    valid = ~np.isnan(corners).any(axis=1)

    roof_edge_set = set()
    for e, t in zip(edges, edge_types):
        if t != "roof":
            continue
        u, v = int(e[0]), int(e[1])
        if valid[u] and valid[v] and u != v:
            roof_edge_set.add(tuple(sorted((u, v))))

    def same_xy_candidates(old_idx):
        xy = corners[int(old_idx), :2]
        d = np.linalg.norm(corners[:, :2] - xy[None, :], axis=1)
        cand = np.where(valid & (d <= xy_tol))[0].astype(int).tolist()

        if len(cand) == 0:
            return [int(old_idx)]

        return cand

    corrected_ring = []

    k = len(ring)

    for i, old_idx in enumerate(ring):
        prev_old = ring[(i - 1) % k]
        next_old = ring[(i + 1) % k]

        cand_this = same_xy_candidates(old_idx)
        cand_prev = same_xy_candidates(prev_old)
        cand_next = same_xy_candidates(next_old)

        scored = []

        for c in cand_this:
            connected_prev = any(
                tuple(sorted((int(c), int(p)))) in roof_edge_set
                for p in cand_prev
                if int(p) != int(c)
            )

            connected_next = any(
                tuple(sorted((int(c), int(n)))) in roof_edge_set
                for n in cand_next
                if int(n) != int(c)
            )

            connectivity_score = int(connected_prev) + int(connected_next)
            z_score = float(corners[int(c), 2])

            scored.append((connectivity_score, z_score, int(c)))

        scored.sort(reverse=True)
        chosen = scored[0][2]
        corrected_ring.append(int(chosen))

    corrected_ring = dedupe_ring_vertex_indices(corrected_ring)

    if len(corrected_ring) < 3:
        return ring

    return corrected_ring


def find_roof_edge_matching_original_xy(edge_old, corners, edges, edge_types, xy_tol=1e-7):
    """
    Find the final roof edge whose XY endpoints match one original footprint
    edge, preserving the original edge direction.

    This is safer for walls than choosing one remapped vertex per footprint
    corner: after Stage-II splitting, the same XY corner can have multiple roof
    vertices at different Z values, one per adjacent roof plane.
    """
    old_u, old_v = int(edge_old[0]), int(edge_old[1])
    if old_u < 0 or old_v < 0 or old_u >= len(corners) or old_v >= len(corners):
        return None

    xy_u = np.asarray(corners[old_u, :2], dtype=float)
    xy_v = np.asarray(corners[old_v, :2], dtype=float)
    if not np.all(np.isfinite(xy_u)) or not np.all(np.isfinite(xy_v)):
        return None

    scale = max(1.0, float(np.linalg.norm(xy_v - xy_u)))
    tol = float(xy_tol) * scale

    best = None

    for e, t in zip(edges, edge_types):
        if t != "roof":
            continue

        a, b = int(e[0]), int(e[1])
        if a < 0 or b < 0 or a >= len(corners) or b >= len(corners) or a == b:
            continue

        xy_a = np.asarray(corners[a, :2], dtype=float)
        xy_b = np.asarray(corners[b, :2], dtype=float)
        if not np.all(np.isfinite(xy_a)) or not np.all(np.isfinite(xy_b)):
            continue

        err_forward = max(
            float(np.linalg.norm(xy_a - xy_u)),
            float(np.linalg.norm(xy_b - xy_v))
        )
        err_reverse = max(
            float(np.linalg.norm(xy_a - xy_v)),
            float(np.linalg.norm(xy_b - xy_u))
        )

        if err_forward <= tol:
            oriented = (a, b)
            err = err_forward
        elif err_reverse <= tol:
            oriented = (b, a)
            err = err_reverse
        else:
            continue

        z1 = float(corners[oriented[0], 2])
        z2 = float(corners[oriented[1], 2])
        if not np.isfinite(z1) or not np.isfinite(z2):
            continue

        score = (-err, min(z1, z2), 0.5 * (z1 + z2), max(z1, z2))
        if best is None or score > best[0]:
            best = (score, oriented)

    if best is None:
        return None

    return best[1]


def polygon_from_ring_indices(corners, ring):
    ring = dedupe_ring_vertex_indices(ring)
    if len(ring) < 3:
        return None

    coords = []
    for vi in ring:
        p = corners[int(vi)]
        if not np.all(np.isfinite(p[:2])):
            return None
        coords.append((float(p[0]), float(p[1])))

    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)

    if poly.is_empty:
        return None

    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)

    if poly.geom_type != "Polygon" or poly.area <= 1e-9:
        return None

    return poly


def split_nested_loop_seed_jobs(corners, loop_jobs):
    base_jobs = []
    enriched = []
    for job in loop_jobs:
        poly = polygon_from_ring_indices(corners, job.get("ring", []))
        if poly is None:
            base_jobs.append(dict(job))
            continue
        enriched.append({
            "job": dict(job),
            "polygon": poly,
            "area": float(poly.area),
            "point": poly.representative_point(),
        })

    inner_jobs = []

    for i, item in enumerate(enriched):
        is_inner = False
        for j, other in enumerate(enriched):
            if i == j:
                continue
            if other["area"] <= item["area"] + 1e-9:
                continue
            if other["polygon"].buffer(1e-7).covers(item["point"]):
                is_inner = True
                break

        if is_inner:
            inner_jobs.append(item["job"])
        else:
            base_jobs.append(item["job"])

    return base_jobs, inner_jobs


def roof_surface_z_at_xy(corners, surface_face, x, y):
    if "plane_a" in surface_face and "plane_b" in surface_face and "plane_c" in surface_face:
        return plane_z_at_xy(
            float(surface_face["plane_a"]),
            float(surface_face["plane_b"]),
            float(surface_face["plane_c"]),
            float(x),
            float(y)
        )

    ring = dedupe_ring_vertex_indices(surface_face.get("vertex_indices", []))
    if len(ring) < 3:
        return None

    verts = np.asarray([corners[int(v)] for v in ring], dtype=float)
    plane = fit_base_plane_least_squares(verts[:, :2], verts[:, 2])
    if plane is None:
        return float(np.median(verts[:, 2]))

    a, b, c = plane
    return plane_z_at_xy(a, b, c, float(x), float(y))


def find_containing_roof_surface_z(corners, surface_faces, x, y, inner_poly=None, inner_ring=None):
    pt = Point(float(x), float(y))
    inner_area = 0.0 if inner_poly is None else float(inner_poly.area)
    inner_set = set(int(v) for v in (inner_ring or []))

    candidates = []
    for sf in surface_faces or []:
        if sf.get("surface_type") != "roof":
            continue

        ring = dedupe_ring_vertex_indices(sf.get("vertex_indices", []))
        if len(ring) < 3:
            continue

        ring_set = set(int(v) for v in ring)
        if len(inner_set) > 0 and ring_set.issubset(inner_set):
            continue

        poly = polygon_from_ring_indices(corners, ring)
        if poly is None:
            continue

        if inner_poly is not None and poly.area <= inner_area + 1e-9:
            continue

        if not poly.buffer(1e-7).covers(pt):
            continue

        z = roof_surface_z_at_xy(corners, sf, x, y)
        if z is None or not np.isfinite(z):
            continue

        candidates.append((float(poly.area), float(z)))

    if len(candidates) == 0:
        return None

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def add_inner_roof_loop_seams(
    corners,
    surface_faces,
    inner_loop_jobs,
    edges,
    edge_types,
    xy_tol=1e-7
):
    corners = np.asarray(corners, dtype=float)
    added_count = 0

    for job in inner_loop_jobs:
        original_ring = [int(v) for v in job.get("ring", [])]
        if len(original_ring) < 3:
            continue

        ring = remap_exterior_ring_to_lifted_roof_vertices(
            ring=original_ring,
            corners=corners,
            edges=edges,
            edge_types=edge_types,
            xy_tol=xy_tol
        )

        if len(ring) < 3:
            continue

        inner_poly = polygon_from_ring_indices(corners, ring)
        if inner_poly is None:
            inner_poly = polygon_from_ring_indices(corners, original_ring)

        support_coords = []
        for roof_idx in ring:
            x, y = corners[int(roof_idx), :2]
            z_support = find_containing_roof_surface_z(
                corners,
                surface_faces,
                x,
                y,
                inner_poly=inner_poly,
                inner_ring=ring
            )

            if z_support is None:
                support_coords = []
                break

            support_coords.append([float(x), float(y), float(z_support)])

        if len(support_coords) != len(ring):
            continue

        support_indices = []
        for coord in support_coords:
            corners = np.vstack([corners, np.asarray([coord], dtype=float)])
            support_indices.append(int(len(corners) - 1))

        k = len(ring)
        use_original_edge_lookup = len(original_ring) == len(ring)

        for i in range(k):
            if use_original_edge_lookup:
                old_edge = (original_ring[i], original_ring[(i + 1) % k])
            else:
                old_edge = (ring[i], ring[(i + 1) % k])

            matched_edge = find_roof_edge_matching_original_xy(
                old_edge,
                corners,
                edges,
                edge_types,
                xy_tol=xy_tol
            )

            if matched_edge is None:
                r1 = int(ring[i])
                r2 = int(ring[(i + 1) % k])
            else:
                r1, r2 = int(matched_edge[0]), int(matched_edge[1])

            b1 = int(support_indices[i])
            b2 = int(support_indices[(i + 1) % k])

            if len({r1, r2, b1, b2}) < 3:
                continue

            if max(
                abs(float(corners[r1, 2] - corners[b1, 2])),
                abs(float(corners[r2, 2] - corners[b2, 2]))
            ) < 1e-6:
                continue

            surface_faces.append({
                "surface_type": "roof_seam",
                "component_id": int(job.get("component_id", -1)),
                "loop_id": int(job.get("loop_id", -1)),
                "ring_order": int(i),
                "inner_roof_loop": True,
                "vertex_indices": [int(r1), int(r2), int(b2), int(b1)],
            })
            added_count += 1

    return corners, added_count


def triangulate_planar_ring_3d(coords3d):
    arr = np.asarray(coords3d, dtype=float)
    if arr.shape[0] < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]

    if arr.shape[0] < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    centroid = arr.mean(axis=0)
    centered = arr - centroid

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis_u = vh[0]
        axis_v = vh[1]
    except np.linalg.LinAlgError:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    if np.linalg.norm(axis_u) < 1e-12 or np.linalg.norm(axis_v) < 1e-12:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)

    uv = np.column_stack([
        centered @ axis_u,
        centered @ axis_v,
    ])

    poly_2d = Polygon(uv)
    if not poly_2d.is_valid:
        poly_2d = poly_2d.buffer(0)

    if poly_2d.is_empty:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    if poly_2d.geom_type == "MultiPolygon":
        poly_2d = max(poly_2d.geoms, key=lambda g: g.area)

    if poly_2d.area <= 1e-12:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    scale = max(1.0, float(np.max(np.ptp(uv, axis=0))))
    lookup_tol = 1e-7 * scale

    vertices = []
    triangles_idx = []

    for tri in triangulate(poly_2d):
        if not poly_2d.buffer(1e-9 * scale).covers(tri):
            continue

        tri_uv = np.asarray(tri.exterior.coords[:-1], dtype=float)
        if tri_uv.shape != (3, 2):
            continue

        base_idx = len(vertices)
        for x, y in tri_uv:
            d2 = np.sum((uv - np.array([x, y], dtype=float)) ** 2, axis=1)
            nearest_idx = int(np.argmin(d2))
            if float(np.sqrt(d2[nearest_idx])) <= lookup_tol:
                xyz = arr[nearest_idx]
            else:
                xyz = centroid + (x * axis_u) + (y * axis_v)
            vertices.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])

        triangles_idx.append([base_idx, base_idx + 1, base_idx + 2])

    if len(triangles_idx) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32)

    return np.asarray(vertices, dtype=float), np.asarray(triangles_idx, dtype=np.int32)


def build_lineset_from_segments(segments_xyz, origin, color, x_shift=None):
    if segments_xyz is None or len(segments_xyz) == 0:
        return None

    if x_shift is None:
        x_shift = np.zeros(3, dtype=float)

    pts = []
    idxs = []
    cols = []

    for seg in segments_xyz:
        arr = np.asarray(seg, dtype=float)
        if arr.shape != (2, 3) or not np.all(np.isfinite(arr)):
            continue

        p1 = (arr[0] - origin) + x_shift
        p2 = (arr[1] - origin) + x_shift
        pts.append(p1)
        pts.append(p2)
        idxs.append([len(pts) - 2, len(pts) - 1])
        cols.append(color)

    if len(pts) == 0:
        return None

    ls = geometry.LineSet(
        points=utility.Vector3dVector(np.asarray(pts, dtype=float)),
        lines=utility.Vector2iVector(np.asarray(idxs, dtype=np.int32))
    )
    ls.colors = utility.Vector3dVector(np.asarray(cols, dtype=float))
    return ls


def estimate_xy_point_spacing(lidar_pts, max_samples=5000):
    """
    Estimate LiDAR spacing in the XY plane using median nearest-neighbor distance.
    """
    if lidar_pts is None or lidar_pts.shape[0] < 2:
        return THRESHOLD_HARD_DEFAULT

    xy = lidar_pts[:, :2]
    n = xy.shape[0]

    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        xy = xy[idx]

    nbrs = NearestNeighbors(n_neighbors=2)
    nbrs.fit(xy)
    dists, _ = nbrs.kneighbors(xy)

    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]

    if nn.size == 0:
        return THRESHOLD_HARD_DEFAULT

    return float(np.median(nn))


def cluster_plane_predictions_by_gap(items, split_gap):
    """
    Cluster sorted z predictions. A new cluster starts when the vertical gap
    exceeds split_gap.
    """
    if len(items) == 0:
        return []

    items_sorted = sorted(items, key=lambda d: d["z_pred"])
    clusters = [[items_sorted[0]]]

    for it in items_sorted[1:]:
        prev = clusters[-1][-1]
        if abs(it["z_pred"] - prev["z_pred"]) > split_gap:
            clusters.append([it])
        else:
            clusters[-1].append(it)

    return clusters


def apply_face_plane_vertex_splitting(
    corners,
    edges,
    edge_types,
    edge_props,
    idx_to_id,
    face_planes,
    lidar_pts,
    comp_id_by_vertex,
    split_gap=None,
    exterior_roof_edge_pairs=None
):
    """
    Assign roof vertex heights from fitted face planes.

    If a vertex is shared by multiple fitted face planes whose predicted Z
    values differ beyond the split threshold, split that vertex into one copy
    per plane-group, rebuild roof edges from the face loops, and additionally
    create roof-face polygons plus seam polygons where one original roof edge
    is duplicated into multiple 3D edges.

    Returns:
        new_corners, new_edges, new_edge_types, new_edge_props, new_idx_to_id,
        correction_records, rebuilt_roof_faces, seam_faces,
        pre_adjustment_roof_edge_segments, split_gap_used
    """
    if len(face_planes) == 0:
        return (
            corners, edges, edge_types, edge_props, idx_to_id,
            [], [], [], [], THRESHOLD_HARD_DEFAULT if split_gap is None else float(split_gap)
        )

    if split_gap is None:
        xy_spacing = estimate_xy_point_spacing(lidar_pts)
        split_gap = float(PLANE_VERTEX_SPLIT_GAP_FACTOR * xy_spacing)
    else:
        split_gap = float(split_gap)

    exterior_roof_edge_pairs = set(exterior_roof_edge_pairs or [])

    valid = ~np.isnan(corners).any(axis=1)

    # Collect plane-based Z proposals per original vertex
    proposals = defaultdict(list)
    for fp in face_planes:
        face_key = (int(fp["component_id"]), int(fp["face_id"]))
        a, b, c = fp["a"], fp["b"], fp["c"]

        for vi in set(int(v) for v in fp["ring"]):
            if vi < 0 or vi >= len(corners):
                continue
            if not valid[vi]:
                continue

            x, y = corners[vi, :2]
            z_pred = plane_z_at_xy(a, b, c, x, y)

            proposals[vi].append({
                "face_key": face_key,
                "z_pred": float(z_pred)
            })

    new_corners = corners.copy()
    new_idx_to_id = dict(idx_to_id or {})
    next_id = (max(new_idx_to_id.values()) + 1) if len(new_idx_to_id) > 0 else 0

    face_vertex_to_new_idx = {}
    correction_records = []

    # Assign / split vertices
    for vi, items in proposals.items():
        z_old = float(new_corners[vi, 2])

        clusters = cluster_plane_predictions_by_gap(items, split_gap)
        cluster_infos = []

        for cluster in clusters:
            z_vals = np.array([it["z_pred"] for it in cluster], dtype=float)
            z_new = float(np.median(z_vals))
            cluster_infos.append({
                "items": cluster,
                "z_new": z_new,
                "support": int(len(cluster))
            })

        # Keep the original vertex index for the cluster closest to old Z
        primary_idx = int(np.argmin([abs(ci["z_new"] - z_old) for ci in cluster_infos]))

        for ci_idx, ci in enumerate(cluster_infos):
            if ci_idx == primary_idx:
                new_vi = int(vi)
                new_corners[new_vi, 2] = ci["z_new"]
            else:
                xyz = new_corners[vi].copy()
                xyz[2] = ci["z_new"]
                new_corners = np.vstack([new_corners, xyz[None, :]])
                new_vi = int(len(new_corners) - 1)
                new_idx_to_id[new_vi] = next_id
                next_id += 1

            for it in ci["items"]:
                face_vertex_to_new_idx[(it["face_key"], int(vi))] = new_vi

            correction_records.append({
                "original_vertex_idx": int(vi),
                "new_vertex_idx": int(new_vi),
                "old_z": z_old,
                "new_z": float(ci["z_new"]),
                "support": int(ci["support"]),
                "cluster_count": int(len(cluster_infos)),
                "split_gap_used": float(split_gap),
                "source_faces": [it["face_key"] for it in ci["items"]]
            })

    # Original roof-edge props lookup
    orig_roof_prop_by_pair = {}
    for e, t, p in zip(edges, edge_types, edge_props):
        if t != "roof":
            continue
        s, v = int(e[0]), int(e[1])
        orig_roof_prop_by_pair[frozenset((s, v))] = dict(p)

    components_with_planes = set(int(fp["component_id"]) for fp in face_planes)

    face_records = []
    source_pair_to_new_edges_initial = defaultdict(list)

    # Prepare roof-face loops from fitted planes before edge-level seam correction.
    for fp in face_planes:
        face_key = (int(fp["component_id"]), int(fp["face_id"]))
        ring_old = [int(v) for v in fp["ring"]]
        ring_new_raw = [int(face_vertex_to_new_idx.get((face_key, vi), vi)) for vi in ring_old]

        if len(ring_old) < 3 or len(ring_new_raw) < 3:
            continue

        face_records.append({
            "face_key": face_key,
            "component_id": int(fp["component_id"]),
            "face_id": int(fp["face_id"]),
            "plane_a": float(fp["a"]),
            "plane_b": float(fp["b"]),
            "plane_c": float(fp["c"]),
            "ring_old": ring_old,
            "ring_new_raw": ring_new_raw,
        })

        k = len(ring_new_raw)
        for i in range(k):
            u_old = int(ring_old[i])
            v_old = int(ring_old[(i + 1) % k])
            u = int(ring_new_raw[i])
            v = int(ring_new_raw[(i + 1) % k])
            if u == v:
                continue

            source_pair = frozenset((u_old, v_old))
            source_pair_to_new_edges_initial[source_pair].append({
                "old_u": u_old,
                "old_v": v_old,
                "new_u": u,
                "new_v": v,
            })

    # Small-gap duplicated roof edges should become truly shared edges, not just
    # separate vertices placed at similar heights. Merge those duplicate
    # endpoints first, then rebuild roof faces/edges from the corrected topology.
    merge_parent = {}

    def _uf_find(x):
        x = int(x)
        if x not in merge_parent:
            merge_parent[x] = x
            return x
        while merge_parent[x] != x:
            merge_parent[x] = merge_parent[merge_parent[x]]
            x = merge_parent[x]
        return x

    def _uf_union(a, b):
        ra = _uf_find(a)
        rb = _uf_find(b)
        if ra != rb:
            merge_parent[rb] = ra

    merge_old_vertex_to_ids = defaultdict(set)
    pre_adjustment_roof_edge_segments = []

    seen_pre_adjustment_edges = set()
    for fr in face_records:
        ring_new_raw = [int(v) for v in fr["ring_new_raw"]]
        k = len(ring_new_raw)
        for i in range(k):
            u = int(ring_new_raw[i])
            v = int(ring_new_raw[(i + 1) % k])
            if u == v:
                continue

            key = tuple(sorted((u, v)))
            if key in seen_pre_adjustment_edges:
                continue
            seen_pre_adjustment_edges.add(key)

            pre_adjustment_roof_edge_segments.append(np.asarray([
                new_corners[u].copy(),
                new_corners[v].copy(),
            ], dtype=float))

    for source_pair, edge_list in source_pair_to_new_edges_initial.items():
        unique_edges = []
        seen_exact = set()
        for e in edge_list:
            et = (int(e["old_u"]), int(e["old_v"]), int(e["new_u"]), int(e["new_v"]))
            if et not in seen_exact:
                seen_exact.add(et)
                unique_edges.append(et)

        if len(unique_edges) < 2:
            continue

        for i_ref in range(len(unique_edges) - 1):
            ref_old_u, ref_old_v, ref_new_u, ref_new_v = unique_edges[i_ref]
            ref_by_old = {
                int(ref_old_u): int(ref_new_u),
                int(ref_old_v): int(ref_new_v),
            }

            for i_other in range(i_ref + 1, len(unique_edges)):
                other_old_u, other_old_v, other_new_u, other_new_v = unique_edges[i_other]
                other_by_old = {
                    int(other_old_u): int(other_new_u),
                    int(other_old_v): int(other_new_v),
                }

                if any(int(old_vi) not in ref_by_old or int(old_vi) not in other_by_old for old_vi in source_pair):
                    continue

                endpoint_gaps = []
                for old_vi in source_pair:
                    r_idx = int(ref_by_old[int(old_vi)])
                    o_idx = int(other_by_old[int(old_vi)])
                    z_pair = np.asarray([
                        new_corners[r_idx, 2],
                        new_corners[o_idx, 2],
                    ], dtype=float)

                    if not np.all(np.isfinite(z_pair)):
                        endpoint_gaps = []
                        break

                    endpoint_gaps.append(float(abs(z_pair[0] - z_pair[1])))

                if len(endpoint_gaps) != 2:
                    continue

                if max(endpoint_gaps) > ROOF_SEAM_MIN_VERTICAL_GAP:
                    continue

                for old_vi in source_pair:
                    r_idx = int(ref_by_old[int(old_vi)])
                    o_idx = int(other_by_old[int(old_vi)])
                    merge_old_vertex_to_ids[int(old_vi)].update([r_idx, o_idx])
                    _uf_find(r_idx)
                    _uf_find(o_idx)
                    _uf_union(r_idx, o_idx)

    merged_vertex_remap = {}
    for old_vi, ids in merge_old_vertex_to_ids.items():
        grouped_members = defaultdict(list)
        for ni in sorted(ids):
            grouped_members[_uf_find(int(ni))].append(int(ni))

        for members in grouped_members.values():
            uniq_members = sorted(set(int(ni) for ni in members))
            if len(uniq_members) == 0:
                continue

            canonical = int(old_vi) if int(old_vi) in uniq_members else int(uniq_members[0])
            z_mid = float(np.mean([new_corners[int(ni), 2] for ni in uniq_members]))

            for ni in uniq_members:
                new_corners[int(ni), 2] = z_mid
                merged_vertex_remap[int(ni)] = canonical

            new_corners[canonical, 2] = z_mid
            merged_vertex_remap[canonical] = canonical

    rebuilt_roof_edges = []
    rebuilt_roof_types = []
    rebuilt_roof_props = []
    rebuilt_roof_faces = []
    seen_pairs = set()

    # For seam creation: which new 3D edge(s) came from each original roof edge?
    source_pair_to_new_edges = defaultdict(list)

    # Rebuild roof edges + roof faces from corrected fitted-face loops.
    for fr in face_records:
        ring_old = [int(v) for v in fr["ring_old"]]
        ring_new_raw = [int(merged_vertex_remap.get(v, v)) for v in fr["ring_new_raw"]]
        ring_new_face = dedupe_ring_vertex_indices(ring_new_raw)

        if len(ring_new_face) < 3:
            continue

        rebuilt_roof_faces.append({
            "surface_type": "roof",
            "component_id": int(fr["component_id"]),
            "face_id": int(fr["face_id"]),
            "vertex_indices": [int(v) for v in ring_new_face],
            "plane_a": float(fr["plane_a"]),
            "plane_b": float(fr["plane_b"]),
            "plane_c": float(fr["plane_c"]),
        })

        k = len(ring_new_raw)
        for i in range(k):
            u_old = int(ring_old[i])
            v_old = int(ring_old[(i + 1) % k])
            u = int(ring_new_raw[i])
            v = int(ring_new_raw[(i + 1) % k])

            if u == v:
                continue

            source_pair = frozenset((u_old, v_old))
            source_pair_to_new_edges[source_pair].append({
                "old_u": u_old,
                "old_v": v_old,
                "new_u": u,
                "new_v": v,
            })

            key = tuple(sorted((u, v)))
            if key in seen_pairs:
                continue

            props = dict(orig_roof_prop_by_pair.get(source_pair, {}))
            if "image_id" not in props:
                props["image_id"] = "plane_split_roof"

            rebuilt_roof_edges.append([u, v])
            rebuilt_roof_types.append("roof")
            rebuilt_roof_props.append(props)
            seen_pairs.add(key)

    # Keep original roof edges for components where no fitted plane exists
    for e, t, p in zip(edges, edge_types, edge_props):
        if t != "roof":
            continue

        s, v = int(e[0]), int(e[1])
        cid_s = comp_id_by_vertex.get(s, None)
        cid_v = comp_id_by_vertex.get(v, None)
        cid = cid_s if cid_s is not None else cid_v

        if cid in components_with_planes:
            continue

        key = tuple(sorted((s, v)))
        if key in seen_pairs:
            continue

        rebuilt_roof_edges.append([s, v])
        rebuilt_roof_types.append("roof")
        rebuilt_roof_props.append(dict(p))
        seen_pairs.add(key)

    # Create seam polygons where one original roof edge has been duplicated
    seam_faces = []

    def _append_roof_seam_between_edges(source_pair, ref, other):
        nonlocal new_corners, next_id

        ref_old_u, ref_old_v, a, b = ref
        other_old_u, other_old_v, other_new_u, other_new_v = other
        other_by_old = {
            int(other_old_u): int(other_new_u),
            int(other_old_v): int(other_new_v),
        }

        if int(ref_old_u) not in other_by_old or int(ref_old_v) not in other_by_old:
            return

        c = other_by_old[int(ref_old_u)]
        d = other_by_old[int(ref_old_v)]

        # Avoid degenerate seam polygons.
        if len({a, b, c, d}) < 3:
            return

        za = float(new_corners[int(a), 2])
        zb = float(new_corners[int(b), 2])
        zc = float(new_corners[int(c), 2])
        zd = float(new_corners[int(d), 2])
        gap_u = za - zc
        gap_v = zb - zd

        # If the two duplicated roof edges cross between their endpoints, one
        # quad becomes a bow-tie in vertical profile. Split it at the crossing
        # point so both opposite seam triangles are generated.
        if gap_u * gap_v < -1e-10:
            t_cross = -gap_u / (gap_v - gap_u)
            if 1e-6 < t_cross < (1.0 - 1e-6):
                p_ref = (
                    new_corners[int(a)] +
                    t_cross * (new_corners[int(b)] - new_corners[int(a)])
                )
                p_other = (
                    new_corners[int(c)] +
                    t_cross * (new_corners[int(d)] - new_corners[int(c)])
                )
                p_cross = 0.5 * (p_ref + p_other)

                new_corners = np.vstack([new_corners, p_cross[None, :]])
                cross_idx = int(len(new_corners) - 1)
                new_idx_to_id[cross_idx] = next_id
                next_id += 1

                seam_faces.append({
                    "surface_type": "roof_seam",
                    "source_edge_old_indices": [int(v) for v in source_pair],
                    "vertex_indices": [int(a), int(cross_idx), int(c)],
                })
                seam_faces.append({
                    "surface_type": "roof_seam",
                    "source_edge_old_indices": [int(v) for v in source_pair],
                    "vertex_indices": [int(cross_idx), int(b), int(d)],
                })
                return

        seam_faces.append({
            "surface_type": "roof_seam",
            "source_edge_old_indices": [int(v) for v in source_pair],
            "vertex_indices": [int(a), int(b), int(d), int(c)],
        })

    for source_pair, edge_list in source_pair_to_new_edges.items():
        if source_pair in exterior_roof_edge_pairs:
            continue

        unique_edges = []
        seen_exact = set()
        for e in edge_list:
            et = (int(e["old_u"]), int(e["old_v"]), int(e["new_u"]), int(e["new_v"]))
            if et not in seen_exact:
                seen_exact.add(et)
                unique_edges.append(et)

        if len(unique_edges) < 2:
            continue

        ref = unique_edges[0]
        for other in unique_edges[1:]:
            _append_roof_seam_between_edges(source_pair, ref, other)

    # Keep non-roof edges untouched
    other_edges = []
    other_types = []
    other_props = []
    for e, t, p in zip(edges, edge_types, edge_props):
        if t == "roof":
            continue
        other_edges.append([int(e[0]), int(e[1])])
        other_types.append(t)
        other_props.append(dict(p))

    # Reassign edge_id cleanly
    for i, p in enumerate(rebuilt_roof_props):
        p["edge_id"] = int(i)

    new_edges = rebuilt_roof_edges + other_edges
    new_edge_types = rebuilt_roof_types + other_types
    new_edge_props = rebuilt_roof_props + other_props

    return (
        new_corners,
        new_edges,
        new_edge_types,
        new_edge_props,
        new_idx_to_id,
        correction_records,
        rebuilt_roof_faces,
        seam_faces,
        pre_adjustment_roof_edge_segments,
        split_gap
    )

def get_next_snapshot_path(snapshot_dir, basename):
    os.makedirs(snapshot_dir, exist_ok=True)

    prefix = f"{basename}_snapshot_"
    existing = []

    for fn in os.listdir(snapshot_dir):
        if fn.startswith(prefix) and fn.lower().endswith(".png"):
            stem = os.path.splitext(fn)[0]
            tail = stem.replace(prefix, "")
            if tail.isdigit():
                existing.append(int(tail))

    next_idx = 1 if not existing else (max(existing) + 1)
    return os.path.join(snapshot_dir, f"{prefix}{next_idx:03d}.png")

class FileSession:
    def __init__(self, geojson_path, laz_path, geotiff_path, output_path, current_num=None, total_num=None):
        self.geojson_path = geojson_path
        self.laz_path = laz_path
        self.geotiff_path = geotiff_path
        self.output_path = output_path
        self.current_num = current_num
        self.total_num = total_num

        self.basename = os.path.splitext(os.path.basename(self.geojson_path))[0]

        self.final_result = None

        self.rmse_final = None

        self.params = {
            "THRESHOLD": THRESHOLD_HARD_DEFAULT,
            "BIN_WIDTH": BIN_WIDTH_HARD_DEFAULT,
            "PRIOR_BIAS": PRIOR_BIAS_HARD_DEFAULT,
            "RANSAC_RESIDUAL": RANSAC_RESIDUAL_HARD_DEFAULT,
            "RANSAC_MIN_SAMPLES": RANSAC_MIN_SAMPLES_HARD_DEFAULT,
            "USE_ICP": USE_ICP_HARD_DEFAULT,
        }

        self.corners = None
        self.edges = None
        self.edge_types = None
        self.edge_props = None
        self.idx_to_id = None

        self.lidar_pts = None
        self.lidar_classes = None

        self.geotiff_img = None
        self.geotiff_extent = None
        self.geotiff_mode = None

        self.clicked_edges = set()
        self.mode = "DELETE"

        self.components_tagged = False
        self.comp_id_by_vertex = {}
        self.prior_heights_by_comp = defaultdict(list)

        self.ransac_lines = []
        self.fit_points_all = []
        self.fit_points_inliers = []

        self.face_centroids = None
        self.fitted_face_planes = []
        self.plane_vertex_corrections = []
        self.pre_adjustment_roof_edge_segments = []
        self.nav_action = None

    def load(self):
        (
            self.corners,
            self.edges,
            self.edge_types,
            self.edge_props,
            self.idx_to_id
        ) = load_skeleton_from_geojson(self.geojson_path)

        self.lidar_pts, self.lidar_classes = load_lidar_from_laz(self.laz_path)
        self.geotiff_img, self.geotiff_extent, self.geotiff_mode = load_geotiff_for_display(self.geotiff_path)

        if self.params["USE_ICP"]:
            print("🔄 Performing ICP alignment...")
            self.lidar_pts = icp_align(self.lidar_pts, self.corners[:, :2])
        else:
            print("⏩ Skipping ICP — using original LiDAR alignment.")

    def tag_components(self):
        G = nx.Graph()
        valid = ~np.isnan(self.corners).any(axis=1)

        for s, t in self.edges:
            if valid[s] and valid[t]:
                G.add_edge(int(s), int(t))

        comps = list(nx.connected_components(G))
        self.comp_id_by_vertex = {}

        for cid, comp in enumerate(comps):
            for vi in comp:
                self.comp_id_by_vertex[int(vi)] = cid

        self.prior_heights_by_comp = defaultdict(list)
        self.components_tagged = True
        print(f"🏷️ Tagged {len(comps)} connected component(s). Per-component PRIOR_BIAS is now active.")

    def accept_and_save_2d_skeleton(self):
        deleted_vertices = set(np.where(np.isnan(self.corners).any(axis=1))[0].tolist())

        if len(deleted_vertices) > 0:
            (
                self.corners,
                self.edges,
                self.edge_types,
                self.edge_props,
                _,
                new_idx_to_id
            ) = remap_after_vertex_deletions(
                self.corners,
                self.edges,
                self.edge_types,
                self.edge_props,
                deleted_vertices,
                self.idx_to_id
            )

            if new_idx_to_id is not None:
                self.idx_to_id = new_idx_to_id

        self.idx_to_id = save_2d_roof_geojson(
            self.corners,
            self.edges,
            self.edge_types,
            self.edge_props,
            self.idx_to_id,
            self.geojson_path
        )

    def run_2d_picker(self):
        fig, ax = plt.subplots(figsize=(9, 9))
        try:
            fig.canvas.manager.set_window_title(f"2D — {os.path.basename(self.geojson_path)}")
        except Exception:
            pass

        print(KEY_HELP)

        mode_text = ax.text(
            0.01, 0.99, f"Mode: {self.mode}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(facecolor="w", edgecolor="k", boxstyle="round,pad=0.3")
        )

        def set_title():
            ax.set_title(
                f"[{self.mode} MODE]  (H=Height, D=Delete, E=Edit, T=Toggle TIFF, C=Tag components, 3=Open 3D, Q=quit)\n"
                f"Fixer defaults t={THRESHOLD_HARD_DEFAULT}  b={BIN_WIDTH_HARD_DEFAULT}  "
                f"r={RANSAC_RESIDUAL_HARD_DEFAULT}  m={RANSAC_MIN_SAMPLES_HARD_DEFAULT}  "
                f"pb={PRIOR_BIAS_HARD_DEFAULT}  ICP={USE_ICP_HARD_DEFAULT}  (edits per-edge & auto-reset)"
            )
            mode_text.set_text(f"Mode: {self.mode}")
            fig.canvas.draw_idle()

        set_title()

        ax.scatter(
            self.lidar_pts[:, 0],
            self.lidar_pts[:, 1],
            c=self.lidar_pts[:, 2],
            s=1,
            cmap="viridis",
            alpha=0.5,
            zorder=0
        )

        geotiff_artist = None
        geotiff_visible = {"on": True}

        if self.geotiff_img is not None:
            if self.geotiff_mode == "rgb":
                geotiff_artist = ax.imshow(
                    self.geotiff_img,
                    extent=self.geotiff_extent,
                    origin="upper",
                    alpha=0.85,
                    zorder=1
                )
            else:
                geotiff_artist = ax.imshow(
                    self.geotiff_img,
                    extent=self.geotiff_extent,
                    origin="upper",
                    cmap="gray",
                    alpha=0.85,
                    zorder=1
                )

        def valid_corner_mask():
            return ~np.isnan(self.corners).any(axis=1)

        corner_scatter = ax.scatter(
            self.corners[valid_corner_mask(), 0],
            self.corners[valid_corner_mask(), 1],
            c="blue",
            s=30,
            zorder=3
        )

        lines = []
        selected_delete_edges = set()
        selected_delete_vertices = set()
        edit_selected_vertices = []
        edit_selected_edge = {"idx": None}

        delete_undo_stack = deque()
        edit_undo_stack = deque()

        delete_vertex_marker = {"artist": None}
        edit_vertex_marker = {"artist": None}

        delete_drag = {
            "active": False,
            "moved": False,
            "start_data": None,
            "start_disp": None,
            "rect_artist": None,
        }

        def clear_artist(ref):
            if ref["artist"] is not None:
                try:
                    ref["artist"].remove()
                except Exception:
                    pass
                ref["artist"] = None

        def clear_delete_rect():
            if delete_drag["rect_artist"] is not None:
                try:
                    delete_drag["rect_artist"].remove()
                except Exception:
                    pass
                delete_drag["rect_artist"] = None

        def clear_mode_selections():
            selected_delete_edges.clear()
            selected_delete_vertices.clear()
            edit_selected_vertices.clear()
            edit_selected_edge["idx"] = None
            clear_artist(delete_vertex_marker)
            clear_artist(edit_vertex_marker)
            clear_delete_rect()

        def update_corner_scatter():
            m = valid_corner_mask()
            if np.any(m):
                corner_scatter.set_offsets(self.corners[m, :2])
            else:
                corner_scatter.set_offsets(np.empty((0, 2)))

        def update_line_and_selection_visuals():
            clear_artist(delete_vertex_marker)
            clear_artist(edit_vertex_marker)

            for idx_ln, ln in enumerate(lines):
                color = "green" if idx_ln in self.clicked_edges else "red"
                width = 1.0

                if self.mode == "DELETE" and idx_ln in selected_delete_edges:
                    color = "black"
                    width = 3.0

                if self.mode == "EDIT" and edit_selected_edge["idx"] == idx_ln:
                    color = "yellow"
                    width = 3.0

                ln.set_color(color)
                ln.set_linewidth(width)

            if self.mode == "DELETE" and len(selected_delete_vertices) > 0:
                pts = []
                for vi in sorted(selected_delete_vertices):
                    if 0 <= vi < len(self.corners) and not np.isnan(self.corners[vi]).any():
                        pts.append(self.corners[vi, :2])
                if len(pts) > 0:
                    pts = np.asarray(pts)
                    delete_vertex_marker["artist"] = ax.scatter(
                        pts[:, 0], pts[:, 1],
                        c="black", s=90, zorder=5
                    )

            if self.mode == "EDIT" and len(edit_selected_vertices) > 0:
                pts = []
                for vi in edit_selected_vertices:
                    if 0 <= vi < len(self.corners) and not np.isnan(self.corners[vi]).any():
                        pts.append(self.corners[vi, :2])
                if len(pts) > 0:
                    pts = np.asarray(pts)
                    edit_vertex_marker["artist"] = ax.scatter(
                        pts[:, 0], pts[:, 1],
                        c="yellow", s=90, edgecolors="black", linewidths=0.8, zorder=5
                    )

            fig.canvas.draw_idle()

        def rebuild_all_lines():
            for ln in lines:
                try:
                    ln.remove()
                except Exception:
                    pass
            lines.clear()

            for s, t in self.edges:
                if np.isnan(self.corners[s]).any() or np.isnan(self.corners[t]).any():
                    continue
                x1, y1 = self.corners[s, :2]
                x2, y2 = self.corners[t, :2]
                ln, = ax.plot([x1, x2], [y1, y2], "-", color="red", linewidth=1)
                lines.append(ln)

            update_corner_scatter()
            update_line_and_selection_visuals()

        rebuild_all_lines()

        def make_full_snapshot():
            return {
                "corners": self.corners.copy(),
                "edges": [e[:] for e in self.edges],
                "edge_types": self.edge_types[:],
                "edge_props": [dict(p) for p in self.edge_props],
                "idx_to_id": dict(self.idx_to_id) if self.idx_to_id is not None else None,
                "clicked_edges": set(self.clicked_edges),
                "ransac_lines": [np.array(x, copy=True) for x in self.ransac_lines],
                "fit_points_all": [np.array(x, copy=True) for x in self.fit_points_all],
                "fit_points_inliers": [np.array(x, copy=True) for x in self.fit_points_inliers],
                "prior_heights_by_comp": defaultdict(
                    list,
                    {k: list(v) for k, v in self.prior_heights_by_comp.items()}
                ),
                "components_tagged": self.components_tagged,
                "comp_id_by_vertex": dict(self.comp_id_by_vertex),
            }

        def restore_full_snapshot(snapshot, msg):
            self.corners = snapshot["corners"].copy()
            self.edges = [e[:] for e in snapshot["edges"]]
            self.edge_types = snapshot["edge_types"][:]
            self.edge_props = [dict(p) for p in snapshot["edge_props"]]
            self.idx_to_id = dict(snapshot["idx_to_id"]) if snapshot["idx_to_id"] is not None else None
            self.clicked_edges = set(snapshot["clicked_edges"])
            self.ransac_lines = [np.array(x, copy=True) for x in snapshot["ransac_lines"]]
            self.fit_points_all = [np.array(x, copy=True) for x in snapshot["fit_points_all"]]
            self.fit_points_inliers = [np.array(x, copy=True) for x in snapshot["fit_points_inliers"]]
            self.prior_heights_by_comp = defaultdict(
                list,
                {k: list(v) for k, v in snapshot["prior_heights_by_comp"].items()}
            )
            self.components_tagged = snapshot["components_tagged"]
            self.comp_id_by_vertex = dict(snapshot["comp_id_by_vertex"])

            clear_mode_selections()
            rebuild_all_lines()
            print(msg)

        def clear_processing_state_due_to_topology_change():
            self.clicked_edges.clear()
            self.ransac_lines.clear()
            self.fit_points_all.clear()
            self.fit_points_inliers.clear()
            self.prior_heights_by_comp = defaultdict(list)
            self.components_tagged = False
            self.comp_id_by_vertex = {}

        def nearest_vertex(x, y):
            m = valid_corner_mask()
            idxs = np.where(m)[0]
            if idxs.size == 0:
                return None, None
            d = np.hypot(self.corners[idxs, 0] - x, self.corners[idxs, 1] - y)
            k = int(np.argmin(d))
            return int(idxs[k]), float(d[k])

        def nearest_edge(x, y):
            P = np.array([x, y])
            best_idx, best_d = None, None
            for idx, (s, t) in enumerate(self.edges):
                if np.isnan(self.corners[s]).any() or np.isnan(self.corners[t]).any():
                    continue
                d = point_segment_distance(P, self.corners[s, :2], self.corners[t, :2])
                if best_d is None or d < best_d:
                    best_d, best_idx = d, idx
            return best_idx, best_d

        def select_delete_by_click(x, y):
            vi, dv = nearest_vertex(x, y)
            ee, de = nearest_edge(x, y)

            selected_delete_edges.clear()
            selected_delete_vertices.clear()

            if vi is not None and (ee is None or dv <= (de if de is not None else 1e9)):
                selected_delete_vertices.add(vi)
            elif ee is not None:
                selected_delete_edges.add(ee)

            update_line_and_selection_visuals()

        def select_delete_by_rectangle(x0, y0, x1, y1):
            xmin, xmax = sorted([x0, x1])
            ymin, ymax = sorted([y0, y1])

            selected_delete_edges.clear()
            selected_delete_vertices.clear()

            for vi in range(len(self.corners)):
                if np.isnan(self.corners[vi]).any():
                    continue
                x, y = self.corners[vi, :2]
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    selected_delete_vertices.add(vi)

            rect_geom = box(xmin, ymin, xmax, ymax)

            for idx, (s, t) in enumerate(self.edges):
                if np.isnan(self.corners[s]).any() or np.isnan(self.corners[t]).any():
                    continue
                seg = LineString([
                    tuple(self.corners[s, :2]),
                    tuple(self.corners[t, :2])
                ])
                if seg.intersects(rect_geom):
                    selected_delete_edges.add(idx)

            update_line_and_selection_visuals()

        def add_edit_vertex(x, y):
            vi, dv = nearest_vertex(x, y)
            if vi is not None and dv is not None and dv <= VERTEX_PICK_RADIUS:
                print("🔁 Point already exists nearby. Skipping duplicate.")
                return

            edit_undo_stack.append(make_full_snapshot())

            new_corner = np.array([[x, y, 0.0]], dtype=float)
            self.corners = np.vstack([self.corners, new_corner])

            edit_selected_vertices.clear()
            edit_selected_edge["idx"] = None

            update_corner_scatter()
            update_line_and_selection_visuals()
            print(f"➕ Added vertex {len(self.corners) - 1}")

        def select_edit_target(x, y):
            vi, dv = nearest_vertex(x, y)
            ee, de = nearest_edge(x, y)

            if vi is not None and dv is not None and dv <= VERTEX_PICK_RADIUS:
                if vi not in edit_selected_vertices:
                    if len(edit_selected_vertices) >= 2:
                        edit_selected_vertices.clear()
                    edit_selected_vertices.append(vi)
                edit_selected_edge["idx"] = None
                update_line_and_selection_visuals()
                return

            if ee is not None and de is not None and de <= EDGE_PICK_DIST:
                edit_selected_edge["idx"] = ee
                edit_selected_vertices.clear()
                update_line_and_selection_visuals()
                return

            edit_selected_vertices.clear()
            edit_selected_edge["idx"] = None
            update_line_and_selection_visuals()

        def connect_edit_selected_vertices():
            if len(edit_selected_vertices) != 2:
                return

            i, j = edit_selected_vertices[0], edit_selected_vertices[1]
            if i == j:
                edit_selected_vertices.clear()
                edit_selected_edge["idx"] = None
                update_line_and_selection_visuals()
                return

            candidate = {int(i), int(j)}
            for s, t in self.edges:
                if {int(s), int(t)} == candidate:
                    print("Line already exists. Skipping duplicate connection.")
                    edit_selected_vertices.clear()
                    edit_selected_edge["idx"] = None
                    update_line_and_selection_visuals()
                    return

            edit_undo_stack.append(make_full_snapshot())

            self.edges.append([int(i), int(j)])
            self.edge_types.append("roof")
            self.edge_props.append({
                "edge_id": len(self.edge_props),
                "image_id": "edited_roof"
            })

            self.components_tagged = False
            self.comp_id_by_vertex = {}

            print(f"Connected vertex {i} to vertex {j}")

            edit_selected_vertices.clear()
            edit_selected_edge["idx"] = None
            rebuild_all_lines()

        def delete_selected_edit_geometry():
            if len(edit_selected_vertices) > 0:
                edit_undo_stack.append(make_full_snapshot())

                delete_set = set(edit_selected_vertices)

                incident_idxs = [
                    k for k, (s, t) in enumerate(self.edges)
                    if s in delete_set or t in delete_set
                ]
                for idx_rm in sorted(incident_idxs, reverse=True):
                    del self.edges[idx_rm]
                    del self.edge_types[idx_rm]
                    del self.edge_props[idx_rm]

                for vi in delete_set:
                    if 0 <= vi < len(self.corners):
                        self.corners[vi, :] = np.nan

                (
                    self.corners,
                    self.edges,
                    self.edge_types,
                    self.edge_props,
                    _,
                    new_idx_to_id
                ) = remap_after_vertex_deletions(
                    self.corners,
                    self.edges,
                    self.edge_types,
                    self.edge_props,
                    delete_set,
                    self.idx_to_id
                )
                if new_idx_to_id is not None:
                    self.idx_to_id = new_idx_to_id

                clear_processing_state_due_to_topology_change()

                edit_selected_vertices.clear()
                edit_selected_edge["idx"] = None
                rebuild_all_lines()
                print("🗑️ Deleted selected edit vertex/vertices.")
                return

            if edit_selected_edge["idx"] is not None:
                edit_undo_stack.append(make_full_snapshot())

                idx_rm = edit_selected_edge["idx"]
                del self.edges[idx_rm]
                del self.edge_types[idx_rm]
                del self.edge_props[idx_rm]

                clear_processing_state_due_to_topology_change()

                edit_selected_vertices.clear()
                edit_selected_edge["idx"] = None
                rebuild_all_lines()
                print("🗑️ Deleted selected edit edge.")
                return

            print("No edit geometry selected.")

        def undo_last_edit():
            if not edit_undo_stack:
                print("Nothing to undo.")
                return
            restore_full_snapshot(edit_undo_stack.pop(), "↩️ Edit undo applied.")

        def delete_selected_edge():
            if len(selected_delete_edges) == 0:
                print("No delete edges selected.")
                return

            delete_undo_stack.append(make_full_snapshot())

            for idx_rm in sorted(selected_delete_edges, reverse=True):
                del self.edges[idx_rm]
                del self.edge_types[idx_rm]
                del self.edge_props[idx_rm]

            clear_processing_state_due_to_topology_change()

            selected_delete_edges.clear()
            selected_delete_vertices.clear()
            rebuild_all_lines()
            print("🗑️ Selected edge(s) deleted. Press U to undo.")

        def delete_selected_vertex():
            if len(selected_delete_vertices) == 0:
                print("No delete vertices selected.")
                return

            delete_undo_stack.append(make_full_snapshot())

            delete_set = set(selected_delete_vertices)

            incident_idxs = [k for k, (s, t) in enumerate(self.edges) if s in delete_set or t in delete_set]
            for idx_rm in sorted(incident_idxs, reverse=True):
                del self.edges[idx_rm]
                del self.edge_types[idx_rm]
                del self.edge_props[idx_rm]

            for vi in delete_set:
                if 0 <= vi < len(self.corners):
                    self.corners[vi, :] = np.nan

            clear_processing_state_due_to_topology_change()

            selected_delete_edges.clear()
            selected_delete_vertices.clear()
            rebuild_all_lines()
            print("🗑️ Selected vertex/vertices deleted. Press U to undo.")

        def undo_last_delete():
            if not delete_undo_stack:
                print("Nothing to undo.")
                return
            restore_full_snapshot(delete_undo_stack.pop(), "↩️ Delete undo applied.")

        def accept_current_2d_changes():
            self.accept_and_save_2d_skeleton()

            delete_undo_stack.clear()
            edit_undo_stack.clear()

            clear_mode_selections()
            self.mode = "DELETE"
            rebuild_all_lines()
            set_title()

        def on_press(evt):
            if evt.inaxes != ax or evt.xdata is None or evt.ydata is None:
                return

            x = float(evt.xdata)
            y = float(evt.ydata)

            if self.mode == "DELETE":
                if evt.button != 1:
                    return

                delete_drag["active"] = True
                delete_drag["moved"] = False
                delete_drag["start_data"] = (x, y)
                delete_drag["start_disp"] = (evt.x, evt.y)
                clear_delete_rect()
                return

            if self.mode == "EDIT":
                if evt.button == 3 and evt.dblclick:
                    vi, dv = nearest_vertex(x, y)
                    if vi is not None and dv is not None and dv <= VERTEX_PICK_RADIUS:
                        if vi not in edit_selected_vertices:
                            if len(edit_selected_vertices) >= 2:
                                edit_selected_vertices.clear()
                            edit_selected_vertices.append(vi)

                    update_line_and_selection_visuals()
                    connect_edit_selected_vertices()
                    return

                if evt.button == 1 and not evt.dblclick:
                    add_edit_vertex(x, y)
                    return

                if evt.button == 3 and not evt.dblclick:
                    select_edit_target(x, y)
                    return

        def on_motion(evt):
            if self.mode != "DELETE":
                return
            if not delete_drag["active"]:
                return
            if evt.inaxes != ax or evt.xdata is None or evt.ydata is None:
                return

            x0, y0 = delete_drag["start_data"]
            dx_pix = abs(evt.x - delete_drag["start_disp"][0])
            dy_pix = abs(evt.y - delete_drag["start_disp"][1])

            if dx_pix >= 5 or dy_pix >= 5:
                delete_drag["moved"] = True

            if not delete_drag["moved"]:
                return

            clear_delete_rect()

            xmin, xmax = sorted([x0, float(evt.xdata)])
            ymin, ymax = sorted([y0, float(evt.ydata)])

            delete_drag["rect_artist"] = ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    fill=False,
                    edgecolor="black",
                    linewidth=1.5,
                    linestyle="--",
                    zorder=10
                )
            )
            fig.canvas.draw_idle()

        def on_release(evt):
            if self.mode != "DELETE":
                return
            if not delete_drag["active"]:
                return

            start = delete_drag["start_data"]
            moved = delete_drag["moved"]

            delete_drag["active"] = False

            if evt.inaxes != ax or evt.xdata is None or evt.ydata is None:
                clear_delete_rect()
                fig.canvas.draw_idle()
                return

            x1 = float(evt.xdata)
            y1 = float(evt.ydata)

            clear_delete_rect()

            if moved:
                select_delete_by_rectangle(start[0], start[1], x1, y1)
            else:
                select_delete_by_click(x1, y1)

            fig.canvas.draw_idle()

        def on_key(evt):
            k = (evt.key or "").lower()

            if k == "d":
                self.mode = "DELETE"
                clear_mode_selections()
                set_title()
                update_line_and_selection_visuals()

            elif k == "e":
                if self.mode == "DELETE" and len(selected_delete_edges) > 0:
                    delete_selected_edge()
                else:
                    self.mode = "EDIT"
                    clear_mode_selections()
                    set_title()
                    update_line_and_selection_visuals()

            elif k == "t":
                if geotiff_artist is not None:
                    geotiff_visible["on"] = not geotiff_visible["on"]
                    geotiff_artist.set_visible(geotiff_visible["on"])
                    fig.canvas.draw_idle()
                    state = "ON" if geotiff_visible["on"] else "OFF"
                    print(f"🖼️ GeoTIFF overlay: {state}")

            elif k == "c":
                self.tag_components()

            elif k == "v" and self.mode == "DELETE":
                delete_selected_vertex()

            elif k == "a" and self.mode in ("DELETE", "EDIT"):
                accept_current_2d_changes()

            elif k == "u" and self.mode == "DELETE":
                undo_last_delete()

            elif k in ("u", "z") and self.mode == "EDIT":
                undo_last_edit()

            elif k == "x" and self.mode == "EDIT":
                delete_selected_edit_geometry()

            elif k == "3":
                self.nav_action = "preview"
                plt.close(fig)

            elif k == "q":
                self.nav_action = "quit"
                plt.close(fig)

        def on_close(evt):
            if self.nav_action is None:
                self.nav_action = "quit"

        fig.canvas.mpl_connect("button_press_event", on_press)
        fig.canvas.mpl_connect("motion_notify_event", on_motion)
        fig.canvas.mpl_connect("button_release_event", on_release)
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)

        ax.set_aspect("equal", adjustable="box")
        plt.show()

    def _snapshot_variant_state(self):
        return {
            "corners": self.corners.copy(),
            "edges": [e[:] for e in self.edges],
            "edge_types": self.edge_types[:],
            "edge_props": [dict(p) for p in self.edge_props],
            "idx_to_id": dict(self.idx_to_id) if self.idx_to_id is not None else None,
            "face_centroids": (
                None if self.face_centroids is None
                else np.array(self.face_centroids, copy=True)
            ),
            "fitted_face_planes": [dict(fp) for fp in self.fitted_face_planes],
            "plane_vertex_corrections": [dict(rec) for rec in self.plane_vertex_corrections],
            "surface_faces": [dict(sf) for sf in getattr(self, "surface_faces", [])],
            "pre_adjustment_roof_edge_segments": [
                np.array(seg, copy=True)
                for seg in getattr(self, "pre_adjustment_roof_edge_segments", [])
            ],
        }


    def _restore_variant_state(self, snap):
        self.corners = snap["corners"].copy()
        self.edges = [e[:] for e in snap["edges"]]
        self.edge_types = snap["edge_types"][:]
        self.edge_props = [dict(p) for p in snap["edge_props"]]
        self.idx_to_id = dict(snap["idx_to_id"]) if snap["idx_to_id"] is not None else None
        self.face_centroids = (
            None if snap["face_centroids"] is None
            else np.array(snap["face_centroids"], copy=True)
        )
        self.fitted_face_planes = [dict(fp) for fp in snap["fitted_face_planes"]]
        self.plane_vertex_corrections = [dict(rec) for rec in snap["plane_vertex_corrections"]]
        self.surface_faces = [dict(sf) for sf in snap.get("surface_faces", [])]
        self.pre_adjustment_roof_edge_segments = [
            np.array(seg, copy=True)
            for seg in snap.get("pre_adjustment_roof_edge_segments", [])
        ]

    def compute_lidar_rmse(self):
        self.rmse_final = None
        self.n_roof_ref_points = 0

        if self.final_result is None:
            print("⚠️ Cannot compute RMSE: final result missing.")
            return

        roof_ref_pts = extract_roof_reference_points(self.lidar_pts, self.lidar_classes)
        self.n_roof_ref_points = int(len(roof_ref_pts))

        if self.n_roof_ref_points == 0:
            print("⚠️ No LiDAR roof reference points found for evaluation.")
            return

        final_vertices, final_triangles = build_mesh_from_surface_faces(
            self.final_result["corners"],
            self.final_result.get("surface_faces", [])
        )

        if final_triangles.shape[0] == 0:
            print("⚠️ Final roof mesh is empty.")
            return

        self.rmse_final = compute_point_to_mesh_rmse(
            roof_ref_pts,
            final_vertices,
            final_triangles,
            max_dist=LIDAR_TO_MESH_MAX_DIST
        )

        print("📏 LiDAR point-to-roof-surface RMSE:")
        print(f"   roof ref pts : {self.n_roof_ref_points}")
        print(f"   Final        : {self.rmse_final}")


    def _build_variant_from_current_roof(self, use_stage2):
        """
        Builds one final 3D variant from the current roof-only state.

        use_stage2=False  -> Stage I only
        use_stage2=True   -> Stage I + Stage II
        """
        self.face_centroids = []
        self.fitted_face_planes = []
        self.plane_vertex_corrections = []
        self.surface_faces = []
        self.pre_adjustment_roof_edge_segments = []

        valid = ~np.isnan(self.corners).any(axis=1)

        # ----------------------------
        # Roof-only graph before Stage II
        # ----------------------------
        G_init = nx.Graph()
        for ei, (s, t) in enumerate(self.edges):
            if self.edge_types[ei] != "roof":
                continue
            if valid[s] and valid[t]:
                G_init.add_edge(int(s), int(t), ei=ei)

        if G_init.number_of_edges() == 0:
            print("⚠️ No roof edges available for variant build.")
            self.face_centroids = np.empty((0, 3))
            return

        comp_id_by_vertex_before_split = {}
        comps_init = list(nx.connected_components(G_init))
        for cid, comp_nodes in enumerate(comps_init):
            for vi in comp_nodes:
                comp_id_by_vertex_before_split[int(vi)] = int(cid)

        # ----------------------------
        # Cache ORIGINAL exterior roof edges so Stage II can suppress roof-seam
        # creation on building exteriors. Exterior walls should follow the final
        # adjusted roof boundary instead.
        # ----------------------------
        exterior_roof_edge_pairs = set()
        exterior_loop_seed_jobs = []
        inner_roof_loop_seed_jobs = []

        for cid, comp_nodes in enumerate(comps_init):
            sub0 = G_init.subgraph(comp_nodes).copy()
            comp_edge_idxs0 = [data["ei"] for _, _, data in sub0.edges(data=True)]

            loops0 = extract_component_loops(
                self.corners,
                self.edges,
                comp_edge_idxs0,
                comp_nodes=comp_nodes
            )

            for loop_id, ring in enumerate(loops0["exteriors"]):
                ring = [int(v) for v in ring]
                if len(ring) < 3:
                    continue

                exterior_loop_seed_jobs.append({
                    "component_id": int(cid),
                    "loop_id": int(loop_id),
                    "ring": ring
                })

                k_ext = len(ring)
                for i in range(k_ext):
                    exterior_roof_edge_pairs.add(
                        frozenset((int(ring[i]), int(ring[(i + 1) % k_ext])))
                    )

            for loop_id, hole_rings in enumerate(loops0.get("holes", [])):
                for hole_id, h_ring in enumerate(hole_rings):
                    h_ring = [int(v) for v in h_ring]
                    if len(h_ring) < 3:
                        continue

                    inner_roof_loop_seed_jobs.append({
                        "component_id": int(cid),
                        "loop_id": int(loop_id),
                        "hole_id": int(hole_id),
                        "ring": h_ring
                    })

        # ----------------------------
        # Stage II (optional)
        # ----------------------------
        if use_stage2:
            for cid, comp_nodes in enumerate(comps_init):
                sub = G_init.subgraph(comp_nodes).copy()
                comp_edge_idxs = [data["ei"] for _, _, data in sub.edges(data=True)]

                face_planes = fit_planes_for_component_faces(
                    self.corners,
                    self.edges,
                    comp_edge_idxs,
                    comp_nodes,
                    self.lidar_pts,
                    component_id=cid
                )
                self.fitted_face_planes.extend(face_planes)

            if ENABLE_PLANE_VERTEX_SPLIT_CORRECTION and len(self.fitted_face_planes) > 0:
                (
                    self.corners,
                    self.edges,
                    self.edge_types,
                    self.edge_props,
                    self.idx_to_id,
                    corrections,
                    rebuilt_roof_faces,
                    seam_faces,
                    pre_adjustment_roof_edge_segments,
                    split_gap_used
                ) = apply_face_plane_vertex_splitting(
                    self.corners,
                    self.edges,
                    self.edge_types,
                    self.edge_props,
                    self.idx_to_id,
                    self.fitted_face_planes,
                    self.lidar_pts,
                    comp_id_by_vertex_before_split,
                    exterior_roof_edge_pairs=exterior_roof_edge_pairs
                )

                for sf in rebuilt_roof_faces:
                    self.surface_faces.append(dict(sf))

                for sf in seam_faces:
                    self.surface_faces.append(dict(sf))

                self.plane_vertex_corrections = corrections
                self.pre_adjustment_roof_edge_segments = pre_adjustment_roof_edge_segments

                print(
                    f"ℹ️ Stage II enabled: split gap used = {split_gap_used:.3f} m"
                )

        # ----------------------------
        # Final roof graph after optional Stage II
        # ----------------------------
        G_final = nx.Graph()
        valid = ~np.isnan(self.corners).any(axis=1)

        for ei, (s, t) in enumerate(self.edges):
            if self.edge_types[ei] != "roof":
                continue
            if valid[s] and valid[t]:
                G_final.add_edge(int(s), int(t), ei=ei)

        if G_final.number_of_edges() == 0:
            print("⚠️ No roof edges remain after variant build.")
            self.face_centroids = np.empty((0, 3))
            return

        lengths_dict = build_vertex_incident_edge_lengths(self.corners[:, :2], self.edges)

        # If no explicit roof surfaces were created yet, derive them from current roof loops
        if not any(sf.get("surface_type") == "roof" for sf in self.surface_faces):
            for cid, comp_nodes in enumerate(nx.connected_components(G_final)):
                sub = G_final.subgraph(comp_nodes).copy()
                comp_edge_idxs = [data["ei"] for _, _, data in sub.edges(data=True)]

                face_defs = extract_component_face_polygons(
                    self.corners,
                    self.edges,
                    comp_edge_idxs,
                    comp_nodes=comp_nodes
                )

                for face_id, fd in enumerate(face_defs):
                    self.surface_faces.append({
                        "surface_type": "roof",
                        "component_id": int(cid),
                        "face_id": int(face_id),
                        "vertex_indices": [int(v) for v in fd["ring"]],
                    })

        # ----------------------------
        # Base + walls from ORIGINAL exterior footprint loops, remapped to the
        # final adjusted roof vertices below.
        #
        # Stage-II splitting can disconnect the roof graph into separate plane
        # islands because roof_seam faces are stored as surfaces, not roof
        # edges. Extracting exteriors from G_final would then treat internal
        # seam boundaries as building exteriors and generate wall/base surfaces
        # inside or beyond the actual footprint.
        # ----------------------------
        final_base_seed_jobs = [dict(job) for job in exterior_loop_seed_jobs]

        if len(final_base_seed_jobs) == 0:
            for cid, comp_nodes in enumerate(nx.connected_components(G_final)):
                sub = G_final.subgraph(comp_nodes).copy()
                comp_edge_idxs = [data["ei"] for _, _, data in sub.edges(data=True)]
                loops = extract_component_loops(
                    self.corners,
                    self.edges,
                    comp_edge_idxs,
                    comp_nodes=comp_nodes
                )

                for loop_id, ring in enumerate(loops["exteriors"]):
                    final_base_seed_jobs.append({
                        "component_id": int(cid),
                        "loop_id": int(loop_id),
                        "ring": [int(v) for v in ring]
                    })

        final_base_seed_jobs, nested_inner_jobs = split_nested_loop_seed_jobs(
            self.corners,
            final_base_seed_jobs
        )
        inner_roof_loop_seed_jobs.extend(nested_inner_jobs)

        if len(inner_roof_loop_seed_jobs) > 0:
            self.corners, added_inner_seams = add_inner_roof_loop_seams(
                self.corners,
                self.surface_faces,
                inner_roof_loop_seed_jobs,
                self.edges,
                self.edge_types,
                xy_tol=1e-7
            )
            if added_inner_seams > 0:
                print(
                    f"ℹ️ Converted {added_inner_seams} inner roof-loop side faces "
                    f"to roof_seam surfaces; no base was generated for those loops."
                )

        for job in final_base_seed_jobs:
            cid = int(job["component_id"])
            loop_id = int(job["loop_id"])
            original_ring = [int(v) for v in job["ring"]]
            ring = original_ring[:]

            # IMPORTANT:
            # Stage-II splitting can create multiple vertices at the same XY.
            # The exterior loop extraction may return the old/lower vertex.
            # Remap the exterior/base loop to the lifted final roof vertices
            # before creating base vertices, wall edges, and wall surfaces.
            ring = remap_exterior_ring_to_lifted_roof_vertices(
                ring=ring,
                corners=self.corners,
                edges=self.edges,
                edge_types=self.edge_types,
                xy_tol=1e-7
            )

            if len(ring) < 3:
                continue

            use_original_edge_lookup = len(original_ring) == len(ring)

            base_verts = []
            roof_ring_xyz = []

            for roof_idx in ring:
                vx, vy, vz_roof = self.corners[roof_idx]
                R = radius_for_vertex(roof_idx, lengths_dict)

                z_v = base_z_at_vertex_upshoot_adaptive(
                    np.array([vx, vy]),
                    self.lidar_pts,
                    self.lidar_classes,
                    R
                )

                if z_v is None:
                    z_v = float(vz_roof)
                    print(
                        f"⚠ Comp {cid} loop {loop_id} vertex {roof_idx}: "
                        f"no pts within R={R:.2f} m → provisional z={z_v:.2f}"
                    )
                else:
                    print(
                        f"✔ Comp {cid} loop {loop_id} vertex {roof_idx}: "
                        f"R={R:.2f} m → provisional base z={z_v:.2f}"
                    )

                base_verts.append([vx, vy, z_v])
                roof_ring_xyz.append([vx, vy, vz_roof])

            base_verts = np.array(base_verts, dtype=float)
            roof_ring_xyz = np.array(roof_ring_xyz, dtype=float)

            # regularize abnormal base vertices
            base_verts, base_fix_logs = regularize_base_ring_vertices(
                base_verts=base_verts,
                roof_ring_xyz=roof_ring_xyz
            )

            if len(base_fix_logs) > 0:
                print(
                    f"✅ Comp {cid} loop {loop_id}: corrected "
                    f"{len(base_fix_logs)} abnormal base vertices."
                )
                for rec in base_fix_logs:
                    print(
                        f"   • ring vertex {rec['ring_vertex_order']}: "
                        f"{rec['old_z']:.2f} -> {rec['new_z']:.2f} "
                        f"(roof={rec['roof_z']:.2f}, method={rec['method']}, reason={rec['reason']})"
                    )
            else:
                print(f"ℹ️ Comp {cid} loop {loop_id}: no abnormal base vertices detected.")

            corner_offset = len(self.corners)
            self.corners = np.vstack([self.corners, base_verts])

            kN = len(ring)
            base_ring_indices = [corner_offset + i for i in range(kN)]
            base_edges = [[corner_offset + i, corner_offset + ((i + 1) % kN)] for i in range(kN)]

            # base edges
            for i_e, e in enumerate(base_edges):
                self.edges.append(e)
                self.edge_types.append("base")
                self.edge_props.append({
                    "edge_id": len(self.edge_props),
                    "image_id": "generated_base",
                    "component_id": int(cid),
                    "loop_id": int(loop_id),
                    "ring_order": int(i_e)
                })

            # base surface  (IMPORTANT: append ONCE, not inside the edge loop)
            self.surface_faces.append({
                "surface_type": "base",
                "component_id": int(cid),
                "loop_id": int(loop_id),
                "vertex_indices": [int(v) for v in base_ring_indices],
            })

            wall_top_edges = []
            for i in range(kN):
                if use_original_edge_lookup:
                    old_edge = (original_ring[i], original_ring[(i + 1) % kN])
                else:
                    old_edge = (ring[i], ring[(i + 1) % kN])

                matched_edge = find_roof_edge_matching_original_xy(
                    old_edge,
                    self.corners,
                    self.edges,
                    self.edge_types,
                    xy_tol=1e-7
                )

                if matched_edge is None:
                    matched_edge = (int(ring[i]), int(ring[(i + 1) % kN]))

                wall_top_edges.append((int(matched_edge[0]), int(matched_edge[1])))

            # Build a lookup of roof edges that are already covered by roof_seam surfaces.
            # If a wall top edge is already part of a roof_seam, do not create a wall face there.
            roof_seam_edge_pairs = set()

            for sf in self.surface_faces:
                if sf.get("surface_type") != "roof_seam":
                    continue

                seam_ring = dedupe_ring_vertex_indices(sf.get("vertex_indices", []))
                if len(seam_ring) < 3:
                    continue

                for j in range(len(seam_ring)):
                    u = int(seam_ring[j])
                    v = int(seam_ring[(j + 1) % len(seam_ring)])
                    if u == v:
                        continue
                    roof_seam_edge_pairs.add(tuple(sorted((u, v))))


            # wall surfaces + centroids
            wall_vertical_pairs = set()
            for i in range(kN):
                b1 = corner_offset + i
                b2 = corner_offset + ((i + 1) % kN)
                r1, r2 = wall_top_edges[i]

                top_roof_edge_key = tuple(sorted((r1, r2)))

                if top_roof_edge_key in roof_seam_edge_pairs:
                    print(
                        f"ℹ️ Skipped wall surface at comp {cid}, loop {loop_id}, ring_order {i} "
                        f"because roof edge ({r1}, {r2}) already belongs to a roof_seam surface."
                    )
                    continue

                for roof_idx, base_idx in ((r1, b1), (r2, b2)):
                    wall_key = (int(roof_idx), int(base_idx))
                    if wall_key in wall_vertical_pairs:
                        continue

                    wall_vertical_pairs.add(wall_key)
                    self.edges.append([int(roof_idx), int(base_idx)])
                    self.edge_types.append("wall")
                    self.edge_props.append({
                        "edge_id": len(self.edge_props),
                        "image_id": "generated_wall",
                        "component_id": int(cid),
                        "loop_id": int(loop_id),
                        "ring_order": int(i)
                    })

                self.surface_faces.append({
                    "surface_type": "wall",
                    "component_id": int(cid),
                    "loop_id": int(loop_id),
                    "ring_order": int(i),
                    "vertex_indices": [int(r1), int(r2), int(b2), int(b1)],
                })

                pts = [self.corners[r1], self.corners[r2], self.corners[b1], self.corners[b2]]
                self.face_centroids.append(np.mean(pts, axis=0))

        self.face_centroids = (
            np.array(self.face_centroids, dtype=float)
            if len(self.face_centroids) > 0
            else np.empty((0, 3))
        )

    def finalize_for_preview(self):
        deleted_vertices = set(np.where(np.isnan(self.corners).any(axis=1))[0].tolist())
        if len(deleted_vertices) > 0:
            (
                self.corners,
                self.edges,
                self.edge_types,
                self.edge_props,
                _,
                new_idx_to_id
            ) = remap_after_vertex_deletions(
                self.corners,
                self.edges,
                self.edge_types,
                self.edge_props,
                deleted_vertices,
                self.idx_to_id
            )
            if new_idx_to_id is not None:
                self.idx_to_id = new_idx_to_id

        if len(self.edges) == 0:
            print("⚠️ No edges remain after deletions; skipping base/walls.")
            self.face_centroids = np.empty((0, 3))
            self.fitted_face_planes = []
            self.plane_vertex_corrections = []
            self.final_result = None
            return

        roof_only_snapshot = self._snapshot_variant_state()
        self._restore_variant_state(roof_only_snapshot)
        self._build_variant_from_current_roof(use_stage2=True)
        self.final_result = self._snapshot_variant_state()
        self._restore_variant_state(self.final_result)

    def _add_variant_to_viewer(
            self,
            vis,
            variant_snap,
            variant_name,
            x_offset,
            lidar_debug_geoms,
            plane_geoms,
            roof_surface_geoms,
            wall_surface_geoms,
            previous_roof_edge_geoms,
            add_planes=True,
            show_corrected_vertices=False,
            z_origin=None,
        ):
        if variant_snap is None:
            return

        corners = np.asarray(variant_snap["corners"], dtype=float)
        edges = [e[:] for e in variant_snap["edges"]]
        edge_types = variant_snap["edge_types"][:]
        fitted_face_planes = variant_snap["fitted_face_planes"]
        plane_vertex_corrections = variant_snap["plane_vertex_corrections"]
        surface_faces = variant_snap.get("surface_faces", [])
        pre_adjustment_roof_edge_segments = variant_snap.get("pre_adjustment_roof_edge_segments", [])

        if corners.size == 0 or len(edges) == 0:
            return

        origin = corners.mean(axis=0)
        if z_origin is not None:
            origin[2] = float(z_origin)
        shift = np.array([x_offset, 0.0, 0.0], dtype=float)

        # -------------------------------------------------
        # LiDAR cloud
        # -------------------------------------------------
        lidar_disp = (self.lidar_pts - origin) + shift
        zs = self.lidar_pts[:, 2]
        zmin, zmax = zs.min(), zs.max()
        norm_z = (zs - zmin) / (zmax - zmin + 1e-8)
        cmap = plt.get_cmap("viridis")
        colors = cmap(norm_z)[:, :3]

        pcd = geometry.PointCloud(utility.Vector3dVector(lidar_disp))
        pcd.colors = utility.Vector3dVector(colors)
        vis.add_geometry(pcd)
        lidar_debug_geoms.append(pcd)

        # -------------------------------------------------
        # Plane meshes/outlines (ONLY for full variant)
        # -------------------------------------------------
        if add_planes:
            for fp in fitted_face_planes:
                mesh = build_plane_mesh_from_polygon(
                    fp["polygon"],
                    fp["a"], fp["b"], fp["c"],
                    origin
                )
                if mesh is not None:
                    mesh.translate(shift, relative=True)
                    vis.add_geometry(mesh)
                    plane_geoms.append(mesh)

                outline = build_plane_outline_lineset(
                    fp["polygon"],
                    fp["a"], fp["b"], fp["c"],
                    origin
                )
                if outline is not None:
                    outline.translate(shift, relative=True)
                    vis.add_geometry(outline)
                    plane_geoms.append(outline)

        # -------------------------------------------------
        # Faces
        # -------------------------------------------------
        for sf in surface_faces:
            surface_type = sf.get("surface_type")
            ring = sf.get("vertex_indices", [])
            if len(ring) < 3:
                continue

            if surface_type == "wall":
                mesh = build_wall_quad_mesh_from_ring(
                    corners,
                    ring,
                    origin,
                    surface_color_from_type(surface_type),
                    x_shift=shift
                )
            else:
                mesh = build_surface_mesh_from_ring(
                    corners,
                    ring,
                    origin,
                    surface_color_from_type(surface_type),
                    x_shift=shift
                )

            if mesh is not None:
                vis.add_geometry(mesh)

                if surface_type in ("roof", "roof_seam"):
                    roof_surface_geoms.append(mesh)

                elif surface_type == "wall":
                    wall_surface_geoms.append(mesh)

        # -------------------------------------------------
        # Semantic wireframe colors
        # -------------------------------------------------
        corners_disp = (corners - origin) + shift
        pts = []
        idxs = []
        cols = []

        for (s, t), etype in zip(edges, edge_types):
            pts.append(corners_disp[s])
            pts.append(corners_disp[t])
            idxs.append([len(pts) - 2, len(pts) - 1])

            if etype == "roof":
                cols.append(ROOF_EDGE_COLOR)
            elif etype == "wall":
                cols.append(WALL_EDGE_COLOR)
            elif etype == "base":
                cols.append(BASE_EDGE_COLOR)
            else:
                cols.append([0.4, 0.4, 0.4])

        if len(pts) > 0:
            ls = geometry.LineSet(
                points=utility.Vector3dVector(np.asarray(pts, dtype=float)),
                lines=utility.Vector2iVector(np.asarray(idxs, dtype=np.int32))
            )
            ls.colors = utility.Vector3dVector(np.asarray(cols, dtype=float))
            vis.add_geometry(ls)

        prev_roof_edges = build_lineset_from_segments(
            pre_adjustment_roof_edge_segments,
            origin=origin,
            color=PRE_ADJUSTMENT_ROOF_EDGE_COLOR,
            x_shift=shift
        )
        if prev_roof_edges is not None:
            vis.add_geometry(prev_roof_edges)
            previous_roof_edge_geoms.append(prev_roof_edges)

        # -------------------------------------------------
        # Corrected/split vertices
        # OFF for both variants now
        # -------------------------------------------------
        if show_corrected_vertices:
            corrected_vertex_ids = sorted(set(
                rec["new_vertex_idx"] for rec in plane_vertex_corrections
            ))
            for vi in corrected_vertex_ids:
                if 0 <= vi < len(corners_disp):
                    sph = geometry.TriangleMesh.create_sphere(radius=0.26)
                    sph.translate(corners_disp[vi], relative=False)
                    sph.paint_uniform_color(PLANE_CORRECTED_VERTEX_COLOR)
                    vis.add_geometry(sph)

        # -------------------------------------------------
        # Inlier point clouds (green)
        # -------------------------------------------------
        for pts_in in self.fit_points_inliers:
            if pts_in is None or len(pts_in) == 0:
                continue
            pcd_in = geometry.PointCloud(
                utility.Vector3dVector((pts_in - origin) + shift)
            )
            pcd_in.paint_uniform_color([0.0, 1.0, 0.0])
            vis.add_geometry(pcd_in)
            lidar_debug_geoms.append(pcd_in)

        # IMPORTANT:
        # no label_marker here

    def run_3d_viewer(self):
        if self.final_result is None:
            print("⚠️ Final result is required for preview.")
            return "redo"

        vis = o3d.visualization.VisualizerWithKeyCallback()
        progress_txt = ""
        if self.current_num is not None and self.total_num is not None:
            progress_txt = f"{self.current_num}/{self.total_num} — "
        vis.create_window(
            f"{progress_txt}3D Preview — {os.path.basename(self.geojson_path)}",
            1700, 850
        )

        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = 5.0
        opt.line_width = 5.0
        opt.show_coordinate_frame = False
        opt.mesh_show_back_face = True

        lidar_debug_geoms = []
        plane_geoms = []
        roof_surface_geoms = []
        wall_surface_geoms = []
        previous_roof_edge_geoms = []

        if self.lidar_pts is not None and len(self.lidar_pts) > 0:
            x_span = float(self.lidar_pts[:, 0].max() - self.lidar_pts[:, 0].min())
        else:
            x_span = 0.0

        final_corners = np.asarray(self.final_result.get("corners", []), dtype=float)
        if final_corners.size > 0:
            model_x_span = float(final_corners[:, 0].max() - final_corners[:, 0].min())
        else:
            model_x_span = 0.0

        if self.lidar_pts is not None and len(self.lidar_pts) > 0:
            preview_z_origin = float(self.lidar_pts[:, 2].mean())
        elif final_corners.size > 0:
            preview_z_origin = float(final_corners[:, 2].mean())
        else:
            preview_z_origin = 0.0

        preview_span = max(x_span, model_x_span)
        gap = max(PREVIEW_MIN_COLUMN_GAP, preview_span * PREVIEW_COLUMN_GAP_FACTOR)

        center_offset = 0.0

        print("👁️ 3D preview layout:")
        print("   CENTER = Final method + GT overlay")

        self._add_variant_to_viewer(
            vis=vis,
            variant_snap=self.final_result,
            variant_name="Final",
            x_offset=center_offset,
            lidar_debug_geoms=lidar_debug_geoms,
            plane_geoms=plane_geoms,
            roof_surface_geoms=roof_surface_geoms,
            wall_surface_geoms=wall_surface_geoms,
            previous_roof_edge_geoms=previous_roof_edge_geoms,
            add_planes=True,
            show_corrected_vertices=False,
            z_origin=preview_z_origin
        )

        nav = {"action": None}
        lidar_visible = {"on": True}
        planes_visible = {"on": True}
        roof_surfaces_visible = {"on": True}
        wall_surfaces_visible = {"on": True}
        previous_roof_edges_visible = {"on": True}

        def cb(action):
            def _cb(v):
                nav["action"] = action
                v.close()
                return False
            return _cb

        def toggle_lidar(v):
            if lidar_visible["on"]:
                for g in lidar_debug_geoms:
                    try:
                        v.remove_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                lidar_visible["on"] = False
                print("👁️ LiDAR/debug overlays: OFF (planes remain visible)")
            else:
                for g in lidar_debug_geoms:
                    try:
                        v.add_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                lidar_visible["on"] = True
                print("👁️ LiDAR/debug overlays: ON")

            v.poll_events()
            v.update_renderer()
            return False

        def toggle_previous_roof_edges(v):
            if previous_roof_edges_visible["on"]:
                for g in previous_roof_edge_geoms:
                    try:
                        v.remove_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                previous_roof_edges_visible["on"] = False
                print("👁️ Previous red roof edges: OFF")
            else:
                for g in previous_roof_edge_geoms:
                    try:
                        v.add_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                previous_roof_edges_visible["on"] = True
                print("👁️ Previous red roof edges: ON")

            v.poll_events()
            v.update_renderer()
            return False

        def toggle_planes(v):
            if planes_visible["on"]:
                for g in plane_geoms:
                    try:
                        v.remove_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                planes_visible["on"] = False
                print("👁️ Fitted blue roof plane overlays: OFF")
            else:
                for g in plane_geoms:
                    try:
                        v.add_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                planes_visible["on"] = True
                print("👁️ Fitted blue roof plane overlays: ON")

            v.poll_events()
            v.update_renderer()
            return False

        def toggle_roof_surfaces(v):
            if roof_surfaces_visible["on"]:
                for g in roof_surface_geoms:
                    try:
                        v.remove_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                roof_surfaces_visible["on"] = False
                print("👁️ Generated light roof surfaces: OFF (fitted blue planes remain visible)")
            else:
                for g in roof_surface_geoms:
                    try:
                        v.add_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass
                roof_surfaces_visible["on"] = True
                print("👁️ Generated light roof surfaces: ON")

            v.poll_events()
            v.update_renderer()
            return False
        
        def toggle_wall_surfaces(v):
            if wall_surfaces_visible["on"]:
                for g in wall_surface_geoms:
                    try:
                        v.remove_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass

                wall_surfaces_visible["on"] = False
                print("👁️ Wall surfaces: OFF")
            else:
                for g in wall_surface_geoms:
                    try:
                        v.add_geometry(g, reset_bounding_box=False)
                    except Exception:
                        pass

                wall_surfaces_visible["on"] = True
                print("👁️ Wall surfaces: ON")

            v.poll_events()
            v.update_renderer()
            return False

        def save_snapshot(v):
            out_path = get_next_snapshot_path(SNAPSHOT_DIR, self.basename)
            v.poll_events()
            v.update_renderer()
            v.capture_screen_image(out_path, do_render=True)
            print(f"📸 Snapshot saved: {out_path}")
            return False

        vis.register_key_callback(ord("N"), cb("next"))
        vis.register_key_callback(ord("P"), cb("prev"))
        vis.register_key_callback(ord("R"), cb("redo"))
        vis.register_key_callback(ord("Q"), cb("quit"))
        vis.register_key_callback(ord("L"), toggle_lidar)
        vis.register_key_callback(ord("O"), toggle_previous_roof_edges)
        vis.register_key_callback(ord("F"), toggle_planes)
        vis.register_key_callback(ord("G"), toggle_roof_surfaces)
        vis.register_key_callback(ord("W"), toggle_wall_surfaces)
        vis.register_key_callback(ord("S"), save_snapshot)

        vis.run()

        try:
            vis.destroy_window()
        except Exception:
            pass

        return nav["action"] if nav["action"] else "quit"

    def save(self):
        if self.final_result is None:
            raise RuntimeError("Final result must exist before saving.")

        save_3d_geojson(
            self.final_result["corners"],
            self.final_result["edges"],
            self.final_result["edge_types"],
            self.final_result["edge_props"],
            self.final_result["idx_to_id"],
            self.output_path,
            surface_faces=self.final_result["surface_faces"],
            face_centroids=self.final_result["face_centroids"]
        )

        upsert_rmse_row(
            RMSE_CSV_PATH,
            {
                "basename": self.basename,
                "n_roof_ref_points": self.n_roof_ref_points,
                "rmse_stage1": "",
                "rmse_full": "" if self.rmse_final is None else self.rmse_final,
                "better_method": "final_only",
            }
        )

        print("✅ Saved final result:")
        print("   •", self.output_path)
        print("✅ Updated RMSE CSV:")
        print("   •", RMSE_CSV_PATH)

    def run(self):
        print(
            f"\n============================\n"
            f"📄 GEOJSON: {self.geojson_path}\n"
            f"🛰️  LAZ   : {self.laz_path}\n"
            f"🖼️  TIFF  : {self.geotiff_path}\n"
            f"💾 OUT    : {self.output_path}\n"
            f"============================\n"
        )

        self.load()
        self.run_2d_picker()

        if self.nav_action == "quit":
            return "quit"
        if self.nav_action != "preview":
            return "redo"

        self.finalize_for_preview()
        self.compute_lidar_rmse()
        nav_3d = self.run_3d_viewer()

        if nav_3d == "next":
            self.save()
            return "next"
        if nav_3d == "prev":
            self.save()
            return "prev"
        if nav_3d == "redo":
            return "redo"

        return "quit"


def collect_filter_basenames(process_only_from_dir):
    if not os.path.isdir(process_only_from_dir):
        raise FileNotFoundError(f"PROCESS_ONLY_FROM_DIR not found: {process_only_from_dir}")

    base_names = set()

    for p in glob.glob(os.path.join(process_only_from_dir, "*.geojson")):
        base_names.add(os.path.splitext(os.path.basename(p))[0])

    for p in glob.glob(os.path.join(process_only_from_dir, "*.laz")):
        base_names.add(os.path.splitext(os.path.basename(p))[0])

    for p in glob.glob(os.path.join(process_only_from_dir, "*.tif")):
        base_names.add(os.path.splitext(os.path.basename(p))[0])

    for p in glob.glob(os.path.join(process_only_from_dir, "*.tiff")):
        base_names.add(os.path.splitext(os.path.basename(p))[0])

    return sorted(base_names)


def collect_triplets(geojson_dir, laz_dir, geotiff_dir, process_only_from_dir=None):
    geos = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(geojson_dir, "*.geojson"))
    }

    lazs = {
        os.path.splitext(os.path.basename(p))[0]: p
        for p in glob.glob(os.path.join(laz_dir, "*.laz"))
    }

    tifs = {}
    for p in glob.glob(os.path.join(geotiff_dir, "*.tif")):
        tifs[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(geotiff_dir, "*.tiff")):
        tifs[os.path.splitext(os.path.basename(p))[0]] = p

    common = set(geos.keys()) & set(lazs.keys()) & set(tifs.keys())

    if process_only_from_dir:
        wanted = set(collect_filter_basenames(process_only_from_dir))
        common = common & wanted

        missing_in_main = sorted(wanted - (set(geos.keys()) | set(lazs.keys()) | set(tifs.keys())))
        if missing_in_main:
            print("⚠️ These names exist in PROCESS_ONLY_FROM_DIR but not in the main data folders:")
            for name in missing_in_main:
                print("   -", name)

    common = sorted(common)
    return [(base, geos[base], lazs[base], tifs[base]) for base in common]


def is_valid_saved_3d_geojson(output_path):
    if not os.path.exists(output_path):
        return False

    if os.path.getsize(output_path) == 0:
        return False

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            return False
        if data.get("type") != "FeatureCollection":
            return False
        if "features" not in data or not isinstance(data["features"], list):
            return False

        return True

    except Exception:
        return False
    
def get_output_path(base, output_dir):
    return os.path.join(output_dir, f"{base}_3d.geojson")


def is_valid_saved_variant_pair(base, output_dir):
    return is_valid_saved_3d_geojson(get_output_path(base, output_dir))


def find_resume_index(triplets, output_dir):
    for idx, (base, _, _, _) in enumerate(triplets):
        if not is_valid_saved_variant_pair(base, output_dir):
            return idx
    return len(triplets)


def main_batch():
    print(KEY_HELP)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    triplets = collect_triplets(
        GEOJSON_DIR,
        LAZ_DIR,
        GEOTIFF_DIR,
        PROCESS_ONLY_FROM_DIR
    )

    if not triplets:
        print("❌ No matching filtered .geojson + .laz + .tif/.tiff triplets found.")
        return

    N = len(triplets)

    completed = 0
    for base, _, _, _ in triplets:
        if is_valid_saved_variant_pair(base, OUTPUT_DIR):
            completed += 1

    resume_idx = find_resume_index(triplets, OUTPUT_DIR)

    if resume_idx >= N:
        print(f"✅ All {N} filtered files are already completed.")
        return

    print(f"Found {N} filtered file triplets.")
    print(f"Already completed: {completed}")
    print(f"Resuming from file {resume_idx + 1}/{N}: {triplets[resume_idx][0]}")

    idx = resume_idx

    while 0 <= idx < N:
        base, geo_path, laz_path, tif_path = triplets[idx]
        out_path = os.path.join(OUTPUT_DIR, f"{base}_3d.geojson")

        session = FileSession(
            geo_path,
            laz_path,
            tif_path,
            out_path,
            current_num=idx + 1,
            total_num=N
        )
        action = session.run()

        if action == "quit":
            print("👋 Exiting batch.")
            break

        elif action == "redo":
            print("🔄 Redoing current file...")
            continue

        elif action == "next":
            if idx == N - 1:
                print("✅ Last filtered file saved. Done.")
                break
            idx += 1

        elif action == "prev":
            if idx == 0:
                print("⏮️ Already at first file; staying here.")
            else:
                idx -= 1

        else:
            print("⚠️ Unknown action; exiting.")
            break


if __name__ == "__main__":
    main_batch()
