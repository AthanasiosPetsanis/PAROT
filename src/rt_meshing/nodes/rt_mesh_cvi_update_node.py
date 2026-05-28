#!/usr/bin/env python3

import os
import heapq
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rospy
import rosgraph
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
try:
    from scipy.spatial import cKDTree as _SciPyKDTree
except Exception:
    _SciPyKDTree = None


_LOG_BLUE = "\033[94m"
_LOG_RESET = "\033[0m"


def _blue_log_text(value) -> str:
    return f"{_LOG_BLUE}{value}{_LOG_RESET}"


def _blue_log_image_id(image_id) -> str:
    try:
        return _blue_log_text("%06d" % int(image_id))
    except Exception:
        return _blue_log_text(str(image_id))


def _blue_log_image_name(name: str) -> str:
    return _blue_log_text(str(name))


def _get_building_params_from_profile():
    keys = ("center_x", "center_y", "center_z", "width", "length", "height")
    defaults = {
        "center_x": 0.0,
        "center_y": 0.0,
        "center_z": 0.0,
        "width": 20.0,
        "length": 30.0,
        "height": 10.0,
    }
    values = {k: rospy.get_param("~building/" + k, defaults[k]) for k in keys}
    active = str(rospy.get_param("~building/active_profile", "manual")).strip()
    if active and active.lower() not in ("manual", "none", "off"):
        profile = rospy.get_param("~building/profiles/" + active, None)
        if isinstance(profile, dict):
            for k in keys:
                if k in profile:
                    values[k] = profile[k]
                else:
                    rospy.logwarn(
                        "[WARN] Building profile '%s' missing key '%s'; using ~building/%s fallback.",
                        active,
                        k,
                        k,
                    )
            rospy.loginfo(
                "Building profile active: %s center=(%.3f, %.3f, %.3f) size=(%.3f, %.3f, %.3f)",
                active,
                float(values["center_x"]),
                float(values["center_y"]),
                float(values["center_z"]),
                float(values["width"]),
                float(values["length"]),
                float(values["height"]),
            )
        else:
            rospy.logwarn(
                "[WARN] Building active_profile '%s' not found under ~building/profiles; using explicit ~building values.",
                active,
            )
    return tuple(float(values[k]) for k in keys)


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Return a 3x3 rotation matrix from quaternion (x,y,z,w)."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = float(np.dot(q, q))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q *= 1.0 / np.sqrt(n)
    x, y, z, w = q.tolist()

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _atomic_savez(path: str, arrays: Dict[str, np.ndarray]) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _edge(ax: float, ay: float, bx: float, by: float, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    # 2D cross product (b - a) x (p - a)
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def rasterize_visible_triangle_indices(
    triangles_world: np.ndarray,
    cam_pos_world: np.ndarray,
    cam_quat_xyzw_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    near_eps: float = 1e-3,
    shutdown_check=None,
) -> np.ndarray:
    """
    Software z-buffer rasterizer.

    Coordinate convention: camera +X forward (depth), +Y right, +Z down.
    """
    triangles_world = np.asarray(triangles_world, dtype=np.float32)
    cam_pos_world = np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
    cam_quat_xyzw_world = np.asarray(cam_quat_xyzw_world, dtype=np.float64).reshape(4)

    h = int(height)
    w = int(width)

    depth_buf = np.full((h, w), np.inf, dtype=np.float32)
    tri_idx_buf = np.full((h, w), -1, dtype=np.int32)

    # World->camera
    r_world_cam = _quat_xyzw_to_rot(
        float(cam_quat_xyzw_world[0]),
        float(cam_quat_xyzw_world[1]),
        float(cam_quat_xyzw_world[2]),
        float(cam_quat_xyzw_world[3]),
    )
    r_cam_world = r_world_cam.T

    verts = triangles_world.reshape(-1, 3).astype(np.float64)
    verts_rel = verts - cam_pos_world[None, :]
    verts_cam = (r_cam_world @ verts_rel.T).T.reshape((-1, 3, 3))

    x = verts_cam[:, :, 0]  # forward/depth
    y = verts_cam[:, :, 1]  # right
    z = verts_cam[:, :, 2]  # down

    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + fx * (y / x)
        v = cy + fy * (z / x)

    num_tris = int(verts_cam.shape[0])
    for tri_idx in range(num_tris):
        if shutdown_check is not None and shutdown_check():
            break

        x0, x1, x2 = float(x[tri_idx, 0]), float(x[tri_idx, 1]), float(x[tri_idx, 2])
        if min(x0, x1, x2) <= near_eps:
            continue

        u0, u1, u2 = float(u[tri_idx, 0]), float(u[tri_idx, 1]), float(u[tri_idx, 2])
        v0, v1, v2 = float(v[tri_idx, 0]), float(v[tri_idx, 1]), float(v[tri_idx, 2])
        if not (
            np.isfinite(u0)
            and np.isfinite(u1)
            and np.isfinite(u2)
            and np.isfinite(v0)
            and np.isfinite(v1)
            and np.isfinite(v2)
        ):
            continue

        xmin = int(np.floor(min(u0, u1, u2)))
        xmax = int(np.ceil(max(u0, u1, u2)))
        ymin = int(np.floor(min(v0, v1, v2)))
        ymax = int(np.ceil(max(v0, v1, v2)))

        if xmax < 0 or ymax < 0 or xmin >= w or ymin >= h:
            continue

        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w - 1, xmax)
        ymax = min(h - 1, ymax)
        if xmin > xmax or ymin > ymax:
            continue

        area = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0)
        if abs(area) < 1e-8:
            continue

        xs = (np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5)[None, :]
        ys = (np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5)[:, None]

        e0 = _edge(u1, v1, u2, v2, xs, ys)
        e1 = _edge(u2, v2, u0, v0, xs, ys)
        e2 = _edge(u0, v0, u1, v1, xs, ys)

        if area > 0.0:
            inside = (e0 >= 0.0) & (e1 >= 0.0) & (e2 >= 0.0)
        else:
            inside = (e0 <= 0.0) & (e1 <= 0.0) & (e2 <= 0.0)

        if not bool(np.any(inside)):
            continue

        w0 = e0 / area
        w1 = e1 / area
        w2 = e2 / area

        inv_x0 = 1.0 / x0
        inv_x1 = 1.0 / x1
        inv_x2 = 1.0 / x2
        inv_depth = w0 * inv_x0 + w1 * inv_x1 + w2 * inv_x2
        depth = 1.0 / inv_depth

        region_depth = depth_buf[ymin : ymax + 1, xmin : xmax + 1]
        region_idx = tri_idx_buf[ymin : ymax + 1, xmin : xmax + 1]

        update = inside & (depth < region_depth)
        if bool(np.any(update)):
            region_depth[update] = depth[update].astype(np.float32)
            region_idx[update] = tri_idx

    return tri_idx_buf


def _parse_pose_line(line: str) -> Optional[Tuple[int, np.ndarray, np.ndarray, str]]:
    # Expected: image_id qw qx qy qz tx ty tz 1 image_000001.jpg
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    try:
        image_id = int(parts[0])
        qw = float(parts[1])
        qx = float(parts[2])
        qy = float(parts[3])
        qz = float(parts[4])
        tx = float(parts[5])
        ty = float(parts[6])
        tz = float(parts[7])
        img_name = parts[9]
    except Exception:
        return None

    cam_pos = np.array([tx, ty, tz], dtype=np.float64)
    cam_quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    return image_id, cam_pos, cam_quat_xyzw, img_name


def _cvi_temperature_color(t: float) -> Tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    points = [
        (0.0, (0.0, 0.0, 1.0)),
        (0.25, (0.0, 1.0, 1.0)),
        (0.5, (0.0, 1.0, 0.0)),
        (0.75, (1.0, 1.0, 0.0)),
        (1.0, (1.0, 0.0, 0.0)),
    ]
    for i in range(len(points) - 1):
        t0, c0 = points[i]
        t1, c1 = points[i + 1]
        if t <= t1:
            if t1 <= t0:
                return c1
            w = (t - t0) / (t1 - t0)
            return (
                c0[0] + w * (c1[0] - c0[0]),
                c0[1] + w * (c1[1] - c0[1]),
                c0[2] + w * (c1[2] - c0[2]),
            )
    return points[-1][1]


def _normalize_minmax_to_range(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    if vals.size == 0:
        return vals.copy()
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-8:
        return np.full_like(vals, float(out_min), dtype=np.float32)
    alpha = (vals - vmin) / float(vmax - vmin)
    return (float(out_min) + alpha * float(out_max - out_min)).astype(np.float32)


def _build_vertices_and_topology(
    triangles: np.ndarray,
    quantize_eps: float,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    flat = triangles.reshape(-1, 3).astype(np.float64)
    eps = float(max(quantize_eps, 0.0))
    if eps > 0.0:
        q = np.round(flat / eps).astype(np.int64)
        _, unique_idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        vertices = flat[unique_idx]
    else:
        vertices, inv = np.unique(flat, axis=0, return_inverse=True)
    tri_vidx = inv.reshape(-1, 3).astype(np.int32)

    n_v = int(vertices.shape[0])
    v_to_tri: List[List[int]] = [[] for _ in range(n_v)]
    v_neighbors: List[set] = [set() for _ in range(n_v)]
    for tid, tri in enumerate(tri_vidx):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        v_to_tri[a].append(tid)
        v_to_tri[b].append(tid)
        v_to_tri[c].append(tid)
        v_neighbors[a].add(b)
        v_neighbors[a].add(c)
        v_neighbors[b].add(a)
        v_neighbors[b].add(c)
        v_neighbors[c].add(a)
        v_neighbors[c].add(b)

    v_to_tri_arr: List[np.ndarray] = []
    for lst in v_to_tri:
        if lst:
            v_to_tri_arr.append(np.asarray(lst, dtype=np.int32))
        else:
            v_to_tri_arr.append(np.zeros((0,), dtype=np.int32))

    v_neighbors_arr: List[np.ndarray] = []
    for nb in v_neighbors:
        if nb:
            v_neighbors_arr.append(np.asarray(sorted(nb), dtype=np.int32))
        else:
            v_neighbors_arr.append(np.zeros((0,), dtype=np.int32))

    return vertices, tri_vidx, v_to_tri_arr, v_neighbors_arr


def _vertex_normal(
    vid: int,
    vertices: np.ndarray,
    tri_vidx: np.ndarray,
    v_to_tri: List[np.ndarray],
    visible_tri_mask: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    tri_ids = v_to_tri[int(vid)]
    if tri_ids.size == 0:
        return None

    if visible_tri_mask is not None:
        vis = tri_ids[visible_tri_mask[tri_ids]]
        if vis.size > 0:
            tri_ids = vis

    n = np.zeros((3,), dtype=np.float64)
    for tid in tri_ids.tolist():
        ia, ib, ic = tri_vidx[int(tid)]
        a = vertices[int(ia)]
        b = vertices[int(ib)]
        c = vertices[int(ic)]
        nn = np.cross(b - a, c - a)
        if np.all(np.isfinite(nn)):
            n += nn

    ln = float(np.linalg.norm(n))
    if ln <= 1e-12:
        return None
    return n / ln


def _barycentric_coords(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[np.ndarray]:
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) <= 1e-14:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.array([u, v, w], dtype=np.float64)


def _split_triangle_along_edge(
    tri: Tuple[int, int, int],
    edge_u: int,
    edge_v: int,
    mid_vid: int,
) -> Optional[List[Tuple[int, int, int]]]:
    t = (int(tri[0]), int(tri[1]), int(tri[2]))
    u = int(edge_u)
    v = int(edge_v)
    m = int(mid_vid)
    for i in range(3):
        a = int(t[i])
        b = int(t[(i + 1) % 3])
        c = int(t[(i + 2) % 3])
        if (a == u and b == v) or (a == v and b == u):
            # Preserve parent winding by following the local (a -> b -> c) order.
            return [(a, m, c), (m, b, c)]
    return None


def _ray_triangle_intersection(
    ray_o: np.ndarray,
    ray_dir: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-9,
) -> Optional[np.ndarray]:
    # Moller-Trumbore intersection.
    e1 = b - a
    e2 = c - a
    pvec = np.cross(ray_dir, e2)
    det = float(np.dot(e1, pvec))
    if abs(det) <= eps:
        return None
    inv_det = 1.0 / det
    tvec = ray_o - a
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < -eps or u > 1.0 + eps:
        return None
    qvec = np.cross(tvec, e1)
    v = float(np.dot(ray_dir, qvec)) * inv_det
    if v < -eps or (u + v) > 1.0 + eps:
        return None
    t = float(np.dot(e2, qvec)) * inv_det
    if t <= eps:
        return None
    return ray_o + t * ray_dir


def _find_invalid_triangles(
    old_vertices: np.ndarray,
    new_vertices: np.ndarray,
    tri_vidx: np.ndarray,
    tri_ids: np.ndarray,
    min_area: float,
) -> np.ndarray:
    bad: List[int] = []
    min_a = float(min_area)
    for tid in tri_ids.tolist():
        ia, ib, ic = tri_vidx[int(tid)]
        oa = old_vertices[int(ia)]
        ob = old_vertices[int(ib)]
        oc = old_vertices[int(ic)]
        na = new_vertices[int(ia)]
        nb = new_vertices[int(ib)]
        nc = new_vertices[int(ic)]

        old_n = np.cross(ob - oa, oc - oa)
        new_n = np.cross(nb - na, nc - na)
        area = float(np.linalg.norm(new_n))
        if (not np.isfinite(area)) or area < min_a:
            bad.append(int(tid))
            continue
        if float(np.dot(old_n, new_n)) <= 0.0:
            bad.append(int(tid))
            continue

    if not bad:
        return np.zeros((0,), dtype=np.int32)
    return np.asarray(sorted(set(bad)), dtype=np.int32)


class RTMeshCVIUpdateNode:
    def __init__(self) -> None:
        rospy.init_node("rt_mesh_cvi_update_node")

        poses_dir = rospy.get_param("~file_paths/poses_dir", "/home/thanos/Documents/IROS_2026/")
        mesh_dir = rospy.get_param(
            "~file_paths/mesh_path",
            "/home/thanos/Documents/IROS_2026/RT_meshing/Mesh_Space/RTmesh_test0/",
        )

        self.poses_path = rospy.get_param(
            "~rt_meshing/cvi_poses_path",
            os.path.join(poses_dir, "poses", "poses.txt"),
        )
        self.mesh_path = rospy.get_param(
            "~rt_meshing/cvi_mesh_path",
            os.path.join(mesh_dir, "mesh_latest.npz"),
        )
        self.geometry_point_source = str(rospy.get_param("~rt_meshing/geometry_point_source", "sfm")).strip().lower()
        if self.geometry_point_source not in ("sfm", "depth"):
            rospy.logwarn("[WARN] Unsupported geometry_point_source '%s'; falling back to sfm.", self.geometry_point_source)
            self.geometry_point_source = "sfm"
        sfm_sparse_path = rospy.get_param(
            "~rt_meshing/mesh_update_sparse_path",
            os.path.join(mesh_dir, "sparse_latest.npz"),
        )
        depth_sparse_path = rospy.get_param(
            "~rt_meshing/mesh_update_depth_path",
            os.path.join(mesh_dir, "depth_latest.npz"),
        )
        self.sparse_path = depth_sparse_path if self.geometry_point_source == "depth" else sfm_sparse_path
        self.sfm_output_path = rospy.get_param(
            "~rt_meshing/sfm_output_npz",
            os.path.join(mesh_dir, "sparse_latest.npz"),
        )
        self.depth_output_path = rospy.get_param(
            "~rt_meshing/depth_output_npz",
            os.path.join(mesh_dir, "depth_latest.npz"),
        )
        self.reset_mesh_on_start = bool(rospy.get_param("~rt_meshing/reset_mesh_on_start", False))

        self.stride = int(rospy.get_param("~rt_meshing/cvi_stride", 1))
        self.poll_hz = float(rospy.get_param("~rt_meshing/cvi_poll_hz", 10.0))
        self.start_at_end = bool(rospy.get_param("~rt_meshing/cvi_start_at_end", False))
        self.frame_id = rospy.get_param("~rt_meshing/sfm_frame_id", "odom_local_ned")

        self.wait_for_controller = bool(rospy.get_param("~rt_meshing/wait_for_controller", True))
        self.controller_node_name = str(
            rospy.get_param("~rt_meshing/controller_node_name", "/controller_node")
        ).strip()
        if not self.controller_node_name:
            self.controller_node_name = "/controller_node"
        if not self.controller_node_name.startswith("/"):
            self.controller_node_name = "/" + self.controller_node_name
        self._ros_master = rosgraph.Master(rospy.get_name())

        fx_full = float(rospy.get_param("~calibration/fx", 381.361145))
        fy_full = float(rospy.get_param("~calibration/fy", 381.361145))
        cx_full = float(rospy.get_param("~calibration/cx", 320.0))
        cy_full = float(rospy.get_param("~calibration/cy", 240.0))
        width_full = int(rospy.get_param("~calibration/width", 640))
        height_full = int(rospy.get_param("~calibration/height", 480))

        render_w = int(rospy.get_param("~rt_meshing/cvi_render_width", max(1, width_full // 2)))
        render_h = int(rospy.get_param("~rt_meshing/cvi_render_height", max(1, height_full // 2)))
        self.width = max(1, render_w)
        self.height = max(1, render_h)

        sx = float(self.width) / float(max(1, width_full))
        sy = float(self.height) / float(max(1, height_full))
        self.fx = fx_full * sx
        self.fy = fy_full * sy
        self.cx = cx_full * sx
        self.cy = cy_full * sy

        self.near_eps = float(rospy.get_param("~rt_meshing/cvi_near_eps", 1e-3))
        self.cvi_update_mode = str(
            rospy.get_param("~rt_meshing/mesh_cvi_update_mode", "geometric")
        ).strip().lower()
        if self.cvi_update_mode not in ("rasterize", "geometric"):
            self.cvi_update_mode = "geometric"
        self.save_debug_images = bool(rospy.get_param("~rt_meshing/cvi_save_debug_images", False))
        self.debug_every = max(1, int(rospy.get_param("~rt_meshing/cvi_debug_every", 1)))
        self.debug_dir = rospy.get_param(
            "~rt_meshing/cvi_debug_dir",
            os.path.join(mesh_dir, "cvi_renders"),
        )
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Mesh geometry update (merged in this node)
        self.mesh_update_enabled = bool(rospy.get_param("~rt_meshing/mesh_update_enabled", True))
        self.max_corr_l0_weight = float(
            rospy.get_param("~rt_meshing/mesh_update_max_corr_l0_weight", 1.0)
        )
        if not np.isfinite(self.max_corr_l0_weight) or self.max_corr_l0_weight <= 0.0:
            self.max_corr_l0_weight = 1.0
        self.max_correspondence_dist_param = float(
            rospy.get_param("~rt_meshing/mesh_update_max_correspondence_dist", 0.75)
        )
        self.max_correspondence_dist = float(self.max_correspondence_dist_param)
        self.split_corr_l0_weight = float(
            rospy.get_param("~rt_meshing/mesh_split_corr_l0_weight", 0.5)
        )
        if (not np.isfinite(self.split_corr_l0_weight)) or self.split_corr_l0_weight <= 0.0:
            self.split_corr_l0_weight = 0.5
        self.split_correspondence_dist = float(
            rospy.get_param("~rt_meshing/mesh_split_correspondence_dist", 0.5 * self.max_correspondence_dist)
        )
        self.quantize_eps = float(rospy.get_param("~rt_meshing/mesh_update_quantize_eps", 1e-6))
        self.min_triangle_area = float(rospy.get_param("~rt_meshing/mesh_update_min_triangle_area", 1e-10))
        self.vertex_miss_threshold = max(1, int(rospy.get_param("~rt_meshing/mesh_update_vertex_miss_threshold", 1)))
        self.shrink_step_m = float(rospy.get_param("~rt_meshing/mesh_update_shrink_step_m", 0.02))
        self.intersection_logic_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_update_intersection_logic_enabled", False)
        )
        self.knn_k = max(1, int(rospy.get_param("~rt_meshing/mesh_update_knn_k", 5)))
        self.knn_min_neighbors = max(1, int(rospy.get_param("~rt_meshing/mesh_update_knn_min_neighbors", 1)))
        self.knn_smooth_ring = max(0, int(rospy.get_param("~rt_meshing/mesh_update_knn_smooth_ring", 1)))
        self.knn_smooth_lambda = float(rospy.get_param("~rt_meshing/mesh_update_knn_smooth_lambda", 0.2))
        self.knn_smooth_iters = max(0, int(rospy.get_param("~rt_meshing/mesh_update_knn_smooth_iters", 1)))
        self.pseudomerge_enabled = bool(rospy.get_param("~rt_meshing/mesh_pseudomerge_enabled", False))
        self.pseudomerge_cos_threshold = float(
            rospy.get_param("~rt_meshing/mesh_pseudomerge_cos_threshold", 0.985)
        )
        self.pseudomerge_cos_threshold = float(np.clip(self.pseudomerge_cos_threshold, -1.0, 1.0))
        self.pseudomerge_step_l0_weight = float(
            rospy.get_param("~rt_meshing/mesh_pseudomerge_step_l0_weight", 0.02)
        )
        if (not np.isfinite(self.pseudomerge_step_l0_weight)) or self.pseudomerge_step_l0_weight < 0.0:
            self.pseudomerge_step_l0_weight = 0.02
        self.pseudomerge_max_step_m = float(
            rospy.get_param("~rt_meshing/mesh_pseudomerge_max_step_m", 0.02)
        )
        if (not np.isfinite(self.pseudomerge_max_step_m)) or self.pseudomerge_max_step_m < 0.0:
            self.pseudomerge_max_step_m = 0.02
        self.pseudomerge_iters = max(0, int(rospy.get_param("~rt_meshing/mesh_pseudomerge_iters", 1)))
        self.pseudomerge_visible_only = bool(rospy.get_param("~rt_meshing/mesh_pseudomerge_visible_only", True))
        self.rollback_enabled = bool(rospy.get_param("~rt_meshing/mesh_update_rollback_enabled", False))
        self.split_enabled = bool(rospy.get_param("~rt_meshing/mesh_split_enabled", True))
        self.split_edge_growth_ratio = float(
            rospy.get_param("~rt_meshing/mesh_split_edge_growth_ratio", 0.8)
        )
        if (not np.isfinite(self.split_edge_growth_ratio)) or self.split_edge_growth_ratio < 0.0:
            self.split_edge_growth_ratio = 0.8
        self.split_support_min_neighbors = max(
            1, int(rospy.get_param("~rt_meshing/mesh_split_support_min_neighbors", 1))
        )
        self.prune_enabled = bool(rospy.get_param("~rt_meshing/mesh_prune_enabled", True))
        self.prune_edge_ratio = float(rospy.get_param("~rt_meshing/mesh_prune_edge_ratio", 0.2))
        if (not np.isfinite(self.prune_edge_ratio)) or self.prune_edge_ratio < 0.0:
            self.prune_edge_ratio = 0.2
        self.prune_min_angle_deg = float(rospy.get_param("~rt_meshing/mesh_prune_min_angle_deg", 0.0))
        if (not np.isfinite(self.prune_min_angle_deg)) or self.prune_min_angle_deg < 0.0:
            self.prune_min_angle_deg = 0.0
        self.prune_min_angle_rad = float(np.deg2rad(self.prune_min_angle_deg))
        self.split_max_inserts_per_parent = max(
            0, int(rospy.get_param("~rt_meshing/mesh_split_max_inserts_per_parent", 4))
        )
        self.split_corner_area_fraction = float(
            rospy.get_param("~rt_meshing/mesh_split_corner_area_fraction", 0.15)
        )
        self.split_corner_area_fraction = float(np.clip(self.split_corner_area_fraction, 1e-4, 0.95))
        self.split_corner_gate = float(1.0 - np.sqrt(self.split_corner_area_fraction))
        self.split_bary_dedup_eps = float(
            rospy.get_param("~rt_meshing/mesh_split_bary_dedup_eps", 0.03)
        )
        self.split_bary_dedup_eps = float(max(self.split_bary_dedup_eps, 1e-6))
        self.rt_mesh_include_bottom = bool(rospy.get_param("~rt_meshing/include_bottom", True))
        mesh_center_z_offset = float(rospy.get_param("~rt_meshing/center_z_offset", 0.0))
        build_cx, build_cy, build_cz, _, _, build_h = _get_building_params_from_profile()
        self.build_height = float(build_h)
        self.build_center = np.array(
            [build_cx, build_cy, -build_cz + mesh_center_z_offset],
            dtype=np.float64,
        )
        self.mesh_view_ground_clearance_m = float(
            rospy.get_param("~rt_meshing/mesh_view_ground_clearance_m", 0.1)
        )
        self.mesh_view_termination_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_termination_enabled", False)
        )
        self.mesh_view_termination_cvi_threshold = float(
            rospy.get_param("~rt_meshing/mesh_view_termination_cvi_threshold", 1.4)
        )

        # Confidence update (kept same idea as previous mesh update node)
        self.conf_ema_alpha = float(rospy.get_param("~rt_meshing/mesh_conf_ema_alpha", 0.2))
        self.conf_min = float(rospy.get_param("~rt_meshing/mesh_conf_min", 1.0))
        self.conf_max = float(rospy.get_param("~rt_meshing/mesh_conf_max", 2.0))
        self.conf_reproj_bad_px = float(
            rospy.get_param(
                "~rt_meshing/mesh_conf_reproj_bad_px",
                float(rospy.get_param("~rt_meshing/sfm_track_confirm_reproj_px", 1.0)),
            )
        )
        self.conf_parallax_good_deg = float(
            rospy.get_param(
                "~rt_meshing/mesh_conf_parallax_good_deg",
                max(2.0, float(rospy.get_param("~rt_meshing/sfm_min_parallax_deg", 1.0))),
            )
        )
        self.conf_dist_bad_m = float(
            rospy.get_param("~rt_meshing/mesh_conf_dist_bad_m", self.max_correspondence_dist)
        )
        self.depth_good_view_observations = max(
            1.0, float(rospy.get_param("~rt_meshing/depth_good_view_observations", 5.0))
        )
        self.depth_good_sample_count = max(
            1.0, float(rospy.get_param("~rt_meshing/depth_good_sample_count", 30.0))
        )
        self.depth_view_conf_weight = max(
            0.0, float(rospy.get_param("~rt_meshing/depth_view_conf_weight", 0.7))
        )
        self.depth_sample_conf_weight = max(
            0.0, float(rospy.get_param("~rt_meshing/depth_sample_conf_weight", 0.3))
        )
        self.depth_voxel_size = max(
            1.0e-6, float(rospy.get_param("~rt_meshing/depth_voxel_size", 0.10))
        )
        self.depth_mesh_patching = bool(rospy.get_param("~rt_meshing/depth_mesh_patching", False))
        self.depth_mesh_patch_intersections_topic = str(
            rospy.get_param(
                "~rt_meshing/depth_mesh_patch_intersections_topic",
                "controller/depth_mesh_patch_intersections",
            )
        )
        self.depth_mesh_patch_cut_loop_topic = str(
            rospy.get_param(
                "~rt_meshing/depth_mesh_patch_cut_loop_topic",
                "controller/depth_mesh_patch_cut_loop",
            )
        )
        self.depth_mesh_patch_failed_rays_topic = str(
            rospy.get_param(
                "~rt_meshing/depth_mesh_patch_failed_rays_topic",
                "controller/depth_mesh_patch_failed_rays",
            )
        )
        self.depth_mesh_patch_cut_max_boundary_edges = max(
            0, int(rospy.get_param("~rt_meshing/depth_mesh_patch_cut_max_boundary_edges", 400))
        )
        self.depth_mesh_patch_cut_diagnostics_enabled = bool(
            rospy.get_param("~rt_meshing/depth_mesh_patch_cut_diagnostics_enabled", False)
        )

        self._pose_fp = None
        self._pose_offset = 0
        self._event_counter = 0
        self._processed_counter = 0
        self._shutdown_requested = False
        self._timing_num_updates = 0
        self._timing_stats_pct: Dict[str, Dict[str, float]] = {}
        self._timing_stats_ms: Dict[str, Dict[str, float]] = {}
        self._geom_timing_stats_ms: Dict[str, Dict[str, float]] = {}
        self._assoc_timing_stats_ms: Dict[str, Dict[str, float]] = {}
        self._assign_timing_stats_ms: Dict[str, Dict[str, float]] = {}
        rospy.on_shutdown(self._on_shutdown)

        # Extra safety against startup race: remove stale artifacts so controller recreates fresh mesh.
        if self.reset_mesh_on_start:
            for p in [self.mesh_path, self.sparse_path, self.sfm_output_path, self.depth_output_path]:
                if not p:
                    continue
                try:
                    if os.path.isfile(p):
                        os.remove(p)
                        rospy.logwarn("reset_mesh_on_start=true: removed stale file %s", p)
                except Exception as exc:
                    rospy.logwarn("reset_mesh_on_start=true: failed removing %s (%s)", p, str(exc))

        # Sparse cache to avoid reloading npz if unchanged.
        self._sparse_cache_mtime: Optional[float] = None
        self._sparse_points = np.zeros((0, 3), dtype=np.float64)
        self._sparse_point_ids = np.zeros((0,), dtype=np.int64)
        self._sparse_point_revs = np.zeros((0,), dtype=np.int64)
        self._sparse_point_revs_valid = False
        self._sparse_track_ids = np.zeros((0,), dtype=np.int64)
        self._sparse_views_support = np.zeros((0,), dtype=np.float64)
        self._sparse_views_total = np.zeros((0,), dtype=np.float64)
        self._sparse_reproj = np.zeros((0,), dtype=np.float64)
        self._sparse_parallax = np.zeros((0,), dtype=np.float64)
        self._sparse_quality_base = np.zeros((0,), dtype=np.float64)
        self._bucket_num_vertices = 0
        self._bucket_sum_xyz = np.zeros((0, 3), dtype=np.float64)
        self._bucket_sum_q = np.zeros((0,), dtype=np.float64)
        self._bucket_count = np.zeros((0,), dtype=np.int32)
        self._bucket_track_assignments: Dict[int, Dict[str, np.ndarray]] = {}
        self._bucket_processed_revs: Dict[int, int] = {}
        self._bucket_has_stable_ids = True
        self._bucket_needs_rebuild = True
        self._depth_patch_vertices = np.zeros((0, 3), dtype=np.float64)
        self._depth_patch_boundary_edges = np.zeros((0, 2), dtype=np.int32)
        self._depth_patch_camera_position = np.zeros((3,), dtype=np.float64)
        self._depth_patch_label = ""
        self._depth_accum_patch_keys = np.zeros((0, 3, 3), dtype=np.int64)
        self._depth_accum_patch_triangles = np.zeros((0, 3, 3), dtype=np.float64)
        self._depth_accum_patch_cvi_raw = np.zeros((0,), dtype=np.float32)
        self._depth_accum_patch_vertex_conf = np.zeros((0, 3), dtype=np.float32)
        self.depth_mesh_patch_intersections_pub = rospy.Publisher(
            self.depth_mesh_patch_intersections_topic,
            Marker,
            queue_size=1,
            latch=True,
        )
        self.depth_mesh_patch_cut_loop_pub = rospy.Publisher(
            self.depth_mesh_patch_cut_loop_topic,
            Marker,
            queue_size=1,
            latch=True,
        )
        self.depth_mesh_patch_failed_rays_pub = rospy.Publisher(
            self.depth_mesh_patch_failed_rays_topic,
            Marker,
            queue_size=1,
            latch=True,
        )

        rospy.loginfo("RTMeshCVIUpdateNode configured:")
        rospy.loginfo("  poses_path: %s", self.poses_path)
        rospy.loginfo("  mesh_path:  %s", self.mesh_path)
        rospy.loginfo("  geometry_point_source: %s", self.geometry_point_source)
        rospy.loginfo("  sparse_path: %s", self.sparse_path)
        rospy.loginfo("  stride:     %d", self.stride)
        rospy.loginfo("  poll_hz:    %.2f", self.poll_hz)
        rospy.loginfo("  start_at_end: %s", str(self.start_at_end))
        rospy.loginfo("  reset_mesh_on_start: %s", str(self.reset_mesh_on_start))
        rospy.loginfo(
            "  intrinsics(cvi): fx=%.3f fy=%.3f cx=%.3f cy=%.3f size=%dx%d",
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.width,
            self.height,
        )
        rospy.loginfo("  mesh_update_enabled: %s", str(self.mesh_update_enabled))
        rospy.loginfo("  depth_mesh_patching: %s", str(self.depth_mesh_patching))
        rospy.loginfo("  cvi_update_mode: %s", self.cvi_update_mode)
        rospy.loginfo(
            "  max_corr=%.3f split_corr=%.3f max_corr_l0_w=%.3f split_corr_l0_w=%.3f miss_thresh=%d shrink=%.4f mode_intersection=%s rollback=%s prune_enabled=%s prune_edge_ratio=%.3f prune_min_angle_deg=%.2f split_enabled=%s split_growth=%.3f split_min_n=%d corner_area=%.3f dedup=%.4f",
            self.max_correspondence_dist,
            self.split_correspondence_dist,
            self.max_corr_l0_weight,
            self.split_corr_l0_weight,
            self.vertex_miss_threshold,
            self.shrink_step_m,
            str(self.intersection_logic_enabled),
            str(self.rollback_enabled),
            str(self.prune_enabled),
            self.prune_edge_ratio,
            self.prune_min_angle_deg,
            str(self.split_enabled),
            self.split_edge_growth_ratio,
            self.split_support_min_neighbors,
            self.split_corner_area_fraction,
            self.split_bary_dedup_eps,
        )
        rospy.loginfo(
            "  knn: k=%d min_neighbors=%d smooth_ring=%d smooth_lambda=%.3f smooth_iters=%d scipy_kdtree=%s",
            self.knn_k,
            self.knn_min_neighbors,
            self.knn_smooth_ring,
            self.knn_smooth_lambda,
            self.knn_smooth_iters,
            "yes" if _SciPyKDTree is not None else "no (numpy fallback)",
        )
        if _SciPyKDTree is None:
            rospy.logwarn(
                "SciPy cKDTree not available; mesh sparse association will use the NumPy all-vertices fallback, which can be much slower. Install scipy to enable the fast KD-tree path."
            )
        rospy.loginfo(
            "  pseudomerge: enabled=%s cos_thresh=%.4f step_l0_w=%.4f max_step_m=%.4f iters=%d visible_only=%s",
            str(self.pseudomerge_enabled),
            self.pseudomerge_cos_threshold,
            self.pseudomerge_step_l0_weight,
            self.pseudomerge_max_step_m,
            self.pseudomerge_iters,
            str(self.pseudomerge_visible_only),
        )
        rospy.loginfo(
            "  conf: alpha=%.3f range=[%.2f, %.2f] reproj_bad=%.2f parallax_good=%.2f dist_bad=%.2f",
            self.conf_ema_alpha,
            self.conf_min,
            self.conf_max,
            self.conf_reproj_bad_px,
            self.conf_parallax_good_deg,
            self.conf_dist_bad_m,
        )
        rospy.loginfo("  save_debug_images: %s", str(self.save_debug_images))
        if self.save_debug_images:
            rospy.loginfo("  debug_dir: %s (every %d updates)", self.debug_dir, self.debug_every)
        if self.wait_for_controller:
            rospy.loginfo("  controller gate: enabled (node=%s)", self.controller_node_name)

    def _on_shutdown(self) -> None:
        self._shutdown_requested = True
        if self._pose_fp is not None:
            try:
                self._pose_fp.close()
            except Exception:
                pass

    def _ensure_pose_file_open(self) -> bool:
        if not os.path.isfile(self.poses_path):
            return False
        if self._pose_fp is None or self._pose_fp.closed:
            self._pose_fp = open(self.poses_path, "r", encoding="utf-8", errors="ignore")
        return True

    def _load_mesh(self) -> Optional[Dict[str, np.ndarray]]:
        if not os.path.isfile(self.mesh_path):
            return None
        try:
            data = np.load(self.mesh_path, allow_pickle=False)
            arrays = {k: data[k] for k in data.files}
            data.close()
            return arrays
        except Exception as exc:
            rospy.logwarn("Failed to load mesh npz '%s': %s", self.mesh_path, str(exc))
            return None

    def _save_mesh(self, arrays: Dict[str, np.ndarray]) -> None:
        try:
            _atomic_savez(self.mesh_path, arrays)
        except Exception as exc:
            rospy.logwarn("Failed to save mesh npz '%s': %s", self.mesh_path, str(exc))

    def _load_sparse_cache(self) -> bool:
        if not os.path.isfile(self.sparse_path):
            rospy.logwarn_throttle(2.0, "Sparse npz not found: %s", self.sparse_path)
            had_sparse = (
                self._sparse_cache_mtime is not None
                or self._sparse_points.shape[0] > 0
                or self._bucket_track_assignments
            )
            self._sparse_points = np.zeros((0, 3), dtype=np.float64)
            self._sparse_point_ids = np.zeros((0,), dtype=np.int64)
            self._sparse_point_revs = np.zeros((0,), dtype=np.int64)
            self._sparse_point_revs_valid = False
            self._sparse_track_ids = np.zeros((0,), dtype=np.int64)
            self._sparse_views_support = np.zeros((0,), dtype=np.float64)
            self._sparse_views_total = np.zeros((0,), dtype=np.float64)
            self._sparse_reproj = np.zeros((0,), dtype=np.float64)
            self._sparse_parallax = np.zeros((0,), dtype=np.float64)
            self._sparse_quality_base = np.zeros((0,), dtype=np.float64)
            self._depth_patch_vertices = np.zeros((0, 3), dtype=np.float64)
            self._depth_patch_boundary_edges = np.zeros((0, 2), dtype=np.int32)
            self._depth_patch_camera_position = np.zeros((3,), dtype=np.float64)
            self._depth_patch_label = ""
            self._depth_accum_patch_keys = np.zeros((0, 3, 3), dtype=np.int64)
            self._depth_accum_patch_triangles = np.zeros((0, 3, 3), dtype=np.float64)
            self._depth_accum_patch_cvi_raw = np.zeros((0,), dtype=np.float32)
            self._depth_accum_patch_vertex_conf = np.zeros((0, 3), dtype=np.float32)
            self._sparse_cache_mtime = None
            if had_sparse:
                self._bucket_needs_rebuild = True
            return had_sparse

        try:
            mtime = os.path.getmtime(self.sparse_path)
        except OSError:
            return False

        if self._sparse_cache_mtime is not None and mtime == self._sparse_cache_mtime:
            return False

        try:
            data = np.load(self.sparse_path, allow_pickle=False)
            points = np.asarray(data.get("points_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64).reshape(-1, 3)
            n = int(points.shape[0])
            point_ids = np.asarray(
                data.get("point_ids", np.full((n,), -1, dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            point_revs_raw = data.get("point_revs", None)
            point_revs_valid = point_revs_raw is not None
            point_revs = np.asarray(
                point_revs_raw if point_revs_raw is not None else np.ones((n,), dtype=np.int64),
                dtype=np.int64,
            ).reshape(-1)
            track_ids = np.asarray(
                data.get("track_ids", np.full((n,), -1, dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            views_support = np.asarray(data.get("views_support", np.ones((n,), dtype=np.float32)), dtype=np.float64).reshape(-1)
            views_total = np.asarray(data.get("views_total", np.maximum(views_support, 1.0)), dtype=np.float64).reshape(-1)
            reproj = np.asarray(
                data.get("reproj_error_px", np.full((n,), float(self.conf_reproj_bad_px), dtype=np.float32)),
                dtype=np.float64,
            ).reshape(-1)
            parallax = np.asarray(
                data.get("parallax_deg", np.full((n,), float(self.conf_parallax_good_deg), dtype=np.float32)),
                dtype=np.float64,
            ).reshape(-1)
            depth_views = np.asarray(
                data.get("depth_view_count", views_support),
                dtype=np.float64,
            ).reshape(-1)
            depth_samples = np.asarray(
                data.get("depth_sample_count", np.ones((n,), dtype=np.float32)),
                dtype=np.float64,
            ).reshape(-1)
            patch_vertices = np.asarray(
                data.get("depth_patch_vertices", np.zeros((0, 3), dtype=np.float32)),
                dtype=np.float64,
            ).reshape(-1, 3)
            patch_boundary_edges = np.asarray(
                data.get("depth_patch_boundary_edges", np.zeros((0, 2), dtype=np.int32)),
                dtype=np.int32,
            ).reshape(-1, 2)
            patch_camera_position = np.asarray(
                data.get("depth_patch_camera_position", np.zeros((3,), dtype=np.float32)),
                dtype=np.float64,
            ).reshape(-1)
            patch_label_arr = np.asarray(data.get("depth_patch_label", np.asarray([""])))
            accumulated_patch_keys = np.asarray(
                data.get("depth_accumulated_patch_triangles", np.zeros((0, 9), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1, 9)
            accumulated_patch_cvi_raw = np.asarray(
                data.get(
                    "depth_accumulated_patch_cvi_raw",
                    np.zeros((accumulated_patch_keys.shape[0],), dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1)
            data.close()

            if point_ids.shape[0] != n:
                point_ids = np.full((n,), -1, dtype=np.int64)
            if point_revs.shape[0] != n:
                point_revs = np.ones((n,), dtype=np.int64)
                point_revs_valid = False
            if track_ids.shape[0] != n:
                track_ids = np.full((n,), -1, dtype=np.int64)
            if views_support.shape[0] != n:
                views_support = np.ones((n,), dtype=np.float64)
            if views_total.shape[0] != n:
                views_total = np.maximum(views_support, 1.0)
            if reproj.shape[0] != n:
                reproj = np.full((n,), float(self.conf_reproj_bad_px), dtype=np.float64)
            if parallax.shape[0] != n:
                parallax = np.full((n,), float(self.conf_parallax_good_deg), dtype=np.float64)
            if depth_views.shape[0] != n:
                depth_views = views_support.copy()
            if depth_samples.shape[0] != n:
                depth_samples = np.ones((n,), dtype=np.float64)
            if patch_camera_position.shape[0] != 3:
                patch_camera_position = np.zeros((3,), dtype=np.float64)

            if self.geometry_point_source == "depth":
                s_views = np.clip(depth_views / float(self.depth_good_view_observations), 0.0, 1.0)
                s_samples = np.clip(depth_samples / float(self.depth_good_sample_count), 0.0, 1.0)
                weight_sum = max(self.depth_view_conf_weight + self.depth_sample_conf_weight, 1.0e-6)
                quality_base = np.clip(
                    (self.depth_view_conf_weight * s_views + self.depth_sample_conf_weight * s_samples)
                    / weight_sum,
                    0.0,
                    1.0,
                ).astype(np.float64)
            else:
                vto = np.maximum(views_total, 1.0)
                vsp = np.clip(views_support, 0.0, vto)
                s_views = vsp / vto
                s_reproj = 1.0 - np.clip(
                    reproj / max(float(self.conf_reproj_bad_px), 1.0e-6),
                    0.0,
                    1.0,
                )
                s_parallax = np.clip(
                    parallax / max(float(self.conf_parallax_good_deg), 1.0e-6),
                    0.0,
                    1.0,
                )
                quality_base = np.clip(
                    0.25 * (s_views + s_reproj + s_parallax),
                    0.0,
                    1.0,
                ).astype(np.float64)

            self._sparse_points = points
            self._sparse_point_ids = point_ids
            self._sparse_point_revs = np.maximum(point_revs, 1)
            self._sparse_point_revs_valid = bool(point_revs_valid)
            self._sparse_track_ids = track_ids
            self._sparse_views_support = views_support
            self._sparse_views_total = views_total
            self._sparse_reproj = reproj
            self._sparse_parallax = parallax
            self._sparse_quality_base = quality_base
            self._depth_patch_vertices = patch_vertices
            self._depth_patch_boundary_edges = patch_boundary_edges
            self._depth_patch_camera_position = patch_camera_position.astype(np.float64, copy=False)
            (
                self._depth_accum_patch_keys,
                self._depth_accum_patch_triangles,
                self._depth_accum_patch_cvi_raw,
                self._depth_accum_patch_vertex_conf,
            ) = self._build_accumulated_depth_patch_mesh(
                accumulated_patch_keys=accumulated_patch_keys,
                accumulated_patch_cvi_raw=accumulated_patch_cvi_raw,
                points=points,
                quality_base=quality_base,
            )
            try:
                self._depth_patch_label = str(patch_label_arr.reshape(-1)[0])
            except Exception:
                self._depth_patch_label = ""
            self._sparse_cache_mtime = mtime
            if not self._sparse_point_revs_valid and n > 0:
                rospy.logwarn_throttle(
                    10.0,
                    "Sparse npz is missing point_revs or has invalid shape; mesh association will fall back to full rebuild until SfM exports valid point revisions.",
                )
            return True
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to load sparse npz '%s': %s", self.sparse_path, str(exc))
            return False

    @staticmethod
    def _point_msg(p: np.ndarray) -> Point:
        arr = np.asarray(p, dtype=np.float64).reshape(3)
        return Point(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    @staticmethod
    def _canonical_depth_patch_key(row: np.ndarray) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        arr = np.asarray(row, dtype=np.int64).reshape(3, 3)
        return tuple(sorted(tuple(int(x) for x in key) for key in arr))

    def _build_accumulated_depth_patch_mesh(
        self,
        accumulated_patch_keys: np.ndarray,
        accumulated_patch_cvi_raw: np.ndarray,
        points: np.ndarray,
        quality_base: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        keys_raw = np.asarray(accumulated_patch_keys, dtype=np.int64).reshape(-1, 9)
        if keys_raw.shape[0] <= 0:
            return (
                np.zeros((0, 3, 3), dtype=np.int64),
                np.zeros((0, 3, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )

        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        q = np.asarray(quality_base, dtype=np.float64).reshape(-1)
        point_keys = np.floor(pts / float(self.depth_voxel_size)).astype(np.int64) if pts.shape[0] > 0 else np.zeros((0, 3), dtype=np.int64)
        key_to_idx: Dict[Tuple[int, int, int], int] = {}
        for idx, key in enumerate(point_keys):
            kt = (int(key[0]), int(key[1]), int(key[2]))
            if kt not in key_to_idx:
                key_to_idx[kt] = int(idx)

        cvi_raw = np.asarray(accumulated_patch_cvi_raw, dtype=np.float32).reshape(-1)
        out_keys: List[np.ndarray] = []
        out_triangles: List[np.ndarray] = []
        out_cvi: List[float] = []
        out_conf: List[np.ndarray] = []
        seen: Set[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = set()
        for row_idx, row in enumerate(keys_raw):
            tri_keys = np.asarray(row, dtype=np.int64).reshape(3, 3)
            canonical = self._canonical_depth_patch_key(tri_keys)
            if canonical in seen or len(set(canonical)) != 3:
                continue
            tri_pts: List[np.ndarray] = []
            tri_conf: List[float] = []
            missing = False
            for key_arr in tri_keys:
                kt = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
                pidx = key_to_idx.get(kt)
                if pidx is None:
                    missing = True
                    break
                tri_pts.append(np.asarray(pts[int(pidx)], dtype=np.float64).reshape(3))
                qv = float(q[int(pidx)]) if int(pidx) < int(q.shape[0]) else 0.0
                tri_conf.append(float(self.conf_min + np.clip(qv, 0.0, 1.0) * (self.conf_max - self.conf_min)))
            if missing or len(tri_pts) != 3:
                continue
            pa, pb, pc = tri_pts
            area2 = float(np.linalg.norm(np.cross(pb - pa, pc - pa)))
            if (not np.isfinite(area2)) or area2 <= 1.0e-12:
                continue
            seen.add(canonical)
            out_keys.append(tri_keys)
            out_triangles.append(np.asarray(tri_pts, dtype=np.float64).reshape(3, 3))
            raw = float(cvi_raw[row_idx]) if row_idx < int(cvi_raw.shape[0]) else 0.0
            out_cvi.append(raw if np.isfinite(raw) and raw > 0.0 else 0.0)
            out_conf.append(np.asarray(tri_conf, dtype=np.float32).reshape(3))

        if not out_triangles:
            return (
                np.zeros((0, 3, 3), dtype=np.int64),
                np.zeros((0, 3, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
            )
        return (
            np.asarray(out_keys, dtype=np.int64).reshape(-1, 3, 3),
            np.asarray(out_triangles, dtype=np.float64).reshape(-1, 3, 3),
            np.asarray(out_cvi, dtype=np.float32).reshape(-1),
            np.asarray(out_conf, dtype=np.float32).reshape(-1, 3),
        )

    def _clear_depth_patch_cut_markers(self) -> None:
        for pub, ns, mid in (
            (self.depth_mesh_patch_intersections_pub, "depth_mesh_patch_intersections", 0),
            (self.depth_mesh_patch_cut_loop_pub, "depth_mesh_patch_cut_loop", 0),
            (self.depth_mesh_patch_failed_rays_pub, "depth_mesh_patch_failed_rays", 0),
        ):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.frame_id
            marker.ns = ns
            marker.id = int(mid)
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0
            pub.publish(marker)

    @staticmethod
    def _closest_ray_mesh_intersection(
        ray_o: np.ndarray,
        target: np.ndarray,
        triangles: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, int]]:
        origin = np.asarray(ray_o, dtype=np.float64).reshape(3)
        tgt = np.asarray(target, dtype=np.float64).reshape(3)
        ray = tgt - origin
        norm = float(np.linalg.norm(ray))
        if (not np.isfinite(norm)) or norm <= 1.0e-12:
            return None
        ray_dir = ray / norm
        best_t = np.inf
        best_p = None
        best_tid = -1
        tri = np.asarray(triangles, dtype=np.float64).reshape(-1, 3, 3)
        for tid in range(int(tri.shape[0])):
            p = _ray_triangle_intersection(origin, ray_dir, tri[tid, 0], tri[tid, 1], tri[tid, 2])
            if p is None:
                continue
            t = float(np.dot(p - origin, ray_dir))
            if t > 1.0e-9 and t < best_t:
                best_t = t
                best_p = p
                best_tid = int(tid)
        if best_p is None or best_tid < 0:
            return None
        return np.asarray(best_p, dtype=np.float64).reshape(3), int(best_tid)

    @staticmethod
    def _nearest_triangle_vertex(
        point: np.ndarray,
        tri_id: int,
        vertices: np.ndarray,
        tri_vidx: np.ndarray,
    ) -> int:
        vids = np.asarray(tri_vidx[int(tri_id)], dtype=np.int32).reshape(3)
        pts = vertices[vids.astype(np.int64)]
        d = np.linalg.norm(pts - np.asarray(point, dtype=np.float64).reshape(1, 3), axis=1)
        return int(vids[int(np.argmin(d))])

    @staticmethod
    def _shortest_vertex_path(
        vertices: np.ndarray,
        neighbors: List[np.ndarray],
        start: int,
        goal: int,
    ) -> List[int]:
        start = int(start)
        goal = int(goal)
        if start == goal:
            return [start]
        n = int(vertices.shape[0])
        if start < 0 or goal < 0 or start >= n or goal >= n:
            return []

        dist = np.full((n,), np.inf, dtype=np.float64)
        prev = np.full((n,), -1, dtype=np.int32)
        dist[start] = 0.0
        heap: List[Tuple[float, int]] = [(0.0, start)]
        visited: Set[int] = set()
        while heap:
            cur_d, cur = heapq.heappop(heap)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == goal:
                break
            if cur_d > float(dist[cur]):
                continue
            for nb in neighbors[cur].tolist():
                nb = int(nb)
                w = float(np.linalg.norm(vertices[nb] - vertices[cur]))
                if (not np.isfinite(w)) or w <= 0.0:
                    continue
                nd = float(cur_d) + w
                if nd < float(dist[nb]):
                    dist[nb] = nd
                    prev[nb] = cur
                    heapq.heappush(heap, (nd, nb))

        if not np.isfinite(dist[goal]):
            return []
        out = [goal]
        cur = goal
        while cur != start:
            cur = int(prev[cur])
            if cur < 0:
                return []
            out.append(cur)
        out.reverse()
        return out

    def _publish_depth_patch_cut_diagnostics(self, triangles: np.ndarray) -> None:
        if (
            self.geometry_point_source != "depth"
            or not bool(self.depth_mesh_patching)
            or not bool(self.depth_mesh_patch_cut_diagnostics_enabled)
        ):
            return
        patch_vertices = np.asarray(self._depth_patch_vertices, dtype=np.float64).reshape(-1, 3)
        boundary_edges = np.asarray(self._depth_patch_boundary_edges, dtype=np.int32).reshape(-1, 2)
        if patch_vertices.shape[0] < 2 or boundary_edges.shape[0] <= 0:
            self._clear_depth_patch_cut_markers()
            return

        max_edges = int(self.depth_mesh_patch_cut_max_boundary_edges)
        edges = boundary_edges
        if max_edges > 0 and edges.shape[0] > max_edges:
            sample = np.linspace(0, edges.shape[0] - 1, max_edges).astype(np.int64)
            edges = edges[sample]

        vertices, tri_vidx, _v_to_tri, v_neighbors = _build_vertices_and_topology(
            triangles=np.asarray(triangles, dtype=np.float64),
            quantize_eps=self.quantize_eps,
        )
        if vertices.shape[0] <= 0:
            self._clear_depth_patch_cut_markers()
            return

        cam = np.asarray(self._depth_patch_camera_position, dtype=np.float64).reshape(3)
        unique_boundary_vids = sorted(set(int(v) for edge in edges.tolist() for v in edge if int(v) >= 0))
        hit_cache: Dict[int, Tuple[np.ndarray, int, int]] = {}
        failed_targets: List[np.ndarray] = []
        for pvid in unique_boundary_vids:
            if pvid < 0 or pvid >= int(patch_vertices.shape[0]):
                continue
            target = patch_vertices[int(pvid)]
            hit = self._closest_ray_mesh_intersection(cam, target, triangles)
            if hit is None:
                failed_targets.append(target.copy())
                continue
            hit_point, hit_tid = hit
            anchor = self._nearest_triangle_vertex(hit_point, hit_tid, vertices, tri_vidx)
            hit_cache[int(pvid)] = (hit_point, int(hit_tid), int(anchor))

        stamp = rospy.Time.now()
        hit_marker = Marker()
        hit_marker.header.stamp = stamp
        hit_marker.header.frame_id = self.frame_id
        hit_marker.ns = "depth_mesh_patch_intersections"
        hit_marker.id = 0
        hit_marker.type = Marker.POINTS
        hit_marker.action = Marker.ADD
        hit_marker.pose.orientation.w = 1.0
        hit_marker.scale.x = 0.09
        hit_marker.scale.y = 0.09
        hit_marker.color.r = 1.0
        hit_marker.color.g = 0.2
        hit_marker.color.b = 1.0
        hit_marker.color.a = 1.0
        hit_marker.points = [self._point_msg(hit_cache[pvid][0]) for pvid in sorted(hit_cache.keys())]
        self.depth_mesh_patch_intersections_pub.publish(hit_marker)

        cut_marker = Marker()
        cut_marker.header.stamp = stamp
        cut_marker.header.frame_id = self.frame_id
        cut_marker.ns = "depth_mesh_patch_cut_loop"
        cut_marker.id = 0
        cut_marker.type = Marker.LINE_LIST
        cut_marker.action = Marker.ADD
        cut_marker.pose.orientation.w = 1.0
        cut_marker.scale.x = 0.05
        cut_marker.color.r = 1.0
        cut_marker.color.g = 0.45
        cut_marker.color.b = 0.0
        cut_marker.color.a = 1.0
        cut_points: List[Point] = []
        path_edges = 0
        missing_paths = 0
        for a, b in edges.tolist():
            ia = int(a)
            ib = int(b)
            if ia not in hit_cache or ib not in hit_cache:
                continue
            hit_a, _tid_a, anchor_a = hit_cache[ia]
            hit_b, _tid_b, anchor_b = hit_cache[ib]
            path = self._shortest_vertex_path(vertices, v_neighbors, anchor_a, anchor_b)
            if not path:
                missing_paths += 1
                continue
            poly = [hit_a] + [vertices[int(vid)] for vid in path] + [hit_b]
            for p0, p1 in zip(poly[:-1], poly[1:]):
                cut_points.append(self._point_msg(p0))
                cut_points.append(self._point_msg(p1))
            path_edges += 1
        cut_marker.points = cut_points
        self.depth_mesh_patch_cut_loop_pub.publish(cut_marker)

        failed_marker = Marker()
        failed_marker.header.stamp = stamp
        failed_marker.header.frame_id = self.frame_id
        failed_marker.ns = "depth_mesh_patch_failed_rays"
        failed_marker.id = 0
        failed_marker.type = Marker.LINE_LIST
        failed_marker.action = Marker.ADD
        failed_marker.pose.orientation.w = 1.0
        failed_marker.scale.x = 0.03
        failed_marker.color.r = 1.0
        failed_marker.color.g = 0.0
        failed_marker.color.b = 0.0
        failed_marker.color.a = 1.0
        failed_points: List[Point] = []
        for target in failed_targets:
            failed_points.append(self._point_msg(cam))
            failed_points.append(self._point_msg(target))
        failed_marker.points = failed_points
        self.depth_mesh_patch_failed_rays_pub.publish(failed_marker)

        rospy.loginfo_throttle(
            1.0,
            "Depth patch cut dry-run: label=%s boundary_edges=%d sampled_edges=%d hits=%d failed_rays=%d path_edges=%d missing_paths=%d",
            _blue_log_text(str(self._depth_patch_label)),
            int(boundary_edges.shape[0]),
            int(edges.shape[0]),
            int(len(hit_cache)),
            int(len(failed_targets)),
            int(path_edges),
            int(missing_paths),
        )

    def _reset_vertex_buckets(self, num_vertices: int) -> None:
        n_v = max(0, int(num_vertices))
        self._bucket_num_vertices = n_v
        self._bucket_sum_xyz = np.zeros((n_v, 3), dtype=np.float64)
        self._bucket_sum_q = np.zeros((n_v,), dtype=np.float64)
        self._bucket_count = np.zeros((n_v,), dtype=np.int32)
        self._bucket_track_assignments = {}
        self._bucket_processed_revs = {}

    def _sparse_keys_for_bucketing(self) -> Tuple[np.ndarray, bool]:
        n = int(self._sparse_points.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.int64), True
        point_ids = np.asarray(self._sparse_point_ids, dtype=np.int64).reshape(-1)
        if point_ids.shape[0] != n:
            point_ids = np.full((n,), -1, dtype=np.int64)
        stable = bool(
            np.all(point_ids >= 0) and np.unique(point_ids).shape[0] == point_ids.shape[0]
        )
        if stable:
            return point_ids.astype(np.int64), True
        # Fallback keys are only valid for full rebuilds of the current sparse snapshot.
        return (-(np.arange(n, dtype=np.int64) + 1)), False

    def _remove_bucket_track_assignment(self, track_key: int) -> np.ndarray:
        rec = self._bucket_track_assignments.pop(int(track_key), None)
        self._bucket_processed_revs.pop(int(track_key), None)
        if rec is None:
            return np.zeros((0,), dtype=np.int32)
        vids = np.asarray(rec.get("vids", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        qs = np.asarray(rec.get("qs", np.zeros((vids.shape[0],), dtype=np.float64)), dtype=np.float64).reshape(-1)
        point = np.asarray(rec.get("point", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
        if vids.shape[0] <= 0:
            return vids
        self._bucket_sum_xyz[vids] -= point.reshape(1, 3)
        self._bucket_sum_q[vids] -= qs
        self._bucket_count[vids] -= 1
        self._bucket_count[vids] = np.maximum(self._bucket_count[vids], 0)
        near_zero = self._bucket_count[vids] <= 0
        if np.any(near_zero):
            reset_vids = vids[near_zero]
            self._bucket_sum_xyz[reset_vids] = 0.0
            self._bucket_sum_q[reset_vids] = 0.0
            self._bucket_count[reset_vids] = 0
        return np.unique(vids).astype(np.int32)

    def _assign_sparse_track_to_vertices(
        self,
        track_key: int,
        sparse_idx: int,
        vertices: np.ndarray,
        vertex_tree,
        assign_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, int]:
        n_v = int(vertices.shape[0])
        if n_v <= 0 or sparse_idx < 0 or sparse_idx >= int(self._sparse_points.shape[0]):
            return np.zeros((0,), dtype=np.int32), 0

        point = np.asarray(self._sparse_points[int(sparse_idx)], dtype=np.float64).reshape(3)
        t0 = time.perf_counter()
        if vertex_tree is not None:
            kq = int(min(self.knn_k, n_v))
            if float(self.max_correspondence_dist) >= 0.0:
                dists, vids = vertex_tree.query(
                    point,
                    k=kq,
                    distance_upper_bound=float(self.max_correspondence_dist),
                )
            else:
                dists, vids = vertex_tree.query(point, k=kq)
            dists = np.atleast_1d(np.asarray(dists, dtype=np.float64))
            vids = np.atleast_1d(np.asarray(vids, dtype=np.int64))
        else:
            diff = vertices - point.reshape(1, 3)
            d2 = np.einsum("ij,ij->i", diff, diff)
            kq = int(min(self.knn_k, d2.shape[0]))
            if kq <= 0:
                return np.zeros((0,), dtype=np.int32), 0
            vids = np.argpartition(d2, kth=kq - 1)[:kq]
            order = np.argsort(d2[vids])
            vids = vids[order].astype(np.int64)
            dists = np.sqrt(d2[vids]).astype(np.float64)
        if assign_stats is not None:
            assign_stats["t_assoc_assign_query_ms"] += (time.perf_counter() - t0) * 1000.0
            assign_stats["assoc_assign_calls"] += 1.0

        t0 = time.perf_counter()
        if float(self.max_correspondence_dist) < 0.0:
            ok = np.isfinite(dists) & (vids >= 0) & (vids < n_v)
        else:
            ok = np.isfinite(dists) & (vids >= 0) & (vids < n_v) & (dists <= float(self.max_correspondence_dist))
        ok_count = int(np.count_nonzero(ok))
        if assign_stats is not None:
            assign_stats["t_assoc_assign_filter_ms"] += (time.perf_counter() - t0) * 1000.0
        if int(np.count_nonzero(ok)) < int(self.knn_min_neighbors):
            if assign_stats is not None:
                assign_stats["assoc_assign_rejected_tracks"] += 1.0
                assign_stats["assoc_assign_candidates_total"] += float(dists.shape[0])
                assign_stats["assoc_assign_kept_total"] += float(ok_count)
            return np.zeros((0,), dtype=np.int32), 0

        vids_ok = vids[ok].astype(np.int32)
        d_ok = dists[ok].astype(np.float64)
        if assign_stats is not None:
            assign_stats["assoc_assign_candidates_total"] += float(dists.shape[0])
            assign_stats["assoc_assign_kept_total"] += float(vids_ok.shape[0])

        t0 = time.perf_counter()
        base_q = (
            float(self._sparse_quality_base[int(sparse_idx)])
            if 0 <= int(sparse_idx) < int(self._sparse_quality_base.shape[0])
            else 0.0
        )
        dist_bad = float(self.conf_dist_bad_m)
        if dist_bad <= 0.0:
            dist_bad = float(self.max_correspondence_dist)
        if dist_bad > 0.0:
            s_dist = 1.0 - np.clip(d_ok / max(dist_bad, 1.0e-6), 0.0, 1.0)
        else:
            s_dist = np.ones_like(d_ok, dtype=np.float64)
        qs = np.clip(base_q + 0.25 * s_dist, 0.0, 1.0).astype(np.float64)
        if assign_stats is not None:
            assign_stats["t_assoc_assign_quality_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._bucket_sum_xyz[vids_ok] += point.reshape(1, 3)
        self._bucket_sum_q[vids_ok] += qs
        self._bucket_count[vids_ok] += 1
        if assign_stats is not None:
            assign_stats["t_assoc_assign_bucket_update_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._bucket_track_assignments[int(track_key)] = {
            "vids": vids_ok.astype(np.int32),
            "qs": qs.astype(np.float64),
            "point": point.astype(np.float64),
            "meta": np.array(
                [
                    float(self._sparse_views_support[int(sparse_idx)]),
                    float(self._sparse_views_total[int(sparse_idx)]),
                    float(self._sparse_reproj[int(sparse_idx)]),
                    float(self._sparse_parallax[int(sparse_idx)]),
                ],
                dtype=np.float64,
            ),
        }
        touched_vids = np.unique(vids_ok).astype(np.int32)
        if assign_stats is not None:
            assign_stats["t_assoc_assign_record_ms"] += (time.perf_counter() - t0) * 1000.0
            assign_stats["assoc_assign_accepted_tracks"] += 1.0
            assign_stats["assoc_assign_unique_vid_total"] += float(touched_vids.shape[0])
        return touched_vids, int(vids_ok.shape[0])

    def _refresh_sparse_track_assignment(
        self,
        track_key: int,
        sparse_idx: int,
        vertices: np.ndarray,
        assign_stats: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, int]:
        rec = self._bucket_track_assignments.get(int(track_key), None)
        if rec is None:
            return self._assign_sparse_track_to_vertices(
                track_key=int(track_key),
                sparse_idx=int(sparse_idx),
                vertices=vertices,
                vertex_tree=None,
                assign_stats=assign_stats,
            )

        vids = np.asarray(rec.get("vids", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        old_qs = np.asarray(rec.get("qs", np.zeros((vids.shape[0],), dtype=np.float64)), dtype=np.float64).reshape(-1)
        old_point = np.asarray(rec.get("point", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
        if vids.shape[0] <= 0:
            return np.zeros((0,), dtype=np.int32), 0

        point = np.asarray(self._sparse_points[int(sparse_idx)], dtype=np.float64).reshape(3)
        t0 = time.perf_counter()
        base_q = (
            float(self._sparse_quality_base[int(sparse_idx)])
            if 0 <= int(sparse_idx) < int(self._sparse_quality_base.shape[0])
            else 0.0
        )
        dist_bad = float(self.conf_dist_bad_m)
        if dist_bad <= 0.0:
            dist_bad = float(self.max_correspondence_dist)
        if vids.shape[0] > 0:
            d_ok = np.linalg.norm(vertices[vids.astype(np.int64)] - point.reshape(1, 3), axis=1).astype(np.float64)
        else:
            d_ok = np.zeros((0,), dtype=np.float64)
        if dist_bad > 0.0:
            s_dist = 1.0 - np.clip(d_ok / max(dist_bad, 1.0e-6), 0.0, 1.0)
        else:
            s_dist = np.ones_like(d_ok, dtype=np.float64)
        qs = np.clip(base_q + 0.25 * s_dist, 0.0, 1.0).astype(np.float64)
        if assign_stats is not None:
            assign_stats["t_assoc_assign_quality_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._bucket_sum_xyz[vids] += (point - old_point).reshape(1, 3)
        if old_qs.shape[0] == qs.shape[0]:
            self._bucket_sum_q[vids] += (qs - old_qs)
        else:
            self._bucket_sum_q[vids] -= old_qs[: vids.shape[0]]
            self._bucket_sum_q[vids] += qs[: vids.shape[0]]
        if assign_stats is not None:
            assign_stats["t_assoc_assign_bucket_update_ms"] += (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self._bucket_track_assignments[int(track_key)] = {
            "vids": vids.astype(np.int32),
            "qs": qs.astype(np.float64),
            "point": point.astype(np.float64),
            "meta": np.array(
                [
                    float(self._sparse_views_support[int(sparse_idx)]),
                    float(self._sparse_views_total[int(sparse_idx)]),
                    float(self._sparse_reproj[int(sparse_idx)]),
                    float(self._sparse_parallax[int(sparse_idx)]),
                ],
                dtype=np.float64,
            ),
        }
        if assign_stats is not None:
            assign_stats["t_assoc_assign_record_ms"] += (time.perf_counter() - t0) * 1000.0
            assign_stats["assoc_assign_calls"] += 1.0
            assign_stats["assoc_assign_accepted_tracks"] += 1.0
            assign_stats["assoc_assign_candidates_total"] += float(vids.shape[0])
            assign_stats["assoc_assign_kept_total"] += float(vids.shape[0])
            assign_stats["assoc_assign_unique_vid_total"] += float(np.unique(vids).shape[0])
        touched_vids = np.unique(vids).astype(np.int32)
        return touched_vids, int(vids.shape[0])

    def _prune_bucket_assignments_by_current_distance(self, vertices: np.ndarray) -> np.ndarray:
        n_v = int(vertices.shape[0])
        touched_mask = np.zeros((n_v,), dtype=bool)
        if n_v <= 0 or not self._bucket_track_assignments:
            return touched_mask
        if float(self.max_correspondence_dist) < 0.0:
            return touched_mask

        new_assignments: Dict[int, Dict[str, np.ndarray]] = {}
        new_processed_revs: Dict[int, int] = {}
        self._bucket_sum_xyz.fill(0.0)
        self._bucket_sum_q.fill(0.0)
        self._bucket_count.fill(0)

        max_dist = float(self.max_correspondence_dist)
        min_neighbors = int(max(1, self.knn_min_neighbors))

        for track_key, rec in list(self._bucket_track_assignments.items()):
            vids = np.asarray(rec.get("vids", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
            qs = np.asarray(rec.get("qs", np.zeros((vids.shape[0],), dtype=np.float64)), dtype=np.float64).reshape(-1)
            point = np.asarray(rec.get("point", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(3)
            meta = np.asarray(rec.get("meta", np.zeros((4,), dtype=np.float64)), dtype=np.float64).reshape(-1)
            if vids.shape[0] <= 0:
                continue

            valid_idx = (vids >= 0) & (vids < n_v)
            vids = vids[valid_idx]
            if qs.shape[0] == valid_idx.shape[0]:
                qs = qs[valid_idx]
            else:
                qs = np.zeros((vids.shape[0],), dtype=np.float64)
            if vids.shape[0] <= 0:
                continue

            dists = np.linalg.norm(vertices[vids.astype(np.int64)] - point.reshape(1, 3), axis=1).astype(np.float64)
            ok = np.isfinite(dists) & (dists <= max_dist)
            vids_kept = vids[ok].astype(np.int32)
            qs_kept = qs[ok].astype(np.float64) if qs.shape[0] == ok.shape[0] else np.zeros((vids_kept.shape[0],), dtype=np.float64)

            touched_mask[vids.astype(np.int64)] = True
            if vids_kept.shape[0] < min_neighbors:
                continue

            self._bucket_sum_xyz[vids_kept] += point.reshape(1, 3)
            self._bucket_sum_q[vids_kept] += qs_kept
            self._bucket_count[vids_kept] += 1
            touched_mask[vids_kept.astype(np.int64)] = True
            new_assignments[int(track_key)] = {
                "vids": vids_kept,
                "qs": qs_kept,
                "point": point.astype(np.float64),
                "meta": meta.astype(np.float64),
            }
            if int(track_key) in self._bucket_processed_revs:
                new_processed_revs[int(track_key)] = int(self._bucket_processed_revs[int(track_key)])

        self._bucket_track_assignments = new_assignments
        self._bucket_processed_revs = new_processed_revs
        return touched_mask

    @staticmethod
    def _empty_assoc_sync_stats() -> Dict[str, float]:
        return {
            "assoc_sparse_tracks": 0.0,
            "assoc_vertices": 0.0,
            "assoc_rebuild": 0.0,
            "assoc_stable_ids": 0.0,
            "assoc_point_ids_stable": 0.0,
            "assoc_bucket_stable_ids": 0.0,
            "assoc_point_revs_valid": 0.0,
            "assoc_added_tracks": 0.0,
            "assoc_changed_tracks": 0.0,
            "assoc_removed_tracks": 0.0,
            "assoc_pairs_changed": 0.0,
            "assoc_assign_calls": 0.0,
            "assoc_assign_accepted_tracks": 0.0,
            "assoc_assign_rejected_tracks": 0.0,
            "assoc_assign_candidates_total": 0.0,
            "assoc_assign_kept_total": 0.0,
            "assoc_assign_unique_vid_total": 0.0,
            "t_assoc_key_extract_ms": 0.0,
            "t_assoc_tree_ms": 0.0,
            "t_assoc_change_detect_ms": 0.0,
            "t_assoc_remove_ms": 0.0,
            "t_assoc_assign_ms": 0.0,
            "t_assoc_assign_query_ms": 0.0,
            "t_assoc_assign_filter_ms": 0.0,
            "t_assoc_assign_quality_ms": 0.0,
            "t_assoc_assign_bucket_update_ms": 0.0,
            "t_assoc_assign_record_ms": 0.0,
        }

    def _sync_sparse_vertex_buckets(
        self,
        vertices: np.ndarray,
        force_rebuild: bool = False,
    ) -> Tuple[np.ndarray, int, Dict[str, float]]:
        n_v = int(vertices.shape[0])
        stats = self._empty_assoc_sync_stats()
        stats["assoc_vertices"] = float(n_v)

        t0 = time.perf_counter()
        keys, stable_ids = self._sparse_keys_for_bucketing()
        key_list = keys.tolist()
        point_ids = np.asarray(self._sparse_point_ids, dtype=np.int64).reshape(-1)
        if point_ids.shape[0] != len(key_list):
            point_ids = np.full((len(key_list),), -1, dtype=np.int64)
        point_revs = np.asarray(self._sparse_point_revs, dtype=np.int64).reshape(-1)
        point_revs_valid = bool(self._sparse_point_revs_valid and point_revs.shape[0] == len(key_list))
        if point_revs.shape[0] != len(key_list):
            point_revs = np.ones((len(key_list),), dtype=np.int64)
        point_ids_stable = bool(
            np.all(point_ids >= 0) and np.unique(point_ids).shape[0] == point_ids.shape[0]
        )
        stats["t_assoc_key_extract_ms"] = (time.perf_counter() - t0) * 1000.0
        stats["assoc_sparse_tracks"] = float(len(key_list))
        stats["assoc_stable_ids"] = 1.0 if point_ids_stable else 0.0
        stats["assoc_point_ids_stable"] = 1.0 if point_ids_stable else 0.0
        stats["assoc_bucket_stable_ids"] = 1.0 if stable_ids else 0.0
        stats["assoc_point_revs_valid"] = 1.0 if point_revs_valid else 0.0
        touched_mask = np.zeros((n_v,), dtype=bool)
        assoc_pairs = 0

        rebuild = bool(
            force_rebuild
            or self._bucket_needs_rebuild
            or int(self._bucket_num_vertices) != n_v
            or (not stable_ids)
            or (not self._bucket_has_stable_ids)
            or (not point_revs_valid)
        )

        t0 = time.perf_counter()
        vertex_tree = _SciPyKDTree(vertices) if (_SciPyKDTree is not None and n_v > 0) else None
        stats["t_assoc_tree_ms"] = (time.perf_counter() - t0) * 1000.0
        if _SciPyKDTree is None and n_v > 0:
            rospy.logwarn_throttle(
                30.0,
                "Mesh sparse association is using the NumPy all-vertices fallback because SciPy cKDTree is unavailable.",
            )

        if rebuild:
            stats["assoc_rebuild"] = 1.0
            old_nonempty = (
                self._bucket_count > 0
                if int(self._bucket_num_vertices) == n_v
                else np.zeros((n_v,), dtype=bool)
            )
            self._reset_vertex_buckets(n_v)
            self._bucket_has_stable_ids = bool(stable_ids)

            t0 = time.perf_counter()
            for sparse_idx, track_key in enumerate(key_list):
                vids_changed, pair_count = self._assign_sparse_track_to_vertices(
                    track_key=int(track_key),
                    sparse_idx=int(sparse_idx),
                    vertices=vertices,
                    vertex_tree=vertex_tree,
                    assign_stats=stats,
                )
                assoc_pairs += int(pair_count)
                if vids_changed.shape[0] > 0:
                    touched_mask[vids_changed] = True
                self._bucket_processed_revs[int(track_key)] = int(point_revs[int(sparse_idx)])
            stats["t_assoc_assign_ms"] = (time.perf_counter() - t0) * 1000.0
            stats["assoc_added_tracks"] = float(len(key_list))
            stats["assoc_pairs_changed"] = float(assoc_pairs)
            new_nonempty = self._bucket_count > 0
            if old_nonempty.shape[0] == touched_mask.shape[0]:
                touched_mask |= old_nonempty
            touched_mask |= new_nonempty
            self._bucket_needs_rebuild = False
            return touched_mask, int(assoc_pairs), stats

        t0 = time.perf_counter()
        new_key_to_idx = {int(k): int(i) for i, k in enumerate(key_list)}
        old_keys = set(self._bucket_track_assignments.keys())
        new_keys = set(new_key_to_idx.keys())
        changed_keys = set()

        for track_key in (old_keys & new_keys):
            sparse_idx = int(new_key_to_idx[int(track_key)])
            rev_now = int(point_revs[sparse_idx])
            rev_prev = int(self._bucket_processed_revs.get(int(track_key), 0))
            if rev_now > rev_prev:
                changed_keys.add(int(track_key))
        stats["t_assoc_change_detect_ms"] = (time.perf_counter() - t0) * 1000.0
        stats["assoc_changed_tracks"] = float(len(changed_keys))

        t0 = time.perf_counter()
        for track_key in sorted(old_keys - new_keys):
            vids_changed = self._remove_bucket_track_assignment(int(track_key))
            if vids_changed.shape[0] > 0:
                touched_mask[vids_changed] = True

        for track_key in sorted(changed_keys):
            vids_changed = self._remove_bucket_track_assignment(int(track_key))
            if vids_changed.shape[0] > 0:
                touched_mask[vids_changed] = True
        stats["t_assoc_remove_ms"] = (time.perf_counter() - t0) * 1000.0
        stats["assoc_removed_tracks"] = float(len(old_keys - new_keys) + len(changed_keys))

        t0 = time.perf_counter()
        for track_key in sorted((new_keys - old_keys) | changed_keys):
            sparse_idx = int(new_key_to_idx[int(track_key)])
            if int(track_key) in changed_keys and int(track_key) in old_keys:
                vids_changed, pair_count = self._refresh_sparse_track_assignment(
                    track_key=int(track_key),
                    sparse_idx=sparse_idx,
                    vertices=vertices,
                    assign_stats=stats,
                )
            else:
                vids_changed, pair_count = self._assign_sparse_track_to_vertices(
                    track_key=int(track_key),
                    sparse_idx=sparse_idx,
                    vertices=vertices,
                    vertex_tree=vertex_tree,
                    assign_stats=stats,
                )
            assoc_pairs += int(pair_count)
            if vids_changed.shape[0] > 0:
                touched_mask[vids_changed] = True
            self._bucket_processed_revs[int(track_key)] = int(point_revs[sparse_idx])
        stats["t_assoc_assign_ms"] = (time.perf_counter() - t0) * 1000.0
        stats["assoc_added_tracks"] = float(len(new_keys - old_keys))
        stats["assoc_pairs_changed"] = float(assoc_pairs)

        self._bucket_has_stable_ids = bool(stable_ids)
        self._bucket_needs_rebuild = False
        return touched_mask, int(assoc_pairs), stats

    def _save_debug_raster(
        self,
        tri_idx_buf: np.ndarray,
        tri_cvi_effective: np.ndarray,
        image_id: int,
        image_name: str,
    ) -> None:
        if not self.save_debug_images:
            return
        self._processed_counter += 1
        if (self._processed_counter % self.debug_every) != 0:
            return
        if tri_cvi_effective.size == 0:
            return

        vmin = float(np.min(tri_cvi_effective))
        vmax = float(np.max(tri_cvi_effective))
        if abs(vmax - vmin) < 1e-8:
            norm = np.zeros_like(tri_cvi_effective, dtype=np.float32)
        else:
            norm = (tri_cvi_effective.astype(np.float32) - vmin) / (vmax - vmin)

        lut = np.zeros((tri_cvi_effective.shape[0], 3), dtype=np.uint8)  # BGR
        for i in range(norm.shape[0]):
            r, g, b = _cvi_temperature_color(float(norm[i]))
            lut[i, 0] = int(np.clip(round(b * 255.0), 0, 255))
            lut[i, 1] = int(np.clip(round(g * 255.0), 0, 255))
            lut[i, 2] = int(np.clip(round(r * 255.0), 0, 255))

        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask = tri_idx_buf >= 0
        if np.any(mask):
            idx = tri_idx_buf[mask]
            valid = (idx >= 0) & (idx < lut.shape[0])
            if np.any(valid):
                yy, xx = np.where(mask)
                img[yy[valid], xx[valid]] = lut[idx[valid]]

        out_name = f"cvi_raster_{int(image_id):06d}.png"
        out_path = os.path.join(self.debug_dir, out_name)
        if not cv2.imwrite(out_path, img):
            rospy.logwarn("Failed to save CVI raster debug image: %s", out_path)
            return
        rospy.loginfo("Saved CVI raster debug image for %s -> %s", _blue_log_image_name(image_name), out_path)

    def _infer_bottom_inactive_triangles(
        self,
        triangles: np.ndarray,
        mesh_l0_value: Optional[float] = None,
    ) -> np.ndarray:
        tri = np.asarray(triangles, dtype=np.float64)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            return np.zeros((0,), dtype=bool)
        num_tris = int(tri.shape[0])
        if num_tris <= 0 or (not bool(self.rt_mesh_include_bottom)):
            return np.zeros((num_tris,), dtype=bool)
        centers = np.mean(tri, axis=1)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cross_norm = np.linalg.norm(cross, axis=1)
        valid = cross_norm > 1.0e-10
        normals = np.zeros_like(cross, dtype=np.float64)
        normals[valid] = cross[valid] / cross_norm[valid][:, None]
        outward = np.einsum("ij,ij->i", normals, centers - self.build_center.reshape(1, 3))
        outward_sign = np.where(outward < 0.0, -1.0, 1.0)
        normals = normals * outward_sign[:, None]
        bottom_z = float(self.build_center[2] + 0.5 * float(self.build_height))
        l0 = None
        try:
            if mesh_l0_value is not None:
                l0 = float(mesh_l0_value)
        except Exception:
            l0 = None
        if l0 is None or (not np.isfinite(l0)) or l0 <= 0.0:
            l0 = 0.5
        tol = float(max(5.0e-2, 1.5 * l0))
        vertex_near_bottom = np.all(np.abs(tri[:, :, 2] - bottom_z) <= tol, axis=1)
        centroid_near_bottom = np.abs(centers[:, 2] - bottom_z) <= tol
        downward_outward = normals[:, 2] >= 0.7
        return np.asarray(valid & vertex_near_bottom & centroid_near_bottom & downward_outward, dtype=bool)

    def _planner_max_region_z(self) -> float:
        bottom_z = float(self.build_center[2] + 0.5 * float(self.build_height))
        return float(bottom_z - float(self.mesh_view_ground_clearance_m))

    def _planner_triangle_mask(self, triangles: np.ndarray) -> np.ndarray:
        tri = np.asarray(triangles, dtype=np.float64)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            return np.zeros((0,), dtype=bool)
        centroids = np.mean(tri, axis=1)
        return np.asarray(centroids[:, 2] <= float(self._planner_max_region_z()), dtype=bool)

    def _cvi_update_active_mask(
        self,
        triangles: np.ndarray,
        tri_cvi_effective: np.ndarray,
        tri_inactive_bottom: np.ndarray,
    ) -> np.ndarray:
        tri = np.asarray(triangles, dtype=np.float64)
        num_tris = int(tri.shape[0]) if tri.ndim == 3 and tri.shape[1:] == (3, 3) else 0
        eligible_mask = self._planner_triangle_mask(tri)
        bottom_mask = np.asarray(tri_inactive_bottom, dtype=bool).reshape(-1)
        if bottom_mask.shape[0] == num_tris:
            eligible_mask = np.asarray(eligible_mask & (~bottom_mask), dtype=bool)
        active_mask = np.asarray(eligible_mask, dtype=bool)
        tri_eff = np.asarray(tri_cvi_effective, dtype=np.float64).reshape(-1)
        if self.mesh_view_termination_enabled and tri_eff.shape[0] == num_tris:
            active_mask = np.asarray(
                active_mask
                & np.isfinite(tri_eff)
                & (tri_eff < float(self.mesh_view_termination_cvi_threshold)),
                dtype=bool,
            )
        return active_mask

    def _compute_geometric_cvi_gain(
        self,
        triangles_world: np.ndarray,
        cam_pos_world: np.ndarray,
        cam_quat_xyzw_world: np.ndarray,
    ) -> np.ndarray:
        triangles = np.asarray(triangles_world, dtype=np.float64).reshape(-1, 3, 3)
        num_tris = int(triangles.shape[0])
        if num_tris <= 0:
            return np.zeros((0,), dtype=np.float32)

        centers = np.mean(triangles, axis=1)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        cross_norm = np.linalg.norm(cross, axis=1)
        area = 0.5 * cross_norm
        valid_area = cross_norm > 1e-10
        if not np.any(valid_area):
            return np.zeros((num_tris,), dtype=np.float32)

        normals = np.zeros_like(cross, dtype=np.float64)
        normals[valid_area] = cross[valid_area] / cross_norm[valid_area][:, None]
        outward_sign = np.sign(np.einsum("ij,ij->i", normals, centers - self.build_center.reshape(1, 3)))
        outward_sign[outward_sign == 0.0] = 1.0
        normals = normals * outward_sign[:, None]

        cam_pos = np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
        rel = centers - cam_pos.reshape(1, 3)
        dist = np.linalg.norm(rel, axis=1)
        valid_dist = dist > 1e-9
        if not np.any(valid_dist):
            return np.zeros((num_tris,), dtype=np.float32)

        r_world_cam = _quat_xyzw_to_rot(
            float(cam_quat_xyzw_world[0]),
            float(cam_quat_xyzw_world[1]),
            float(cam_quat_xyzw_world[2]),
            float(cam_quat_xyzw_world[3]),
        )
        r_cam_world = r_world_cam.T
        centers_cam = (r_cam_world @ rel.T).T
        x = centers_cam[:, 0]
        y = centers_cam[:, 1]
        z = centers_cam[:, 2]

        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.fx * (y / x)
            v = self.cy + self.fy * (z / x)

        in_frustum = (
            valid_area
            & valid_dist
            & np.isfinite(u)
            & np.isfinite(v)
            & (x > self.near_eps)
            & (u >= 0.0)
            & (u < float(self.width))
            & (v >= 0.0)
            & (v < float(self.height))
        )
        if not np.any(in_frustum):
            return np.zeros((num_tris,), dtype=np.float32)

        tri_to_cam = cam_pos.reshape(1, 3) - centers
        tri_to_cam_unit = np.zeros_like(tri_to_cam, dtype=np.float64)
        tri_to_cam_unit[valid_dist] = tri_to_cam[valid_dist] / dist[valid_dist][:, None]
        facing = np.maximum(0.0, np.einsum("ij,ij->i", normals, tri_to_cam_unit))
        visible_mask = in_frustum & (facing > 0.0)
        gains = np.zeros((num_tris,), dtype=np.float64)
        if np.any(visible_mask):
            gains[visible_mask] = (
                area[visible_mask]
                * facing[visible_mask]
                / (dist[visible_mask] * dist[visible_mask] + 1.0e-6)
            )
        return gains.astype(np.float32)

    def _compute_tri_cvi_public_and_effective(
        self,
        tri_cvi_raw: np.ndarray,
        tri_vertex_conf: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tri_cvi_raw = np.asarray(tri_cvi_raw, dtype=np.float32).reshape(-1)
        tri_conf = np.mean(np.asarray(tri_vertex_conf, dtype=np.float32), axis=1).astype(np.float32)
        if self.cvi_update_mode == "geometric":
            tri_cvi_public = _normalize_minmax_to_range(tri_cvi_raw, 1.0, 2.0)
            tri_cvi_effective = (tri_cvi_public * tri_conf).astype(np.float32)
        else:
            tri_cvi_public = tri_cvi_raw.astype(np.float32)
            tri_cvi_effective = (tri_cvi_raw * tri_conf).astype(np.float32)
        return tri_conf.astype(np.float32), tri_cvi_public.astype(np.float32), tri_cvi_effective.astype(np.float32)

    def _depth_patch_planning_active(self) -> bool:
        return (
            self.geometry_point_source == "depth"
            and bool(self.depth_mesh_patching)
        )

    def _previous_depth_patch_cvi_by_key(self, arrays: Dict[str, np.ndarray]) -> Dict[
        Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
        float,
    ]:
        keys = np.asarray(
            arrays.get("depth_planning_patch_keys", np.zeros((0, 9), dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1, 9)
        cvi = np.asarray(
            arrays.get("depth_planning_patch_cvi_raw", np.zeros((keys.shape[0],), dtype=np.float32)),
            dtype=np.float64,
        ).reshape(-1)
        out: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]], float] = {}
        for idx, row in enumerate(keys):
            canonical = self._canonical_depth_patch_key(row)
            if len(set(canonical)) != 3:
                continue
            raw = float(cvi[idx]) if idx < int(cvi.shape[0]) else 0.0
            if np.isfinite(raw) and raw > 0.0:
                out[canonical] = max(float(out.get(canonical, 0.0)), raw)
        return out

    def _append_depth_patch_to_mesh_state(
        self,
        arrays: Dict[str, np.ndarray],
        triangles: np.ndarray,
        tri_ids: np.ndarray,
        tri_cvi_raw: np.ndarray,
        tri_vertex_conf: np.ndarray,
        tri_vertex_miss_count: np.ndarray,
        tri_inactive_bottom: np.ndarray,
        next_tri_id: int,
        patch_cvi_override: Optional[np.ndarray] = None,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        triangles = np.asarray(triangles, dtype=np.float32).reshape(-1, 3, 3)
        tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)
        tri_cvi_raw = np.asarray(tri_cvi_raw, dtype=np.float32).reshape(-1)
        tri_vertex_conf = np.asarray(tri_vertex_conf, dtype=np.float32).reshape(-1, 3)
        tri_vertex_miss_count = np.asarray(tri_vertex_miss_count, dtype=np.int32).reshape(-1, 3)
        tri_inactive_bottom = np.asarray(tri_inactive_bottom, dtype=bool).reshape(-1)
        n_base = int(triangles.shape[0])
        empty_keys = np.zeros((0, 9), dtype=np.int64)
        empty_cvi = np.zeros((0,), dtype=np.float32)

        if (
            not self._depth_patch_planning_active()
            or self._depth_accum_patch_triangles.shape[0] <= 0
            or self._depth_accum_patch_keys.shape[0] != self._depth_accum_patch_triangles.shape[0]
        ):
            return (
                triangles,
                tri_ids,
                tri_cvi_raw,
                tri_vertex_conf,
                tri_vertex_miss_count,
                tri_inactive_bottom,
                np.zeros((n_base,), dtype=np.int32),
                empty_keys,
                empty_cvi,
            )

        patch_triangles = np.asarray(self._depth_accum_patch_triangles, dtype=np.float32).reshape(-1, 3, 3)
        patch_keys = np.asarray(self._depth_accum_patch_keys, dtype=np.int64).reshape(-1, 3, 3)
        patch_conf = np.asarray(self._depth_accum_patch_vertex_conf, dtype=np.float32).reshape(-1, 3)
        if patch_conf.shape[0] != patch_triangles.shape[0]:
            patch_conf = np.full((patch_triangles.shape[0], 3), float(self.conf_min), dtype=np.float32)
        if patch_cvi_override is not None and np.asarray(patch_cvi_override).reshape(-1).shape[0] == patch_triangles.shape[0]:
            patch_cvi = np.asarray(patch_cvi_override, dtype=np.float32).reshape(-1).copy()
        else:
            patch_cvi = np.asarray(self._depth_accum_patch_cvi_raw, dtype=np.float32).reshape(-1)
            if patch_cvi.shape[0] != patch_triangles.shape[0]:
                patch_cvi = np.zeros((patch_triangles.shape[0],), dtype=np.float32)
            prev_cvi = self._previous_depth_patch_cvi_by_key(arrays)
            if prev_cvi:
                patch_cvi = patch_cvi.copy()
                for idx, key_row in enumerate(patch_keys.reshape(-1, 9)):
                    canonical = self._canonical_depth_patch_key(key_row)
                    patch_cvi[idx] = max(float(patch_cvi[idx]), float(prev_cvi.get(canonical, 0.0)))

        patch_n = int(patch_triangles.shape[0])
        if patch_n <= 0:
            return (
                triangles,
                tri_ids,
                tri_cvi_raw,
                tri_vertex_conf,
                tri_vertex_miss_count,
                tri_inactive_bottom,
                np.zeros((n_base,), dtype=np.int32),
                empty_keys,
                empty_cvi,
            )
        patch_id_start = int(max(int(next_tri_id), int(np.max(tri_ids)) + 1 if tri_ids.size else 0))
        patch_ids = np.arange(patch_id_start, patch_id_start + patch_n, dtype=np.int64)
        combined_triangles = np.concatenate([triangles, patch_triangles], axis=0)
        combined_ids = np.concatenate([tri_ids, patch_ids], axis=0)
        combined_cvi = np.concatenate([tri_cvi_raw, patch_cvi.astype(np.float32)], axis=0)
        combined_conf = np.concatenate([tri_vertex_conf, patch_conf.astype(np.float32)], axis=0)
        combined_miss = np.concatenate(
            [tri_vertex_miss_count, np.zeros((patch_n, 3), dtype=np.int32)],
            axis=0,
        )
        combined_inactive = np.concatenate(
            [tri_inactive_bottom, np.zeros((patch_n,), dtype=bool)],
            axis=0,
        )
        tri_source = np.concatenate(
            [np.zeros((n_base,), dtype=np.int32), np.ones((patch_n,), dtype=np.int32)],
            axis=0,
        )
        return (
            combined_triangles,
            combined_ids,
            combined_cvi,
            combined_conf,
            combined_miss,
            combined_inactive,
            tri_source,
            patch_keys.reshape(-1, 9).astype(np.int64),
            patch_cvi.astype(np.float32),
        )

    def _init_pose_offset_from_state_or_file(self) -> None:
        file_size = os.path.getsize(self.poses_path)
        if self.start_at_end:
            self._pose_offset = int(file_size)
            return

        arrays = self._load_mesh()
        if arrays is None:
            self._pose_offset = 0
            return

        saved = arrays.get("cvi_state_pose_offset")
        if saved is None:
            self._pose_offset = 0
            return
        try:
            saved_off = int(np.array(saved).reshape(()).item())
        except Exception:
            saved_off = 0

        if saved_off < 0 or saved_off > file_size:
            self._pose_offset = 0
        else:
            self._pose_offset = saved_off

    def _sync_pose_offset_if_truncated(self) -> None:
        try:
            file_size = os.path.getsize(self.poses_path)
        except OSError:
            return
        if self._pose_offset > file_size:
            rospy.logwarn(
                "poses.txt appears truncated (offset %d > size %d); resetting offset to 0",
                self._pose_offset,
                file_size,
            )
            self._pose_offset = 0
            if self._pose_fp is not None:
                self._pose_fp.seek(0, os.SEEK_SET)

    def _extract_mesh_state(
        self,
        arrays: Dict[str, np.ndarray],
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Optional[float]]]:
        triangles = arrays.get("triangles")
        if triangles is None:
            rospy.logwarn("Mesh npz has no 'triangles'; skipping update.")
            return None

        triangles = np.asarray(triangles, dtype=np.float32)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            rospy.logwarn("Invalid triangles shape %s; skipping.", str(triangles.shape))
            return None
        source_raw = arrays.get("tri_source", None)
        if source_raw is not None and np.asarray(source_raw).reshape(-1).shape[0] == int(triangles.shape[0]):
            tri_source = np.asarray(source_raw, dtype=np.int32).reshape(-1)
            base_mask = tri_source == 0
            if np.any(~base_mask):
                triangles = triangles[base_mask]
        else:
            base_mask = None
        num_tris = int(triangles.shape[0])

        tri_ids = arrays.get("tri_ids")
        if tri_ids is not None and base_mask is not None and np.asarray(tri_ids).reshape(-1).shape[0] == int(base_mask.shape[0]):
            tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)[base_mask]
        if tri_ids is None or np.asarray(tri_ids).reshape(-1).shape[0] != num_tris:
            tri_ids = np.arange(num_tris, dtype=np.int64)
        else:
            tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)

        tri_cvi_raw = arrays.get("tri_cvi_raw")
        if tri_cvi_raw is None:
            tri_cvi_raw = arrays.get("tri_cvi")
        if tri_cvi_raw is not None and base_mask is not None and np.asarray(tri_cvi_raw).reshape(-1).shape[0] == int(base_mask.shape[0]):
            tri_cvi_raw = np.asarray(tri_cvi_raw, dtype=np.float32).reshape(-1)[base_mask]
        if tri_cvi_raw is None or np.asarray(tri_cvi_raw).reshape(-1).shape[0] != num_tris:
            tri_cvi_raw = np.zeros((num_tris,), dtype=np.float32)
        else:
            tri_cvi_raw = np.asarray(tri_cvi_raw, dtype=np.float32).reshape(-1)

        tri_vertex_conf = arrays.get("tri_vertex_conf")
        if tri_vertex_conf is not None and base_mask is not None and np.asarray(tri_vertex_conf).shape == (int(base_mask.shape[0]), 3):
            tri_vertex_conf = np.asarray(tri_vertex_conf, dtype=np.float32)[base_mask]
        if tri_vertex_conf is None or np.asarray(tri_vertex_conf).shape != (num_tris, 3):
            tri_vertex_conf = np.ones((num_tris, 3), dtype=np.float32)
        else:
            tri_vertex_conf = np.asarray(tri_vertex_conf, dtype=np.float32)

        tri_vertex_miss_count = arrays.get("tri_vertex_miss_count")
        if tri_vertex_miss_count is not None and base_mask is not None and np.asarray(tri_vertex_miss_count).shape == (int(base_mask.shape[0]), 3):
            tri_vertex_miss_count = np.asarray(tri_vertex_miss_count, dtype=np.int32)[base_mask]
        if tri_vertex_miss_count is None or np.asarray(tri_vertex_miss_count).shape != (num_tris, 3):
            tri_vertex_miss_count = np.zeros((num_tris, 3), dtype=np.int32)
        else:
            tri_vertex_miss_count = np.asarray(tri_vertex_miss_count, dtype=np.int32)

        tri_inactive_bottom = arrays.get("tri_inactive_bottom")
        if tri_inactive_bottom is not None and base_mask is not None and np.asarray(tri_inactive_bottom).reshape(-1).shape[0] == int(base_mask.shape[0]):
            tri_inactive_bottom = np.asarray(tri_inactive_bottom, dtype=bool).reshape(-1)[base_mask]
        if tri_inactive_bottom is None or np.asarray(tri_inactive_bottom).reshape(-1).shape[0] != num_tris:
            tri_inactive_bottom = self._infer_bottom_inactive_triangles(
                triangles,
                mesh_l0_value=arrays.get("mesh_l0", None),
            )
        else:
            tri_inactive_bottom = np.asarray(tri_inactive_bottom, dtype=bool).reshape(-1)

        next_tri_id = arrays.get("next_tri_id")
        if next_tri_id is None:
            next_tri_id = int(tri_ids.max()) + 1 if tri_ids.size else num_tris
        else:
            next_tri_id = int(np.array(next_tri_id).reshape(()).item())

        mesh_l0_value = None
        self.max_correspondence_dist = float(self.max_correspondence_dist_param)
        mesh_l0 = arrays.get("mesh_l0")
        if mesh_l0 is not None:
            try:
                l0 = float(np.array(mesh_l0, dtype=np.float64).reshape(()).item())
                if np.isfinite(l0) and l0 > 0.0:
                    mesh_l0_value = l0
                    if float(self.max_correspondence_dist_param) >= 0.0:
                        self.max_correspondence_dist = self.max_corr_l0_weight * l0
                    else:
                        self.max_correspondence_dist = float(self.max_correspondence_dist_param)
                    self.split_correspondence_dist = self.split_corr_l0_weight * l0
            except Exception:
                mesh_l0_value = None

        return (
            triangles,
            tri_ids,
            tri_cvi_raw,
            tri_vertex_conf,
            tri_vertex_miss_count,
            tri_inactive_bottom,
            int(next_tri_id),
            mesh_l0_value,
        )

    def _pack_mesh_state(
        self,
        arrays: Dict[str, np.ndarray],
        triangles: np.ndarray,
        tri_ids: np.ndarray,
        tri_cvi_raw: np.ndarray,
        tri_cvi_public: np.ndarray,
        next_tri_id: int,
        tri_vertex_conf: np.ndarray,
        tri_vertex_miss_count: np.ndarray,
        tri_inactive_bottom: np.ndarray,
        tri_conf: np.ndarray,
        tri_cvi_effective: np.ndarray,
        geom_stats: Dict[str, float],
        pose_offset: Optional[int] = None,
        tri_source: Optional[np.ndarray] = None,
        depth_planning_patch_keys: Optional[np.ndarray] = None,
        depth_planning_patch_cvi_raw: Optional[np.ndarray] = None,
    ) -> None:
        arrays["triangles"] = triangles.astype(np.float32)
        arrays["points"] = np.mean(triangles.astype(np.float32), axis=1).astype(np.float32)
        arrays["tri_ids"] = tri_ids.astype(np.int64)
        arrays["next_tri_id"] = np.int64(int(next_tri_id))
        arrays["tri_vertex_conf"] = tri_vertex_conf.astype(np.float32)
        arrays["tri_conf"] = tri_conf.astype(np.float32)
        arrays["tri_vertex_miss_count"] = tri_vertex_miss_count.astype(np.int32)
        arrays["tri_inactive_bottom"] = np.asarray(tri_inactive_bottom, dtype=bool).reshape(-1)
        arrays["tri_cvi_raw"] = tri_cvi_raw.astype(np.float32)
        arrays["tri_cvi"] = tri_cvi_public.astype(np.float32)
        arrays["tri_cvi_effective"] = tri_cvi_effective.astype(np.float32)
        n_tri = int(np.asarray(triangles).shape[0])
        if tri_source is None or np.asarray(tri_source).reshape(-1).shape[0] != n_tri:
            arrays["tri_source"] = np.zeros((n_tri,), dtype=np.int32)
        else:
            arrays["tri_source"] = np.asarray(tri_source, dtype=np.int32).reshape(-1)
        if depth_planning_patch_keys is None:
            arrays["depth_planning_patch_keys"] = np.zeros((0, 9), dtype=np.int64)
        else:
            arrays["depth_planning_patch_keys"] = np.asarray(depth_planning_patch_keys, dtype=np.int64).reshape(-1, 9)
        if depth_planning_patch_cvi_raw is None:
            arrays["depth_planning_patch_cvi_raw"] = np.zeros((0,), dtype=np.float32)
        else:
            arrays["depth_planning_patch_cvi_raw"] = np.asarray(depth_planning_patch_cvi_raw, dtype=np.float32).reshape(-1)
        arrays["cvi_update_mode"] = np.array(self.cvi_update_mode)
        if pose_offset is not None:
            arrays["cvi_state_pose_offset"] = np.int64(int(pose_offset))
        arrays["mesh_cvi_update_last_visible_vertices"] = np.int64(int(geom_stats["visible_vertices"]))
        arrays["mesh_cvi_update_last_supported_vertices"] = np.int64(int(geom_stats["supported_vertices"]))
        arrays["mesh_cvi_update_last_shrunk_vertices"] = np.int64(int(geom_stats["shrunk_vertices"]))
        arrays["mesh_cvi_update_last_rolled_back_vertices"] = np.int64(int(geom_stats["rolled_back_vertices"]))
        arrays["mesh_cvi_update_last_pseudo_merged_edges"] = np.int64(int(geom_stats["pseudo_merged_edges"]))
        arrays["mesh_cvi_update_last_associated_points"] = np.int64(int(geom_stats["associated_points"]))
        arrays["mesh_cvi_update_last_near_vertex_points"] = np.int64(int(geom_stats["near_vertex_points"]))
        arrays["mesh_cvi_update_last_interior_points"] = np.int64(int(geom_stats["interior_points"]))
        arrays["mesh_cvi_update_last_pruned_edges"] = np.int64(int(geom_stats["pruned_edges"]))
        arrays["mesh_cvi_update_last_pruned_triangles"] = np.int64(int(geom_stats["pruned_triangles"]))
        arrays["mesh_cvi_update_last_split_parents"] = np.int64(int(geom_stats["split_parents"]))
        arrays["mesh_cvi_update_last_new_triangles"] = np.int64(int(geom_stats["new_triangles"]))
        arrays["mesh_cvi_update_last_unix"] = np.float64(time.time())

    def _update_mesh_from_sparse_only(self) -> bool:
        if (not self.mesh_update_enabled) or self.intersection_logic_enabled:
            return False
        arrays = self._load_mesh()
        if arrays is None:
            return False

        mesh_state = self._extract_mesh_state(arrays)
        if mesh_state is None:
            return False

        (
            triangles,
            tri_ids,
            tri_cvi_raw,
            tri_vertex_conf,
            tri_vertex_miss_count,
            tri_inactive_bottom,
            next_tri_id,
            mesh_l0_value,
        ) = mesh_state

        (
            triangles,
            tri_ids,
            tri_cvi_raw,
            tri_inactive_bottom,
            next_tri_id,
            tri_vertex_conf,
            tri_vertex_miss_count,
            geom_stats,
        ) = self._apply_geometry_update(
            triangles=triangles,
            tri_ids=tri_ids,
            tri_cvi_raw=tri_cvi_raw,
            tri_inactive_bottom=tri_inactive_bottom,
            next_tri_id=next_tri_id,
            tri_vertex_conf=tri_vertex_conf,
            tri_vertex_miss_count=tri_vertex_miss_count,
            tri_idx_buf=None,
            cam_pos_world=None,
            cam_quat_xyzw_world=None,
            mesh_l0_value=mesh_l0_value,
            sparse_changed=True,
            allow_shrink=False,
        )

        tri_conf, tri_cvi_public, tri_cvi_effective = self._compute_tri_cvi_public_and_effective(
            tri_cvi_raw=tri_cvi_raw,
            tri_vertex_conf=tri_vertex_conf,
        )
        pose_offset = arrays.get("cvi_state_pose_offset", None)
        try:
            pose_offset_int = int(np.array(pose_offset).reshape(()).item()) if pose_offset is not None else None
        except Exception:
            pose_offset_int = None
        self._pack_mesh_state(
            arrays=arrays,
            triangles=triangles,
            tri_ids=tri_ids,
            tri_cvi_raw=tri_cvi_raw,
            tri_cvi_public=tri_cvi_public,
            next_tri_id=next_tri_id,
            tri_vertex_conf=tri_vertex_conf,
            tri_vertex_miss_count=tri_vertex_miss_count,
            tri_inactive_bottom=tri_inactive_bottom,
            tri_conf=tri_conf,
            tri_cvi_effective=tri_cvi_effective,
            geom_stats=geom_stats,
            pose_offset=pose_offset_int,
        )
        self._save_mesh(arrays)
        rospy.loginfo(
            "Mesh sparse-trigger update: moved buckets applied, vis_v=%d sup_v=%d shr_v=%d",
            int(geom_stats["visible_vertices"]),
            int(geom_stats["supported_vertices"]),
            int(geom_stats["shrunk_vertices"]),
        )
        return True

    def _is_controller_running(self) -> bool:
        if not self.wait_for_controller:
            return True
        try:
            pubs, subs, srvs = self._ros_master.getSystemState()
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "Controller gate: failed ROS master query (%s)",
                str(exc),
            )
            return False

        target = self.controller_node_name
        for _topic, nodes in pubs:
            if target in nodes:
                return True
        for _topic, nodes in subs:
            if target in nodes:
                return True
        for _service, nodes in srvs:
            if target in nodes:
                return True
        return False

    def _point_quality(self, sparse_idx: int, corr_dist: float) -> float:
        base_q = (
            float(self._sparse_quality_base[int(sparse_idx)])
            if 0 <= int(sparse_idx) < int(self._sparse_quality_base.shape[0])
            else 0.0
        )
        dist_bad = float(self.conf_dist_bad_m)
        if dist_bad <= 0.0:
            dist_bad = float(self.max_correspondence_dist)
        if dist_bad > 0.0:
            s_dist = 1.0 - float(np.clip(corr_dist / max(dist_bad, 1e-6), 0.0, 1.0))
        else:
            s_dist = 1.0
        return float(np.clip(base_q + 0.25 * s_dist, 0.0, 1.0))

    def _update_running_timing_pct_stats(
        self,
        timing_items: List[Tuple[str, float]],
        cycle_total_ms: float,
    ) -> Dict[str, Dict[str, float]]:
        denom = max(float(cycle_total_ms), 1e-9)
        self._timing_num_updates += 1
        out: Dict[str, Dict[str, float]] = {}

        for key, val_ms in timing_items:
            pct = 100.0 * float(val_ms) / denom
            stat_pct = self._timing_stats_pct.get(key)
            if stat_pct is None:
                stat_pct = {"n": 1.0, "mean": pct, "m2": 0.0}
            else:
                n = float(stat_pct["n"]) + 1.0
                mean_old = float(stat_pct["mean"])
                m2_old = float(stat_pct["m2"])
                delta = pct - mean_old
                mean_new = mean_old + (delta / n)
                delta2 = pct - mean_new
                m2_new = m2_old + delta * delta2
                stat_pct = {"n": n, "mean": mean_new, "m2": m2_new}
            self._timing_stats_pct[key] = stat_pct

            stat_ms = self._timing_stats_ms.get(key)
            if stat_ms is None:
                stat_ms = {"n": 1.0, "mean": float(val_ms), "m2": 0.0}
            else:
                nms = float(stat_ms["n"]) + 1.0
                mean_ms_old = float(stat_ms["mean"])
                m2_ms_old = float(stat_ms["m2"])
                delta_ms = float(val_ms) - mean_ms_old
                mean_ms_new = mean_ms_old + (delta_ms / nms)
                delta2_ms = float(val_ms) - mean_ms_new
                m2_ms_new = m2_ms_old + delta_ms * delta2_ms
                stat_ms = {"n": nms, "mean": mean_ms_new, "m2": m2_ms_new}
            self._timing_stats_ms[key] = stat_ms

            n_pct = float(stat_pct["n"])
            var_pct = float(stat_pct["m2"]) / (n_pct - 1.0) if n_pct > 1.0 else 0.0
            std_pct = float(np.sqrt(max(var_pct, 0.0)))

            n_ms = float(stat_ms["n"])
            var_ms = float(stat_ms["m2"]) / (n_ms - 1.0) if n_ms > 1.0 else 0.0
            std_ms = float(np.sqrt(max(var_ms, 0.0)))

            out[key] = {
                "mean_pct": float(stat_pct["mean"]),
                "std_pct": std_pct,
                "mean_ms": float(stat_ms["mean"]),
                "std_ms": std_ms,
            }

        return out

    @staticmethod
    def _update_running_mean_stats(
        stats_store: Dict[str, Dict[str, float]],
        timing_items: List[Tuple[str, float]],
    ) -> Dict[str, float]:
        for key, val_ms in timing_items:
            stat_ms = stats_store.get(key)
            if stat_ms is None:
                stat_ms = {"n": 1.0, "mean": float(val_ms)}
            else:
                nms = float(stat_ms["n"]) + 1.0
                mean_ms_old = float(stat_ms["mean"])
                mean_ms_new = mean_ms_old + ((float(val_ms) - mean_ms_old) / nms)
                stat_ms = {"n": nms, "mean": mean_ms_new}
            stats_store[key] = stat_ms
        return {key: float(stat["mean"]) for key, stat in stats_store.items()}

    def _apply_geometry_update(
        self,
        triangles: np.ndarray,
        tri_ids: np.ndarray,
        tri_cvi_raw: np.ndarray,
        tri_inactive_bottom: np.ndarray,
        next_tri_id: int,
        tri_vertex_conf: np.ndarray,
        tri_vertex_miss_count: np.ndarray,
        tri_idx_buf: Optional[np.ndarray],
        cam_pos_world: Optional[np.ndarray],
        cam_quat_xyzw_world: Optional[np.ndarray],
        mesh_l0_value: Optional[float] = None,
        sparse_changed: bool = False,
        allow_shrink: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, Dict[str, float]]:
        t_geom_total = time.perf_counter()

        t0 = time.perf_counter()
        triangles = np.asarray(triangles, dtype=np.float64)
        num_tris = int(triangles.shape[0])
        tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)
        tri_cvi_raw = np.asarray(tri_cvi_raw, dtype=np.float32).reshape(-1)
        tri_inactive_bottom = np.asarray(tri_inactive_bottom, dtype=bool).reshape(-1)
        if tri_ids.shape[0] != num_tris:
            tri_ids = np.arange(num_tris, dtype=np.int64)
        if tri_cvi_raw.shape[0] != num_tris:
            tri_cvi_raw = np.zeros((num_tris,), dtype=np.float32)
        if tri_inactive_bottom.shape[0] != num_tris:
            tri_inactive_bottom = self._infer_bottom_inactive_triangles(
                triangles,
                mesh_l0_value=mesh_l0_value,
            )

        tri_vertex_conf = np.asarray(tri_vertex_conf, dtype=np.float64)
        if tri_vertex_conf.shape != (num_tris, 3):
            tri_vertex_conf = np.full((num_tris, 3), float(self.conf_min), dtype=np.float64)
        tri_vertex_miss_count = np.asarray(tri_vertex_miss_count, dtype=np.int32)
        if tri_vertex_miss_count.shape != (num_tris, 3):
            tri_vertex_miss_count = np.zeros((num_tris, 3), dtype=np.int32)
        t_prepare_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        vertices, tri_vidx, v_to_tri, _v_neighbors = _build_vertices_and_topology(
            triangles=triangles,
            quantize_eps=self.quantize_eps,
        )
        t_topology_ms = (time.perf_counter() - t0) * 1000.0
        n_v = int(vertices.shape[0])
        if n_v <= 0:
            t_geom_total_ms = (time.perf_counter() - t_geom_total) * 1000.0
            return (
                triangles.astype(np.float32),
                tri_ids.astype(np.int64),
                tri_cvi_raw.astype(np.float32),
                tri_inactive_bottom.astype(bool),
                int(next_tri_id),
                tri_vertex_conf.astype(np.float32),
                tri_vertex_miss_count.astype(np.int32),
                {
                    "visible_vertices": 0.0,
                    "supported_vertices": 0.0,
                    "shrunk_vertices": 0.0,
                    "rolled_back_vertices": 0.0,
                    "pseudo_merged_edges": 0.0,
                    "associated_points": 0.0,
                    "near_vertex_points": 0.0,
                    "interior_points": 0.0,
                    "pruned_edges": 0.0,
                    "pruned_triangles": 0.0,
                    "split_parents": 0.0,
                    "new_triangles": 0.0,
                    "assoc_sparse_tracks": 0.0,
                    "assoc_vertices": 0.0,
                    "assoc_rebuild": 0.0,
                    "assoc_stable_ids": 0.0,
                    "assoc_point_ids_stable": 0.0,
                    "assoc_bucket_stable_ids": 0.0,
                    "assoc_point_revs_valid": 0.0,
                    "assoc_added_tracks": 0.0,
                    "assoc_changed_tracks": 0.0,
                    "assoc_removed_tracks": 0.0,
                    "assoc_pairs_changed": 0.0,
                    "assoc_assign_calls": 0.0,
                    "assoc_assign_accepted_tracks": 0.0,
                    "assoc_assign_rejected_tracks": 0.0,
                    "assoc_assign_candidates_total": 0.0,
                    "assoc_assign_kept_total": 0.0,
                    "assoc_assign_unique_vid_total": 0.0,
                    "t_geom_total_ms": float(t_geom_total_ms),
                    "t_geom_prepare_ms": float(t_prepare_ms),
                    "t_geom_topology_ms": float(t_topology_ms),
                    "t_geom_vertex_state_ms": 0.0,
                    "t_geom_visibility_ms": 0.0,
                    "t_geom_assoc_ms": 0.0,
                    "t_assoc_key_extract_ms": 0.0,
                    "t_assoc_tree_ms": 0.0,
                    "t_assoc_change_detect_ms": 0.0,
                    "t_assoc_remove_ms": 0.0,
                    "t_assoc_assign_ms": 0.0,
                    "t_assoc_assign_query_ms": 0.0,
                    "t_assoc_assign_filter_ms": 0.0,
                    "t_assoc_assign_quality_ms": 0.0,
                    "t_assoc_assign_bucket_update_ms": 0.0,
                    "t_assoc_assign_record_ms": 0.0,
                    "t_geom_vertex_update_ms": 0.0,
                    "t_geom_conf_ema_ms": 0.0,
                    "t_geom_pseudomerge_ms": 0.0,
                    "t_geom_rollback_ms": 0.0,
                    "t_geom_prune_ms": 0.0,
                    "t_geom_split_ms": 0.0,
                    "t_geom_pack_ms": 0.0,
                },
            )

        # Vertex-level confidence and miss.
        t0 = time.perf_counter()
        v_conf_sum = np.zeros((n_v,), dtype=np.float64)
        v_conf_cnt = np.zeros((n_v,), dtype=np.float64)
        for c in range(3):
            np.add.at(v_conf_sum, tri_vidx[:, c], tri_vertex_conf[:, c])
            np.add.at(v_conf_cnt, tri_vidx[:, c], 1.0)
        v_conf_old = np.full((n_v,), float(self.conf_min), dtype=np.float64)
        m_conf = v_conf_cnt > 0.0
        v_conf_old[m_conf] = v_conf_sum[m_conf] / np.maximum(v_conf_cnt[m_conf], 1e-12)
        v_conf_old = np.clip(v_conf_old, float(self.conf_min), float(self.conf_max))

        v_miss_old = np.zeros((n_v,), dtype=np.int32)
        for c in range(3):
            np.maximum.at(v_miss_old, tri_vidx[:, c], tri_vertex_miss_count[:, c])
        t_vertex_state_ms = (time.perf_counter() - t0) * 1000.0

        # Visible triangles and visible/in-frustum vertices.
        t0 = time.perf_counter()
        visible_tri_mask = np.zeros((num_tris,), dtype=bool)
        visible_vid_mask = np.zeros((n_v,), dtype=bool)
        frustum_vid_mask = np.zeros((n_v,), dtype=bool)
        target_vid_mask = np.zeros((n_v,), dtype=bool)
        target_vids = np.zeros((0,), dtype=np.int32)
        valid_idx = np.zeros((0,), dtype=np.int32)
        ui = np.zeros((0,), dtype=np.int32)
        vi = np.zeros((0,), dtype=np.int32)
        sparse_idx_vis = np.zeros((0,), dtype=np.int32)
        sparse_points_vis = np.zeros((0, 3), dtype=np.float64)
        r_cam_world = None
        if (
            tri_idx_buf is not None
            and cam_pos_world is not None
            and cam_quat_xyzw_world is not None
        ):
            visible_ids = tri_idx_buf[tri_idx_buf >= 0].astype(np.int32)
            if visible_ids.size > 0:
                visible_tri_mask[np.unique(visible_ids)] = True

            if np.any(visible_tri_mask):
                visible_vid_mask[tri_vidx[visible_tri_mask].reshape(-1)] = True

            r_world_cam = _quat_xyzw_to_rot(
                float(cam_quat_xyzw_world[0]),
                float(cam_quat_xyzw_world[1]),
                float(cam_quat_xyzw_world[2]),
                float(cam_quat_xyzw_world[3]),
            )
            r_cam_world = r_world_cam.T
            verts_rel = vertices - cam_pos_world.reshape(1, 3)
            verts_cam = (r_cam_world @ verts_rel.T).T
            x_v = verts_cam[:, 0]
            y_v = verts_cam[:, 1]
            z_v = verts_cam[:, 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                u_v = self.cx + self.fx * (y_v / x_v)
                v_v = self.cy + self.fy * (z_v / x_v)
            frustum_vid_mask = (
                np.isfinite(u_v)
                & np.isfinite(v_v)
                & (x_v > self.near_eps)
                & (u_v >= 0.0)
                & (u_v < float(self.width))
                & (v_v >= 0.0)
                & (v_v < float(self.height))
            )
            target_vid_mask = visible_vid_mask & frustum_vid_mask
            target_vids = np.where(target_vid_mask)[0].astype(np.int32)

            points = self._sparse_points
            if points.shape[0] > 0:
                p_rel = points - cam_pos_world.reshape(1, 3)
                p_cam = (r_cam_world @ p_rel.T).T
                x = p_cam[:, 0]
                y = p_cam[:, 1]
                z = p_cam[:, 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    u = self.cx + self.fx * (y / x)
                    v = self.cy + self.fy * (z / x)
                ui = np.floor(u).astype(np.int32)
                vi = np.floor(v).astype(np.int32)
                valid = (
                    np.isfinite(u)
                    & np.isfinite(v)
                    & (x > self.near_eps)
                    & (ui >= 0)
                    & (ui < self.width)
                    & (vi >= 0)
                    & (vi < self.height)
                )
                valid_idx = np.where(valid)[0].astype(np.int32)
                sparse_idx_vis = valid_idx
                sparse_points_vis = (
                    points[sparse_idx_vis]
                    if sparse_idx_vis.size > 0
                    else np.zeros((0, 3), dtype=np.float64)
                )
        t_visibility_ms = (time.perf_counter() - t0) * 1000.0

        # Sparse association to currently visible triangles through raster (legacy intersection mode)
        # OR persistent sparse-to-vertex buckets (default mode).
        t0 = time.perf_counter()
        tri_has_intersection = np.zeros((num_tris,), dtype=bool)
        near_vertex_buckets: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        accepted_points_by_tri: Dict[int, List[Tuple[np.ndarray, float]]] = {}
        assoc_points = 0
        near_points = 0
        interior_points = 0
        bucket_changed_mask = np.zeros((n_v,), dtype=bool)
        bucket_nonempty_mask = np.zeros((n_v,), dtype=bool)
        assoc_sync_stats = self._empty_assoc_sync_stats()
        assoc_sync_stats["assoc_vertices"] = float(n_v)
        assoc_sync_stats["assoc_sparse_tracks"] = float(int(self._sparse_points.shape[0]))

        if self.intersection_logic_enabled:
            points = self._sparse_points
            if valid_idx.size > 0 and np.any(visible_tri_mask) and r_cam_world is not None and cam_pos_world is not None:
                for sp_idx in valid_idx.tolist():
                    tid = int(tri_idx_buf[int(vi[sp_idx]), int(ui[sp_idx])])
                    if tid < 0 or tid >= num_tris or (not bool(visible_tri_mask[int(tid)])):
                        continue

                    ia, ib, ic = tri_vidx[tid]
                    a = vertices[int(ia)]
                    b = vertices[int(ib)]
                    c = vertices[int(ic)]
                    sparse_p = points[sp_idx]
                    ray_dir = sparse_p - cam_pos_world
                    ray_n = float(np.linalg.norm(ray_dir))
                    if ray_n <= 1e-12:
                        continue
                    ray_dir = ray_dir / ray_n
                    pint = _ray_triangle_intersection(cam_pos_world, ray_dir, a, b, c)
                    if pint is None:
                        continue
                    bary = _barycentric_coords(pint, a, b, c)
                    if bary is None:
                        continue
                    if float(np.min(bary)) < -1e-4:
                        continue
                    corr_dist = float(np.linalg.norm(sparse_p - pint))
                    if float(self.max_correspondence_dist) >= 0.0 and corr_dist > float(self.max_correspondence_dist):
                        continue
                    tri_has_intersection[tid] = True
                    assoc_points += 1
                    q = self._point_quality(sp_idx, corr_dist)
                    accepted_points_by_tri.setdefault(tid, []).append((sparse_p.copy(), float(q)))

                    vmax = int(np.argmax(bary))
                    if float(bary[vmax]) >= self.split_corner_gate:
                        gvid = int(tri_vidx[tid, vmax])
                        if target_vid_mask[gvid]:
                            near_vertex_buckets.setdefault(gvid, []).append((sparse_p, q))
                            near_points += 1
                    else:
                        interior_points += 1
        else:
            if sparse_changed or self._bucket_needs_rebuild or int(self._bucket_num_vertices) != n_v:
                bucket_changed_mask, _assoc_pairs_changed, assoc_sync_stats = self._sync_sparse_vertex_buckets(
                    vertices=vertices,
                    force_rebuild=bool(self._bucket_needs_rebuild),
                )
            if int(self._bucket_num_vertices) == n_v:
                bucket_recheck_mask = self._prune_bucket_assignments_by_current_distance(vertices)
                if bucket_recheck_mask.shape[0] == bucket_changed_mask.shape[0]:
                    bucket_changed_mask |= bucket_recheck_mask
            if int(self._bucket_num_vertices) == n_v:
                bucket_nonempty_mask = self._bucket_count > 0
                assoc_points = int(np.sum(self._bucket_count))
                near_points = int(assoc_points)

            # Mark visible triangles touched by currently valid sparse points for diagnostics.
            if valid_idx.size > 0 and np.any(visible_tri_mask):
                tid_hits = tri_idx_buf[vi[sparse_idx_vis], ui[sparse_idx_vis]]
                tid_hits = tid_hits[(tid_hits >= 0) & (tid_hits < num_tris)]
                if tid_hits.size > 0:
                    tri_has_intersection[np.unique(tid_hits).astype(np.int32)] = True
        t_assoc_ms = (time.perf_counter() - t0) * 1000.0

        # Move vertices whose sparse buckets changed; shrink only visible vertices with empty buckets.
        t0 = time.perf_counter()
        proposed = vertices.copy()
        v_conf_new = v_conf_old.copy()
        v_miss_new = v_miss_old.copy()
        moved_vid = np.zeros((n_v,), dtype=bool)
        supported_vid = np.zeros((n_v,), dtype=bool)
        shrunk_vid = np.zeros((n_v,), dtype=bool)
        conf_touched = np.zeros((n_v,), dtype=bool)
        conf_target = np.zeros((n_v,), dtype=np.float64)

        if self.intersection_logic_enabled:
            for vid in target_vids.tolist():
                vals = near_vertex_buckets.get(int(vid), None)
                if vals is None or len(vals) == 0:
                    tri_ids_vid = v_to_tri[int(vid)]
                    best_point: Optional[np.ndarray] = None
                    best_q = 0.0
                    best_d2 = np.inf
                    if tri_ids_vid.size > 0:
                        vref = vertices[int(vid)]
                        for tid in tri_ids_vid.tolist():
                            if not visible_tri_mask[int(tid)]:
                                continue
                            for sp, sq in accepted_points_by_tri.get(int(tid), []):
                                d = sp - vref
                                d2 = float(np.dot(d, d))
                                if d2 < best_d2:
                                    best_d2 = d2
                                    best_point = sp
                                    best_q = float(sq)
                    if best_point is not None:
                        vals = [(best_point, best_q)]

                if vals is not None and len(vals) > 0:
                    pts = np.asarray([v_[0] for v_ in vals], dtype=np.float64)
                    qs = np.asarray([v_[1] for v_ in vals], dtype=np.float64)
                    proposed[int(vid)] = np.mean(pts, axis=0)
                    moved_vid[int(vid)] = True
                    supported_vid[int(vid)] = True
                    v_miss_new[int(vid)] = 0
                    conf_target[int(vid)] = float(
                        self.conf_min + float(np.mean(qs)) * (self.conf_max - self.conf_min)
                    )
                    conf_touched[int(vid)] = True
                    continue

                tri_ids_vid = v_to_tri[int(vid)]
                has_tri_points = False
                if tri_ids_vid.size > 0:
                    has_tri_points = bool(
                        np.any(visible_tri_mask[tri_ids_vid] & tri_has_intersection[tri_ids_vid])
                    )
                if has_tri_points or (not allow_shrink):
                    v_miss_new[int(vid)] = 0
                    continue

                v_miss_new[int(vid)] = int(v_miss_old[int(vid)] + 1)
                if int(v_miss_new[int(vid)]) >= int(self.vertex_miss_threshold):
                    dir_to_center = self.build_center - proposed[int(vid)]
                    dn = float(np.linalg.norm(dir_to_center))
                    if dn > 1e-9:
                        proposed[int(vid)] = proposed[int(vid)] + float(self.shrink_step_m) * (dir_to_center / dn)
                        moved_vid[int(vid)] = True
                        shrunk_vid[int(vid)] = True
        else:
            changed_vids = np.where(bucket_changed_mask)[0].astype(np.int32)
            for vid in changed_vids.tolist():
                cnt = int(self._bucket_count[int(vid)]) if int(vid) < int(self._bucket_count.shape[0]) else 0
                if cnt <= 0:
                    continue
                pts_mean = self._bucket_sum_xyz[int(vid)] / float(max(cnt, 1))
                q_mean = self._bucket_sum_q[int(vid)] / float(max(cnt, 1))
                proposed[int(vid)] = pts_mean
                moved_vid[int(vid)] = True
                if target_vid_mask[int(vid)]:
                    supported_vid[int(vid)] = True
                    v_miss_new[int(vid)] = 0
                conf_target[int(vid)] = float(
                    self.conf_min + float(q_mean) * (self.conf_max - self.conf_min)
                )
                conf_touched[int(vid)] = True

            for vid in target_vids.tolist():
                if bucket_nonempty_mask[int(vid)]:
                    supported_vid[int(vid)] = True
                    v_miss_new[int(vid)] = 0
                    continue
                if not allow_shrink:
                    continue
                v_miss_new[int(vid)] = int(v_miss_old[int(vid)] + 1)
                if int(v_miss_new[int(vid)]) >= int(self.vertex_miss_threshold):
                    dir_to_center = self.build_center - proposed[int(vid)]
                    dn = float(np.linalg.norm(dir_to_center))
                    if dn > 1e-9:
                        proposed[int(vid)] = proposed[int(vid)] + float(self.shrink_step_m) * (dir_to_center / dn)
                        moved_vid[int(vid)] = True
                        shrunk_vid[int(vid)] = True

        # KNN mode smoothing: apply Laplacian smoothing on supported vertices plus ring neighbors.
        if (not self.intersection_logic_enabled) and self.knn_smooth_iters > 0 and self.knn_smooth_lambda > 0.0:
            seed_mask = moved_vid.copy()
            if np.any(seed_mask):
                smooth_mask = seed_mask.copy()
                frontier = np.where(seed_mask)[0]
                for _ in range(int(self.knn_smooth_ring)):
                    if frontier.size == 0:
                        break
                    neigh_chunks = []
                    for vid in frontier.tolist():
                        nb = _v_neighbors[int(vid)]
                        if nb.size > 0:
                            neigh_chunks.append(nb)
                    if not neigh_chunks:
                        break
                    neigh = np.unique(np.concatenate(neigh_chunks)).astype(np.int32)
                    neigh = neigh[~smooth_mask[neigh]]
                    if neigh.size == 0:
                        break
                    smooth_mask[neigh] = True
                    frontier = neigh

                smooth_vids = np.where(smooth_mask)[0]
                lam = float(np.clip(self.knn_smooth_lambda, 0.0, 1.0))
                for _ in range(int(self.knn_smooth_iters)):
                    prev = proposed.copy()
                    for vid in smooth_vids.tolist():
                        nb = _v_neighbors[int(vid)]
                        if nb.size == 0:
                            continue
                        avg_nb = np.mean(prev[nb], axis=0)
                        proposed[int(vid)] = (1.0 - lam) * prev[int(vid)] + lam * avg_nb
                moved_vid[smooth_vids] = True
        t_vertex_update_ms = (time.perf_counter() - t0) * 1000.0

        # Pseudo-merge: nudge shared edges of nearly coplanar adjacent triangles.
        t0 = time.perf_counter()
        pseudo_merged_edges = 0
        l0_ref = (
            float(mesh_l0_value)
            if (mesh_l0_value is not None and np.isfinite(mesh_l0_value) and mesh_l0_value > 0.0)
            else max(2.0 * float(self.split_correspondence_dist), 1e-6)
        )
        pseudo_step = min(
            float(self.pseudomerge_max_step_m),
            float(self.pseudomerge_step_l0_weight) * float(l0_ref),
        )
        if (
            self.pseudomerge_enabled
            and self.pseudomerge_iters > 0
            and pseudo_step > 0.0
            and num_tris > 0
        ):
            # Edge adjacency is topology-only and can be built once.
            edge_to_tris: Dict[Tuple[int, int], List[int]] = {}
            for tid in range(num_tris):
                a, b, c = int(tri_vidx[tid, 0]), int(tri_vidx[tid, 1]), int(tri_vidx[tid, 2])
                for u, v in ((a, b), (b, c), (c, a)):
                    key = (u, v) if u < v else (v, u)
                    edge_to_tris.setdefault(key, []).append(int(tid))
            edge_pairs = [
                (k, v)
                for k, v in edge_to_tris.items()
                if len(v) == 2
                and (
                    (not self.pseudomerge_visible_only)
                    or (visible_tri_mask[int(v[0])] and visible_tri_mask[int(v[1])])
                )
            ]

            for _ in range(int(self.pseudomerge_iters)):
                delta_sum = np.zeros((n_v, 3), dtype=np.float64)
                delta_cnt = np.zeros((n_v,), dtype=np.float64)
                accepted_this_iter = 0

                for edge_key, adj in edge_pairs:
                    va, vb = int(edge_key[0]), int(edge_key[1])
                    t0i, t1i = int(adj[0]), int(adj[1])

                    ia0, ib0, ic0 = tri_vidx[t0i]
                    ia1, ib1, ic1 = tri_vidx[t1i]
                    p0a, p0b, p0c = proposed[int(ia0)], proposed[int(ib0)], proposed[int(ic0)]
                    p1a, p1b, p1c = proposed[int(ia1)], proposed[int(ib1)], proposed[int(ic1)]
                    n0 = np.cross(p0b - p0a, p0c - p0a)
                    n1 = np.cross(p1b - p1a, p1c - p1a)
                    ln0 = float(np.linalg.norm(n0))
                    ln1 = float(np.linalg.norm(n1))
                    if ln0 <= 1e-12 or ln1 <= 1e-12:
                        continue
                    n0 = n0 / ln0
                    n1 = n1 / ln1
                    cos_cur = float(np.dot(n0, n1))
                    if cos_cur < float(self.pseudomerge_cos_threshold):
                        continue

                    m = n0 + n1
                    lm = float(np.linalg.norm(m))
                    if lm <= 1e-12:
                        continue
                    m = m / lm

                    touched = np.unique(
                        np.concatenate((v_to_tri[int(va)], v_to_tri[int(vb)])).astype(np.int32)
                    )
                    if touched.size == 0:
                        continue

                    best_delta = None
                    best_gain = 0.0
                    for sign in (1.0, -1.0):
                        d = float(sign) * float(pseudo_step) * m
                        new_va = proposed[int(va)] + d
                        new_vb = proposed[int(vb)] + d

                        # Validate local neighborhood around moved edge.
                        valid_local = True
                        for tid in touched.tolist():
                            ia, ib, ic = tri_vidx[int(tid)]
                            oa = proposed[int(ia)]
                            ob = proposed[int(ib)]
                            oc = proposed[int(ic)]
                            na = new_va if int(ia) == va else (new_vb if int(ia) == vb else oa)
                            nb = new_va if int(ib) == va else (new_vb if int(ib) == vb else ob)
                            nc = new_va if int(ic) == va else (new_vb if int(ic) == vb else oc)

                            old_n = np.cross(ob - oa, oc - oa)
                            new_n = np.cross(nb - na, nc - na)
                            area = float(np.linalg.norm(new_n))
                            if (not np.isfinite(area)) or area < float(self.min_triangle_area):
                                valid_local = False
                                break
                            if float(np.dot(old_n, new_n)) <= 0.0:
                                valid_local = False
                                break
                        if not valid_local:
                            continue

                        # Keep only moves that improve local coplanarity.
                        p0a2 = new_va if int(ia0) == va else (new_vb if int(ia0) == vb else p0a)
                        p0b2 = new_va if int(ib0) == va else (new_vb if int(ib0) == vb else p0b)
                        p0c2 = new_va if int(ic0) == va else (new_vb if int(ic0) == vb else p0c)
                        p1a2 = new_va if int(ia1) == va else (new_vb if int(ia1) == vb else p1a)
                        p1b2 = new_va if int(ib1) == va else (new_vb if int(ib1) == vb else p1b)
                        p1c2 = new_va if int(ic1) == va else (new_vb if int(ic1) == vb else p1c)
                        n0_new = np.cross(p0b2 - p0a2, p0c2 - p0a2)
                        n1_new = np.cross(p1b2 - p1a2, p1c2 - p1a2)
                        ln0_new = float(np.linalg.norm(n0_new))
                        ln1_new = float(np.linalg.norm(n1_new))
                        if ln0_new <= 1e-12 or ln1_new <= 1e-12:
                            continue
                        cos_new = float(np.dot(n0_new / ln0_new, n1_new / ln1_new))
                        gain = cos_new - cos_cur
                        if gain > best_gain:
                            best_gain = gain
                            best_delta = d

                    if best_delta is None:
                        continue

                    delta_sum[int(va)] += best_delta
                    delta_sum[int(vb)] += best_delta
                    delta_cnt[int(va)] += 1.0
                    delta_cnt[int(vb)] += 1.0
                    accepted_this_iter += 1

                if accepted_this_iter <= 0:
                    break
                apply_vid = np.where(delta_cnt > 0.0)[0]
                if apply_vid.size == 0:
                    break
                proposed[apply_vid] = proposed[apply_vid] + (
                    delta_sum[apply_vid] / delta_cnt[apply_vid].reshape(-1, 1)
                )
                moved_vid[apply_vid] = True
                pseudo_merged_edges += int(accepted_this_iter)
        t_pseudomerge_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if np.any(conf_touched):
            alpha = float(np.clip(self.conf_ema_alpha, 0.0, 1.0))
            v_conf_new[conf_touched] = (1.0 - alpha) * v_conf_old[conf_touched] + alpha * conf_target[conf_touched]
            v_conf_new = np.clip(v_conf_new, float(self.conf_min), float(self.conf_max))
        t_conf_ema_ms = (time.perf_counter() - t0) * 1000.0

        # Safety rollback on moved vertices before splitting.
        t0 = time.perf_counter()
        rolled_back = np.zeros((n_v,), dtype=bool)
        if self.rollback_enabled:
            moved_vids = np.where(moved_vid)[0]
            if moved_vids.size > 0:
                affected_tri = set()
                for vid in moved_vids.tolist():
                    affected_tri.update(v_to_tri[int(vid)].tolist())
                if affected_tri:
                    affected_arr = np.asarray(sorted(affected_tri), dtype=np.int32)
                    for _ in range(3):
                        invalid = _find_invalid_triangles(
                            old_vertices=vertices,
                            new_vertices=proposed,
                            tri_vidx=tri_vidx,
                            tri_ids=affected_arr,
                            min_area=self.min_triangle_area,
                        )
                        if invalid.size == 0:
                            break
                        bad_vid = np.unique(tri_vidx[invalid].reshape(-1)).astype(np.int32)
                        proposed[bad_vid] = vertices[bad_vid]
                        v_conf_new[bad_vid] = v_conf_old[bad_vid]
                        v_miss_new[bad_vid] = v_miss_old[bad_vid]
                        moved_vid[bad_vid] = False
                        supported_vid[bad_vid] = False
                        shrunk_vid[bad_vid] = False
                        rolled_back[bad_vid] = True
        t_rollback_ms = (time.perf_counter() - t0) * 1000.0

        # Prune tiny edges first (same cycle, before splitting).
        t0 = time.perf_counter()
        pruned_edges = 0
        pruned_triangles = 0
        if self.prune_enabled and np.any(visible_tri_mask):
            prune_thresh = float(self.prune_edge_ratio) * l0_ref
            use_angle_prune = self.prune_min_angle_rad > 0.0
            if prune_thresh > 0.0 or use_angle_prune:
                edge_to_tris: Dict[Tuple[int, int], List[int]] = {}
                candidate_edges_len: Dict[Tuple[int, int], float] = {}
                candidate_edges_angle: Dict[Tuple[int, int], float] = {}
                for tid in range(num_tris):
                    t = tri_vidx[tid]
                    e01 = (int(t[0]), int(t[1]))
                    e12 = (int(t[1]), int(t[2]))
                    e20 = (int(t[2]), int(t[0]))
                    for ea, eb in (e01, e12, e20):
                        key = (ea, eb) if ea < eb else (eb, ea)
                        edge_to_tris.setdefault(key, []).append(int(tid))
                        if not visible_tri_mask[int(tid)]:
                            continue
                        if prune_thresh > 0.0:
                            elen = float(np.linalg.norm(proposed[int(ea)] - proposed[int(eb)]))
                            if np.isfinite(elen) and elen < prune_thresh:
                                prev = candidate_edges_len.get(key, None)
                                if prev is None or elen < prev:
                                    candidate_edges_len[key] = elen

                    # Additional prune candidate: collapse opposite edge of a sliver triangle.
                    if use_angle_prune and visible_tri_mask[int(tid)]:
                        va, vb, vc = int(t[0]), int(t[1]), int(t[2])
                        pa = proposed[va]
                        pb = proposed[vb]
                        pc = proposed[vc]
                        # Edge lengths opposite angles at (a,b,c): (|b-c|, |a-c|, |a-b|).
                        la = float(np.linalg.norm(pb - pc))
                        lb = float(np.linalg.norm(pa - pc))
                        lc = float(np.linalg.norm(pa - pb))
                        if (
                            np.isfinite(la)
                            and np.isfinite(lb)
                            and np.isfinite(lc)
                            and la > 1e-12
                            and lb > 1e-12
                            and lc > 1e-12
                        ):
                            # Law of cosines
                            cos_a = float(np.clip((lb * lb + lc * lc - la * la) / (2.0 * lb * lc), -1.0, 1.0))
                            cos_b = float(np.clip((la * la + lc * lc - lb * lb) / (2.0 * la * lc), -1.0, 1.0))
                            cos_c = float(np.clip((la * la + lb * lb - lc * lc) / (2.0 * la * lb), -1.0, 1.0))
                            ang_a = float(np.arccos(cos_a))
                            ang_b = float(np.arccos(cos_b))
                            ang_c = float(np.arccos(cos_c))
                            angs = [ang_a, ang_b, ang_c]
                            min_i = int(np.argmin(angs))
                            min_ang = float(angs[min_i])
                            if min_ang < self.prune_min_angle_rad:
                                # Opposite edge to minimum angle.
                                if min_i == 0:
                                    ea, eb = vb, vc
                                elif min_i == 1:
                                    ea, eb = va, vc
                                else:
                                    ea, eb = va, vb
                                key = (ea, eb) if ea < eb else (eb, ea)
                                prev = candidate_edges_angle.get(key, None)
                                if prev is None or min_ang < prev:
                                    candidate_edges_angle[key] = min_ang

                # Angle candidates first (smaller angle first), then short-edge candidates (shorter first).
                edge_keys = set(candidate_edges_len.keys()) | set(candidate_edges_angle.keys())
                edge_order = sorted(
                    edge_keys,
                    key=lambda k: (
                        0 if k in candidate_edges_angle else 1,
                        candidate_edges_angle[k] if k in candidate_edges_angle else candidate_edges_len[k],
                    ),
                )
                used_vid = np.zeros((n_v,), dtype=bool)
                used_tri = np.zeros((num_tris,), dtype=bool)
                selected_collapses: List[Tuple[int, int, np.ndarray]] = []
                expected_remove_tris: Set[int] = set()

                for edge_key in edge_order:
                    va, vb = int(edge_key[0]), int(edge_key[1])
                    adj = edge_to_tris.get(edge_key, [])
                    # Skip boundary/non-manifold edges.
                    if len(adj) != 2:
                        continue
                    t0i, t1i = int(adj[0]), int(adj[1])
                    # Only visible triangles.
                    if (not visible_tri_mask[t0i]) or (not visible_tri_mask[t1i]):
                        continue
                    if used_vid[va] or used_vid[vb]:
                        continue
                    if used_tri[t0i] or used_tri[t1i]:
                        continue

                    # Local pre-check: reject this collapse if it would create
                    # degenerate triangles outside the two adjacent edge triangles.
                    keep = int(min(va, vb))
                    rem = int(max(va, vb))
                    touched = np.where(np.any(tri_vidx == rem, axis=1))[0]
                    if touched.size == 0:
                        continue
                    tri_local = tri_vidx[touched].copy()
                    tri_local[tri_local == rem] = keep
                    deg_local = (
                        (tri_local[:, 0] == tri_local[:, 1])
                        | (tri_local[:, 1] == tri_local[:, 2])
                        | (tri_local[:, 2] == tri_local[:, 0])
                    )
                    deg_ids = touched[deg_local].astype(np.int32)
                    adj_set = {int(t0i), int(t1i)}
                    unexpected = [int(tid) for tid in deg_ids.tolist() if int(tid) not in adj_set]
                    if unexpected:
                        continue

                    selected_collapses.append((va, vb, deg_ids))
                    for tid in deg_ids.tolist():
                        expected_remove_tris.add(int(tid))
                    used_vid[va] = True
                    used_vid[vb] = True
                    used_tri[t0i] = True
                    used_tri[t1i] = True

                if selected_collapses:
                    tri_vidx_before = tri_vidx.copy()
                    tri_ids_before = tri_ids.copy()
                    tri_cvi_before = tri_cvi_raw.copy()
                    tri_inactive_bottom_before = tri_inactive_bottom.copy()
                    vis_before = visible_tri_mask.copy()

                    for va, vb, _deg_ids in selected_collapses:
                        keep = int(min(va, vb))
                        rem = int(max(va, vb))
                        midpoint = 0.5 * (proposed[va] + proposed[vb])
                        proposed[keep] = midpoint
                        v_conf_new[keep] = 0.5 * (v_conf_new[va] + v_conf_new[vb])
                        v_miss_new[keep] = int(min(int(v_miss_new[va]), int(v_miss_new[vb])))
                        tri_vidx[tri_vidx == rem] = keep

                    keep_tri_mask = np.ones((num_tris,), dtype=bool)
                    if expected_remove_tris:
                        rm_idx = np.asarray(sorted(expected_remove_tris), dtype=np.int32)
                        keep_tri_mask[rm_idx] = False

                    # Sanity guard: if any additional degenerate triangle appears
                    # beyond the expected removals, cancel prune for this cycle.
                    deg_all = (
                        (tri_vidx[:, 0] == tri_vidx[:, 1])
                        | (tri_vidx[:, 1] == tri_vidx[:, 2])
                        | (tri_vidx[:, 2] == tri_vidx[:, 0])
                    )
                    unexpected_deg = np.where(deg_all & keep_tri_mask)[0]
                    if unexpected_deg.size > 0:
                        tri_vidx = tri_vidx_before
                        tri_ids = tri_ids_before
                        tri_cvi_raw = tri_cvi_before
                        tri_inactive_bottom = tri_inactive_bottom_before
                        visible_tri_mask = vis_before
                        pruned_edges = 0
                        pruned_triangles = 0
                    else:
                        pruned_edges = int(len(selected_collapses))
                        pruned_triangles = int(np.count_nonzero(~keep_tri_mask))

                        if np.any(keep_tri_mask):
                            tri_vidx = tri_vidx[keep_tri_mask]
                            tri_ids = tri_ids_before[keep_tri_mask]
                            tri_cvi_raw = tri_cvi_before[keep_tri_mask]
                            tri_inactive_bottom = tri_inactive_bottom_before[keep_tri_mask]
                            visible_tri_mask = vis_before[keep_tri_mask]
                            num_tris = int(tri_vidx.shape[0])
                        else:
                            # Full collapse would destroy mesh; rollback prune.
                            tri_vidx = tri_vidx_before
                            tri_ids = tri_ids_before
                            tri_cvi_raw = tri_cvi_before
                            tri_inactive_bottom = tri_inactive_bottom_before
                            visible_tri_mask = vis_before
                            pruned_edges = 0
                            pruned_triangles = 0
        t_prune_ms = (time.perf_counter() - t0) * 1000.0

        # Split overgrown edges using midpoint support from current sparse points.
        t0 = time.perf_counter()
        vertices_list = [proposed[i].copy() for i in range(n_v)]
        v_conf_list = [float(v_conf_new[i]) for i in range(n_v)]
        v_miss_list = [int(v_miss_new[i]) for i in range(n_v)]

        tri_vidx_new: List[Tuple[int, int, int]] = []
        tri_ids_new: List[int] = []
        tri_cvi_new: List[float] = []
        tri_inactive_bottom_new: List[bool] = []
        next_tid = int(max(int(next_tri_id), int(np.max(tri_ids)) + 1 if tri_ids.size else 0))
        split_parents = 0
        new_triangles = 0
        tri_replacement: Dict[int, List[Tuple[int, int, int]]] = {}
        split_tri_used = np.zeros((num_tris,), dtype=bool)

        edge_len_thresh = (1.0 + float(self.split_edge_growth_ratio)) * l0_ref
        support_radius = float(self.split_correspondence_dist)

        if (
            self.split_enabled
            and np.any(visible_tri_mask)
            and support_radius > 0.0
            and sparse_points_vis.shape[0] > 0
        ):
            edge_to_tris: Dict[Tuple[int, int], List[int]] = {}
            candidate_edges: Dict[Tuple[int, int], float] = {}

            for tid in range(num_tris):
                t = tri_vidx[tid]
                e01 = (int(t[0]), int(t[1]))
                e12 = (int(t[1]), int(t[2]))
                e20 = (int(t[2]), int(t[0]))
                for ea, eb in (e01, e12, e20):
                    key = (ea, eb) if ea < eb else (eb, ea)
                    edge_to_tris.setdefault(key, []).append(int(tid))
                    if not visible_tri_mask[int(tid)]:
                        continue
                    elen = float(np.linalg.norm(proposed[int(ea)] - proposed[int(eb)]))
                    if np.isfinite(elen) and elen > edge_len_thresh:
                        prev = candidate_edges.get(key, None)
                        if prev is None or elen > prev:
                            candidate_edges[key] = elen

            split_tree = (
                _SciPyKDTree(sparse_points_vis)
                if (_SciPyKDTree is not None and sparse_points_vis.shape[0] > 0)
                else None
            )
            split_support_min_n = int(self.split_support_min_neighbors)
            edge_order = sorted(candidate_edges.items(), key=lambda kv: kv[1], reverse=True)

            for edge_key, _ in edge_order:
                va, vb = int(edge_key[0]), int(edge_key[1])
                adj_tri = edge_to_tris.get(edge_key, [])
                if not adj_tri:
                    continue
                if any(split_tri_used[int(tid)] for tid in adj_tri):
                    continue
                if not any(bool(visible_tri_mask[int(tid)]) for tid in adj_tri):
                    continue

                midpoint = 0.5 * (proposed[va] + proposed[vb])
                if split_tree is not None:
                    nb_local = split_tree.query_ball_point(midpoint, r=support_radius)
                else:
                    dd = sparse_points_vis - midpoint.reshape(1, 3)
                    d2 = np.einsum("ij,ij->i", dd, dd)
                    nb_local = np.where(d2 <= support_radius * support_radius)[0].tolist()
                if len(nb_local) < split_support_min_n:
                    continue

                # Create midpoint vertex (on edge). Keep it only if all adjacent splits are valid.
                gnew = len(vertices_list)
                q_sum = 0.0
                for li in nb_local:
                    pidx = int(sparse_idx_vis[int(li)])
                    corr_d = float(np.linalg.norm(sparse_points_vis[int(li)] - midpoint))
                    q_sum += float(self._point_quality(pidx, corr_d))
                q_mean = q_sum / float(max(1, len(nb_local)))
                vertices_list.append(midpoint.astype(np.float64))
                cnew = float(self.conf_min + q_mean * (self.conf_max - self.conf_min))
                v_conf_list.append(float(np.clip(cnew, self.conf_min, self.conf_max)))
                v_miss_list.append(0)

                local_repl: Dict[int, List[Tuple[int, int, int]]] = {}
                split_ok = True
                for tid in adj_tri:
                    parent = tri_vidx[int(tid)]
                    children = _split_triangle_along_edge(
                        tri=(int(parent[0]), int(parent[1]), int(parent[2])),
                        edge_u=va,
                        edge_v=vb,
                        mid_vid=gnew,
                    )
                    if children is None or len(children) != 2:
                        split_ok = False
                        break

                    pa = proposed[int(parent[0])]
                    pb = proposed[int(parent[1])]
                    pc = proposed[int(parent[2])]
                    pn = np.cross(pb - pa, pc - pa)
                    if float(np.linalg.norm(pn)) <= 1e-12:
                        split_ok = False
                        break

                    valid_children: List[Tuple[int, int, int]] = []
                    for ch in children:
                        a = vertices_list[int(ch[0])]
                        b = vertices_list[int(ch[1])]
                        c = vertices_list[int(ch[2])]
                        cn = np.cross(b - a, c - a)
                        area = float(np.linalg.norm(cn))
                        if (not np.isfinite(area)) or area < float(self.min_triangle_area):
                            split_ok = False
                            break
                        if float(np.dot(cn, pn)) <= 0.0:
                            split_ok = False
                            break
                        valid_children.append((int(ch[0]), int(ch[1]), int(ch[2])))
                    if (not split_ok) or len(valid_children) != 2:
                        split_ok = False
                        break
                    local_repl[int(tid)] = valid_children

                if not split_ok:
                    vertices_list.pop()
                    v_conf_list.pop()
                    v_miss_list.pop()
                    continue

                for tid, children in local_repl.items():
                    tri_replacement[int(tid)] = children
                    split_tri_used[int(tid)] = True

        for tid in range(num_tris):
            repl = tri_replacement.get(int(tid), None)
            if repl is None:
                t = tri_vidx[tid]
                tri_vidx_new.append((int(t[0]), int(t[1]), int(t[2])))
                tri_ids_new.append(int(tri_ids[tid]))
                tri_cvi_new.append(float(tri_cvi_raw[tid]))
                tri_inactive_bottom_new.append(bool(tri_inactive_bottom[tid]))
                continue

            split_parents += 1
            new_triangles += len(repl)
            child_cvi = float(tri_cvi_raw[tid]) / float(max(1, len(repl)))
            for ch in repl:
                tri_vidx_new.append((int(ch[0]), int(ch[1]), int(ch[2])))
                tri_ids_new.append(int(next_tid))
                tri_cvi_new.append(child_cvi)
                tri_inactive_bottom_new.append(bool(tri_inactive_bottom[tid]))
                next_tid += 1
        t_split_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        tri_vidx_new_arr = np.asarray(tri_vidx_new, dtype=np.int32).reshape(-1, 3)
        vertices_final = np.asarray(vertices_list, dtype=np.float64)
        v_conf_final = np.asarray(v_conf_list, dtype=np.float64)
        v_miss_final = np.asarray(v_miss_list, dtype=np.int32)

        tri_new = vertices_final[tri_vidx_new_arr].astype(np.float32)
        tri_vertex_conf_new = v_conf_final[tri_vidx_new_arr].astype(np.float32)
        tri_vertex_miss_new = v_miss_final[tri_vidx_new_arr].astype(np.int32)
        tri_ids_out = np.asarray(tri_ids_new, dtype=np.int64).reshape(-1)
        tri_cvi_out = np.asarray(tri_cvi_new, dtype=np.float32).reshape(-1)
        tri_inactive_bottom_out = np.asarray(tri_inactive_bottom_new, dtype=bool).reshape(-1)
        t_pack_ms = (time.perf_counter() - t0) * 1000.0
        t_geom_total_ms = (time.perf_counter() - t_geom_total) * 1000.0

        if pruned_edges > 0 or pruned_triangles > 0 or split_parents > 0 or new_triangles > 0:
            self._bucket_needs_rebuild = True

        stats = {
            "visible_vertices": float(target_vids.shape[0]),
            "supported_vertices": float(np.count_nonzero(supported_vid)),
            "shrunk_vertices": float(np.count_nonzero(shrunk_vid)),
            "rolled_back_vertices": float(np.count_nonzero(rolled_back)),
            "pseudo_merged_edges": float(pseudo_merged_edges),
            "associated_points": float(assoc_points),
            "near_vertex_points": float(near_points),
            "interior_points": float(interior_points),
            "pruned_edges": float(pruned_edges),
            "pruned_triangles": float(pruned_triangles),
            "split_parents": float(split_parents),
            "new_triangles": float(new_triangles),
            "assoc_sparse_tracks": float(assoc_sync_stats["assoc_sparse_tracks"]),
            "assoc_vertices": float(assoc_sync_stats["assoc_vertices"]),
            "assoc_rebuild": float(assoc_sync_stats["assoc_rebuild"]),
            "assoc_stable_ids": float(assoc_sync_stats["assoc_stable_ids"]),
            "assoc_point_ids_stable": float(assoc_sync_stats["assoc_point_ids_stable"]),
            "assoc_bucket_stable_ids": float(assoc_sync_stats["assoc_bucket_stable_ids"]),
            "assoc_point_revs_valid": float(assoc_sync_stats["assoc_point_revs_valid"]),
            "assoc_added_tracks": float(assoc_sync_stats["assoc_added_tracks"]),
            "assoc_changed_tracks": float(assoc_sync_stats["assoc_changed_tracks"]),
            "assoc_removed_tracks": float(assoc_sync_stats["assoc_removed_tracks"]),
            "assoc_pairs_changed": float(assoc_sync_stats["assoc_pairs_changed"]),
            "assoc_assign_calls": float(assoc_sync_stats["assoc_assign_calls"]),
            "assoc_assign_accepted_tracks": float(assoc_sync_stats["assoc_assign_accepted_tracks"]),
            "assoc_assign_rejected_tracks": float(assoc_sync_stats["assoc_assign_rejected_tracks"]),
            "assoc_assign_candidates_total": float(assoc_sync_stats["assoc_assign_candidates_total"]),
            "assoc_assign_kept_total": float(assoc_sync_stats["assoc_assign_kept_total"]),
            "assoc_assign_unique_vid_total": float(assoc_sync_stats["assoc_assign_unique_vid_total"]),
            "t_geom_total_ms": float(t_geom_total_ms),
            "t_geom_prepare_ms": float(t_prepare_ms),
            "t_geom_topology_ms": float(t_topology_ms),
            "t_geom_vertex_state_ms": float(t_vertex_state_ms),
            "t_geom_visibility_ms": float(t_visibility_ms),
            "t_geom_assoc_ms": float(t_assoc_ms),
            "t_assoc_key_extract_ms": float(assoc_sync_stats["t_assoc_key_extract_ms"]),
            "t_assoc_tree_ms": float(assoc_sync_stats["t_assoc_tree_ms"]),
            "t_assoc_change_detect_ms": float(assoc_sync_stats["t_assoc_change_detect_ms"]),
            "t_assoc_remove_ms": float(assoc_sync_stats["t_assoc_remove_ms"]),
            "t_assoc_assign_ms": float(assoc_sync_stats["t_assoc_assign_ms"]),
            "t_assoc_assign_query_ms": float(assoc_sync_stats["t_assoc_assign_query_ms"]),
            "t_assoc_assign_filter_ms": float(assoc_sync_stats["t_assoc_assign_filter_ms"]),
            "t_assoc_assign_quality_ms": float(assoc_sync_stats["t_assoc_assign_quality_ms"]),
            "t_assoc_assign_bucket_update_ms": float(assoc_sync_stats["t_assoc_assign_bucket_update_ms"]),
            "t_assoc_assign_record_ms": float(assoc_sync_stats["t_assoc_assign_record_ms"]),
            "t_geom_vertex_update_ms": float(t_vertex_update_ms),
            "t_geom_conf_ema_ms": float(t_conf_ema_ms),
            "t_geom_pseudomerge_ms": float(t_pseudomerge_ms),
            "t_geom_rollback_ms": float(t_rollback_ms),
            "t_geom_prune_ms": float(t_prune_ms),
            "t_geom_split_ms": float(t_split_ms),
            "t_geom_pack_ms": float(t_pack_ms),
        }
        return (
            tri_new,
            tri_ids_out,
            tri_cvi_out,
            tri_inactive_bottom_out,
            int(next_tid),
            tri_vertex_conf_new,
            tri_vertex_miss_new,
            stats,
        )

    def spin(self) -> None:
        # Wait for inputs.
        try:
            while not rospy.is_shutdown() and not self._shutdown_requested:
                poses_ok = os.path.isfile(self.poses_path)
                mesh_ok = os.path.isfile(self.mesh_path)
                ctrl_ok = self._is_controller_running()
                if poses_ok and mesh_ok and ctrl_ok:
                    break

                if not poses_ok or not mesh_ok:
                    rospy.loginfo_throttle(
                        2.0,
                        "Waiting for poses (%s) and mesh (%s) to exist...",
                        self.poses_path,
                        self.mesh_path,
                    )
                if not ctrl_ok:
                    rospy.loginfo_throttle(
                        2.0,
                        "Mesh CVI update node gated; waiting for controller node %s...",
                        self.controller_node_name,
                    )
                time.sleep(0.2)
        except KeyboardInterrupt:
            return

        if rospy.is_shutdown():
            return

        if not self._ensure_pose_file_open():
            rospy.logerr("Pose file does not exist: %s", self.poses_path)
            return

        self._init_pose_offset_from_state_or_file()
        self._pose_fp.seek(self._pose_offset, os.SEEK_SET)
        rospy.loginfo("Starting pose tail at byte offset %d", self._pose_offset)

        rate = rospy.Rate(max(0.1, self.poll_hz))
        try:
            while not rospy.is_shutdown() and not self._shutdown_requested:
                if not self._is_controller_running():
                    rospy.loginfo_throttle(
                        1.0,
                        "Mesh CVI update node gated; waiting for controller node %s...",
                        self.controller_node_name,
                    )
                    try:
                        rate.sleep()
                    except rospy.ROSInterruptException:
                        break
                    continue

                if not self._ensure_pose_file_open():
                    rate.sleep()
                    continue

                sparse_changed_outer = self._load_sparse_cache() if self.mesh_update_enabled else False
                if sparse_changed_outer:
                    self._update_mesh_from_sparse_only()

                self._sync_pose_offset_if_truncated()
                self._pose_fp.seek(self._pose_offset, os.SEEK_SET)

                processed_any = False
                while True:
                    if rospy.is_shutdown() or self._shutdown_requested:
                        break
                    line = self._pose_fp.readline()
                    if not line:
                        break

                    self._pose_offset = self._pose_fp.tell()
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    pose = _parse_pose_line(line)
                    if pose is None:
                        continue

                    self._event_counter += 1
                    if self.stride > 1 and (self._event_counter % self.stride) != 0:
                        continue

                    processed_any = True
                    image_id, cam_pos, cam_quat_xyzw, img_name = pose

                    t_cycle_start = time.perf_counter()
                    t_load_mesh_ms = 0.0
                    t_parse_arrays_ms = 0.0
                    t_raster_ms = 0.0
                    t_cvi_update_ms = 0.0
                    t_sparse_cache_ms = 0.0
                    t_geom_update_ms = 0.0
                    t_effective_cvi_ms = 0.0
                    t_pack_arrays_ms = 0.0
                    t_save_mesh_ms = 0.0
                    t_debug_save_ms = 0.0

                    t0 = time.perf_counter()
                    arrays = self._load_mesh()
                    t_load_mesh_ms = (time.perf_counter() - t0) * 1000.0
                    if arrays is None:
                        rospy.logwarn("Mesh file missing/unreadable; skipping update.")
                        continue

                    t0 = time.perf_counter()
                    mesh_state = self._extract_mesh_state(arrays)
                    t_parse_arrays_ms = (time.perf_counter() - t0) * 1000.0
                    if mesh_state is None:
                        continue
                    (
                        triangles,
                        tri_ids,
                        tri_cvi_raw,
                        tri_vertex_conf,
                        tri_vertex_miss_count,
                        tri_inactive_bottom,
                        next_tri_id,
                        mesh_l0_value,
                    ) = mesh_state
                    sparse_changed_cycle = False
                    if self.mesh_update_enabled or self._depth_patch_planning_active():
                        t0 = time.perf_counter()
                        sparse_changed_cycle = self._load_sparse_cache()
                        t_sparse_cache_ms = (time.perf_counter() - t0) * 1000.0

                    base_num_tris = int(triangles.shape[0])
                    (
                        cvi_triangles,
                        cvi_tri_ids,
                        cvi_tri_raw,
                        cvi_tri_vertex_conf,
                        cvi_tri_vertex_miss_count,
                        cvi_tri_inactive_bottom,
                        cvi_tri_source,
                        depth_planning_patch_keys,
                        depth_planning_patch_cvi_raw,
                    ) = self._append_depth_patch_to_mesh_state(
                        arrays=arrays,
                        triangles=triangles,
                        tri_ids=tri_ids,
                        tri_cvi_raw=tri_cvi_raw,
                        tri_vertex_conf=tri_vertex_conf,
                        tri_vertex_miss_count=tri_vertex_miss_count,
                        tri_inactive_bottom=tri_inactive_bottom,
                        next_tri_id=next_tri_id,
                    )
                    num_tris = int(cvi_triangles.shape[0])
                    _tri_conf_current, _tri_cvi_public_current, tri_cvi_effective_current = self._compute_tri_cvi_public_and_effective(
                        tri_cvi_raw=cvi_tri_raw,
                        tri_vertex_conf=cvi_tri_vertex_conf,
                    )
                    cvi_update_active_mask = self._cvi_update_active_mask(
                        triangles=cvi_triangles,
                        tri_cvi_effective=tri_cvi_effective_current,
                        tri_inactive_bottom=cvi_tri_inactive_bottom,
                    )

                    # 1) Rasterize once for mesh-update visibility. In geometric CVI mode,
                    # patch triangles do not need rasterization; they use vectorized frustum gain.
                    t0 = time.perf_counter()
                    raster_triangles = triangles if self.cvi_update_mode == "geometric" else cvi_triangles
                    tri_idx_buf = rasterize_visible_triangle_indices(
                        triangles_world=raster_triangles,
                        cam_pos_world=cam_pos,
                        cam_quat_xyzw_world=cam_quat_xyzw,
                        fx=self.fx,
                        fy=self.fy,
                        cx=self.cx,
                        cy=self.cy,
                        width=self.width,
                        height=self.height,
                        near_eps=self.near_eps,
                        shutdown_check=lambda: rospy.is_shutdown() or self._shutdown_requested,
                    )
                    t_raster_ms = (time.perf_counter() - t0) * 1000.0
                    if rospy.is_shutdown() or self._shutdown_requested:
                        break
                    if self.cvi_update_mode != "geometric" and num_tris > base_num_tris:
                        base_tri_idx_buf = tri_idx_buf.copy()
                        base_tri_idx_buf[base_tri_idx_buf >= base_num_tris] = -1
                    else:
                        base_tri_idx_buf = tri_idx_buf

                    # 2) CVI raw update.
                    t0 = time.perf_counter()
                    if self.cvi_update_mode == "geometric":
                        gains = self._compute_geometric_cvi_gain(
                            triangles_world=cvi_triangles,
                            cam_pos_world=cam_pos,
                            cam_quat_xyzw_world=cam_quat_xyzw,
                        )
                        if gains.shape[0] == cvi_update_active_mask.shape[0]:
                            gains = gains.copy()
                            gains[~cvi_update_active_mask] = 0.0
                        cvi_tri_raw = cvi_tri_raw + gains
                    else:
                        visible = tri_idx_buf[tri_idx_buf >= 0].astype(np.int64)
                        if visible.size > 0:
                            counts = np.bincount(visible, minlength=num_tris).astype(np.float32)
                            if counts.shape[0] == cvi_update_active_mask.shape[0]:
                                counts = counts.copy()
                                counts[~cvi_update_active_mask] = 0.0
                            cvi_tri_raw = cvi_tri_raw + counts / float(self.width * self.height)
                    t_cvi_update_ms = (time.perf_counter() - t0) * 1000.0
                    tri_cvi_raw = cvi_tri_raw[:base_num_tris].astype(np.float32, copy=False)
                    patch_cvi_raw_after_update = cvi_tri_raw[base_num_tris:].astype(np.float32, copy=False)

                    # 3) Mesh geometry update from visible/in-frustum vertices to sparse cloud.
                    geom_stats = {
                        "visible_vertices": 0.0,
                        "supported_vertices": 0.0,
                        "shrunk_vertices": 0.0,
                        "rolled_back_vertices": 0.0,
                        "pseudo_merged_edges": 0.0,
                        "associated_points": 0.0,
                        "near_vertex_points": 0.0,
                        "interior_points": 0.0,
                        "pruned_edges": 0.0,
                        "pruned_triangles": 0.0,
                        "split_parents": 0.0,
                        "new_triangles": 0.0,
                        "t_geom_total_ms": 0.0,
                        "t_geom_prepare_ms": 0.0,
                        "t_geom_topology_ms": 0.0,
                        "t_geom_vertex_state_ms": 0.0,
                        "t_geom_visibility_ms": 0.0,
                        "t_geom_assoc_ms": 0.0,
                        "t_geom_vertex_update_ms": 0.0,
                        "t_geom_conf_ema_ms": 0.0,
                        "t_geom_pseudomerge_ms": 0.0,
                        "t_geom_rollback_ms": 0.0,
                        "t_geom_prune_ms": 0.0,
                        "t_geom_split_ms": 0.0,
                        "t_geom_pack_ms": 0.0,
                    }
                    if self.mesh_update_enabled:
                        t0 = time.perf_counter()
                        (
                            triangles,
                            tri_ids,
                            tri_cvi_raw,
                            tri_inactive_bottom,
                            next_tri_id,
                            tri_vertex_conf,
                            tri_vertex_miss_count,
                            geom_stats,
                        ) = self._apply_geometry_update(
                            triangles=triangles,
                            tri_ids=tri_ids,
                            tri_cvi_raw=tri_cvi_raw,
                            tri_inactive_bottom=tri_inactive_bottom,
                            next_tri_id=next_tri_id,
                            tri_vertex_conf=tri_vertex_conf,
                            tri_vertex_miss_count=tri_vertex_miss_count,
                            tri_idx_buf=base_tri_idx_buf,
                            cam_pos_world=cam_pos,
                            cam_quat_xyzw_world=cam_quat_xyzw,
                            mesh_l0_value=mesh_l0_value,
                            sparse_changed=bool(sparse_changed_cycle),
                            allow_shrink=True,
                        )
                        t_geom_update_ms = (time.perf_counter() - t0) * 1000.0
                    if (
                        self.geometry_point_source == "depth"
                        and bool(self.depth_mesh_patching)
                        and bool(sparse_changed_cycle)
                    ):
                        self._publish_depth_patch_cut_diagnostics(triangles)

                    t0 = time.perf_counter()
                    (
                        final_triangles,
                        final_tri_ids,
                        final_tri_cvi_raw,
                        final_tri_vertex_conf,
                        final_tri_vertex_miss_count,
                        final_tri_inactive_bottom,
                        final_tri_source,
                        final_depth_patch_keys,
                        final_depth_patch_cvi_raw,
                    ) = self._append_depth_patch_to_mesh_state(
                        arrays=arrays,
                        triangles=triangles,
                        tri_ids=tri_ids,
                        tri_cvi_raw=tri_cvi_raw,
                        tri_vertex_conf=tri_vertex_conf,
                        tri_vertex_miss_count=tri_vertex_miss_count,
                        tri_inactive_bottom=tri_inactive_bottom,
                        next_tri_id=next_tri_id,
                        patch_cvi_override=patch_cvi_raw_after_update,
                    )
                    tri_conf, tri_cvi_public, tri_cvi_effective = self._compute_tri_cvi_public_and_effective(
                        tri_cvi_raw=final_tri_cvi_raw,
                        tri_vertex_conf=final_tri_vertex_conf,
                    )
                    t_effective_cvi_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    self._pack_mesh_state(
                        arrays=arrays,
                        triangles=final_triangles,
                        tri_ids=final_tri_ids,
                        tri_cvi_raw=final_tri_cvi_raw,
                        tri_cvi_public=tri_cvi_public,
                        next_tri_id=next_tri_id,
                        tri_vertex_conf=final_tri_vertex_conf,
                        tri_vertex_miss_count=final_tri_vertex_miss_count,
                        tri_inactive_bottom=final_tri_inactive_bottom,
                        tri_conf=tri_conf,
                        tri_cvi_effective=tri_cvi_effective,
                        geom_stats=geom_stats,
                        pose_offset=int(self._pose_offset),
                        tri_source=final_tri_source,
                        depth_planning_patch_keys=final_depth_patch_keys,
                        depth_planning_patch_cvi_raw=final_depth_patch_cvi_raw,
                    )
                    t_pack_arrays_ms = (time.perf_counter() - t0) * 1000.0

                    t0 = time.perf_counter()
                    self._save_mesh(arrays)
                    t_save_mesh_ms = (time.perf_counter() - t0) * 1000.0
                    t0 = time.perf_counter()
                    self._save_debug_raster(
                        tri_idx_buf=tri_idx_buf,
                        tri_cvi_effective=tri_cvi_effective,
                        image_id=image_id,
                        image_name=img_name,
                    )
                    t_debug_save_ms = (time.perf_counter() - t0) * 1000.0
                    t_cycle_total_ms = (time.perf_counter() - t_cycle_start) * 1000.0

                    rospy.loginfo(
                        "Mesh+CVI update: image_id=%s name=%s vis_v=%d sup_v=%d shr_v=%d rb_v=%d pm_edge=%d assoc=%d near=%d interior=%d pr_edge=%d pr_tri=%d split_par=%d new_tri=%d tris=%d",
                        _blue_log_image_id(image_id),
                        _blue_log_image_name(img_name),
                        int(geom_stats["visible_vertices"]),
                        int(geom_stats["supported_vertices"]),
                        int(geom_stats["shrunk_vertices"]),
                        int(geom_stats["rolled_back_vertices"]),
                        int(geom_stats["pseudo_merged_edges"]),
                        int(geom_stats["associated_points"]),
                        int(geom_stats["near_vertex_points"]),
                        int(geom_stats["interior_points"]),
                        int(geom_stats["pruned_edges"]),
                        int(geom_stats["pruned_triangles"]),
                        int(geom_stats["split_parents"]),
                        int(geom_stats["new_triangles"]),
                        int(final_triangles.shape[0]),
                    )
                    top_level_items = [
                        ("cycle_total", t_cycle_total_ms),
                        ("load_mesh", t_load_mesh_ms),
                        ("parse_arrays", t_parse_arrays_ms),
                        ("rasterize", t_raster_ms),
                        ("cvi_accum", t_cvi_update_ms),
                        ("sparse_cache", t_sparse_cache_ms),
                        ("geom_update_total", t_geom_update_ms),
                        ("effective_cvi", t_effective_cvi_ms),
                        ("pack_arrays", t_pack_arrays_ms),
                        ("save_mesh", t_save_mesh_ms),
                        ("save_debug", t_debug_save_ms),
                    ]
                    top_level_sum_ms = float(
                        sum(v for k, v in top_level_items if k != "cycle_total")
                    )
                    top_level_other_ms = max(0.0, float(t_cycle_total_ms) - top_level_sum_ms)
                    top_level_items.append(("other_untracked", top_level_other_ms))

                    geom_items = [
                        ("geom_prepare", float(geom_stats["t_geom_prepare_ms"])),
                        ("geom_topology", float(geom_stats["t_geom_topology_ms"])),
                        ("geom_vertex_state", float(geom_stats["t_geom_vertex_state_ms"])),
                        ("geom_visibility", float(geom_stats["t_geom_visibility_ms"])),
                        ("geom_assoc", float(geom_stats["t_geom_assoc_ms"])),
                        ("geom_vertex_update", float(geom_stats["t_geom_vertex_update_ms"])),
                        ("geom_pseudomerge", float(geom_stats["t_geom_pseudomerge_ms"])),
                        ("geom_conf_ema", float(geom_stats["t_geom_conf_ema_ms"])),
                        ("geom_rollback", float(geom_stats["t_geom_rollback_ms"])),
                        ("geom_prune", float(geom_stats["t_geom_prune_ms"])),
                        ("geom_split", float(geom_stats["t_geom_split_ms"])),
                        ("geom_pack", float(geom_stats["t_geom_pack_ms"])),
                    ]
                    geom_sum_ms = float(sum(v for _k, v in geom_items))
                    geom_other_ms = max(0.0, float(t_geom_update_ms) - geom_sum_ms)
                    geom_items.append(("geom_other_untracked", geom_other_ms))

                    denom = max(float(t_cycle_total_ms), 1e-9)
                    pct_stats = self._update_running_timing_pct_stats(
                        timing_items=top_level_items,
                        cycle_total_ms=t_cycle_total_ms,
                    )
                    mean_cycle_total_ms = max(
                        float(pct_stats.get("cycle_total", {}).get("mean_ms", t_cycle_total_ms)),
                        1.0e-9,
                    )
                    running_top_items = sorted(
                        [
                            (k, float(v["mean_ms"]))
                            for k, v in pct_stats.items()
                            if k != "cycle_total"
                        ],
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    timing_so_far_lines = "\n".join(
                        [
                            f"  {k}: {100.0 * (v / mean_cycle_total_ms):.2f}% ({v:.2f} ms)"
                            for k, v in [("cycle_total", mean_cycle_total_ms)] + running_top_items
                        ]
                    )
                    geom_means = self._update_running_mean_stats(
                        self._geom_timing_stats_ms,
                        [("geom_total", float(t_geom_update_ms))] + geom_items,
                    )
                    mean_geom_total_ms = max(float(geom_means.get("geom_total", t_geom_update_ms)), 1.0e-9)
                    geom_sorted = sorted(
                        [(k, float(geom_means.get(k, 0.0))) for k, _v in geom_items],
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    geom_lines = "\n".join(
                        [
                            f"  {k}: {100.0 * (v / mean_geom_total_ms):.2f}% ({v:.2f} ms)"
                            for k, v in geom_sorted
                        ]
                    )
                    assoc_lines = "  assoc breakdown unavailable"
                    assoc_summary = ""
                    assign_lines = ""
                    assign_summary = ""
                    assoc_total_mean_ms = 0.0
                    assign_total_mean_ms = 0.0
                    if not self.intersection_logic_enabled:
                        assoc_items = [
                            ("assoc_key_extract", float(geom_stats.get("t_assoc_key_extract_ms", 0.0))),
                            ("assoc_tree", float(geom_stats.get("t_assoc_tree_ms", 0.0))),
                            ("assoc_change_detect", float(geom_stats.get("t_assoc_change_detect_ms", 0.0))),
                            ("assoc_remove", float(geom_stats.get("t_assoc_remove_ms", 0.0))),
                            ("assoc_assign", float(geom_stats.get("t_assoc_assign_ms", 0.0))),
                        ]
                        assoc_sum_ms = float(sum(v for _k, v in assoc_items))
                        assoc_other_ms = max(0.0, float(geom_stats["t_geom_assoc_ms"]) - assoc_sum_ms)
                        assoc_items.append(("assoc_other_untracked", assoc_other_ms))
                        assoc_means = self._update_running_mean_stats(
                            self._assoc_timing_stats_ms,
                            [("assoc_total", float(geom_stats["t_geom_assoc_ms"]))] + assoc_items,
                        )
                        assoc_total_mean_ms = max(float(assoc_means.get("assoc_total", geom_stats["t_geom_assoc_ms"])), 1.0e-9)
                        assoc_sorted = sorted(
                            [(k, float(assoc_means.get(k, 0.0))) for k, _v in assoc_items],
                            key=lambda kv: kv[1],
                            reverse=True,
                        )
                        assoc_lines = "\n".join(
                            [
                                f"  {k}: {100.0 * (v / assoc_total_mean_ms):.2f}% ({v:.2f} ms)"
                                for k, v in assoc_sorted
                            ]
                        )
                        assoc_summary = (
                            "\nAssociation stats: tracks=%d verts=%d rebuild=%d point_ids_stable=%d bucket_ids_stable=%d point_revs_valid=%d "
                            "added=%d changed=%d removed=%d assoc_pairs_changed=%d associated_points=%d"
                        ) % (
                            int(geom_stats.get("assoc_sparse_tracks", 0.0)),
                            int(geom_stats.get("assoc_vertices", 0.0)),
                            int(geom_stats.get("assoc_rebuild", 0.0)),
                            int(geom_stats.get("assoc_point_ids_stable", 0.0)),
                            int(geom_stats.get("assoc_bucket_stable_ids", 0.0)),
                            int(geom_stats.get("assoc_point_revs_valid", 0.0)),
                            int(geom_stats.get("assoc_added_tracks", 0.0)),
                            int(geom_stats.get("assoc_changed_tracks", 0.0)),
                            int(geom_stats.get("assoc_removed_tracks", 0.0)),
                            int(geom_stats.get("assoc_pairs_changed", 0.0)),
                            int(geom_stats.get("associated_points", 0.0)),
                        )
                        assign_items = [
                            ("assign_query", float(geom_stats.get("t_assoc_assign_query_ms", 0.0))),
                            ("assign_quality", float(geom_stats.get("t_assoc_assign_quality_ms", 0.0))),
                            ("assign_bucket_update", float(geom_stats.get("t_assoc_assign_bucket_update_ms", 0.0))),
                            ("assign_record", float(geom_stats.get("t_assoc_assign_record_ms", 0.0))),
                            ("assign_filter", float(geom_stats.get("t_assoc_assign_filter_ms", 0.0))),
                        ]
                        assign_sum_ms = float(sum(v for _k, v in assign_items))
                        assign_other_ms = max(0.0, float(geom_stats.get("t_assoc_assign_ms", 0.0)) - assign_sum_ms)
                        assign_items.append(("assign_other_untracked", assign_other_ms))
                        assign_means = self._update_running_mean_stats(
                            self._assign_timing_stats_ms,
                            [("assign_total", float(geom_stats.get("t_assoc_assign_ms", 0.0)))] + assign_items,
                        )
                        assign_total_mean_ms = max(float(assign_means.get("assign_total", geom_stats.get("t_assoc_assign_ms", 0.0))), 1.0e-9)
                        assign_sorted = sorted(
                            [(k, float(assign_means.get(k, 0.0))) for k, _v in assign_items],
                            key=lambda kv: kv[1],
                            reverse=True,
                        )
                        assign_lines = "\n".join(
                            [
                                f"  {k}: {100.0 * (v / assign_total_mean_ms):.2f}% ({v:.2f} ms)"
                                for k, v in assign_sorted
                            ]
                        )
                        assign_summary = (
                            "\nAssign stats: calls=%d accepted=%d rejected=%d candidates=%d kept=%d unique_vids=%d"
                        ) % (
                            int(geom_stats.get("assoc_assign_calls", 0.0)),
                            int(geom_stats.get("assoc_assign_accepted_tracks", 0.0)),
                            int(geom_stats.get("assoc_assign_rejected_tracks", 0.0)),
                            int(geom_stats.get("assoc_assign_candidates_total", 0.0)),
                            int(geom_stats.get("assoc_assign_kept_total", 0.0)),
                            int(geom_stats.get("assoc_assign_unique_vid_total", 0.0)),
                        )
                    rospy.loginfo(
                        "Mesh+CVI timing averages: image_id=%s name=%s updates=%d cvi_mode=%s\nTop-level running means (share from mean_ms / mean_cycle_total):\n%s\nGeometry running means:\n%s\nAssociation running means:\n%s%s\nAssign running means:\n%s%s",
                        _blue_log_image_id(image_id),
                        _blue_log_image_name(img_name),
                        int(self._timing_num_updates),
                        self.cvi_update_mode,
                        timing_so_far_lines,
                        geom_lines,
                        assoc_lines,
                        assoc_summary,
                        assign_lines if assign_lines else "  assign breakdown unavailable",
                        assign_summary,
                    )

                if not processed_any:
                    rospy.loginfo_throttle(1.0, "Mesh+CVI node idle; waiting for new image poses...")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    node = RTMeshCVIUpdateNode()
    node.spin()
