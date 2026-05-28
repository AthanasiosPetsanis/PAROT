#!/usr/bin/env python3

import os
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
import rosgraph


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
    # np.savez appends ".npz" if the filename doesn't already end with it.
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

    Coordinate convention: we assume the camera's +X axis is forward (depth),
    +Y is right, +Z is down (consistent with typical NED/FRD usage).
    """
    triangles_world = np.asarray(triangles_world, dtype=np.float32)
    cam_pos_world = np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
    cam_quat_xyzw_world = np.asarray(cam_quat_xyzw_world, dtype=np.float64).reshape(4)

    h = int(height)
    w = int(width)

    depth_buf = np.full((h, w), np.inf, dtype=np.float32)
    tri_idx_buf = np.full((h, w), -1, dtype=np.int32)

    # World->camera
    R_world_cam = _quat_xyzw_to_rot(
        float(cam_quat_xyzw_world[0]),
        float(cam_quat_xyzw_world[1]),
        float(cam_quat_xyzw_world[2]),
        float(cam_quat_xyzw_world[3]),
    )
    R_cam_world = R_world_cam.T

    verts = triangles_world.reshape(-1, 3).astype(np.float64)
    verts_rel = verts - cam_pos_world[None, :]
    verts_cam = (R_cam_world @ verts_rel.T).T.reshape((-1, 3, 3))

    x = verts_cam[:, :, 0]  # forward/depth
    y = verts_cam[:, :, 1]  # right
    z = verts_cam[:, :, 2]  # down

    # Project vertices
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

        if not (np.isfinite(u0) and np.isfinite(u1) and np.isfinite(u2) and np.isfinite(v0) and np.isfinite(v1) and np.isfinite(v2)):
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

        # inv_depth should be > 0 for valid pixels.
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


class RTMeshCVINode:
    def __init__(self) -> None:
        rospy.init_node("rt_mesh_cvi_node")

        poses_dir = rospy.get_param("~file_paths/poses_dir", "/home/thanos/Documents/IROS_2026/")
        mesh_dir = rospy.get_param("~file_paths/mesh_path", "/home/thanos/Documents/IROS_2026/RT_meshing/Mesh_Space/RTmesh_test0/")

        self.poses_path = rospy.get_param(
            "~rt_meshing/cvi_poses_path",
            os.path.join(poses_dir, "poses", "poses.txt"),
        )
        self.mesh_path = rospy.get_param(
            "~rt_meshing/cvi_mesh_path",
            os.path.join(mesh_dir, "mesh_latest.npz"),
        )

        self.stride = int(rospy.get_param("~rt_meshing/cvi_stride", 1))
        self.poll_hz = float(rospy.get_param("~rt_meshing/cvi_poll_hz", 10.0))
        self.start_at_end = bool(rospy.get_param("~rt_meshing/cvi_start_at_end", False))
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

        # Default CVI rasterization at half resolution for speed.
        render_w = int(rospy.get_param("~rt_meshing/cvi_render_width", max(1, width_full // 2)))
        render_h = int(rospy.get_param("~rt_meshing/cvi_render_height", max(1, height_full // 2)))
        self.width = max(1, render_w)
        self.height = max(1, render_h)

        # Keep same camera FOV when using a reduced render resolution.
        sx = float(self.width) / float(max(1, width_full))
        sy = float(self.height) / float(max(1, height_full))
        self.fx = fx_full * sx
        self.fy = fy_full * sy
        self.cx = cx_full * sx
        self.cy = cy_full * sy

        self.near_eps = float(rospy.get_param("~rt_meshing/cvi_near_eps", 1e-3))
        self.save_debug_images = bool(rospy.get_param("~rt_meshing/cvi_save_debug_images", False))
        self.debug_every = max(1, int(rospy.get_param("~rt_meshing/cvi_debug_every", 1)))
        self.debug_dir = rospy.get_param(
            "~rt_meshing/cvi_debug_dir",
            os.path.join(mesh_dir, "cvi_renders"),
        )
        if self.save_debug_images:
            os.makedirs(self.debug_dir, exist_ok=True)

        self._pose_fp = None
        self._pose_offset = 0
        self._event_counter = 0
        self._processed_counter = 0
        self._shutdown_requested = False
        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("RTMeshCVINode configured:")
        rospy.loginfo("  poses_path: %s", self.poses_path)
        rospy.loginfo("  mesh_path:  %s", self.mesh_path)
        rospy.loginfo("  stride:     %d", self.stride)
        rospy.loginfo("  poll_hz:    %.2f", self.poll_hz)
        rospy.loginfo("  start_at_end: %s", str(self.start_at_end))
        rospy.loginfo(
            "  intrinsics(full): fx=%.3f fy=%.3f cx=%.3f cy=%.3f size=%dx%d",
            fx_full,
            fy_full,
            cx_full,
            cy_full,
            width_full,
            height_full,
        )
        rospy.loginfo(
            "  intrinsics(cvi):  fx=%.3f fy=%.3f cx=%.3f cy=%.3f size=%dx%d",
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.width,
            self.height,
        )
        rospy.loginfo("  save_debug_images: %s", str(self.save_debug_images))
        if self.save_debug_images:
            rospy.loginfo("  debug_dir: %s (every %d processed updates)", self.debug_dir, self.debug_every)
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
        except Exception as e:
            rospy.logwarn("Failed to load mesh npz '%s': %s", self.mesh_path, str(e))
            return None

    def _save_mesh(self, arrays: Dict[str, np.ndarray]) -> None:
        try:
            _atomic_savez(self.mesh_path, arrays)
        except Exception as e:
            rospy.logwarn("Failed to save mesh npz '%s': %s", self.mesh_path, str(e))

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

    def _init_pose_offset_from_state_or_file(self) -> None:
        # Use persisted offset (if present) unless explicitly starting at end.
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
            rospy.logwarn("poses.txt appears truncated (offset %d > size %d); resetting offset to 0", self._pose_offset, file_size)
            self._pose_offset = 0
            if self._pose_fp is not None:
                self._pose_fp.seek(0, os.SEEK_SET)

    def _is_controller_running(self) -> bool:
        if not self.wait_for_controller:
            return True
        try:
            pubs, subs, srvs = self._ros_master.getSystemState()
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "CVI controller gate: failed ROS master query (%s)",
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

    def spin(self) -> None:
        # Wait for inputs.
        try:
            while not rospy.is_shutdown() and not self._shutdown_requested:
                poses_ok = os.path.isfile(self.poses_path)
                mesh_ok = os.path.isfile(self.mesh_path)
                controller_ok = self._is_controller_running()
                if poses_ok and mesh_ok and controller_ok:
                    break
                if not poses_ok or not mesh_ok:
                    rospy.loginfo_throttle(
                        2.0,
                        "Waiting for poses (%s) and mesh (%s) to exist...",
                        self.poses_path,
                        self.mesh_path,
                    )
                if not controller_ok:
                    rospy.loginfo_throttle(
                        2.0,
                        "CVI node gated; waiting for controller node %s...",
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
                        "CVI node gated; waiting for controller node %s...",
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
                    # stride=N means: process events N, 2N, 3N, ... (drop the rest)
                    if self.stride > 1 and (self._event_counter % self.stride) != 0:
                        continue

                    processed_any = True
                    image_id, cam_pos, cam_quat_xyzw, img_name = pose
                    rospy.loginfo(
                        "CVI update: image_id=%s name=%s",
                        _blue_log_image_id(image_id),
                        _blue_log_image_name(img_name),
                    )

                    if rospy.is_shutdown() or self._shutdown_requested:
                        break

                    arrays = self._load_mesh()
                    if arrays is None:
                        rospy.logwarn("Mesh file missing/unreadable; skipping CVI update.")
                        continue

                    triangles = arrays.get("triangles")
                    if triangles is None:
                        rospy.logwarn("Mesh npz has no 'triangles'; skipping.")
                        continue

                    triangles = np.asarray(triangles, dtype=np.float32)
                    num_tris = int(triangles.shape[0])

                    tri_ids = arrays.get("tri_ids")
                    if tri_ids is None or np.asarray(tri_ids).reshape(-1).shape[0] != num_tris:
                        tri_ids = np.arange(num_tris, dtype=np.int64)
                    else:
                        tri_ids = np.asarray(tri_ids, dtype=np.int64).reshape(-1)
                    arrays["tri_ids"] = tri_ids

                    tri_cvi_raw = arrays.get("tri_cvi_raw")
                    if tri_cvi_raw is None:
                        tri_cvi_raw = arrays.get("tri_cvi")
                    if tri_cvi_raw is None or np.asarray(tri_cvi_raw).reshape(-1).shape[0] != num_tris:
                        tri_cvi_raw = np.zeros(num_tris, dtype=np.float32)
                    else:
                        tri_cvi_raw = np.asarray(tri_cvi_raw, dtype=np.float32).reshape(-1)

                    next_tri_id = arrays.get("next_tri_id")
                    if next_tri_id is None:
                        arrays["next_tri_id"] = np.int64(int(tri_ids.max()) + 1 if tri_ids.size else num_tris)

                    tri_idx_buf = rasterize_visible_triangle_indices(
                        triangles_world=triangles,
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

                    if rospy.is_shutdown() or self._shutdown_requested:
                        break

                    visible = tri_idx_buf[tri_idx_buf >= 0].astype(np.int64)
                    if visible.size:
                        counts = np.bincount(visible, minlength=num_tris).astype(np.float32)
                        tri_cvi_raw = tri_cvi_raw + counts / float(self.width * self.height)

                    # Re-load latest mesh before write to reduce cross-node clobbering.
                    latest = self._load_mesh()
                    if latest is None:
                        continue
                    latest_triangles = latest.get("triangles")
                    if latest_triangles is None or np.asarray(latest_triangles).shape != triangles.shape:
                        rospy.logwarn_throttle(
                            1.0,
                            "CVI save skipped: mesh triangles changed concurrently (shape mismatch).",
                        )
                        continue

                    tri_vertex_conf = latest.get("tri_vertex_conf")
                    if tri_vertex_conf is None:
                        tri_vertex_conf = np.ones((num_tris, 3), dtype=np.float32)
                    else:
                        tri_vertex_conf = np.asarray(tri_vertex_conf, dtype=np.float32)
                        if tri_vertex_conf.shape != (num_tris, 3):
                            tri_vertex_conf = np.ones((num_tris, 3), dtype=np.float32)
                    tri_conf = np.mean(tri_vertex_conf, axis=1).astype(np.float32)
                    tri_cvi_effective = (tri_cvi_raw * tri_conf).astype(np.float32)

                    latest["tri_ids"] = tri_ids
                    latest["tri_vertex_conf"] = tri_vertex_conf.astype(np.float32)
                    latest["tri_conf"] = tri_conf.astype(np.float32)
                    latest["tri_cvi_raw"] = tri_cvi_raw.astype(np.float32)
                    latest["tri_cvi"] = tri_cvi_raw.astype(np.float32)  # backward-compatible alias
                    latest["tri_cvi_effective"] = tri_cvi_effective.astype(np.float32)
                    latest["cvi_state_pose_offset"] = np.int64(self._pose_offset)

                    self._save_mesh(latest)
                    self._save_debug_raster(
                        tri_idx_buf=tri_idx_buf,
                        tri_cvi_effective=tri_cvi_effective,
                        image_id=image_id,
                        image_name=img_name,
                    )

                if not processed_any:
                    rospy.loginfo_throttle(1.0, "CVI node idle; waiting for new image poses...")
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break
        except KeyboardInterrupt:
            return


if __name__ == "__main__":
    node = RTMeshCVINode()
    node.spin()
