#!/usr/bin/env python3

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy


def _atomic_savez(path: str, arrays: Dict[str, np.ndarray]) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


def _as_scalar_int(x, default: int = 0) -> int:
    if x is None:
        return int(default)
    try:
        return int(np.array(x).reshape(()).item())
    except Exception:
        return int(default)


def _as_scalar_float(x, default: float = 0.0) -> float:
    if x is None:
        return float(default)
    try:
        return float(np.array(x).reshape(()).item())
    except Exception:
        return float(default)


def _closest_point_barycentric(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Closest point on triangle ABC to point P.
    Returns:
      q: closest point
      bary: barycentric weights (wa, wb, wc)
      d2: squared distance ||p-q||^2
    """
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        q = a
        bary = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return q, bary, float(np.dot(p - q, p - q))

    bp = p - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        q = b
        bary = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        return q, bary, float(np.dot(p - q, p - q))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / max(d1 - d3, 1e-12)
        q = a + v * ab
        bary = np.array([1.0 - v, v, 0.0], dtype=np.float64)
        return q, bary, float(np.dot(p - q, p - q))

    cp = p - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        q = c
        bary = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return q, bary, float(np.dot(p - q, p - q))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / max(d2 - d6, 1e-12)
        q = a + w * ac
        bary = np.array([1.0 - w, 0.0, w], dtype=np.float64)
        return q, bary, float(np.dot(p - q, p - q))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1e-12)
        q = b + w * (c - b)
        bary = np.array([0.0, 1.0 - w, w], dtype=np.float64)
        return q, bary, float(np.dot(p - q, p - q))

    denom = max(va + vb + vc, 1e-12)
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w
    q = a + ab * v + ac * w
    bary = np.array([u, v, w], dtype=np.float64)
    return q, bary, float(np.dot(p - q, p - q))


def _triangles_valid_orientation(
    triangles_old: np.ndarray,
    triangles_new: np.ndarray,
    min_area: float,
) -> bool:
    old_n = np.cross(
        triangles_old[:, 1, :] - triangles_old[:, 0, :],
        triangles_old[:, 2, :] - triangles_old[:, 0, :],
    )
    new_n = np.cross(
        triangles_new[:, 1, :] - triangles_new[:, 0, :],
        triangles_new[:, 2, :] - triangles_new[:, 0, :],
    )
    new_area = np.linalg.norm(new_n, axis=1)
    if np.any(~np.isfinite(new_area)):
        return False
    if np.any(new_area < float(min_area)):
        return False

    dots = np.einsum("ij,ij->i", old_n, new_n)
    return not bool(np.any(dots <= 0.0))


class RTMeshUpdateNode:
    def __init__(self) -> None:
        rospy.init_node("rt_mesh_update_node")

        self.enabled = bool(rospy.get_param("~rt_meshing/mesh_update_enabled", False))
        if not self.enabled:
            rospy.logwarn("rt_mesh_update_node disabled (set rt_meshing/mesh_update_enabled=true).")
            return

        mesh_dir = rospy.get_param(
            "~file_paths/mesh_path",
            "/home/thanos/Documents/IROS_2026/RT_meshing/Mesh_Space/RTmesh_test0/",
        )
        self.mesh_path = rospy.get_param(
            "~rt_meshing/mesh_update_mesh_path",
            os.path.join(mesh_dir, "mesh_latest.npz"),
        )
        self.sparse_path = rospy.get_param(
            "~rt_meshing/mesh_update_sparse_path",
            os.path.join(mesh_dir, "sparse_latest.npz"),
        )

        self.poll_hz = float(rospy.get_param("~rt_meshing/mesh_update_poll_hz", 2.0))
        self.reset_state_on_start = bool(rospy.get_param("~rt_meshing/mesh_update_reset_state_on_start", False))
        self.start_at_end = bool(rospy.get_param("~rt_meshing/mesh_update_start_at_end", False))

        self.k_nearest_vertices = max(1, int(rospy.get_param("~rt_meshing/mesh_update_k_nearest_vertices", 3)))
        self.vertex_ring = max(0, int(rospy.get_param("~rt_meshing/mesh_update_vertex_ring", 1)))
        self.max_new_points_per_cycle = max(
            1, int(rospy.get_param("~rt_meshing/mesh_update_max_new_points_per_cycle", 1500))
        )
        self.max_correspondence_dist = float(rospy.get_param("~rt_meshing/mesh_update_max_correspondence_dist", 0.75))
        self.learning_rate = float(rospy.get_param("~rt_meshing/mesh_update_learning_rate", 0.2))
        self.max_step_m = float(rospy.get_param("~rt_meshing/mesh_update_max_step_m", 0.04))
        self.quantize_eps = float(rospy.get_param("~rt_meshing/mesh_update_quantize_eps", 1e-6))

        self.temperature_start = float(rospy.get_param("~rt_meshing/mesh_update_temperature_start", 1.0))
        self.temperature_end = float(rospy.get_param("~rt_meshing/mesh_update_temperature_end", 0.15))
        self.temperature_decay = float(rospy.get_param("~rt_meshing/mesh_update_temperature_decay", 0.995))

        self.flip_check_enabled = bool(rospy.get_param("~rt_meshing/mesh_update_flip_check_enabled", True))
        self.min_triangle_area = float(rospy.get_param("~rt_meshing/mesh_update_min_triangle_area", 1e-10))
        self.line_search_min_scale = float(rospy.get_param("~rt_meshing/mesh_update_line_search_min_scale", 0.125))
        self.laplacian_lambda = float(rospy.get_param("~rt_meshing/mesh_update_laplacian_lambda", 0.0))
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

        self._state_key_last_count = "mesh_update_state_last_sparse_count"
        self._state_key_temp = "mesh_update_temperature"
        self._state_key_iter = "mesh_update_iteration"

        rospy.loginfo("RTMeshUpdateNode configured:")
        rospy.loginfo("  mesh: %s", self.mesh_path)
        rospy.loginfo("  sparse: %s", self.sparse_path)
        rospy.loginfo("  k_nearest=%d ring=%d max_new_per_cycle=%d", self.k_nearest_vertices, self.vertex_ring, self.max_new_points_per_cycle)
        rospy.loginfo("  lr=%.4f max_step=%.4f max_corr=%.4f", self.learning_rate, self.max_step_m, self.max_correspondence_dist)
        rospy.loginfo(
            "  conf: alpha=%.3f range=[%.2f, %.2f] reproj_bad=%.3f parallax_good=%.3f dist_bad=%.3f",
            self.conf_ema_alpha,
            self.conf_min,
            self.conf_max,
            self.conf_reproj_bad_px,
            self.conf_parallax_good_deg,
            self.conf_dist_bad_m,
        )
        rospy.loginfo(
            "  temp: start=%.4f end=%.4f decay=%.6f",
            self.temperature_start,
            self.temperature_end,
            self.temperature_decay,
        )

    def _load_npz(self, path: str) -> Optional[Tuple[Dict[str, np.ndarray], float]]:
        if not os.path.isfile(path):
            return None
        try:
            mtime = os.path.getmtime(path)
            data = np.load(path, allow_pickle=False)
            arrays = {k: data[k] for k in data.files}
            data.close()
            return arrays, float(mtime)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "Failed to load npz %s (%s)", path, str(exc))
            return None

    def _save_npz(self, path: str, arrays: Dict[str, np.ndarray]) -> bool:
        try:
            _atomic_savez(path, arrays)
            return True
        except Exception as exc:
            rospy.logwarn("Failed to save npz %s (%s)", path, str(exc))
            return False

    @staticmethod
    def _build_vertices_and_topology(
        triangles: np.ndarray,
        quantize_eps: float,
        ring_depth: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
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
        v_neighbors: List[set] = [set() for _ in range(n_v)]
        v_to_tri: List[List[int]] = [[] for _ in range(n_v)]
        for t_idx, tri in enumerate(tri_vidx):
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            v_to_tri[a].append(t_idx)
            v_to_tri[b].append(t_idx)
            v_to_tri[c].append(t_idx)
            v_neighbors[a].add(b)
            v_neighbors[a].add(c)
            v_neighbors[b].add(a)
            v_neighbors[b].add(c)
            v_neighbors[c].add(a)
            v_neighbors[c].add(b)

        v_neighbors_arr: List[np.ndarray] = []
        for s in v_neighbors:
            if s:
                v_neighbors_arr.append(np.asarray(sorted(s), dtype=np.int32))
            else:
                v_neighbors_arr.append(np.zeros((0,), dtype=np.int32))

        v_to_tri_arr: List[np.ndarray] = []
        for lst in v_to_tri:
            if lst:
                v_to_tri_arr.append(np.asarray(lst, dtype=np.int32))
            else:
                v_to_tri_arr.append(np.zeros((0,), dtype=np.int32))

        v_to_candidate_tris: List[np.ndarray] = []
        for vid in range(n_v):
            active = {vid}
            frontier = {vid}
            for _ in range(int(ring_depth)):
                nxt = set()
                for v in frontier:
                    for nb in v_neighbors_arr[v].tolist():
                        if nb not in active:
                            nxt.add(int(nb))
                if not nxt:
                    break
                active.update(nxt)
                frontier = nxt
            tri_set = set()
            for v in active:
                tri_set.update(v_to_tri_arr[v].tolist())
            if tri_set:
                v_to_candidate_tris.append(np.asarray(sorted(tri_set), dtype=np.int32))
            else:
                v_to_candidate_tris.append(np.zeros((0,), dtype=np.int32))

        return vertices, tri_vidx, v_neighbors_arr, v_to_tri_arr, v_to_candidate_tris

    def _fit_mesh_batch(
        self,
        triangles: np.ndarray,
        tri_vertex_conf: np.ndarray,
        sparse_points: np.ndarray,
        sparse_reproj_error_px: np.ndarray,
        sparse_parallax_deg: np.ndarray,
        sparse_views_support: np.ndarray,
        sparse_views_total: np.ndarray,
        temperature: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
        triangles = np.asarray(triangles, dtype=np.float64)
        tri_vertex_conf = np.asarray(tri_vertex_conf, dtype=np.float64)
        if tri_vertex_conf.shape != (triangles.shape[0], 3):
            tri_vertex_conf = np.full((triangles.shape[0], 3), float(self.conf_min), dtype=np.float64)
        points = np.asarray(sparse_points, dtype=np.float64).reshape(-1, 3)
        rep = np.asarray(sparse_reproj_error_px, dtype=np.float64).reshape(-1)
        par = np.asarray(sparse_parallax_deg, dtype=np.float64).reshape(-1)
        vsp = np.asarray(sparse_views_support, dtype=np.float64).reshape(-1)
        vto = np.asarray(sparse_views_total, dtype=np.float64).reshape(-1)

        n_pts = int(points.shape[0])
        if n_pts <= 0:
            tri_conf = np.mean(tri_vertex_conf, axis=1).astype(np.float32)
            return (
                triangles.astype(np.float32),
                tri_vertex_conf.astype(np.float32),
                tri_conf,
                {"accepted_points": 0, "avg_corr_dist": 0.0, "step_scale": 0.0},
            )
        if rep.shape[0] != n_pts:
            rep = np.full((n_pts,), float(self.conf_reproj_bad_px), dtype=np.float64)
        if par.shape[0] != n_pts:
            par = np.full((n_pts,), float(self.conf_parallax_good_deg), dtype=np.float64)
        if vsp.shape[0] != n_pts:
            vsp = np.full((n_pts,), 1.0, dtype=np.float64)
        if vto.shape[0] != n_pts:
            vto = np.maximum(vsp, 1.0)

        (
            vertices,
            tri_vidx,
            v_neighbors,
            _v_to_tri,
            v_to_candidate_tris,
        ) = self._build_vertices_and_topology(
            triangles=triangles,
            quantize_eps=self.quantize_eps,
            ring_depth=self.vertex_ring,
        )
        if vertices.shape[0] == 0:
            tri_conf = np.mean(tri_vertex_conf, axis=1).astype(np.float32)
            return (
                triangles.astype(np.float32),
                tri_vertex_conf.astype(np.float32),
                tri_conf,
                {"accepted_points": 0, "avg_corr_dist": 0.0, "step_scale": 0.0},
            )

        tri_vertex_conf = np.clip(tri_vertex_conf, float(self.conf_min), float(self.conf_max))

        v_conf_sum = np.zeros((vertices.shape[0],), dtype=np.float64)
        v_conf_cnt = np.zeros((vertices.shape[0],), dtype=np.float64)
        for c in range(3):
            np.add.at(v_conf_sum, tri_vidx[:, c], tri_vertex_conf[:, c])
            np.add.at(v_conf_cnt, tri_vidx[:, c], 1.0)
        v_conf_old = np.full((vertices.shape[0],), float(self.conf_min), dtype=np.float64)
        valid_vc = v_conf_cnt > 0.0
        v_conf_old[valid_vc] = v_conf_sum[valid_vc] / np.maximum(v_conf_cnt[valid_vc], 1e-12)
        v_conf_old = np.clip(v_conf_old, float(self.conf_min), float(self.conf_max))

        delta_sum = np.zeros_like(vertices, dtype=np.float64)
        weight_sum = np.zeros((vertices.shape[0],), dtype=np.float64)
        conf_target_sum = np.zeros((vertices.shape[0],), dtype=np.float64)
        conf_target_wsum = np.zeros((vertices.shape[0],), dtype=np.float64)

        max_corr_d2 = float(self.max_correspondence_dist * self.max_correspondence_dist)
        accepted = 0
        corr_dist_sum = 0.0
        k = min(int(self.k_nearest_vertices), int(vertices.shape[0]))

        for i, p in enumerate(points):
            d2_v = np.sum((vertices - p[None, :]) ** 2, axis=1)
            if k == vertices.shape[0]:
                nn_vids = np.arange(vertices.shape[0], dtype=np.int32)
            else:
                nn_vids = np.argpartition(d2_v, k - 1)[:k].astype(np.int32)

            tri_candidates = set()
            for vid in nn_vids.tolist():
                tri_candidates.update(v_to_candidate_tris[int(vid)].tolist())
            if not tri_candidates:
                continue

            best_d2 = np.inf
            best_tri = -1
            best_q = None
            best_bary = None
            for t_idx in tri_candidates:
                ia, ib, ic = tri_vidx[int(t_idx)]
                a = vertices[int(ia)]
                b = vertices[int(ib)]
                c = vertices[int(ic)]
                q, bary, d2 = _closest_point_barycentric(p, a, b, c)
                if d2 < best_d2:
                    best_d2 = float(d2)
                    best_tri = int(t_idx)
                    best_q = q
                    best_bary = bary
            if best_tri < 0 or best_q is None or best_bary is None:
                continue
            if best_d2 > max_corr_d2:
                continue

            d = p - best_q
            ia, ib, ic = tri_vidx[best_tri]
            wa, wb, wc = best_bary.tolist()
            delta_sum[int(ia)] += wa * d
            delta_sum[int(ib)] += wb * d
            delta_sum[int(ic)] += wc * d
            weight_sum[int(ia)] += wa
            weight_sum[int(ib)] += wb
            weight_sum[int(ic)] += wc

            views_total = float(max(vto[i], 1.0))
            views_support = float(max(vsp[i], 0.0))
            views_support = min(views_support, views_total)
            s_views = views_support / views_total
            s_reproj = 1.0 - float(np.clip(rep[i] / max(self.conf_reproj_bad_px, 1e-6), 0.0, 1.0))
            s_parallax = float(np.clip(par[i] / max(self.conf_parallax_good_deg, 1e-6), 0.0, 1.0))
            corr_dist = float(np.sqrt(best_d2))
            s_dist = 1.0 - float(np.clip(corr_dist / max(self.conf_dist_bad_m, 1e-6), 0.0, 1.0))
            q = (s_views + s_reproj + s_parallax + s_dist) * 0.25
            c_target = float(self.conf_min + q * (self.conf_max - self.conf_min))

            conf_target_sum[int(ia)] += wa * c_target
            conf_target_sum[int(ib)] += wb * c_target
            conf_target_sum[int(ic)] += wc * c_target
            conf_target_wsum[int(ia)] += wa
            conf_target_wsum[int(ib)] += wb
            conf_target_wsum[int(ic)] += wc
            accepted += 1
            corr_dist_sum += corr_dist

        if accepted <= 0:
            tri_conf = np.mean(tri_vertex_conf, axis=1).astype(np.float32)
            return (
                triangles.astype(np.float32),
                tri_vertex_conf.astype(np.float32),
                tri_conf,
                {"accepted_points": 0, "avg_corr_dist": 0.0, "step_scale": 0.0},
            )

        active = weight_sum > 1e-12
        if not np.any(active):
            tri_conf = np.mean(tri_vertex_conf, axis=1).astype(np.float32)
            return (
                triangles.astype(np.float32),
                tri_vertex_conf.astype(np.float32),
                tri_conf,
                {"accepted_points": 0, "avg_corr_dist": 0.0, "step_scale": 0.0},
            )

        lr_eff = float(self.learning_rate) * float(np.clip(temperature, self.temperature_end, self.temperature_start))
        delta = np.zeros_like(vertices, dtype=np.float64)
        delta[active] = lr_eff * (delta_sum[active] / weight_sum[active][:, None])

        if self.laplacian_lambda > 0.0:
            lap = np.zeros_like(vertices, dtype=np.float64)
            active_idx = np.where(active)[0]
            for vid in active_idx.tolist():
                nbs = v_neighbors[int(vid)]
                if nbs.size == 0:
                    continue
                lap[int(vid)] = np.mean(vertices[nbs], axis=0) - vertices[int(vid)]
            delta[active] += float(self.laplacian_lambda) * float(temperature) * lap[active]

        cap = max(1e-6, float(self.max_step_m) * float(np.clip(temperature, self.temperature_end, self.temperature_start)))
        dn = np.linalg.norm(delta, axis=1)
        over = dn > cap
        if np.any(over):
            scale = cap / np.maximum(dn[over], 1e-12)
            delta[over] *= scale[:, None]

        old_triangles = vertices[tri_vidx]
        step_scale = 1.0
        if self.flip_check_enabled:
            step_scale = 0.0
            test_scale = 1.0
            min_scale = max(1e-4, float(self.line_search_min_scale))
            while test_scale >= min_scale:
                test_vertices = vertices + test_scale * delta
                test_triangles = test_vertices[tri_vidx]
                if _triangles_valid_orientation(
                    triangles_old=old_triangles,
                    triangles_new=test_triangles,
                    min_area=self.min_triangle_area,
                ):
                    step_scale = float(test_scale)
                    break
                test_scale *= 0.5
            if step_scale <= 0.0:
                tri_conf = np.mean(tri_vertex_conf, axis=1).astype(np.float32)
                return (
                    triangles.astype(np.float32),
                    tri_vertex_conf.astype(np.float32),
                    tri_conf,
                    {
                        "accepted_points": int(accepted),
                        "avg_corr_dist": float(corr_dist_sum / float(max(accepted, 1))),
                        "step_scale": 0.0,
                    },
                )

        new_vertices = vertices + step_scale * delta
        new_triangles = new_vertices[tri_vidx].astype(np.float32)
        v_conf_new = v_conf_old.copy()
        touched = conf_target_wsum > 1e-12
        if np.any(touched):
            c_target_v = np.zeros_like(v_conf_new)
            c_target_v[touched] = conf_target_sum[touched] / np.maximum(conf_target_wsum[touched], 1e-12)
            alpha = float(np.clip(self.conf_ema_alpha, 0.0, 1.0))
            v_conf_new[touched] = (1.0 - alpha) * v_conf_old[touched] + alpha * c_target_v[touched]
            v_conf_new = np.clip(v_conf_new, float(self.conf_min), float(self.conf_max))

        new_tri_vertex_conf = v_conf_new[tri_vidx].astype(np.float32)
        tri_conf = np.mean(new_tri_vertex_conf, axis=1).astype(np.float32)
        return (
            new_triangles,
            new_tri_vertex_conf,
            tri_conf,
            {
                "accepted_points": int(accepted),
                "avg_corr_dist": float(corr_dist_sum / float(max(accepted, 1))),
                "step_scale": float(step_scale),
            },
        )

    def _ensure_startup_state(self) -> None:
        sparse_loaded = self._load_npz(self.sparse_path)
        mesh_loaded = self._load_npz(self.mesh_path)
        if sparse_loaded is None or mesh_loaded is None:
            return
        sparse_arrays, _ = sparse_loaded
        mesh_arrays, _ = mesh_loaded

        points = np.asarray(sparse_arrays.get("points_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
        sparse_count = int(points.shape[0])
        triangles = np.asarray(mesh_arrays.get("triangles", np.zeros((0, 3, 3), dtype=np.float32)), dtype=np.float32)
        num_tris = int(triangles.shape[0]) if triangles.ndim == 3 and triangles.shape[1:] == (3, 3) else 0
        if num_tris > 0:
            tri_vertex_conf = np.asarray(mesh_arrays.get("tri_vertex_conf", None), dtype=np.float32) if "tri_vertex_conf" in mesh_arrays else None
            if tri_vertex_conf is None or tri_vertex_conf.shape != (num_tris, 3):
                tri_vertex_conf = np.full((num_tris, 3), float(self.conf_min), dtype=np.float32)
            mesh_arrays["tri_vertex_conf"] = tri_vertex_conf
            mesh_arrays["tri_conf"] = np.mean(tri_vertex_conf, axis=1).astype(np.float32)
            if "tri_cvi_raw" not in mesh_arrays:
                if "tri_cvi" in mesh_arrays:
                    mesh_arrays["tri_cvi_raw"] = np.asarray(mesh_arrays["tri_cvi"], dtype=np.float32).reshape(-1)
                else:
                    mesh_arrays["tri_cvi_raw"] = np.zeros((num_tris,), dtype=np.float32)
            tri_cvi_raw = np.asarray(mesh_arrays["tri_cvi_raw"], dtype=np.float32).reshape(-1)
            if tri_cvi_raw.shape[0] != num_tris:
                tri_cvi_raw = np.zeros((num_tris,), dtype=np.float32)
                mesh_arrays["tri_cvi_raw"] = tri_cvi_raw
            mesh_arrays["tri_cvi_effective"] = (tri_cvi_raw * mesh_arrays["tri_conf"]).astype(np.float32)

        has_state = self._state_key_last_count in mesh_arrays and self._state_key_temp in mesh_arrays
        if not self.reset_state_on_start and has_state:
            # Still persist any missing confidence arrays initialized above.
            self._save_npz(self.mesh_path, mesh_arrays)
            return

        last_count = sparse_count if self.start_at_end else 0
        mesh_arrays[self._state_key_last_count] = np.int64(last_count)
        mesh_arrays[self._state_key_temp] = np.float32(self.temperature_start)
        mesh_arrays[self._state_key_iter] = np.int64(0)
        mesh_arrays["mesh_update_state_updated_unix"] = np.float64(time.time())
        self._save_npz(self.mesh_path, mesh_arrays)
        rospy.loginfo(
            "Mesh-update state initialized: last_sparse_count=%d temp=%.4f",
            last_count,
            self.temperature_start,
        )

    def _process_once(self) -> bool:
        sparse_loaded = self._load_npz(self.sparse_path)
        mesh_loaded = self._load_npz(self.mesh_path)
        if sparse_loaded is None or mesh_loaded is None:
            return False
        sparse_arrays, _ = sparse_loaded
        mesh_arrays, _ = mesh_loaded

        triangles = mesh_arrays.get("triangles", None)
        if triangles is None:
            rospy.logwarn_throttle(2.0, "Mesh update: mesh npz has no 'triangles'.")
            return False
        triangles = np.asarray(triangles, dtype=np.float32)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
            rospy.logwarn_throttle(2.0, "Mesh update: invalid triangles shape %s", str(triangles.shape))
            return False
        num_tris = int(triangles.shape[0])
        tri_vertex_conf = np.asarray(
            mesh_arrays.get("tri_vertex_conf", np.full((num_tris, 3), float(self.conf_min), dtype=np.float32)),
            dtype=np.float32,
        )
        if tri_vertex_conf.shape != (num_tris, 3):
            tri_vertex_conf = np.full((num_tris, 3), float(self.conf_min), dtype=np.float32)

        sparse_pts = np.asarray(
            sparse_arrays.get("points_xyz", np.zeros((0, 3), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 3)
        sparse_reproj = np.asarray(
            sparse_arrays.get(
                "reproj_error_px",
                np.full((sparse_pts.shape[0],), float(self.conf_reproj_bad_px), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        sparse_parallax = np.asarray(
            sparse_arrays.get(
                "parallax_deg",
                np.full((sparse_pts.shape[0],), float(self.conf_parallax_good_deg), dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        sparse_views_support = np.asarray(
            sparse_arrays.get(
                "views_support",
                np.full((sparse_pts.shape[0],), 1.0, dtype=np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        sparse_views_total = np.asarray(
            sparse_arrays.get(
                "views_total",
                np.maximum(sparse_views_support, 1.0).astype(np.float32),
            ),
            dtype=np.float32,
        ).reshape(-1)
        sparse_count = int(sparse_pts.shape[0])
        if sparse_count <= 0:
            return False
        if sparse_reproj.shape[0] != sparse_count:
            sparse_reproj = np.full((sparse_count,), float(self.conf_reproj_bad_px), dtype=np.float32)
        if sparse_parallax.shape[0] != sparse_count:
            sparse_parallax = np.full((sparse_count,), float(self.conf_parallax_good_deg), dtype=np.float32)
        if sparse_views_support.shape[0] != sparse_count:
            sparse_views_support = np.full((sparse_count,), 1.0, dtype=np.float32)
        if sparse_views_total.shape[0] != sparse_count:
            sparse_views_total = np.maximum(sparse_views_support, 1.0).astype(np.float32)

        last_count = _as_scalar_int(mesh_arrays.get(self._state_key_last_count, None), default=0)
        if last_count < 0:
            last_count = 0
        if last_count > sparse_count:
            # Sparse map was reset/rebuilt.
            last_count = 0

        if sparse_count <= last_count:
            return False

        end = min(sparse_count, last_count + self.max_new_points_per_cycle)
        batch = sparse_pts[last_count:end].astype(np.float64)
        batch_reproj = sparse_reproj[last_count:end].astype(np.float64)
        batch_parallax = sparse_parallax[last_count:end].astype(np.float64)
        batch_views_support = sparse_views_support[last_count:end].astype(np.float64)
        batch_views_total = sparse_views_total[last_count:end].astype(np.float64)

        temp = _as_scalar_float(mesh_arrays.get(self._state_key_temp, None), default=self.temperature_start)
        temp = float(np.clip(temp, self.temperature_end, self.temperature_start))
        iteration = _as_scalar_int(mesh_arrays.get(self._state_key_iter, None), default=0)

        tri_new, tri_vertex_conf_new, tri_conf_new, stats = self._fit_mesh_batch(
            triangles=triangles,
            tri_vertex_conf=tri_vertex_conf,
            sparse_points=batch,
            sparse_reproj_error_px=batch_reproj,
            sparse_parallax_deg=batch_parallax,
            sparse_views_support=batch_views_support,
            sparse_views_total=batch_views_total,
            temperature=temp,
        )
        temp_new = max(float(self.temperature_end), float(temp) * float(self.temperature_decay))
        iter_new = int(iteration + 1)

        # Re-load latest mesh right before save, then only overwrite geometry + updater state.
        latest_loaded = self._load_npz(self.mesh_path)
        if latest_loaded is None:
            return False
        latest_arrays, _ = latest_loaded
        latest_tri = np.asarray(latest_arrays.get("triangles", np.zeros((0, 3, 3), dtype=np.float32)), dtype=np.float32)
        if latest_tri.shape != tri_new.shape:
            rospy.logwarn_throttle(
                1.0,
                "Mesh update skipped save: triangles shape changed concurrently (%s -> %s).",
                str(tri_new.shape),
                str(latest_tri.shape),
            )
            return False

        latest_arrays["triangles"] = tri_new.astype(np.float32)
        latest_arrays["points"] = np.mean(tri_new, axis=1).astype(np.float32)
        latest_arrays["tri_vertex_conf"] = tri_vertex_conf_new.astype(np.float32)
        latest_arrays["tri_conf"] = tri_conf_new.astype(np.float32)
        tri_cvi_raw = np.asarray(
            latest_arrays.get("tri_cvi_raw", latest_arrays.get("tri_cvi", np.zeros((num_tris,), dtype=np.float32))),
            dtype=np.float32,
        ).reshape(-1)
        if tri_cvi_raw.shape[0] != num_tris:
            tri_cvi_raw = np.zeros((num_tris,), dtype=np.float32)
        latest_arrays["tri_cvi_raw"] = tri_cvi_raw.astype(np.float32)
        latest_arrays["tri_cvi"] = tri_cvi_raw.astype(np.float32)
        latest_arrays["tri_cvi_effective"] = (tri_cvi_raw * tri_conf_new).astype(np.float32)
        latest_arrays[self._state_key_last_count] = np.int64(end)
        latest_arrays[self._state_key_temp] = np.float32(temp_new)
        latest_arrays[self._state_key_iter] = np.int64(iter_new)
        latest_arrays["mesh_update_last_batch_points"] = np.int64(int(batch.shape[0]))
        latest_arrays["mesh_update_last_accepted_points"] = np.int64(int(stats["accepted_points"]))
        latest_arrays["mesh_update_last_avg_corr_dist"] = np.float32(float(stats["avg_corr_dist"]))
        latest_arrays["mesh_update_last_step_scale"] = np.float32(float(stats["step_scale"]))
        latest_arrays["mesh_update_state_updated_unix"] = np.float64(time.time())

        if not self._save_npz(self.mesh_path, latest_arrays):
            return False

        rospy.loginfo(
            "Mesh update: sparse[%d:%d] accepted=%d avg_corr=%.4f step=%.3f temp=%.4f->%.4f",
            last_count,
            end,
            int(stats["accepted_points"]),
            float(stats["avg_corr_dist"]),
            float(stats["step_scale"]),
            float(temp),
            float(temp_new),
        )
        return True

    def spin(self) -> None:
        if not self.enabled:
            return

        self._ensure_startup_state()

        rate = rospy.Rate(max(0.2, self.poll_hz))
        while not rospy.is_shutdown():
            processed = self._process_once()
            if not processed:
                rospy.loginfo_throttle(1.0, "Mesh update node idle; waiting for new sparse points...")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break


if __name__ == "__main__":
    try:
        node = RTMeshUpdateNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
