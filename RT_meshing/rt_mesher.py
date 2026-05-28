import numpy as np


class RTBoxMesh:
    def __init__(
        self,
        center,
        size,
        target_edge_m=0.5,
        include_bottom=True,
    ):
        self.center = np.asarray(center, dtype=np.float32)
        self.size = np.asarray(size, dtype=np.float32)
        self.target_edge_m = float(max(target_edge_m, 1e-6))
        self.include_bottom = bool(include_bottom)

    def _face_grid(self, origin, u_dir, v_dir):
        lu = float(np.linalg.norm(u_dir))
        lv = float(np.linalg.norm(v_dir))
        nu = max(1, int(np.ceil(lu / self.target_edge_m)))
        nv = max(1, int(np.ceil(lv / self.target_edge_m)))
        triangles = []
        points = []
        for i in range(nu):
            u0 = i / nu
            u1 = (i + 1) / nu
            for j in range(nv):
                v0 = j / nv
                v1 = (j + 1) / nv
                p00 = origin + u_dir * u0 + v_dir * v0
                p10 = origin + u_dir * u1 + v_dir * v0
                p11 = origin + u_dir * u1 + v_dir * v1
                p01 = origin + u_dir * u0 + v_dir * v1
                triangles.append([p00, p10, p11])
                triangles.append([p00, p11, p01])
                points.append((p00 + p11) * 0.5)
        return triangles, points

    def build(self):
        cx, cy, cz = self.center.tolist()
        sx, sy, sz = self.size.tolist()
        hx, hy, hz = 0.5 * sx, 0.5 * sy, 0.5 * sz

        triangles = []
        points = []

        # +X face
        origin = np.array([cx + hx, cy - hy, cz - hz], dtype=np.float32)
        u_dir = np.array([0.0, 2.0 * hy, 0.0], dtype=np.float32)
        v_dir = np.array([0.0, 0.0, 2.0 * hz], dtype=np.float32)
        tris, pts = self._face_grid(origin, u_dir, v_dir)
        triangles.extend(tris)
        points.extend(pts)

        # -X face
        origin = np.array([cx - hx, cy + hy, cz - hz], dtype=np.float32)
        u_dir = np.array([0.0, -2.0 * hy, 0.0], dtype=np.float32)
        v_dir = np.array([0.0, 0.0, 2.0 * hz], dtype=np.float32)
        tris, pts = self._face_grid(origin, u_dir, v_dir)
        triangles.extend(tris)
        points.extend(pts)

        # +Y face
        origin = np.array([cx + hx, cy + hy, cz - hz], dtype=np.float32)
        u_dir = np.array([-2.0 * hx, 0.0, 0.0], dtype=np.float32)
        v_dir = np.array([0.0, 0.0, 2.0 * hz], dtype=np.float32)
        tris, pts = self._face_grid(origin, u_dir, v_dir)
        triangles.extend(tris)
        points.extend(pts)

        # -Y face
        origin = np.array([cx - hx, cy - hy, cz - hz], dtype=np.float32)
        u_dir = np.array([2.0 * hx, 0.0, 0.0], dtype=np.float32)
        v_dir = np.array([0.0, 0.0, 2.0 * hz], dtype=np.float32)
        tris, pts = self._face_grid(origin, u_dir, v_dir)
        triangles.extend(tris)
        points.extend(pts)

        # -Z face (top in the current NED-like world convention used in this pipeline)
        origin = np.array([cx - hx, cy + hy, cz - hz], dtype=np.float32)
        u_dir = np.array([2.0 * hx, 0.0, 0.0], dtype=np.float32)
        v_dir = np.array([0.0, -2.0 * hy, 0.0], dtype=np.float32)
        tris, pts = self._face_grid(origin, u_dir, v_dir)
        triangles.extend(tris)
        points.extend(pts)

        # +Z face (bottom in the current NED-like world convention)
        if self.include_bottom:
            origin = np.array([cx - hx, cy - hy, cz + hz], dtype=np.float32)
            u_dir = np.array([2.0 * hx, 0.0, 0.0], dtype=np.float32)
            v_dir = np.array([0.0, 2.0 * hy, 0.0], dtype=np.float32)
            tris, pts = self._face_grid(origin, u_dir, v_dir)
            triangles.extend(tris)
            points.extend(pts)

        return np.asarray(triangles, dtype=np.float32), np.asarray(points, dtype=np.float32)
