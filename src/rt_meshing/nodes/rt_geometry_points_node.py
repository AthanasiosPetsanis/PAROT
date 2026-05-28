#!/usr/bin/env python3

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rospy
import rosgraph
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber as FilterSubscriber
from sensor_msgs.msg import Image
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker


_AXIS_FRD_TO_CV = np.array(
    [
        [0.0, 1.0, 0.0],  # x_cv <- y_frd
        [0.0, 0.0, 1.0],  # y_cv <- z_frd
        [1.0, 0.0, 0.0],  # z_cv <- x_frd
    ],
    dtype=np.float64,
)


_LOG_BLUE = "\033[94m"
_LOG_RESET = "\033[0m"


def _blue_log_text(value) -> str:
    return f"{_LOG_BLUE}{value}{_LOG_RESET}"


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


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _ang_diff_deg(a: float, b: float) -> float:
    return abs(math.degrees(_wrap_pi(a - b)))


def _yaw_from_cam_quat_xyzw(cam_quat_xyzw_world: np.ndarray) -> float:
    r_world_cam = _quat_xyzw_to_rot(
        float(cam_quat_xyzw_world[0]),
        float(cam_quat_xyzw_world[1]),
        float(cam_quat_xyzw_world[2]),
        float(cam_quat_xyzw_world[3]),
    )
    # Camera forward in world (FRD +X axis).
    fwd = r_world_cam[:, 0]
    return math.atan2(float(fwd[1]), float(fwd[0]))


def _atomic_savez(path: str, arrays: Dict[str, np.ndarray]) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez(tmp_path, **arrays)
    os.replace(tmp_path, path)


@dataclass
class PoseEntry:
    image_id: int
    img_name: str
    cam_pos_world: np.ndarray
    cam_quat_xyzw_world: np.ndarray
    yaw_world: float


@dataclass
class Keyframe:
    kf_id: int
    image_id: int
    img_name: str
    cam_pos_world: np.ndarray
    cam_quat_xyzw_world: np.ndarray
    yaw_world: float
    kpts_uv: np.ndarray
    desc: np.ndarray
    p_mat: np.ndarray
    r_wc_cv: np.ndarray
    t_wc_cv: np.ndarray
    image_h: int
    image_w: int
    kp_presence: np.ndarray
    feature_sets: Optional[Dict[str, Dict[str, object]]] = None
    image_gray: Optional[np.ndarray] = None
    roi_mask: Optional[np.ndarray] = None
    stream_backend: str = ""


@dataclass
class TrackRecord:
    track_id: int
    observations: List[Tuple[int, int]]  # (kf_id, kp_idx)


@dataclass
class DepthVoxelRecord:
    sum_xyz: np.ndarray
    sample_count: int
    view_count: int
    point_idx: int = -1


class RTSFMSparseNode:
    def __init__(self) -> None:
        rospy.init_node("rt_geometry_points_node")

        self.geometry_point_source = str(rospy.get_param("~rt_meshing/geometry_point_source", "sfm")).strip().lower()
        if self.geometry_point_source not in ("sfm", "depth"):
            rospy.logwarn("[WARN] Unsupported geometry_point_source '%s'; falling back to sfm.", self.geometry_point_source)
            self.geometry_point_source = "sfm"
        self.enabled = bool(rospy.get_param("~rt_meshing/sfm_enabled", False))
        if not self.enabled:
            rospy.logwarn("[WARN] Geometry point node disabled (set rt_meshing/sfm_enabled=true to run).")
            return

        img_base_dir = rospy.get_param("~file_paths/img_dir", "/home/thanos/Documents/IROS_2026/")
        poses_base_dir = rospy.get_param("~file_paths/poses_dir", "/home/thanos/Documents/IROS_2026/")
        mesh_dir = rospy.get_param(
            "~file_paths/mesh_path",
            "/home/thanos/Documents/IROS_2026/RT_meshing/Mesh_Space/RTmesh_test0/",
        )

        self.images_dir = rospy.get_param(
            "~rt_meshing/sfm_images_dir",
            os.path.join(img_base_dir, "images"),
        )
        self.poses_path = rospy.get_param(
            "~rt_meshing/sfm_poses_path",
            os.path.join(poses_base_dir, "poses", "poses.txt"),
        )
        sfm_output_path = rospy.get_param(
            "~rt_meshing/sfm_output_npz",
            os.path.join(mesh_dir, "sparse_latest.npz"),
        )
        depth_output_path = rospy.get_param(
            "~rt_meshing/depth_output_npz",
            os.path.join(mesh_dir, "depth_latest.npz"),
        )
        self.output_path = depth_output_path if self.geometry_point_source == "depth" else sfm_output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        self.frame_id = rospy.get_param("~rt_meshing/sfm_frame_id", "odom_local_ned")
        self.points_topic = rospy.get_param("~rt_meshing/sfm_points_topic", "controller/sfm_sparse_points")
        self.depth_raw_points_topic = rospy.get_param(
            "~rt_meshing/depth_raw_points_topic",
            "controller/depth_raw_points",
        )
        self.wait_for_controller = bool(rospy.get_param("~rt_meshing/wait_for_controller", True))
        self.controller_node_name = str(
            rospy.get_param("~rt_meshing/controller_node_name", "/controller_node")
        ).strip()
        if not self.controller_node_name:
            self.controller_node_name = "/controller_node"
        if not self.controller_node_name.startswith("/"):
            self.controller_node_name = "/" + self.controller_node_name
        self._ros_master = rosgraph.Master(rospy.get_name())
        self.bridge = CvBridge()
        self.depth_source_mode = str(rospy.get_param("~rt_meshing/depth_source_mode", "ros_topic")).strip().lower()
        if self.depth_source_mode not in ("ros_topic", "saved_frames"):
            rospy.logwarn("[WARN] Unsupported depth_source_mode '%s'; falling back to ros_topic.", self.depth_source_mode)
            self.depth_source_mode = "ros_topic"
        self.depth_topic = str(
            rospy.get_param("~rt_meshing/depth_topic", "/airsim_node/drone_1/front_center/DepthPerspective")
        )
        self.depth_camera_pose_topic = str(
            rospy.get_param("~rt_meshing/depth_camera_pose_topic", "controller/camera_pose")
        )
        self.depth_capture_topic = str(
            rospy.get_param("~rt_meshing/depth_capture_topic", "controller/depth_capture")
        )
        self.depth_capture_pose_topic = str(
            rospy.get_param("~rt_meshing/depth_capture_pose_topic", "controller/depth_capture_pose")
        )
        self.depth_capture_sync_slop_s = max(
            0.0, float(rospy.get_param("~rt_meshing/depth_capture_sync_slop_s", 0.05))
        )
        self.depth_dir = rospy.get_param(
            "~rt_meshing/depth_dir",
            os.path.join(img_base_dir, "depth"),
        )

        self.clip_to_building = bool(rospy.get_param("~rt_meshing/sfm_clip_to_building", True))
        self.building_padding = float(rospy.get_param("~rt_meshing/sfm_building_padding", 0.0))
        self.use_building_roi_mask = bool(rospy.get_param("~rt_meshing/sfm_use_building_roi_mask", False))
        self.roi_mask_margin_px = int(rospy.get_param("~rt_meshing/sfm_roi_mask_margin_px", 4))
        self.roi_mask_fallback_full = bool(rospy.get_param("~rt_meshing/sfm_roi_mask_fallback_full_image", True))
        build_cx, build_cy, build_cz, build_w, build_l, build_h = _get_building_params_from_profile()
        # Match controller mesh convention:
        # mesh center uses z = -building.center_z + rt_meshing.center_z_offset.
        mesh_center_z_offset = float(rospy.get_param("~rt_meshing/center_z_offset", 0.0))
        build_cz_mesh = -build_cz + mesh_center_z_offset
        self.build_min = np.array(
            [
                build_cx - 0.5 * build_w - self.building_padding,
                build_cy - 0.5 * build_l - self.building_padding,
                build_cz_mesh - 0.5 * build_h - self.building_padding,
            ],
            dtype=np.float64,
        )
        self.build_max = np.array(
            [
                build_cx + 0.5 * build_w + self.building_padding,
                build_cy + 0.5 * build_l + self.building_padding,
                build_cz_mesh + 0.5 * build_h + self.building_padding,
            ],
            dtype=np.float64,
        )
        hx = 0.5 * build_w
        hy = 0.5 * build_l
        hz = 0.5 * build_h
        self.build_corners_world = np.array(
            [
                [build_cx - hx, build_cy - hy, build_cz_mesh - hz],
                [build_cx - hx, build_cy - hy, build_cz_mesh + hz],
                [build_cx - hx, build_cy + hy, build_cz_mesh - hz],
                [build_cx - hx, build_cy + hy, build_cz_mesh + hz],
                [build_cx + hx, build_cy - hy, build_cz_mesh - hz],
                [build_cx + hx, build_cy - hy, build_cz_mesh + hz],
                [build_cx + hx, build_cy + hy, build_cz_mesh - hz],
                [build_cx + hx, build_cy + hy, build_cz_mesh + hz],
            ],
            dtype=np.float64,
        )

        self.poll_hz = float(rospy.get_param("~rt_meshing/sfm_poll_hz", 5.0))
        self.stride = max(1, int(rospy.get_param("~rt_meshing/sfm_stride", 5)))
        self.pose_image_id_offset = int(rospy.get_param("~rt_meshing/sfm_pose_image_id_offset", -1))
        self.reset_on_start = bool(rospy.get_param("~rt_meshing/sfm_reset_on_start", True))
        self.reset_on_pose_rewind = bool(rospy.get_param("~rt_meshing/sfm_reset_on_pose_rewind", True))
        self.skip_existing_poses_on_start = bool(
            rospy.get_param("~rt_meshing/sfm_skip_existing_poses_on_start", True)
        )

        self.window_size = max(2, int(rospy.get_param("~rt_meshing/sfm_window_size", 10)))
        self.match_neighbors = max(1, int(rospy.get_param("~rt_meshing/sfm_match_neighbors", 4)))

        self.max_features = max(64, int(rospy.get_param("~rt_meshing/sfm_max_features", 1500)))
        requested_feature_backend = str(rospy.get_param("~rt_meshing/sfm_feature_backend", "sift_orb"))
        self.feature_backend, self.feature_backend_components = self._parse_feature_backend(
            requested_feature_backend
        )
        if not self.feature_backend_components:
            rospy.logwarn(
                "[WARN] Unsupported SFM feature backend '%s'; falling back to sift_orb.",
                requested_feature_backend,
            )
            self.feature_backend = "sift_orb"
            self.feature_backend_components = ["orb", "sift"]
        if str(requested_feature_backend).lower().strip() != self.feature_backend:
            rospy.loginfo(
                "SFM feature backend normalized: requested=%s active=%s",
                str(requested_feature_backend),
                self.feature_backend,
            )
        self.sfm_gray_normalize = bool(rospy.get_param("~rt_meshing/sfm_gray_normalize", True))
        self.sfm_sharpen = bool(rospy.get_param("~rt_meshing/sfm_sharpen", True))
        self.sfm_sharpen_amount = float(rospy.get_param("~rt_meshing/sfm_sharpen_amount", 0.7))
        self.sfm_sharpen_sigma = float(rospy.get_param("~rt_meshing/sfm_sharpen_sigma", 1.0))
        self.klt_backend_enabled = "klt" in self.feature_backend_components
        self.alternating_feature_backend = self.feature_backend == "sift_klt_alt"
        self.alternating_feature_streams = ("sift", "klt")
        self.alternating_feature_idx = 0
        self._current_feature_stream = self.feature_backend
        self.klt_grid_step = max(2, int(rospy.get_param("~rt_meshing/sfm_klt_grid_step", 12)))
        self.klt_max_points = max(1, int(rospy.get_param("~rt_meshing/sfm_klt_max_points", 2000)))
        self.klt_border = max(0, int(rospy.get_param("~rt_meshing/sfm_klt_border", 8)))
        self.klt_win_size = max(3, int(rospy.get_param("~rt_meshing/sfm_klt_win_size", 21)))
        if (self.klt_win_size % 2) == 0:
            self.klt_win_size += 1
        self.klt_max_level = max(0, int(rospy.get_param("~rt_meshing/sfm_klt_max_level", 3)))
        self.klt_fb_px = max(0.0, float(rospy.get_param("~rt_meshing/sfm_klt_fb_px", 1.5)))
        self.klt_max_error = float(rospy.get_param("~rt_meshing/sfm_klt_max_error", 30.0))
        self.klt_max_tracks_per_pair = max(
            1,
            int(rospy.get_param("~rt_meshing/sfm_klt_max_tracks_per_pair", 1000)),
        )
        self.orb_patch_size = max(2, int(rospy.get_param("~rt_meshing/sfm_orb_patch_size", 31)))
        self.orb_edge_threshold = max(
            1,
            int(rospy.get_param("~rt_meshing/sfm_orb_edge_threshold", self.orb_patch_size)),
        )
        self.min_keypoints = max(8, int(rospy.get_param("~rt_meshing/sfm_min_keypoints", 80)))
        self.ratio_test = float(rospy.get_param("~rt_meshing/sfm_ratio_test", 0.75))
        self.ransac_px = float(rospy.get_param("~rt_meshing/sfm_ransac_px", 1.5))
        self.min_pair_matches = max(8, int(rospy.get_param("~rt_meshing/sfm_min_pair_matches", 30)))
        self.min_pair_inliers = max(8, int(rospy.get_param("~rt_meshing/sfm_min_pair_inliers", 15)))
        self.min_parallax_deg = float(rospy.get_param("~rt_meshing/sfm_min_parallax_deg", 1.0))
        self.max_reproj_px = float(rospy.get_param("~rt_meshing/sfm_max_reproj_px", 2.0))
        self.near_depth = float(rospy.get_param("~rt_meshing/sfm_near_depth", 0.05))
        self.min_track_views = max(2, int(rospy.get_param("~rt_meshing/sfm_min_track_views", 3)))
        self.track_confirm_reproj_px = float(
            rospy.get_param("~rt_meshing/sfm_track_confirm_reproj_px", self.max_reproj_px)
        )

        self.enable_motion_gating = bool(rospy.get_param("~rt_meshing/sfm_enable_motion_gating", False))
        self.min_translation_m = float(rospy.get_param("~rt_meshing/sfm_min_translation_m", 0.2))
        self.min_yaw_deg = float(rospy.get_param("~rt_meshing/sfm_min_yaw_deg", 5.0))
        self.enable_max_gap_gate = bool(rospy.get_param("~rt_meshing/sfm_enable_max_gap_gate", False))
        self.max_translation_m = float(rospy.get_param("~rt_meshing/sfm_max_translation_m", 5.0))
        self.max_yaw_deg = float(rospy.get_param("~rt_meshing/sfm_max_yaw_deg", 60.0))

        self.voxel_size = float(rospy.get_param("~rt_meshing/sfm_voxel_size", 0.10))
        self.max_points = max(1, int(rospy.get_param("~rt_meshing/sfm_max_points", 250000)))
        self.save_every_kf = max(1, int(rospy.get_param("~rt_meshing/sfm_save_every_keyframes", 1)))
        self.publish_every_kf = max(1, int(rospy.get_param("~rt_meshing/sfm_publish_every_keyframes", 1)))
        self.depth_frame_stride = max(1, int(rospy.get_param("~rt_meshing/depth_frame_stride", 1)))
        self.depth_pixel_stride = max(1, int(rospy.get_param("~rt_meshing/depth_pixel_stride", 1)))
        self.depth_near_m = max(0.0, float(rospy.get_param("~rt_meshing/depth_near_m", 0.02)))
        self.depth_voxel_size = max(1.0e-6, float(rospy.get_param("~rt_meshing/depth_voxel_size", self.voxel_size)))
        self.depth_min_view_observations = max(
            1, int(rospy.get_param("~rt_meshing/depth_min_view_observations", 2))
        )
        self.depth_mesh_patching = bool(rospy.get_param("~rt_meshing/depth_mesh_patching", False))
        self.depth_mesh_patch_topic = str(
            rospy.get_param("~rt_meshing/depth_mesh_patch_topic", "controller/depth_mesh_patch")
        )
        self.depth_mesh_patch_boundary_topic = str(
            rospy.get_param("~rt_meshing/depth_mesh_patch_boundary_topic", "controller/depth_mesh_patch_boundary")
        )
        self.depth_mesh_accumulated_patch_topic = str(
            rospy.get_param(
                "~rt_meshing/depth_mesh_accumulated_patch_topic",
                "controller/depth_mesh_accumulated_patch",
            )
        )
        self.depth_mesh_patch_max_triangles_viz = max(
            0, int(rospy.get_param("~rt_meshing/depth_mesh_patch_max_triangles_viz", 50000))
        )
        self.depth_mesh_accumulated_patch_max_triangles_viz = max(
            0, int(rospy.get_param("~rt_meshing/depth_mesh_accumulated_patch_max_triangles_viz", 50000))
        )
        self.depth_mesh_patch_edge_rule = str(
            rospy.get_param("~rt_meshing/depth_mesh_patch_edge_rule", "chebyshev")
        ).strip().lower()
        if self.depth_mesh_patch_edge_rule in ("long_shadows", "long_shadow", "old"):
            self.depth_mesh_patch_edge_rule = "grid"
        if self.depth_mesh_patch_edge_rule not in ("grid", "chebyshev"):
            rospy.logwarn(
                "[WARN] Unsupported depth_mesh_patch_edge_rule '%s'; falling back to chebyshev.",
                self.depth_mesh_patch_edge_rule,
            )
            self.depth_mesh_patch_edge_rule = "chebyshev"
        self.depth_mesh_patch_edge_chebyshev = max(
            1, int(rospy.get_param("~rt_meshing/depth_mesh_patch_edge_chebyshev", 1))
        )
        self.depth_fx = float(rospy.get_param("~rt_meshing/depth_fx", 554.2562866210938))
        self.depth_fy = float(rospy.get_param("~rt_meshing/depth_fy", 554.2562866210938))
        self.depth_cx = float(rospy.get_param("~rt_meshing/depth_cx", 320.0))
        self.depth_cy = float(rospy.get_param("~rt_meshing/depth_cy", 240.0))

        fx = float(rospy.get_param("~calibration/fx", 381.361145))
        fy = float(rospy.get_param("~calibration/fy", 381.361145))
        cx = float(rospy.get_param("~calibration/cx", 320.0))
        cy = float(rospy.get_param("~calibration/cy", 240.0))
        self.k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        self.feature_detectors: Dict[str, object] = {}
        self.feature_matchers: Dict[str, object] = {}
        self.feature_detector_order: List[str] = []
        if "orb" in self.feature_backend_components:
            self.feature_detector_order.append("orb")
            self.feature_detectors["orb"] = cv2.ORB_create(
                nfeatures=self.max_features,
                edgeThreshold=self.orb_edge_threshold,
                patchSize=self.orb_patch_size,
            )
            self.feature_matchers["orb"] = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if "sift" in self.feature_backend_components:
            self.feature_detector_order.append("sift")
            self.feature_detectors["sift"] = self._create_sift_detector(self.max_features)
            self.feature_matchers["sift"] = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.combined_feature_backend = len(self.feature_detector_order) > 1 or self.klt_backend_enabled
        self.orb = self.feature_detectors.get("orb", None)
        self.matcher = self.feature_matchers.get(
            "orb" if "orb" in self.feature_backend_components else "sift",
            None,
        )
        self.points_pub = rospy.Publisher(self.points_topic, PointCloud2, queue_size=1, latch=True)
        self.depth_raw_points_pub = rospy.Publisher(
            self.depth_raw_points_topic,
            PointCloud2,
            queue_size=1,
            latch=True,
        )
        self.depth_mesh_patch_pub = rospy.Publisher(
            self.depth_mesh_patch_topic,
            Marker,
            queue_size=1,
            latch=True,
        )
        self.depth_mesh_patch_boundary_pub = rospy.Publisher(
            self.depth_mesh_patch_boundary_topic,
            Marker,
            queue_size=1,
            latch=True,
        )
        self.depth_mesh_accumulated_patch_pub = rospy.Publisher(
            self.depth_mesh_accumulated_patch_topic,
            Marker,
            queue_size=1,
            latch=True,
        )

        self.next_pose_offset = 0
        self.accepted_keyframes = 0
        self.last_accepted_pose: Optional[PoseEntry] = None
        self.keyframes: List[Keyframe] = []
        self.next_keyframe_uid = 0
        self.next_track_id = 0
        self.next_point_id = 1
        self.tracks: Dict[int, TrackRecord] = {}
        self.obs_to_track: Dict[Tuple[int, int], int] = {}
        self.track_last_triangulated_kf: Dict[int, int] = {}
        self.points_xyz: List[np.ndarray] = []
        self.obs_count: List[int] = []
        self.point_ids: List[int] = []
        self.point_revs: List[int] = []
        self.point_track_ids: List[int] = []
        self.point_views_support: List[float] = []
        self.point_views_total: List[float] = []
        self.point_reproj_error_px: List[float] = []
        self.point_parallax_deg: List[float] = []
        self.track_to_point_idx: Dict[int, int] = {}
        self.voxel_to_idx: Dict[Tuple[int, int, int], int] = {}
        self.depth_voxels: Dict[Tuple[int, int, int], DepthVoxelRecord] = {}
        self.depth_mesh_accumulated_triangles: Dict[
            Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
            Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
        ] = {}
        self.depth_mesh_accumulated_triangle_cvi_raw: Dict[
            Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]],
            float,
        ] = {}
        self.depth_point_view_counts: List[int] = []
        self.depth_point_sample_counts: List[int] = []
        self.depth_input_frames = 0
        self.depth_processed_frames = 0
        self.latest_depth_patch: Optional[Dict[str, np.ndarray]] = None
        self.depth_capture_pose_sub = None
        self.depth_capture_image_sub = None
        self.depth_capture_sync = None
        self.depth_controller_gate_open = not self.wait_for_controller

        if self.reset_on_start:
            self._clear_sparse_state()
            self._save_sparse_npz()
        else:
            rospy.logwarn(
                "sfm_reset_on_start=false: keeping existing sparse output and reprocessing from first image."
            )
            self._load_existing_sparse_npz()
            self._publish_sparse_cloud()

        if self.geometry_point_source == "depth" and self.depth_source_mode == "ros_topic":
            self.depth_capture_image_sub = FilterSubscriber(
                self.depth_capture_topic,
                Image,
                queue_size=1,
            )
            self.depth_capture_pose_sub = FilterSubscriber(
                self.depth_capture_pose_topic,
                PoseStamped,
                queue_size=1,
            )
            self.depth_capture_sync = ApproximateTimeSynchronizer(
                [self.depth_capture_image_sub, self.depth_capture_pose_sub],
                queue_size=2,
                slop=float(self.depth_capture_sync_slop_s),
            )
            self.depth_capture_sync.registerCallback(self._depth_capture_callback)

        # Prevent processing stale pose/image history from previous runs when launched before controller.
        # We skip currently-existing entries and only process new entries appended after startup.
        if self.reset_on_start and self.wait_for_controller and self.skip_existing_poses_on_start:
            startup_entries = self._load_pose_entries()
            self.next_pose_offset = len(startup_entries)
            if self.next_pose_offset > 0:
                rospy.logwarn(
                    "SFM startup: skipping %d pre-existing pose entries to avoid stale-run processing.",
                    self.next_pose_offset,
                )

        if self.geometry_point_source == "depth":
            rospy.loginfo(
                "RT depth geometry mode started. source_mode=%s output=%s capture_topic=%s capture_pose_topic=%s depth_dir=%s voxel=%.3f min_views=%d pixel_stride=%d mesh_patching=%s edge_rule=%s",
                self.depth_source_mode,
                self.output_path,
                self.depth_capture_topic,
                self.depth_capture_pose_topic,
                self.depth_dir,
                self.depth_voxel_size,
                self.depth_min_view_observations,
                self.depth_pixel_stride,
                str(self.depth_mesh_patching),
                self.depth_mesh_patch_edge_rule,
            )
        else:
            rospy.loginfo(
                "RT SfM sparse node started. poses=%s images=%s stride=%d pose_image_offset=%d window=%d neighbors=%d backend=%s gray_normalize=%s sharpen=%s topic=%s",
                self.poses_path,
                self.images_dir,
                self.stride,
                self.pose_image_id_offset,
                self.window_size,
                self.match_neighbors,
                self.feature_backend,
                str(self.sfm_gray_normalize),
                str(self.sfm_sharpen),
                self.points_topic,
            )
        if self.wait_for_controller:
            rospy.loginfo("SFM gate enabled: waiting for controller node %s", self.controller_node_name)
        if self.clip_to_building:
            rospy.loginfo(
                "SFM building clipping enabled. min=%s max=%s",
                self.build_min.tolist(),
                self.build_max.tolist(),
            )
        if self.use_building_roi_mask:
            rospy.loginfo(
                "SFM ROI mask enabled using projected 3D box rectangle (margin_px=%d, fallback_full=%s).",
                self.roi_mask_margin_px,
                str(self.roi_mask_fallback_full),
            )
        if self.min_track_views > 2:
            rospy.loginfo(
                "SFM multi-view confirmation enabled: min_track_views=%d, reproj_px=%.2f",
                self.min_track_views,
                self.track_confirm_reproj_px,
            )

    @staticmethod
    def _parse_feature_backend(name: str) -> Tuple[str, List[str]]:
        value = str(name).lower().strip().replace("-", "_")
        aliases = {
            "shift_orb": "sift_orb",
            "orb_sift": "sift_orb",
            "sift_klt_alternating": "sift_klt_alt",
            "klt_sift_alt": "sift_klt_alt",
            "klt_sift_alternating": "sift_klt_alt",
            "orb_sift_klt": "sift_orb_klt",
            "shift_orb_klt": "sift_orb_klt",
            "klt_orb_sift": "sift_orb_klt",
            "klt_sift_orb": "sift_orb_klt",
        }
        value = aliases.get(value, value)
        if value == "sift_klt_alt":
            return "sift_klt_alt", ["sift", "klt"]
        tokens = [tok for tok in value.split("_") if tok]
        valid = {"orb", "sift", "klt"}
        if not tokens or any(tok not in valid for tok in tokens):
            return value, []

        requested = set(tokens)
        components: List[str] = []
        # Keep detector order stable so existing ORB/SIFT offsets and behavior remain predictable.
        if "orb" in requested:
            components.append("orb")
        if "sift" in requested:
            components.append("sift")
        if "klt" in requested:
            components.append("klt")

        if components == ["orb"]:
            return "orb", components
        if components == ["sift"]:
            return "sift", components
        if components == ["klt"]:
            return "klt", components
        if components == ["orb", "sift"]:
            return "sift_orb", components
        if components == ["orb", "klt"]:
            return "orb_klt", components
        if components == ["sift", "klt"]:
            return "sift_klt", components
        if components == ["orb", "sift", "klt"]:
            return "sift_orb_klt", components
        return value, components

    @staticmethod
    def _create_sift_detector(max_features: int):
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(nfeatures=int(max_features))
        xfeatures = getattr(cv2, "xfeatures2d", None)
        if xfeatures is not None and hasattr(xfeatures, "SIFT_create"):
            return xfeatures.SIFT_create(nfeatures=int(max_features))
        raise RuntimeError(
            "OpenCV SIFT is not available. Install an OpenCV build with SIFT support "
            "or set rt_meshing/sfm_feature_backend to 'orb'."
        )

    @staticmethod
    def _normalize_gray_image(gray: np.ndarray) -> np.ndarray:
        arr = np.asarray(gray, dtype=np.uint8)
        lo = int(np.min(arr))
        hi = int(np.max(arr))
        if hi <= lo:
            return arr.copy()
        out = ((arr.astype(np.float32) - float(lo)) * (255.0 / float(hi - lo))).clip(0, 255)
        return out.astype(np.uint8)

    @staticmethod
    def _sharpen_gray_image(gray: np.ndarray, amount: float, sigma: float) -> np.ndarray:
        arr = np.asarray(gray, dtype=np.uint8)
        sigma = max(1.0e-6, float(sigma))
        amount = max(0.0, float(amount))
        if amount <= 0.0:
            return arr.copy()
        blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma, sigmaY=sigma)
        out = cv2.addWeighted(arr, 1.0 + amount, blurred, -amount, 0.0)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _preprocess_feature_image(self, gray: np.ndarray) -> np.ndarray:
        img = np.asarray(gray, dtype=np.uint8).copy()
        if self.sfm_gray_normalize:
            img = self._normalize_gray_image(img)
        if self.sfm_sharpen:
            img = self._sharpen_gray_image(
                img,
                amount=self.sfm_sharpen_amount,
                sigma=self.sfm_sharpen_sigma,
            )
        return img

    @staticmethod
    def _keyframe_feature_sets(kf: Keyframe) -> Dict[str, Dict[str, object]]:
        if isinstance(kf.feature_sets, dict) and kf.feature_sets:
            return kf.feature_sets
        feature_sets = getattr(kf, "eval_feature_sets", None)
        if isinstance(feature_sets, dict) and feature_sets:
            return feature_sets
        return {
            "default": {
                "kpts_uv": kf.kpts_uv,
                "desc": kf.desc,
                "offset": 0,
                "count": int(0 if kf.kpts_uv is None else kf.kpts_uv.shape[0]),
            }
        }

    def _next_alternating_feature_stream(self) -> str:
        streams = tuple(getattr(self, "alternating_feature_streams", ("sift", "klt")))
        if not streams:
            return "sift"
        idx = int(getattr(self, "alternating_feature_idx", 0))
        return str(streams[idx % len(streams)])

    def _mark_alternating_feature_attempt_consumed(self) -> None:
        if bool(getattr(self, "alternating_feature_backend", False)):
            self.alternating_feature_idx = int(getattr(self, "alternating_feature_idx", 0)) + 1

    def _active_detector_order(self) -> List[str]:
        if bool(getattr(self, "alternating_feature_backend", False)):
            stream = str(getattr(self, "_current_feature_stream", "sift")).lower()
            return [stream] if stream in self.feature_detector_order else []
        return list(self.feature_detector_order)

    def _current_stream_uses_klt(self) -> bool:
        if bool(getattr(self, "alternating_feature_backend", False)):
            return str(getattr(self, "_current_feature_stream", "")).lower() == "klt"
        return bool(self.klt_backend_enabled)

    def _extract_feature_sets_for_image(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> Tuple[Dict[str, Dict[str, object]], np.ndarray, Dict[str, int]]:
        feature_sets: Dict[str, Dict[str, object]] = {}
        combined_kpts: List[np.ndarray] = []
        per_backend_counts: Dict[str, int] = {}
        offset = 0

        for feature_name in self._active_detector_order():
            detector = self.feature_detectors.get(feature_name, None)
            if detector is None:
                continue
            kps, desc = detector.detectAndCompute(img, mask)
            count = int(0 if not kps or desc is None else len(kps))
            per_backend_counts[feature_name] = count
            if count <= 0:
                feature_sets[feature_name] = {
                    "kpts_uv": np.zeros((0, 2), dtype=np.float32),
                    "desc": desc,
                    "offset": int(offset),
                    "count": 0,
                }
                continue

            kpts_uv = np.asarray([kp.pt for kp in kps], dtype=np.float32)
            feature_sets[feature_name] = {
                "kpts_uv": kpts_uv,
                "desc": desc,
                "offset": int(offset),
                "count": int(count),
            }
            combined_kpts.append(kpts_uv)
            offset += count

        if combined_kpts:
            kpts_uv_all = np.vstack(combined_kpts).astype(np.float32, copy=False)
        else:
            kpts_uv_all = np.zeros((0, 2), dtype=np.float32)
        return feature_sets, kpts_uv_all, per_backend_counts

    def _match_feature_set_ratio(
        self,
        matcher,
        ref_data: Dict[str, object],
        cur_data: Dict[str, object],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        ref_desc = ref_data.get("desc", None)
        cur_desc = cur_data.get("desc", None)
        ref_kpts = np.asarray(ref_data.get("kpts_uv", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        cur_kpts = np.asarray(cur_data.get("kpts_uv", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if matcher is None or ref_desc is None or cur_desc is None or len(ref_desc) < 2 or len(cur_desc) < 2:
            empty_idx = np.zeros((0,), dtype=np.int32)
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_idx, empty_idx, empty_pts, empty_pts, 0

        knn = matcher.knnMatch(cur_desc, ref_desc, k=2)
        good = []
        for m_n in knn:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        if not good:
            empty_idx = np.zeros((0,), dtype=np.int32)
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_idx, empty_idx, empty_pts, empty_pts, 0

        cur_local_idx = np.asarray([m.queryIdx for m in good], dtype=np.int32)
        ref_local_idx = np.asarray([m.trainIdx for m in good], dtype=np.int32)
        cur_global_idx = cur_local_idx + int(cur_data.get("offset", 0) or 0)
        ref_global_idx = ref_local_idx + int(ref_data.get("offset", 0) or 0)
        return (
            ref_global_idx.astype(np.int32, copy=False),
            cur_global_idx.astype(np.int32, copy=False),
            ref_kpts[ref_local_idx],
            cur_kpts[cur_local_idx],
            int(len(good)),
        )

    def _match_pair_ratio_candidates(
        self,
        ref: Keyframe,
        cur: Keyframe,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        ref_sets = self._keyframe_feature_sets(ref)
        cur_sets = self._keyframe_feature_sets(cur)
        all_ref_idx = []
        all_cur_idx = []
        all_pts_ref = []
        all_pts_cur = []
        total_matches = 0

        feature_names = self.feature_detector_order if self.feature_detector_order else ["default"]
        for feature_name in feature_names:
            ref_data = ref_sets.get(feature_name, None)
            cur_data = cur_sets.get(feature_name, None)
            if ref_data is None or cur_data is None:
                continue
            matcher = self.feature_matchers.get(feature_name, self.matcher)
            ref_idx, cur_idx, pts_ref, pts_cur, num_matches = self._match_feature_set_ratio(
                matcher,
                ref_data,
                cur_data,
            )
            total_matches += int(num_matches)
            if ref_idx.size <= 0:
                continue
            all_ref_idx.append(ref_idx)
            all_cur_idx.append(cur_idx)
            all_pts_ref.append(pts_ref)
            all_pts_cur.append(pts_cur)

        if not all_ref_idx:
            empty_idx = np.zeros((0,), dtype=np.int32)
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_idx, empty_idx, empty_pts, empty_pts, int(total_matches)

        return (
            np.concatenate(all_ref_idx).astype(np.int32, copy=False),
            np.concatenate(all_cur_idx).astype(np.int32, copy=False),
            np.vstack(all_pts_ref).astype(np.float32, copy=False),
            np.vstack(all_pts_cur).astype(np.float32, copy=False),
            int(total_matches),
        )

    def _make_klt_grid_points(
        self,
        mask: Optional[np.ndarray],
        image_h: int,
        image_w: int,
    ) -> np.ndarray:
        h = int(image_h)
        w = int(image_w)
        if h <= 0 or w <= 0:
            return np.zeros((0, 2), dtype=np.float32)

        step = max(2, int(self.klt_grid_step))
        border = max(0, int(self.klt_border))
        x0 = min(max(border, 0), max(0, w - 1))
        y0 = min(max(border, 0), max(0, h - 1))
        x1 = max(x0 + 1, w - border)
        y1 = max(y0 + 1, h - border)
        xs = np.arange(x0, x1, step, dtype=np.float32)
        ys = np.arange(y0, y1, step, dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        grid_x, grid_y = np.meshgrid(xs, ys)
        pts = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)]).astype(np.float32)
        if mask is not None:
            m = np.asarray(mask, dtype=np.uint8)
            uv = np.rint(pts).astype(np.int32)
            valid = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < h)
                & (m[uv[:, 1], uv[:, 0]] > 0)
            )
            pts = pts[valid]

        if pts.shape[0] > int(self.klt_max_points):
            sample = np.linspace(0, pts.shape[0] - 1, int(self.klt_max_points)).astype(np.int32)
            pts = pts[sample]
        return pts.astype(np.float32, copy=False)

    def _append_or_reuse_keypoints(
        self,
        kf: Keyframe,
        pts_uv: np.ndarray,
    ) -> np.ndarray:
        pts = np.asarray(pts_uv, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] == 0:
            return np.zeros((0,), dtype=np.int32)

        existing = np.asarray(kf.kpts_uv, dtype=np.float32).reshape(-1, 2)
        reuse_radius = max(0.0, min(2.0, float(self.track_confirm_reproj_px)))
        reuse_radius_sq = reuse_radius * reuse_radius
        out_idx: List[int] = []
        append_pts: List[np.ndarray] = []

        for p in pts:
            reuse_idx: Optional[int] = None
            if reuse_radius > 0.0 and existing.shape[0] > 0:
                d2 = np.sum((existing - p.reshape(1, 2)) ** 2, axis=1)
                best = int(np.argmin(d2))
                if float(d2[best]) <= reuse_radius_sq:
                    reuse_idx = best
            if reuse_idx is not None:
                out_idx.append(int(reuse_idx))
                continue

            new_idx = int(existing.shape[0])
            out_idx.append(new_idx)
            append_pts.append(p.astype(np.float32, copy=True))
            existing = np.vstack([existing, p.reshape(1, 2).astype(np.float32)])

        if append_pts:
            kf.kpts_uv = existing.astype(np.float32, copy=False)
            kf.kp_presence = self._build_kp_presence_mask(
                kf.kpts_uv,
                image_h=int(kf.image_h),
                image_w=int(kf.image_w),
            )
            if isinstance(kf.feature_sets, dict) and "klt" in kf.feature_sets:
                klt_data = kf.feature_sets["klt"]
                offset = int(klt_data.get("offset", 0) or 0)
                klt_data["kpts_uv"] = kf.kpts_uv[offset:].astype(np.float32, copy=False)
                klt_data["count"] = int(klt_data["kpts_uv"].shape[0])

        return np.asarray(out_idx, dtype=np.int32)

    def _match_pair_klt_candidates(
        self,
        ref: Keyframe,
        cur: Keyframe,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        empty_idx = np.zeros((0,), dtype=np.int32)
        empty_pts = np.zeros((0, 2), dtype=np.float32)
        if not self.klt_backend_enabled:
            return empty_idx, empty_idx, empty_pts, empty_pts, 0
        if ref.image_gray is None or cur.image_gray is None:
            return empty_idx, empty_idx, empty_pts, empty_pts, 0

        pts_ref_all = np.asarray(ref.kpts_uv, dtype=np.float32).reshape(-1, 2)
        if pts_ref_all.shape[0] <= 0:
            return empty_idx, empty_idx, empty_pts, empty_pts, 0

        ref_img = np.asarray(ref.image_gray, dtype=np.uint8)
        cur_img = np.asarray(cur.image_gray, dtype=np.uint8)
        pts0 = pts_ref_all.reshape(-1, 1, 2)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        pts1, st, err = cv2.calcOpticalFlowPyrLK(
            ref_img,
            cur_img,
            pts0,
            None,
            winSize=(int(self.klt_win_size), int(self.klt_win_size)),
            maxLevel=int(self.klt_max_level),
            criteria=criteria,
        )
        if pts1 is None or st is None:
            return empty_idx, empty_idx, empty_pts, empty_pts, 0

        keep = st.reshape(-1).astype(bool)
        pts1_flat = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)

        if self.klt_fb_px > 0.0 and np.any(keep):
            pts0_back, st_back, _err_back = cv2.calcOpticalFlowPyrLK(
                cur_img,
                ref_img,
                pts1,
                None,
                winSize=(int(self.klt_win_size), int(self.klt_win_size)),
                maxLevel=int(self.klt_max_level),
                criteria=criteria,
            )
            if pts0_back is None or st_back is None:
                keep[:] = False
            else:
                back_flat = np.asarray(pts0_back, dtype=np.float32).reshape(-1, 2)
                fb_err = np.linalg.norm(back_flat - pts_ref_all, axis=1)
                keep &= st_back.reshape(-1).astype(bool) & np.isfinite(fb_err) & (fb_err <= self.klt_fb_px)

        h = int(cur.image_h)
        w = int(cur.image_w)
        in_bounds = (
            np.isfinite(pts1_flat[:, 0])
            & np.isfinite(pts1_flat[:, 1])
            & (pts1_flat[:, 0] >= 0.0)
            & (pts1_flat[:, 0] < float(w))
            & (pts1_flat[:, 1] >= 0.0)
            & (pts1_flat[:, 1] < float(h))
        )
        keep &= in_bounds

        if cur.roi_mask is not None and np.any(keep):
            m = np.asarray(cur.roi_mask, dtype=np.uint8)
            uv = np.rint(pts1_flat).astype(np.int32)
            mask_ok = np.zeros((pts1_flat.shape[0],), dtype=bool)
            valid_uv = (
                (uv[:, 0] >= 0)
                & (uv[:, 0] < w)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < h)
            )
            mask_ok[valid_uv] = m[uv[valid_uv, 1], uv[valid_uv, 0]] > 0
            keep &= mask_ok

        if err is not None and self.klt_max_error > 0.0 and np.any(keep):
            err_flat = np.asarray(err, dtype=np.float32).reshape(-1)
            keep &= np.isfinite(err_flat) & (err_flat <= self.klt_max_error)

        ref_idx = np.nonzero(keep)[0].astype(np.int32)
        if ref_idx.size <= 0:
            return empty_idx, empty_idx, empty_pts, empty_pts, 0
        pts_ref = pts_ref_all[ref_idx].astype(np.float32, copy=False)
        pts_cur = pts1_flat[ref_idx].astype(np.float32, copy=False)
        if ref_idx.size > int(self.klt_max_tracks_per_pair):
            sample = np.linspace(0, ref_idx.size - 1, int(self.klt_max_tracks_per_pair)).astype(np.int32)
            ref_idx = ref_idx[sample]
            pts_ref = pts_ref[sample]
            pts_cur = pts_cur[sample]

        cur_placeholder_idx = np.full((int(ref_idx.size),), -1, dtype=np.int32)
        return ref_idx, cur_placeholder_idx, pts_ref, pts_cur, int(ref_idx.size)

    def _match_pair_candidates(
        self,
        ref: Keyframe,
        cur: Keyframe,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
        all_ref_idx = []
        all_cur_idx = []
        all_pts_ref = []
        all_pts_cur = []
        all_is_klt = []
        total_matches = 0

        if bool(getattr(self, "alternating_feature_backend", False)):
            if str(getattr(ref, "stream_backend", "")).lower() != str(getattr(cur, "stream_backend", "")).lower():
                empty_idx = np.zeros((0,), dtype=np.int32)
                empty_pts = np.zeros((0, 2), dtype=np.float32)
                return empty_idx, empty_idx, empty_pts, empty_pts, 0, np.zeros((0,), dtype=bool)
            use_klt = str(getattr(cur, "stream_backend", "")).lower() == "klt"
            if use_klt:
                ref_idx, cur_idx, pts_ref, pts_cur, count = self._match_pair_klt_candidates(ref, cur)
                total_matches += int(count)
                if ref_idx.size > 0:
                    all_ref_idx.append(ref_idx)
                    all_cur_idx.append(cur_idx)
                    all_pts_ref.append(pts_ref)
                    all_pts_cur.append(pts_cur)
                    all_is_klt.append(np.ones((int(ref_idx.size),), dtype=bool))
            else:
                ref_idx, cur_idx, pts_ref, pts_cur, count = self._match_pair_ratio_candidates(ref, cur)
                total_matches += int(count)
                if ref_idx.size > 0:
                    all_ref_idx.append(ref_idx)
                    all_cur_idx.append(cur_idx)
                    all_pts_ref.append(pts_ref)
                    all_pts_cur.append(pts_cur)
                    all_is_klt.append(np.zeros((int(ref_idx.size),), dtype=bool))
        else:
            ref_idx, cur_idx, pts_ref, pts_cur, count = self._match_pair_ratio_candidates(ref, cur)
            total_matches += int(count)
            if ref_idx.size > 0:
                all_ref_idx.append(ref_idx)
                all_cur_idx.append(cur_idx)
                all_pts_ref.append(pts_ref)
                all_pts_cur.append(pts_cur)
                all_is_klt.append(np.zeros((int(ref_idx.size),), dtype=bool))

            ref_idx, cur_idx, pts_ref, pts_cur, count = self._match_pair_klt_candidates(ref, cur)
            total_matches += int(count)
            if ref_idx.size > 0:
                all_ref_idx.append(ref_idx)
                all_cur_idx.append(cur_idx)
                all_pts_ref.append(pts_ref)
                all_pts_cur.append(pts_cur)
                all_is_klt.append(np.ones((int(ref_idx.size),), dtype=bool))

        if not all_ref_idx:
            empty_idx = np.zeros((0,), dtype=np.int32)
            empty_pts = np.zeros((0, 2), dtype=np.float32)
            return empty_idx, empty_idx, empty_pts, empty_pts, int(total_matches), np.zeros((0,), dtype=bool)

        return (
            np.concatenate(all_ref_idx).astype(np.int32, copy=False),
            np.concatenate(all_cur_idx).astype(np.int32, copy=False),
            np.vstack(all_pts_ref).astype(np.float32, copy=False),
            np.vstack(all_pts_cur).astype(np.float32, copy=False),
            int(total_matches),
            np.concatenate(all_is_klt).astype(bool, copy=False),
        )

    def _inside_building(self, p: np.ndarray) -> bool:
        if not self.clip_to_building:
            return True
        return bool(np.all(p >= self.build_min) and np.all(p <= self.build_max))

    @staticmethod
    def _pose_msg_to_camera_state(msg: PoseStamped) -> Tuple[np.ndarray, np.ndarray]:
        p = msg.pose.position
        q = msg.pose.orientation
        return (
            np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64),
            np.array([float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64),
        )

    def _depth_capture_callback(self, msg: Image, camera_pose: PoseStamped) -> None:
        if self.geometry_point_source != "depth" or self.depth_source_mode != "ros_topic":
            return
        if not bool(self.depth_controller_gate_open):
            return
        try:
            depth = np.asarray(self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), dtype=np.float32)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "[WARN] Failed to decode depth image: %s", str(exc))
            return
        cam_pos, cam_quat = self._pose_msg_to_camera_state(camera_pose)
        label = str(getattr(msg.header, "frame_id", "") or "")
        if not label.startswith("image_"):
            image_id = int(camera_pose.header.seq or msg.header.seq or 0)
            label = "image_%06d" % image_id if image_id > 0 else "capture"
        self._process_depth_frame(
            depth,
            cam_pos_world=cam_pos,
            cam_quat_xyzw_world=cam_quat,
            label=label,
            apply_frame_stride=False,
        )

    def _depth_samples_from_frame(
        self,
        depth: np.ndarray,
        cam_pos_world: np.ndarray,
        cam_quat_xyzw_world: np.ndarray,
    ) -> Dict[str, object]:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.size == 0:
            return {
                "points_world": np.zeros((0, 3), dtype=np.float64),
                "grid_rows": np.zeros((0,), dtype=np.int32),
                "grid_cols": np.zeros((0,), dtype=np.int32),
                "grid_shape": (0, 0),
            }
        stride = int(self.depth_pixel_stride)
        sampled = depth[::stride, ::stride]
        grid_shape = tuple(int(x) for x in sampled.shape[:2])
        valid = np.isfinite(sampled) & (sampled > float(self.depth_near_m))
        if not np.any(valid):
            return {
                "points_world": np.zeros((0, 3), dtype=np.float64),
                "grid_rows": np.zeros((0,), dtype=np.int32),
                "grid_cols": np.zeros((0,), dtype=np.int32),
                "grid_shape": grid_shape,
            }

        vv, uu = np.nonzero(valid)
        u = (uu.astype(np.float64) * float(stride))
        v = (vv.astype(np.float64) * float(stride))
        ray_cv = np.column_stack(
            [
                (u - float(self.depth_cx)) / max(float(self.depth_fx), 1.0e-9),
                (v - float(self.depth_cy)) / max(float(self.depth_fy), 1.0e-9),
                np.ones((u.shape[0],), dtype=np.float64),
            ]
        )
        ray_norm = np.linalg.norm(ray_cv, axis=1)
        good_ray = np.isfinite(ray_norm) & (ray_norm > 1.0e-12)
        if not np.any(good_ray):
            return {
                "points_world": np.zeros((0, 3), dtype=np.float64),
                "grid_rows": np.zeros((0,), dtype=np.int32),
                "grid_cols": np.zeros((0,), dtype=np.int32),
                "grid_shape": grid_shape,
            }
        vv = vv[good_ray]
        uu = uu[good_ray]
        ray_cv = ray_cv[good_ray] / ray_norm[good_ray, None]
        depth_m = sampled[vv, uu].astype(np.float64)
        points_cv = ray_cv * depth_m[:, None]
        # AirSim camera pose here follows the FRD camera convention used by saved poses:
        # +X forward, +Y right, +Z down. Image/depth rays are CV +Z forward.
        points_frd = points_cv @ _AXIS_FRD_TO_CV
        r_world_cam = _quat_xyzw_to_rot(
            float(cam_quat_xyzw_world[0]),
            float(cam_quat_xyzw_world[1]),
            float(cam_quat_xyzw_world[2]),
            float(cam_quat_xyzw_world[3]),
        )
        points_world = points_frd @ r_world_cam.T + np.asarray(cam_pos_world, dtype=np.float64).reshape(1, 3)
        finite = np.all(np.isfinite(points_world), axis=1)
        if not np.any(finite):
            return {
                "points_world": np.zeros((0, 3), dtype=np.float64),
                "grid_rows": np.zeros((0,), dtype=np.int32),
                "grid_cols": np.zeros((0,), dtype=np.int32),
                "grid_shape": grid_shape,
            }
        points_world = points_world[finite]
        vv = vv[finite]
        uu = uu[finite]
        if self.clip_to_building:
            inside = np.all((points_world >= self.build_min[None, :]) & (points_world <= self.build_max[None, :]), axis=1)
            points_world = points_world[inside]
            vv = vv[inside]
            uu = uu[inside]
        return {
            "points_world": points_world.astype(np.float64, copy=False),
            "grid_rows": vv.astype(np.int32, copy=False),
            "grid_cols": uu.astype(np.int32, copy=False),
            "grid_shape": grid_shape,
        }

    def _depth_points_from_frame(
        self,
        depth: np.ndarray,
        cam_pos_world: np.ndarray,
        cam_quat_xyzw_world: np.ndarray,
    ) -> np.ndarray:
        samples = self._depth_samples_from_frame(depth, cam_pos_world, cam_quat_xyzw_world)
        return np.asarray(samples["points_world"], dtype=np.float64).reshape(-1, 3)

    def _fuse_depth_points(self, points: np.ndarray, return_point_keys: bool = False):
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] == 0:
            empty_keys = np.zeros((0, 3), dtype=np.int64)
            return (0, empty_keys) if bool(return_point_keys) else 0
        keys = np.floor(pts / float(self.depth_voxel_size)).astype(np.int64)
        unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, pts)

        usable_updates = 0
        for key_arr, count, sum_xyz in zip(unique_keys, counts, sums):
            key = tuple(int(x) for x in key_arr.tolist())
            rec = self.depth_voxels.get(key)
            if rec is None:
                rec = DepthVoxelRecord(
                    sum_xyz=np.asarray(sum_xyz, dtype=np.float64).copy(),
                    sample_count=int(count),
                    view_count=1,
                )
                self.depth_voxels[key] = rec
            else:
                rec.sum_xyz += np.asarray(sum_xyz, dtype=np.float64)
                rec.sample_count += int(count)
                rec.view_count += 1

            if rec.view_count < int(self.depth_min_view_observations):
                continue
            centroid = rec.sum_xyz / float(max(1, rec.sample_count))
            if rec.point_idx < 0:
                rec.point_idx = len(self.points_xyz)
                self.points_xyz.append(centroid.copy())
                self.obs_count.append(int(rec.sample_count))
                self.point_ids.append(int(self.next_point_id))
                self.next_point_id += 1
                self.point_revs.append(1)
                self.point_track_ids.append(-1)
                self.point_views_support.append(float(rec.view_count))
                self.point_views_total.append(float(rec.view_count))
                self.point_reproj_error_px.append(0.0)
                self.point_parallax_deg.append(float(max(self.min_parallax_deg, 1.0)))
                self.depth_point_view_counts.append(int(rec.view_count))
                self.depth_point_sample_counts.append(int(rec.sample_count))
            else:
                idx = int(rec.point_idx)
                self.points_xyz[idx] = centroid.copy()
                self.obs_count[idx] = int(rec.sample_count)
                self.point_revs[idx] = max(1, int(self.point_revs[idx]) + 1)
                self.point_views_support[idx] = float(rec.view_count)
                self.point_views_total[idx] = float(rec.view_count)
                self.depth_point_view_counts[idx] = int(rec.view_count)
                self.depth_point_sample_counts[idx] = int(rec.sample_count)
            usable_updates += 1
        return (int(usable_updates), keys) if bool(return_point_keys) else int(usable_updates)

    @staticmethod
    def _point_msg(p: np.ndarray) -> Point:
        arr = np.asarray(p, dtype=np.float64).reshape(3)
        return Point(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    @staticmethod
    def _voxel_key_tuple(key: np.ndarray) -> Tuple[int, int, int]:
        arr = np.asarray(key, dtype=np.int64).reshape(3)
        return (int(arr[0]), int(arr[1]), int(arr[2]))

    @staticmethod
    def _canonical_triangle_key(
        a: Tuple[int, int, int],
        b: Tuple[int, int, int],
        c: Tuple[int, int, int],
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
        return tuple(sorted((a, b, c)))

    def _centroid_for_depth_voxel_key(self, key: Tuple[int, int, int]) -> Optional[np.ndarray]:
        rec = self.depth_voxels.get(tuple(int(x) for x in key))
        if rec is None or int(rec.sample_count) <= 0:
            return None
        return np.asarray(rec.sum_xyz, dtype=np.float64).reshape(3) / float(max(1, rec.sample_count))

    @staticmethod
    def _dark_to_green_color(t: float, alpha: float = 0.36) -> ColorRGBA:
        t = float(np.clip(t, 0.0, 1.0))
        return ColorRGBA(
            r=0.0,
            g=float(0.04 + 0.96 * t),
            b=float(0.015 * (1.0 - t)),
            a=float(alpha),
        )

    def _clear_depth_mesh_patch_markers(self) -> None:
        for pub, ns, mid in (
            (self.depth_mesh_patch_pub, "depth_mesh_patch", 0),
            (self.depth_mesh_patch_boundary_pub, "depth_mesh_patch_boundary", 0),
        ):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.frame_id
            marker.ns = ns
            marker.id = int(mid)
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0
            pub.publish(marker)

    def _publish_depth_mesh_accumulated_patch(self, label: str) -> None:
        if self.geometry_point_source != "depth" or not bool(self.depth_mesh_patching):
            return
        if not self.depth_mesh_accumulated_triangles:
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.frame_id
            marker.ns = "depth_mesh_accumulated_patch"
            marker.id = 0
            marker.action = Marker.DELETE
            marker.pose.orientation.w = 1.0
            self.depth_mesh_accumulated_patch_pub.publish(marker)
            return

        tris = list(self.depth_mesh_accumulated_triangles.values())
        if (
            self.depth_mesh_accumulated_patch_max_triangles_viz > 0
            and len(tris) > self.depth_mesh_accumulated_patch_max_triangles_viz
        ):
            sample_idx = np.linspace(
                0,
                len(tris) - 1,
                int(self.depth_mesh_accumulated_patch_max_triangles_viz),
            ).astype(np.int64)
            tris = [tris[int(i)] for i in sample_idx]

        cvi_values = np.asarray(
            list(self.depth_mesh_accumulated_triangle_cvi_raw.values()),
            dtype=np.float64,
        )
        cvi_min = float(np.min(cvi_values)) if cvi_values.size > 0 else 0.0
        cvi_max = float(np.max(cvi_values)) if cvi_values.size > 0 else 0.0
        cvi_span = cvi_max - cvi_min
        if (not np.isfinite(cvi_min)) or (not np.isfinite(cvi_max)) or cvi_span <= 1.0e-12:
            cvi_min = 0.0
            cvi_span = 1.0

        points: List[Point] = []
        colors: List[ColorRGBA] = []
        valid_tris = 0
        for tri_keys in tris:
            tri_pts = []
            for key in tri_keys:
                p = self._centroid_for_depth_voxel_key(key)
                if p is None:
                    tri_pts = []
                    break
                tri_pts.append(p)
            if len(tri_pts) != 3:
                continue
            pa, pb, pc = tri_pts
            area2 = float(np.linalg.norm(np.cross(pb - pa, pc - pa)))
            if (not np.isfinite(area2)) or area2 <= 1.0e-12:
                continue
            points.extend([self._point_msg(pa), self._point_msg(pb), self._point_msg(pc)])
            raw_cvi = float(self.depth_mesh_accumulated_triangle_cvi_raw.get(self._canonical_triangle_key(*tri_keys), 0.0))
            colors.extend([self._dark_to_green_color((raw_cvi - cvi_min) / cvi_span)] * 3)
            valid_tris += 1

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "depth_mesh_accumulated_patch"
        marker.id = 0
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD if points else Marker.DELETE
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = points
        marker.colors = colors
        self.depth_mesh_accumulated_patch_pub.publish(marker)

        rospy.loginfo_throttle(
            0.5,
            "Depth accumulated patch image=%s triangles=%d viz_triangles=%d",
            _blue_log_text(str(label)),
            int(len(self.depth_mesh_accumulated_triangles)),
            int(valid_tris),
        )

    @staticmethod
    def _largest_boundary_component(edges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not edges:
            return []
        adj: Dict[int, Set[int]] = {}
        for a, b in edges:
            adj.setdefault(int(a), set()).add(int(b))
            adj.setdefault(int(b), set()).add(int(a))

        unseen: Set[int] = set(adj.keys())
        best_nodes: Set[int] = set()
        while unseen:
            start = unseen.pop()
            comp = {start}
            stack = [start]
            while stack:
                node = stack.pop()
                for nb in adj.get(node, set()):
                    if nb in comp:
                        continue
                    comp.add(nb)
                    unseen.discard(nb)
                    stack.append(nb)
            if len(comp) > len(best_nodes):
                best_nodes = comp
        return [(a, b) for a, b in edges if int(a) in best_nodes and int(b) in best_nodes]

    @staticmethod
    def _unique_valid_cell_vertices(values: List[int]) -> List[int]:
        out: List[int] = []
        seen: Set[int] = set()
        for value in values:
            v = int(value)
            if v < 0 or v in seen:
                continue
            seen.add(v)
            out.append(v)
        return out

    def _publish_depth_mesh_patch(
        self,
        samples: Dict[str, object],
        point_keys: np.ndarray,
        label: str,
    ) -> None:
        if self.geometry_point_source != "depth" or not bool(self.depth_mesh_patching):
            return

        points = np.asarray(samples.get("points_world", np.zeros((0, 3))), dtype=np.float64).reshape(-1, 3)
        rows = np.asarray(samples.get("grid_rows", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        cols = np.asarray(samples.get("grid_cols", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        keys = np.asarray(point_keys, dtype=np.int64).reshape(-1, 3)
        grid_shape = tuple(int(x) for x in samples.get("grid_shape", (0, 0)))
        if (
            points.shape[0] == 0
            or keys.shape[0] != points.shape[0]
            or rows.shape[0] != points.shape[0]
            or cols.shape[0] != points.shape[0]
            or len(grid_shape) != 2
            or grid_shape[0] <= 1
            or grid_shape[1] <= 1
        ):
            self._clear_depth_mesh_patch_markers()
            return

        h, w = int(grid_shape[0]), int(grid_shape[1])
        grid_to_vertex = np.full((h, w), -1, dtype=np.int32)
        key_to_local: Dict[Tuple[int, int, int], int] = {}
        patch_vertices: List[np.ndarray] = []
        patch_vertex_keys: List[Tuple[int, int, int]] = []

        for i, key_arr in enumerate(keys):
            r = int(rows[i])
            c = int(cols[i])
            if r < 0 or r >= h or c < 0 or c >= w:
                continue
            key = tuple(int(x) for x in key_arr.tolist())
            rec = self.depth_voxels.get(key)
            if rec is None or int(rec.sample_count) <= 0:
                continue
            local_idx = key_to_local.get(key)
            if local_idx is None:
                local_idx = len(patch_vertices)
                key_to_local[key] = int(local_idx)
                patch_vertices.append(np.asarray(rec.sum_xyz, dtype=np.float64) / float(max(1, rec.sample_count)))
                patch_vertex_keys.append(key)
            grid_to_vertex[r, c] = int(local_idx)

        if len(patch_vertices) < 3:
            self.latest_depth_patch = None
            self._clear_depth_mesh_patch_markers()
            return

        vertices = np.asarray(patch_vertices, dtype=np.float64).reshape(-1, 3)
        vertex_keys = np.asarray(patch_vertex_keys, dtype=np.int64).reshape(-1, 3)
        triangles: List[Tuple[int, int, int]] = []
        triangle_keys: Set[Tuple[int, int, int]] = set()
        area_eps = 1.0e-12

        def edge_allowed(a: int, b: int) -> bool:
            ia = int(a)
            ib = int(b)
            if ia == ib:
                return False
            if self.depth_mesh_patch_edge_rule == "grid":
                return True
            if ia < 0 or ib < 0 or ia >= int(vertex_keys.shape[0]) or ib >= int(vertex_keys.shape[0]):
                return False
            delta = np.abs(vertex_keys[ia] - vertex_keys[ib])
            if not np.all(np.isfinite(delta)):
                return False
            cheb = int(np.max(delta))
            return cheb == int(self.depth_mesh_patch_edge_chebyshev)

        def add_tri(a: int, b: int, c: int) -> None:
            ia, ib, ic = int(a), int(b), int(c)
            if ia == ib or ib == ic or ia == ic:
                return
            if not (edge_allowed(ia, ib) and edge_allowed(ib, ic) and edge_allowed(ic, ia)):
                return
            pa, pb, pc = vertices[ia], vertices[ib], vertices[ic]
            area2 = float(np.linalg.norm(np.cross(pb - pa, pc - pa)))
            if (not np.isfinite(area2)) or area2 <= area_eps:
                return
            key = tuple(sorted((ia, ib, ic)))
            if key in triangle_keys:
                return
            triangle_keys.add(key)
            triangles.append((ia, ib, ic))

        for r in range(h - 1):
            row0 = grid_to_vertex[r]
            row1 = grid_to_vertex[r + 1]
            for c in range(w - 1):
                # Image-cell order: top-left, top-right, bottom-right, bottom-left.
                cell = self._unique_valid_cell_vertices(
                    [int(row0[c]), int(row0[c + 1]), int(row1[c + 1]), int(row1[c])]
                )
                if len(cell) == 3:
                    add_tri(cell[0], cell[1], cell[2])
                elif len(cell) >= 4:
                    add_tri(cell[0], cell[1], cell[2])
                    add_tri(cell[0], cell[2], cell[3])

        if not triangles:
            self.latest_depth_patch = None
            self._clear_depth_mesh_patch_markers()
            return

        edge_counts: Dict[Tuple[int, int], int] = {}
        cam_pos = np.asarray(
            samples.get("cam_pos_world", np.zeros((3,), dtype=np.float64)),
            dtype=np.float64,
        ).reshape(3)
        for tri in triangles:
            tri_voxel_keys = tuple(self._voxel_key_tuple(vertex_keys[int(vid)]) for vid in tri)
            canonical_tri_key = self._canonical_triangle_key(*tri_voxel_keys)
            self.depth_mesh_accumulated_triangles.setdefault(canonical_tri_key, tri_voxel_keys)
            pa, pb, pc = (vertices[int(tri[0])], vertices[int(tri[1])], vertices[int(tri[2])])
            normal_cross = np.cross(pb - pa, pc - pa)
            area2 = float(np.linalg.norm(normal_cross))
            center = (pa + pb + pc) / 3.0
            dist = float(np.linalg.norm(center - cam_pos))
            gain = 0.0
            if np.isfinite(area2) and area2 > area_eps and np.isfinite(dist) and dist > 1.0e-9:
                normal = normal_cross / area2
                tri_to_cam = (cam_pos - center) / dist
                facing = abs(float(np.dot(normal, tri_to_cam)))
                gain = float(0.5 * area2 * facing / (dist * dist + 1.0e-6))
            if np.isfinite(gain) and gain > 0.0:
                self.depth_mesh_accumulated_triangle_cvi_raw[canonical_tri_key] = (
                    float(self.depth_mesh_accumulated_triangle_cvi_raw.get(canonical_tri_key, 0.0))
                    + gain
                )
            for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                edge = (int(min(a, b)), int(max(a, b)))
                edge_counts[edge] = int(edge_counts.get(edge, 0)) + 1
        boundary_edges = [edge for edge, count in edge_counts.items() if int(count) == 1]
        boundary_edges = self._largest_boundary_component(boundary_edges)
        self.latest_depth_patch = {
            "depth_patch_vertices": vertices.astype(np.float32, copy=False),
            "depth_patch_vertex_keys": vertex_keys.astype(np.int64, copy=False),
            "depth_patch_triangles": np.asarray(triangles, dtype=np.int32).reshape(-1, 3),
            "depth_patch_boundary_edges": np.asarray(boundary_edges, dtype=np.int32).reshape(-1, 2),
            "depth_patch_camera_position": np.asarray(
                samples.get("cam_pos_world", np.zeros((3,), dtype=np.float64)),
                dtype=np.float32,
            ).reshape(3),
            "depth_patch_camera_quaternion_xyzw": np.asarray(
                samples.get("cam_quat_xyzw_world", np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)),
                dtype=np.float32,
            ).reshape(4),
            "depth_patch_label": np.asarray([str(label)]),
        }

        viz_triangles = triangles
        if self.depth_mesh_patch_max_triangles_viz > 0 and len(viz_triangles) > self.depth_mesh_patch_max_triangles_viz:
            sample_idx = np.linspace(
                0,
                len(viz_triangles) - 1,
                int(self.depth_mesh_patch_max_triangles_viz),
            ).astype(np.int64)
            viz_triangles = [viz_triangles[int(i)] for i in sample_idx]

        stamp = rospy.Time.now()
        patch_marker = Marker()
        patch_marker.header.stamp = stamp
        patch_marker.header.frame_id = self.frame_id
        patch_marker.ns = "depth_mesh_patch"
        patch_marker.id = 0
        patch_marker.type = Marker.TRIANGLE_LIST
        patch_marker.action = Marker.ADD
        patch_marker.pose.orientation.w = 1.0
        patch_marker.scale.x = 1.0
        patch_marker.scale.y = 1.0
        patch_marker.scale.z = 1.0
        patch_marker.color.r = 0.0
        patch_marker.color.g = 0.85
        patch_marker.color.b = 1.0
        patch_marker.color.a = 0.35
        patch_marker.points = [
            self._point_msg(vertices[int(vid)])
            for tri in viz_triangles
            for vid in tri
        ]
        self.depth_mesh_patch_pub.publish(patch_marker)

        boundary_marker = Marker()
        boundary_marker.header.stamp = stamp
        boundary_marker.header.frame_id = self.frame_id
        boundary_marker.ns = "depth_mesh_patch_boundary"
        boundary_marker.id = 0
        boundary_marker.type = Marker.LINE_LIST
        boundary_marker.action = Marker.ADD
        boundary_marker.pose.orientation.w = 1.0
        boundary_marker.scale.x = 0.04
        boundary_marker.color.r = 1.0
        boundary_marker.color.g = 0.95
        boundary_marker.color.b = 0.05
        boundary_marker.color.a = 1.0
        boundary_marker.points = [
            self._point_msg(vertices[int(vid)])
            for edge in boundary_edges
            for vid in edge
        ]
        self.depth_mesh_patch_boundary_pub.publish(boundary_marker)
        self._publish_depth_mesh_accumulated_patch(label)

        rospy.loginfo_throttle(
            0.5,
            "Depth patch image=%s vertices=%d triangles=%d viz_triangles=%d boundary_edges=%d edge_rule=%s",
            _blue_log_text(str(label)),
            int(vertices.shape[0]),
            int(len(triangles)),
            int(len(viz_triangles)),
            int(len(boundary_edges)),
            self.depth_mesh_patch_edge_rule,
        )

    def _process_depth_frame(
        self,
        depth: np.ndarray,
        cam_pos_world: np.ndarray,
        cam_quat_xyzw_world: np.ndarray,
        label: str,
        apply_frame_stride: bool = True,
    ) -> bool:
        self.depth_input_frames += 1
        if bool(apply_frame_stride) and ((self.depth_input_frames - 1) % int(self.depth_frame_stride)) != 0:
            return True
        samples = self._depth_samples_from_frame(depth, cam_pos_world, cam_quat_xyzw_world)
        samples["cam_pos_world"] = np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
        samples["cam_quat_xyzw_world"] = np.asarray(cam_quat_xyzw_world, dtype=np.float64).reshape(4)
        points = np.asarray(samples["points_world"], dtype=np.float64).reshape(-1, 3)
        self._publish_depth_raw_cloud(points)
        usable_updates, point_keys = self._fuse_depth_points(points, return_point_keys=True)
        self._publish_depth_mesh_patch(samples, point_keys, label)
        self.depth_processed_frames += 1
        self.accepted_keyframes = int(self.depth_processed_frames)
        if (self.depth_processed_frames % self.publish_every_kf) == 0:
            self._publish_sparse_cloud()
        if (self.depth_processed_frames % self.save_every_kf) == 0:
            self._save_sparse_npz()
        rospy.loginfo_throttle(
            0.5,
            "Depth image=%s processed=%d roi_points=%d voxel_records=%d usable_updates=%d usable_points=%d",
            _blue_log_text(str(label)),
            int(self.depth_processed_frames),
            int(points.shape[0]),
            len(self.depth_voxels),
            int(usable_updates),
            len(self.points_xyz),
        )
        return True

    def _depth_path_for_pose(self, entry: PoseEntry) -> str:
        return os.path.join(self.depth_dir, f"depth_{int(entry.image_id):06d}.npy")

    def _process_saved_depth_pose_entry(self, entry: PoseEntry, seq_idx: int) -> bool:
        path = self._depth_path_for_pose(entry)
        if not os.path.isfile(path):
            rospy.logwarn_throttle(2.0, "[WARN] Depth waiting for saved frame: %s", path)
            return False
        try:
            depth = np.load(path, allow_pickle=False)
        except Exception as exc:
            rospy.logwarn("[WARN] Failed loading saved depth frame '%s': %s", path, str(exc))
            return True
        return self._process_depth_frame(
            depth,
            cam_pos_world=entry.cam_pos_world,
            cam_quat_xyzw_world=entry.cam_quat_xyzw_world,
            label=entry.img_name,
        )

    def _build_kp_presence_mask(self, kpts_uv: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
        mask = np.zeros((int(image_h), int(image_w)), dtype=np.uint8)
        if kpts_uv.size == 0:
            return mask
        uv = np.rint(kpts_uv).astype(np.int32)
        valid = (
            (uv[:, 0] >= 0)
            & (uv[:, 0] < int(image_w))
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < int(image_h))
        )
        if np.any(valid):
            pts = uv[valid]
            mask[pts[:, 1], pts[:, 0]] = 255

        rad = max(0, int(math.ceil(float(self.track_confirm_reproj_px))))
        if rad > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
            mask = cv2.dilate(mask, k, iterations=1)
        return mask

    def _confirm_points_multiview(self, points: np.ndarray, keyframes: List[Keyframe]) -> np.ndarray:
        if points.size == 0:
            return points
        if self.min_track_views <= 2:
            return points
        if len(keyframes) < self.min_track_views:
            return np.zeros((0, 3), dtype=np.float32)

        pts = points.astype(np.float64, copy=False)
        x_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
        counts = np.zeros((pts.shape[0],), dtype=np.int16)
        needed = int(self.min_track_views)
        total_views = len(keyframes)

        for i, kf in enumerate(keyframes):
            proj = (kf.p_mat @ x_h.T).T
            z = proj[:, 2]
            valid = z > self.near_depth
            if np.any(valid):
                idx = np.where(valid)[0]
                uv = np.rint(proj[idx, :2] / z[idx, None]).astype(np.int32)
                inside = (
                    (uv[:, 0] >= 0)
                    & (uv[:, 0] < int(kf.image_w))
                    & (uv[:, 1] >= 0)
                    & (uv[:, 1] < int(kf.image_h))
                )
                if np.any(inside):
                    idx = idx[inside]
                    uv = uv[inside]
                else:
                    idx = np.zeros((0,), dtype=np.int64)
                if idx.size > 0:
                    hit = kf.kp_presence[uv[:, 1], uv[:, 0]] > 0
                    if np.any(hit):
                        counts[idx[hit]] += 1

            # Early impossible-case pruning.
            remaining = total_views - (i + 1)
            possible = counts + remaining >= needed
            if not np.any(possible):
                return np.zeros((0, 3), dtype=np.float32)

        keep = counts >= needed
        if not np.any(keep):
            return np.zeros((0, 3), dtype=np.float32)
        return points[keep]

    def _make_building_roi_mask(
        self,
        r_wc_cv: np.ndarray,
        t_wc_cv: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> Optional[np.ndarray]:
        if not self.use_building_roi_mask:
            return None
        if img_h <= 0 or img_w <= 0:
            return None

        xc = (r_wc_cv @ self.build_corners_world.T + t_wc_cv.reshape(3, 1)).T
        z = xc[:, 2]
        valid = z > self.near_depth
        if int(np.count_nonzero(valid)) == 0:
            return np.full((img_h, img_w), 255, dtype=np.uint8) if self.roi_mask_fallback_full else None

        proj = (self.k_mat @ xc[valid].T).T
        denom = proj[:, 2:3]
        finite = np.abs(denom[:, 0]) > 1e-12
        if int(np.count_nonzero(finite)) == 0:
            return np.full((img_h, img_w), 255, dtype=np.uint8) if self.roi_mask_fallback_full else None

        uv = proj[finite, :2] / denom[finite]
        u_min = int(np.floor(np.min(uv[:, 0]) - self.roi_mask_margin_px))
        v_min = int(np.floor(np.min(uv[:, 1]) - self.roi_mask_margin_px))
        u_max = int(np.ceil(np.max(uv[:, 0]) + self.roi_mask_margin_px))
        v_max = int(np.ceil(np.max(uv[:, 1]) + self.roi_mask_margin_px))

        u_min = max(0, u_min)
        v_min = max(0, v_min)
        u_max = min(img_w - 1, u_max)
        v_max = min(img_h - 1, v_max)
        if u_min > u_max or v_min > v_max:
            return np.full((img_h, img_w), 255, dtype=np.uint8) if self.roi_mask_fallback_full else None

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[v_min : v_max + 1, u_min : u_max + 1] = 255
        return mask

    def _clear_sparse_state(self) -> None:
        self.points_xyz = []
        self.obs_count = []
        self.point_ids = []
        self.point_revs = []
        self.point_track_ids = []
        self.point_views_support = []
        self.point_views_total = []
        self.point_reproj_error_px = []
        self.point_parallax_deg = []
        self.depth_voxels = {}
        self.depth_mesh_accumulated_triangles = {}
        self.depth_mesh_accumulated_triangle_cvi_raw = {}
        self.latest_depth_patch = None
        self.depth_point_view_counts = []
        self.depth_point_sample_counts = []
        self.depth_input_frames = 0
        self.depth_processed_frames = 0
        self.track_to_point_idx = {}
        self.voxel_to_idx = {}
        self.next_pose_offset = 0
        self.accepted_keyframes = 0
        self.last_accepted_pose = None
        self.keyframes = []
        self.next_keyframe_uid = 0
        self.next_track_id = 0
        self.next_point_id = 1
        self.tracks = {}
        self.obs_to_track = {}
        self.track_last_triangulated_kf = {}
        if hasattr(self, "alternating_feature_idx"):
            self.alternating_feature_idx = 0
        if hasattr(self, "_current_feature_stream"):
            self._current_feature_stream = self.feature_backend

    def _register_keyframe_scaffold(self, kf: Keyframe) -> None:
        _ = kf.kf_id

    @staticmethod
    def _obs_key(kf_id: int, kp_idx: int) -> Tuple[int, int]:
        return (int(kf_id), int(kp_idx))

    @staticmethod
    def _track_conflicts_with_obs(track: TrackRecord, obs: Tuple[int, int]) -> bool:
        kf_id, kp_idx = obs
        for tkf, tkp in track.observations:
            if tkf == kf_id and tkp != kp_idx:
                return True
        return False

    def _create_track(self, obs_a: Tuple[int, int], obs_b: Tuple[int, int]) -> int:
        track_id = int(self.next_track_id)
        self.next_track_id += 1
        if obs_a == obs_b:
            observations = [obs_a]
        else:
            observations = [obs_a, obs_b]
        self.tracks[track_id] = TrackRecord(track_id=track_id, observations=observations)
        for obs in observations:
            self.obs_to_track[obs] = track_id
        return track_id

    def _add_obs_to_track(self, track_id: int, obs: Tuple[int, int]) -> bool:
        track = self.tracks.get(int(track_id), None)
        if track is None:
            return False
        existing_track = self.obs_to_track.get(obs, None)
        if existing_track is not None:
            return int(existing_track) == int(track_id)
        if self._track_conflicts_with_obs(track, obs):
            return False
        track.observations.append(obs)
        self.obs_to_track[obs] = int(track_id)
        return True

    @staticmethod
    def _tracks_conflict(track_a: TrackRecord, track_b: TrackRecord) -> bool:
        a_map: Dict[int, int] = {}
        for kf_id, kp_idx in track_a.observations:
            prev = a_map.get(int(kf_id), None)
            if prev is None:
                a_map[int(kf_id)] = int(kp_idx)
            elif prev != int(kp_idx):
                return True
        for kf_id, kp_idx in track_b.observations:
            a_kp = a_map.get(int(kf_id), None)
            if a_kp is not None and a_kp != int(kp_idx):
                return True
        return False

    def _merge_tracks(self, track_id_a: int, track_id_b: int) -> Optional[int]:
        if int(track_id_a) == int(track_id_b):
            return int(track_id_a)
        ta = self.tracks.get(int(track_id_a), None)
        tb = self.tracks.get(int(track_id_b), None)
        if ta is None or tb is None:
            return None
        if self._tracks_conflict(ta, tb):
            return None

        if len(ta.observations) >= len(tb.observations):
            dst_id, src_id = int(track_id_a), int(track_id_b)
        else:
            dst_id, src_id = int(track_id_b), int(track_id_a)
        dst = self.tracks[dst_id]
        src = self.tracks[src_id]

        dst_obs_set = set(dst.observations)
        for obs in src.observations:
            if obs in dst_obs_set:
                self.obs_to_track[obs] = dst_id
                continue
            dst.observations.append(obs)
            dst_obs_set.add(obs)
            self.obs_to_track[obs] = dst_id

        del self.tracks[src_id]

        src_pt_idx = self.track_to_point_idx.get(src_id, None)
        dst_pt_idx = self.track_to_point_idx.get(dst_id, None)
        if src_pt_idx is not None:
            if dst_pt_idx is None:
                self.track_to_point_idx[dst_id] = src_pt_idx
                if 0 <= int(src_pt_idx) < len(self.point_track_ids):
                    self.point_track_ids[int(src_pt_idx)] = dst_id
            else:
                s_idx = int(src_pt_idx)
                d_idx = int(dst_pt_idx)
                if (
                    0 <= s_idx < len(self.points_xyz)
                    and 0 <= d_idx < len(self.points_xyz)
                    and s_idx != d_idx
                ):
                    c_s = max(1, int(self.obs_count[s_idx]))
                    c_d = max(1, int(self.obs_count[d_idx]))
                    self.points_xyz[d_idx] = (self.points_xyz[d_idx] * c_d + self.points_xyz[s_idx] * c_s) / float(
                        c_d + c_s
                    )
                    self.obs_count[d_idx] = c_d + c_s
                    self.obs_count[s_idx] = 0
                    if s_idx < len(self.point_track_ids):
                        self.point_track_ids[s_idx] = -1
            del self.track_to_point_idx[src_id]
        return dst_id

    def _register_pair_observations_scaffold(
        self,
        ref: Keyframe,
        cur: Keyframe,
        ref_kp_idx: np.ndarray,
        cur_kp_idx: np.ndarray,
    ) -> None:
        if ref_kp_idx.shape[0] == 0 or cur_kp_idx.shape[0] == 0:
            return
        n = min(int(ref_kp_idx.shape[0]), int(cur_kp_idx.shape[0]))
        for i in range(n):
            obs_ref = self._obs_key(ref.kf_id, int(ref_kp_idx[i]))
            obs_cur = self._obs_key(cur.kf_id, int(cur_kp_idx[i]))
            track_ref = self.obs_to_track.get(obs_ref, None)
            track_cur = self.obs_to_track.get(obs_cur, None)

            if track_ref is None and track_cur is None:
                self._create_track(obs_ref, obs_cur)
                continue

            if track_ref is not None and track_cur is None:
                self._add_obs_to_track(int(track_ref), obs_cur)
                continue

            if track_ref is None and track_cur is not None:
                self._add_obs_to_track(int(track_cur), obs_ref)
                continue

            if int(track_ref) != int(track_cur):
                self._merge_tracks(int(track_ref), int(track_cur))

    def _prune_tracks_to_active_keyframes(self, active_kf_ids: List[int]) -> None:
        if not self.tracks:
            return
        active = set(int(kf_id) for kf_id in active_kf_ids)
        for track_id in list(self.tracks.keys()):
            track = self.tracks.get(track_id, None)
            if track is None:
                continue
            old_obs = list(track.observations)
            filtered: List[Tuple[int, int]] = []
            seen_obs = set()
            kf_to_kp: Dict[int, int] = {}
            for obs in old_obs:
                kf_id, kp_idx = int(obs[0]), int(obs[1])
                if kf_id not in active:
                    continue
                if obs in seen_obs:
                    continue
                prev_kp = kf_to_kp.get(kf_id, None)
                if prev_kp is not None and prev_kp != kp_idx:
                    continue
                seen_obs.add(obs)
                kf_to_kp[kf_id] = kp_idx
                filtered.append(obs)

            for obs in old_obs:
                if self.obs_to_track.get(obs, None) == track_id:
                    del self.obs_to_track[obs]

            if len(filtered) < 2:
                pt_idx = self.track_to_point_idx.pop(track_id, None)
                if pt_idx is not None and 0 <= int(pt_idx) < len(self.point_track_ids):
                    self.point_track_ids[int(pt_idx)] = -1
                del self.tracks[track_id]
                continue

            track.observations = filtered
            for obs in filtered:
                self.obs_to_track[obs] = track_id

        for track_id in list(self.track_last_triangulated_kf.keys()):
            if track_id not in self.tracks:
                del self.track_last_triangulated_kf[track_id]

    def _triangulate_track_point(
        self,
        track_id: int,
        kf_by_id: Dict[int, Keyframe],
    ) -> Optional[Tuple[np.ndarray, float, float, int, int]]:
        track = self.tracks.get(int(track_id), None)
        if track is None:
            return None

        observations: List[Tuple[Keyframe, np.ndarray]] = []
        seen_kf = set()
        for obs in track.observations:
            kf_id, kp_idx = int(obs[0]), int(obs[1])
            if kf_id in seen_kf:
                continue
            kf = kf_by_id.get(kf_id, None)
            if kf is None:
                continue
            if kp_idx < 0 or kp_idx >= int(kf.kpts_uv.shape[0]):
                continue
            seen_kf.add(kf_id)
            observations.append((kf, kf.kpts_uv[kp_idx].astype(np.float64)))

        if len(observations) < max(2, int(self.min_track_views)):
            return None

        x = self._triangulate_track_dlt(observations)
        if x is None:
            return None

        support, inlier_obs, mean_err = self._track_support_for_point(x, observations)
        if support < int(self.min_track_views):
            return None

        # One inlier-only DLT refinement keeps this lightweight but much less pair-biased.
        if len(inlier_obs) >= max(2, int(self.min_track_views)):
            x_refined = self._triangulate_track_dlt(inlier_obs)
            if x_refined is not None:
                x = x_refined
                support, inlier_obs, mean_err = self._track_support_for_point(x, observations)
                if support < int(self.min_track_views):
                    return None

        parallax_deg = self._max_track_parallax_deg(x, inlier_obs if len(inlier_obs) >= 2 else observations)
        if parallax_deg < self.min_parallax_deg:
            return None

        return (
            x.astype(np.float32),
            float(mean_err),
            float(parallax_deg),
            int(support),
            int(len(observations)),
        )

    @staticmethod
    def _triangulate_track_dlt(observations: List[Tuple[Keyframe, np.ndarray]]) -> Optional[np.ndarray]:
        if len(observations) < 2:
            return None
        rows = []
        for kf, uv in observations:
            u = float(uv[0])
            v = float(uv[1])
            p = kf.p_mat
            rows.append(u * p[2, :] - p[0, :])
            rows.append(v * p[2, :] - p[1, :])
        if len(rows) < 4:
            return None
        a_mat = np.asarray(rows, dtype=np.float64)
        try:
            _, _, vt = np.linalg.svd(a_mat, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        x_h = vt[-1, :]
        w = float(x_h[3])
        if abs(w) <= 1e-12:
            return None
        return (x_h[:3] / w).astype(np.float64)

    def _track_support_for_point(
        self,
        x: np.ndarray,
        observations: List[Tuple[Keyframe, np.ndarray]],
    ) -> Tuple[int, List[Tuple[Keyframe, np.ndarray]], float]:
        x_hg = np.array([x[0], x[1], x[2], 1.0], dtype=np.float64)
        support = 0
        inlier_obs: List[Tuple[Keyframe, np.ndarray]] = []
        inlier_err_sum = 0.0
        for kf, uv in observations:
            xc = kf.r_wc_cv @ x + kf.t_wc_cv
            if xc[2] <= self.near_depth:
                continue
            proj = kf.p_mat @ x_hg
            if abs(float(proj[2])) <= 1e-12:
                continue
            uv_hat = proj[:2] / proj[2]
            err = float(np.linalg.norm(uv_hat - uv))
            if err <= self.track_confirm_reproj_px:
                support += 1
                inlier_obs.append((kf, uv))
                inlier_err_sum += err
        mean_err = float(inlier_err_sum / float(max(support, 1)))
        return support, inlier_obs, mean_err

    @staticmethod
    def _max_track_parallax_deg(
        x: np.ndarray,
        observations: List[Tuple[Keyframe, np.ndarray]],
    ) -> float:
        if len(observations) < 2:
            return 0.0
        max_parallax = 0.0
        for i in range(len(observations)):
            c1 = observations[i][0].cam_pos_world
            v1 = x - c1
            n1 = float(np.linalg.norm(v1))
            if n1 <= 1e-12:
                continue
            v1n = v1 / n1
            for j in range(i + 1, len(observations)):
                c2 = observations[j][0].cam_pos_world
                v2 = x - c2
                n2 = float(np.linalg.norm(v2))
                if n2 <= 1e-12:
                    continue
                v2n = v2 / n2
                cosang = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
                parallax = float(np.degrees(np.arccos(cosang)))
                if parallax > max_parallax:
                    max_parallax = parallax
        return max_parallax

    def _triangulate_candidate_tracks(
        self,
        cur_kf: Keyframe,
        candidate_track_ids: List[int],
    ) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not candidate_track_ids:
            empty = np.zeros((0,), dtype=np.float32)
            return [], np.zeros((0, 3), dtype=np.float32), empty, empty, empty, empty

        kf_by_id: Dict[int, Keyframe] = {kf.kf_id: kf for kf in self.keyframes}
        kf_by_id[cur_kf.kf_id] = cur_kf

        out_track_ids: List[int] = []
        pts = []
        reproj = []
        parallax = []
        views = []
        views_total = []
        for track_id in candidate_track_ids:
            if int(self.track_last_triangulated_kf.get(track_id, -1)) == int(cur_kf.kf_id):
                continue
            p = self._triangulate_track_point(track_id=track_id, kf_by_id=kf_by_id)
            self.track_last_triangulated_kf[track_id] = int(cur_kf.kf_id)
            if p is not None:
                point_xyz, mean_err, parallax_deg, support_views, total_views = p
                out_track_ids.append(int(track_id))
                pts.append(point_xyz)
                reproj.append(float(mean_err))
                parallax.append(float(parallax_deg))
                views.append(float(support_views))
                views_total.append(float(total_views))
        if not pts:
            empty = np.zeros((0,), dtype=np.float32)
            return [], np.zeros((0, 3), dtype=np.float32), empty, empty, empty, empty
        return (
            out_track_ids,
            np.asarray(pts, dtype=np.float32),
            np.asarray(reproj, dtype=np.float32),
            np.asarray(parallax, dtype=np.float32),
            np.asarray(views, dtype=np.float32),
            np.asarray(views_total, dtype=np.float32),
        )

    def _load_existing_sparse_npz(self) -> None:
        if not os.path.isfile(self.output_path):
            return
        try:
            data = np.load(self.output_path, allow_pickle=False)
            pts = np.asarray(data.get("points_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
            obs = np.asarray(data.get("obs_count", np.ones((pts.shape[0],), dtype=np.int32)), dtype=np.int32)
            point_ids = np.asarray(
                data.get("point_ids", np.arange(1, pts.shape[0] + 1, dtype=np.int64)),
                dtype=np.int64,
            )
            point_revs = np.asarray(
                data.get("point_revs", np.ones((pts.shape[0],), dtype=np.int64)),
                dtype=np.int64,
            )
            track_ids = np.asarray(
                data.get("track_ids", np.full((pts.shape[0],), -1, dtype=np.int64)),
                dtype=np.int64,
            )
            v_sup = np.asarray(
                data.get("views_support", np.clip(obs.astype(np.float32), 1.0, float(self.window_size))),
                dtype=np.float32,
            )
            v_tot = np.asarray(
                data.get(
                    "views_total",
                    np.maximum(v_sup.astype(np.float32), np.asarray(self.min_track_views, dtype=np.float32)),
                ),
                dtype=np.float32,
            )
            e_rep = np.asarray(
                data.get(
                    "reproj_error_px",
                    np.full((pts.shape[0],), float(self.track_confirm_reproj_px), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            p_deg = np.asarray(
                data.get(
                    "parallax_deg",
                    np.full((pts.shape[0],), float(max(self.min_parallax_deg, 1.0)), dtype=np.float32),
                ),
                dtype=np.float32,
            )
            depth_views = np.asarray(
                data.get("depth_view_count", np.rint(v_sup).astype(np.int32)),
                dtype=np.int32,
            )
            depth_samples = np.asarray(
                data.get("depth_sample_count", np.maximum(obs, 1)),
                dtype=np.int32,
            )
            accumulated_patch_raw = np.asarray(
                data.get("depth_accumulated_patch_triangles", np.zeros((0, 9), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1, 9)
            accumulated_patch_cvi_raw = np.asarray(
                data.get("depth_accumulated_patch_cvi_raw", np.zeros((accumulated_patch_raw.shape[0],), dtype=np.float32)),
                dtype=np.float64,
            ).reshape(-1)
            if obs.shape[0] != pts.shape[0]:
                obs = np.ones((pts.shape[0],), dtype=np.int32)
            if point_ids.shape[0] != pts.shape[0]:
                point_ids = np.arange(1, pts.shape[0] + 1, dtype=np.int64)
            if point_revs.shape[0] != pts.shape[0]:
                point_revs = np.ones((pts.shape[0],), dtype=np.int64)
            if track_ids.shape[0] != pts.shape[0]:
                track_ids = np.full((pts.shape[0],), -1, dtype=np.int64)
            if v_sup.shape[0] != pts.shape[0]:
                v_sup = np.clip(obs.astype(np.float32), 1.0, float(self.window_size))
            if v_tot.shape[0] != pts.shape[0]:
                v_tot = np.maximum(v_sup.astype(np.float32), np.asarray(self.min_track_views, dtype=np.float32))
            if e_rep.shape[0] != pts.shape[0]:
                e_rep = np.full((pts.shape[0],), float(self.track_confirm_reproj_px), dtype=np.float32)
            if p_deg.shape[0] != pts.shape[0]:
                p_deg = np.full((pts.shape[0],), float(max(self.min_parallax_deg, 1.0)), dtype=np.float32)
            if depth_views.shape[0] != pts.shape[0]:
                depth_views = np.rint(v_sup).astype(np.int32)
            if depth_samples.shape[0] != pts.shape[0]:
                depth_samples = np.maximum(obs, 1).astype(np.int32)
            self.points_xyz = []
            self.obs_count = []
            self.point_ids = []
            self.point_revs = []
            self.point_track_ids = []
            self.point_views_support = []
            self.point_views_total = []
            self.point_reproj_error_px = []
            self.point_parallax_deg = []
            self.track_to_point_idx = {}
            self.depth_voxels = {}
            self.depth_mesh_accumulated_triangles = {}
            self.depth_mesh_accumulated_triangle_cvi_raw = {}
            self.depth_point_view_counts = []
            self.depth_point_sample_counts = []
            for i in range(int(pts.shape[0])):
                p = pts[i].astype(np.float64)
                if not self._inside_building(p):
                    continue
                self.points_xyz.append(p)
                self.obs_count.append(int(obs[i]))
                pid = int(point_ids[i])
                self.point_ids.append(pid)
                self.point_revs.append(max(1, int(point_revs[i])))
                tid = int(track_ids[i])
                self.point_track_ids.append(tid)
                self.point_views_support.append(float(v_sup[i]))
                self.point_views_total.append(float(v_tot[i]))
                self.point_reproj_error_px.append(float(e_rep[i]))
                self.point_parallax_deg.append(float(p_deg[i]))
                if self.geometry_point_source == "depth":
                    view_count = max(1, int(depth_views[i]))
                    sample_count = max(1, int(depth_samples[i]))
                    point_idx = len(self.points_xyz) - 1
                    self.depth_point_view_counts.append(view_count)
                    self.depth_point_sample_counts.append(sample_count)
                    key = tuple(np.floor(p / self.depth_voxel_size).astype(np.int64).tolist())
                    self.depth_voxels[key] = DepthVoxelRecord(
                        sum_xyz=p * float(sample_count),
                        sample_count=sample_count,
                        view_count=view_count,
                        point_idx=point_idx,
                    )
                if tid >= 0 and tid not in self.track_to_point_idx:
                    self.track_to_point_idx[tid] = len(self.points_xyz) - 1
            for row_idx, row in enumerate(accumulated_patch_raw):
                tri_keys = (
                    (int(row[0]), int(row[1]), int(row[2])),
                    (int(row[3]), int(row[4]), int(row[5])),
                    (int(row[6]), int(row[7]), int(row[8])),
                )
                if len(set(tri_keys)) != 3:
                    continue
                canonical_tri_key = self._canonical_triangle_key(*tri_keys)
                self.depth_mesh_accumulated_triangles.setdefault(canonical_tri_key, tri_keys)
                if row_idx < int(accumulated_patch_cvi_raw.shape[0]):
                    raw_cvi = float(accumulated_patch_cvi_raw[row_idx])
                    if np.isfinite(raw_cvi) and raw_cvi > 0.0:
                        self.depth_mesh_accumulated_triangle_cvi_raw[canonical_tri_key] = raw_cvi
            if self.track_to_point_idx:
                self.next_track_id = max(int(self.next_track_id), max(self.track_to_point_idx.keys()) + 1)
            if self.point_ids:
                self.next_point_id = max(int(self.next_point_id), max(self.point_ids) + 1)
            if self.voxel_size > 0.0:
                for i, p in enumerate(self.points_xyz):
                    key = tuple(np.floor(p / self.voxel_size).astype(np.int64).tolist())
                    self.voxel_to_idx[key] = i
            rospy.loginfo("Loaded existing sparse points: %d", len(self.points_xyz))
        except Exception as exc:
            rospy.logwarn("Failed to load sparse npz (%s). Starting empty.", str(exc))
            self.points_xyz = []
            self.obs_count = []
            self.point_ids = []
            self.point_revs = []
            self.point_track_ids = []
            self.point_views_support = []
            self.point_views_total = []
            self.point_reproj_error_px = []
            self.point_parallax_deg = []
            self.track_to_point_idx = {}
            self.voxel_to_idx = {}
            self.depth_voxels = {}
            self.depth_mesh_accumulated_triangles = {}
            self.depth_mesh_accumulated_triangle_cvi_raw = {}
            self.depth_point_view_counts = []
            self.depth_point_sample_counts = []
            self.next_point_id = 1

    def _save_sparse_npz(self) -> None:
        if self.points_xyz:
            pts = np.asarray(self.points_xyz, dtype=np.float32)
            obs = np.asarray(self.obs_count, dtype=np.int32)
        else:
            pts = np.zeros((0, 3), dtype=np.float32)
            obs = np.zeros((0,), dtype=np.int32)
        if self.point_ids and len(self.point_ids) == int(pts.shape[0]):
            point_ids = np.asarray(self.point_ids, dtype=np.int64)
        else:
            point_ids = np.arange(1, pts.shape[0] + 1, dtype=np.int64)
        if self.point_revs and len(self.point_revs) == int(pts.shape[0]):
            point_revs = np.asarray(self.point_revs, dtype=np.int64)
        else:
            point_revs = np.ones((pts.shape[0],), dtype=np.int64)
        if self.point_track_ids and len(self.point_track_ids) == int(pts.shape[0]):
            track_ids = np.asarray(self.point_track_ids, dtype=np.int64)
        else:
            track_ids = np.full((pts.shape[0],), -1, dtype=np.int64)
        if self.point_views_support and len(self.point_views_support) == int(pts.shape[0]):
            views_support = np.asarray(self.point_views_support, dtype=np.float32)
        else:
            views_support = np.full((pts.shape[0],), float(self.min_track_views), dtype=np.float32)
        if self.point_views_total and len(self.point_views_total) == int(pts.shape[0]):
            views_total = np.asarray(self.point_views_total, dtype=np.float32)
        else:
            views_total = np.maximum(views_support, np.asarray(float(self.min_track_views), dtype=np.float32))
        if self.point_reproj_error_px and len(self.point_reproj_error_px) == int(pts.shape[0]):
            reproj_error_px = np.asarray(self.point_reproj_error_px, dtype=np.float32)
        else:
            reproj_error_px = np.full((pts.shape[0],), float(self.track_confirm_reproj_px), dtype=np.float32)
        if self.point_parallax_deg and len(self.point_parallax_deg) == int(pts.shape[0]):
            parallax_deg = np.asarray(self.point_parallax_deg, dtype=np.float32)
        else:
            parallax_deg = np.full((pts.shape[0],), float(max(self.min_parallax_deg, 1.0)), dtype=np.float32)
        if self.depth_point_view_counts and len(self.depth_point_view_counts) == int(pts.shape[0]):
            depth_view_count = np.asarray(self.depth_point_view_counts, dtype=np.int32)
        else:
            depth_view_count = np.rint(views_support).astype(np.int32)
        if self.depth_point_sample_counts and len(self.depth_point_sample_counts) == int(pts.shape[0]):
            depth_sample_count = np.asarray(self.depth_point_sample_counts, dtype=np.int32)
        else:
            depth_sample_count = np.maximum(obs, 1).astype(np.int32)
        accumulated_tri_items = list(self.depth_mesh_accumulated_triangles.items())
        if accumulated_tri_items:
            accumulated_triangles = np.asarray(
                [
                    [coord for key in tri_keys for coord in key]
                    for _canonical_tri_key, tri_keys in accumulated_tri_items
                ],
                dtype=np.int64,
            ).reshape(-1, 9)
            accumulated_patch_cvi_raw = np.asarray(
                [
                    float(self.depth_mesh_accumulated_triangle_cvi_raw.get(canonical_tri_key, 0.0))
                    for canonical_tri_key, _tri_keys in accumulated_tri_items
                ],
                dtype=np.float32,
            ).reshape(-1)
        else:
            accumulated_triangles = np.zeros((0, 9), dtype=np.int64)
            accumulated_patch_cvi_raw = np.zeros((0,), dtype=np.float32)

        arrays = {
            "points_xyz": pts,
            "point_ids": point_ids,
            "point_revs": point_revs,
            "obs_count": obs,
            "track_ids": track_ids,
            "views_support": views_support,
            "views_total": views_total,
            "reproj_error_px": reproj_error_px,
            "parallax_deg": parallax_deg,
            "depth_view_count": depth_view_count,
            "depth_sample_count": depth_sample_count,
            "geometry_point_source": np.asarray([self.geometry_point_source]),
            "depth_accumulated_patch_triangles": accumulated_triangles,
            "depth_accumulated_patch_cvi_raw": accumulated_patch_cvi_raw,
            "last_pose_offset": np.array([self.next_pose_offset], dtype=np.int64),
            "accepted_keyframes": np.array([self.accepted_keyframes], dtype=np.int64),
            "updated_unix": np.array([time.time()], dtype=np.float64),
        }
        if self.geometry_point_source == "depth" and bool(self.depth_mesh_patching) and self.latest_depth_patch:
            for key, value in self.latest_depth_patch.items():
                arrays[key] = value
        _atomic_savez(self.output_path, arrays)

    def _publish_sparse_cloud(self) -> None:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        pts = (
            np.asarray(self.points_xyz, dtype=np.float32)
            if self.points_xyz
            else np.zeros((0, 3), dtype=np.float32)
        )
        msg = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        self.points_pub.publish(msg)

    def _publish_depth_raw_cloud(self, points: np.ndarray) -> None:
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        msg = point_cloud2.create_cloud_xyz32(header, pts.tolist())
        self.depth_raw_points_pub.publish(msg)

    def _load_pose_entries(self) -> List[PoseEntry]:
        if not os.path.isfile(self.poses_path):
            return []
        entries: List[PoseEntry] = []
        seen_ids = set()
        with open(self.poses_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    image_id = int(parts[0])
                    if image_id in seen_ids:
                        continue
                    qw = float(parts[1])
                    qx = float(parts[2])
                    qy = float(parts[3])
                    qz = float(parts[4])
                    tx = float(parts[5])
                    ty = float(parts[6])
                    tz = float(parts[7])
                    img_name = parts[9]
                except Exception:
                    continue

                resolved_img_name = self._resolve_pose_image_name(image_id=image_id, img_name=img_name)
                cam_pos = np.array([tx, ty, tz], dtype=np.float64)
                cam_quat = np.array([qx, qy, qz, qw], dtype=np.float64)
                yaw = _yaw_from_cam_quat_xyzw(cam_quat)
                entries.append(
                    PoseEntry(
                        image_id=image_id,
                        img_name=resolved_img_name,
                        cam_pos_world=cam_pos,
                        cam_quat_xyzw_world=cam_quat,
                        yaw_world=yaw,
                    )
                )
                seen_ids.add(image_id)
        entries.sort(key=lambda e: e.image_id)
        return entries

    def _resolve_pose_image_name(self, image_id: int, img_name: str) -> str:
        offset = int(self.pose_image_id_offset)
        if offset == 0:
            return str(img_name)
        shifted_id = int(image_id) + offset
        if shifted_id <= 0:
            return str(img_name)
        return f"image_{shifted_id:06d}.jpg"

    def _is_controller_running(self) -> bool:
        if not self.wait_for_controller:
            return True
        try:
            pubs, subs, srvs = self._ros_master.getSystemState()
        except Exception as exc:
            rospy.logwarn_throttle(
                5.0,
                "SFM controller gate: failed ROS master query (%s)",
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

    def _build_projection(self, entry: PoseEntry) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_world_cam_frd = _quat_xyzw_to_rot(
            float(entry.cam_quat_xyzw_world[0]),
            float(entry.cam_quat_xyzw_world[1]),
            float(entry.cam_quat_xyzw_world[2]),
            float(entry.cam_quat_xyzw_world[3]),
        )
        r_wc_frd = r_world_cam_frd.T
        t_wc_frd = -r_wc_frd @ entry.cam_pos_world.reshape(3)

        r_wc_cv = _AXIS_FRD_TO_CV @ r_wc_frd
        t_wc_cv = _AXIS_FRD_TO_CV @ t_wc_frd

        p_mat = self.k_mat @ np.hstack([r_wc_cv, t_wc_cv.reshape(3, 1)])
        return p_mat.astype(np.float64), r_wc_cv.astype(np.float64), t_wc_cv.astype(np.float64)

    def _passes_gates(self, entry: PoseEntry) -> bool:
        if self.last_accepted_pose is None:
            return True
        if not self.enable_motion_gating and not self.enable_max_gap_gate:
            return True

        prev = self.last_accepted_pose
        trans = float(np.linalg.norm(entry.cam_pos_world - prev.cam_pos_world))
        yaw_delta = _ang_diff_deg(entry.yaw_world, prev.yaw_world)

        if self.enable_motion_gating and trans < self.min_translation_m and yaw_delta < self.min_yaw_deg:
            return False
        if self.enable_max_gap_gate and (trans > self.max_translation_m or yaw_delta > self.max_yaw_deg):
            return False
        return True

    def _match_pair_inliers(
        self,
        ref: Keyframe,
        cur: Keyframe,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        ref_kp_idx, cur_kp_idx, pts_ref, pts_cur, num_matches, is_klt = self._match_pair_candidates(ref, cur)
        if num_matches < self.min_pair_matches:
            return None

        f_mat, inlier_mask = cv2.findFundamentalMat(
            pts_ref,
            pts_cur,
            cv2.FM_RANSAC,
            self.ransac_px,
            0.99,
        )
        if f_mat is None or inlier_mask is None:
            return None

        inlier = inlier_mask.ravel().astype(bool)
        pts_ref_in = pts_ref[inlier]
        pts_cur_in = pts_cur[inlier]
        ref_kp_idx_in = ref_kp_idx[inlier]
        cur_kp_idx_in = cur_kp_idx[inlier]
        is_klt_in = is_klt[inlier]
        if pts_ref_in.shape[0] < self.min_pair_inliers:
            return None

        if np.any(is_klt_in):
            klt_rows = np.nonzero(is_klt_in)[0]
            assigned = self._append_or_reuse_keypoints(cur, pts_cur_in[klt_rows])
            if assigned.shape[0] != klt_rows.shape[0]:
                return None
            cur_kp_idx_in = cur_kp_idx_in.copy()
            cur_kp_idx_in[klt_rows] = assigned

        valid_idx = cur_kp_idx_in >= 0
        if int(np.count_nonzero(valid_idx)) < self.min_pair_inliers:
            return None
        if not np.all(valid_idx):
            ref_kp_idx_in = ref_kp_idx_in[valid_idx]
            cur_kp_idx_in = cur_kp_idx_in[valid_idx]
            pts_ref_in = pts_ref_in[valid_idx]
            pts_cur_in = pts_cur_in[valid_idx]

        return ref_kp_idx_in, cur_kp_idx_in, pts_ref_in, pts_cur_in

    def _triangulate_observations(
        self,
        ref: Keyframe,
        cur: Keyframe,
        pts_ref_in: np.ndarray,
        pts_cur_in: np.ndarray,
    ) -> np.ndarray:
        if pts_ref_in.size == 0 or pts_cur_in.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        x_h = cv2.triangulatePoints(ref.p_mat, cur.p_mat, pts_ref_in.T, pts_cur_in.T)
        w = x_h[3, :]
        valid_w = np.abs(w) > 1e-12
        if not np.any(valid_w):
            return np.zeros((0, 3), dtype=np.float32)

        x = (x_h[:3, valid_w] / w[valid_w]).T.astype(np.float64)
        pts_ref_in = pts_ref_in[valid_w]
        pts_cur_in = pts_cur_in[valid_w]

        xc_ref = (ref.r_wc_cv @ x.T + ref.t_wc_cv.reshape(3, 1)).T
        xc_cur = (cur.r_wc_cv @ x.T + cur.t_wc_cv.reshape(3, 1)).T
        valid_depth = (xc_ref[:, 2] > self.near_depth) & (xc_cur[:, 2] > self.near_depth)
        if not np.any(valid_depth):
            return np.zeros((0, 3), dtype=np.float32)

        x = x[valid_depth]
        pts_ref_in = pts_ref_in[valid_depth]
        pts_cur_in = pts_cur_in[valid_depth]

        x_hg = np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
        proj_ref = (ref.p_mat @ x_hg.T).T
        proj_cur = (cur.p_mat @ x_hg.T).T
        uv_ref = proj_ref[:, :2] / proj_ref[:, 2:3]
        uv_cur = proj_cur[:, :2] / proj_cur[:, 2:3]
        err_ref = np.linalg.norm(uv_ref - pts_ref_in.astype(np.float64), axis=1)
        err_cur = np.linalg.norm(uv_cur - pts_cur_in.astype(np.float64), axis=1)
        valid_reproj = (err_ref <= self.max_reproj_px) & (err_cur <= self.max_reproj_px)
        if not np.any(valid_reproj):
            return np.zeros((0, 3), dtype=np.float32)

        x = x[valid_reproj]
        if x.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        v1 = x - ref.cam_pos_world.reshape(1, 3)
        v2 = x - cur.cam_pos_world.reshape(1, 3)
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        valid_norm = (n1 > 1e-12) & (n2 > 1e-12)
        if not np.any(valid_norm):
            return np.zeros((0, 3), dtype=np.float32)

        v1 = v1[valid_norm] / n1[valid_norm][:, None]
        v2 = v2[valid_norm] / n2[valid_norm][:, None]
        x = x[valid_norm]
        cosang = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
        parallax_deg = np.degrees(np.arccos(cosang))
        valid_parallax = parallax_deg >= self.min_parallax_deg
        if not np.any(valid_parallax):
            return np.zeros((0, 3), dtype=np.float32)

        x = x[valid_parallax]
        return x.astype(np.float32)

    def _triangulate_pair(self, ref: Keyframe, cur: Keyframe) -> np.ndarray:
        matched = self._match_pair_inliers(ref, cur)
        if matched is None:
            return np.zeros((0, 3), dtype=np.float32)
        _, _, pts_ref_in, pts_cur_in = matched
        return self._triangulate_observations(ref, cur, pts_ref_in, pts_cur_in)

    def _integrate_track_points(
        self,
        track_ids: List[int],
        new_points: np.ndarray,
        reproj_error_px: np.ndarray,
        parallax_deg: np.ndarray,
        views_support: np.ndarray,
        views_total: np.ndarray,
    ) -> int:
        if new_points.size == 0 or not track_ids:
            return 0
        pts = np.asarray(new_points, dtype=np.float64)
        n = min(int(pts.shape[0]), int(len(track_ids)))
        if n <= 0:
            return 0
        pts = pts[:n]
        tids = np.asarray(track_ids[:n], dtype=np.int64)
        rep = np.asarray(reproj_error_px, dtype=np.float32)[:n]
        par = np.asarray(parallax_deg, dtype=np.float32)[:n]
        vsp = np.asarray(views_support, dtype=np.float32)[:n]
        vto = np.asarray(views_total, dtype=np.float32)[:n]

        # 1) Basic validity gate.
        finite_mask = (
            np.all(np.isfinite(pts), axis=1)
            & np.isfinite(rep)
            & np.isfinite(par)
            & np.isfinite(vsp)
            & np.isfinite(vto)
        )
        if not np.any(finite_mask):
            return 0
        pts = pts[finite_mask]
        tids = tids[finite_mask]
        rep = rep[finite_mask]
        par = par[finite_mask]
        vsp = vsp[finite_mask]
        vto = vto[finite_mask]

        # 2) Building clip gate.
        if self.clip_to_building:
            in_building = np.all((pts >= self.build_min[None, :]) & (pts <= self.build_max[None, :]), axis=1)
            if not np.any(in_building):
                return 0
            pts = pts[in_building]
            tids = tids[in_building]
            rep = rep[in_building]
            par = par[in_building]
            vsp = vsp[in_building]
            vto = vto[in_building]

        if pts.shape[0] == 0:
            return 0

        added_or_updated = 0
        for tid, p, e_r, p_d, v_s, v_t in zip(
            tids.tolist(),
            pts.astype(np.float64),
            rep.tolist(),
            par.tolist(),
            vsp.tolist(),
            vto.tolist(),
        ):
            tid = int(tid)
            e_r = float(e_r)
            p_d = float(p_d)
            v_s = float(v_s)
            v_t = float(max(v_t, v_s, 1.0))
            idx = self.track_to_point_idx.get(tid, None)
            if idx is None or idx < 0 or idx >= len(self.points_xyz):
                if len(self.points_xyz) >= self.max_points:
                    continue
                idx = len(self.points_xyz)
                self.points_xyz.append(p)
                self.obs_count.append(1)
                self.point_ids.append(int(self.next_point_id))
                self.next_point_id += 1
                self.point_revs.append(1)
                self.point_track_ids.append(tid)
                self.point_views_support.append(v_s)
                self.point_views_total.append(v_t)
                self.point_reproj_error_px.append(e_r)
                self.point_parallax_deg.append(p_d)
                self.track_to_point_idx[tid] = idx
                added_or_updated += 1
                continue

            c = max(1, int(self.obs_count[idx]))
            self.points_xyz[idx] = (self.points_xyz[idx] * c + p) / float(c + 1)
            self.obs_count[idx] = c + 1
            if idx < len(self.point_revs):
                self.point_revs[idx] = max(1, int(self.point_revs[idx]) + 1)
            if idx < len(self.point_track_ids):
                self.point_track_ids[idx] = tid
            if idx < len(self.point_views_support):
                self.point_views_support[idx] = (self.point_views_support[idx] * c + v_s) / float(c + 1)
            if idx < len(self.point_views_total):
                self.point_views_total[idx] = (self.point_views_total[idx] * c + v_t) / float(c + 1)
            if idx < len(self.point_reproj_error_px):
                self.point_reproj_error_px[idx] = (self.point_reproj_error_px[idx] * c + e_r) / float(c + 1)
            if idx < len(self.point_parallax_deg):
                self.point_parallax_deg[idx] = (self.point_parallax_deg[idx] * c + p_d) / float(c + 1)
            added_or_updated += 1

        return added_or_updated

    def _integrate_points(self, new_points: np.ndarray) -> int:
        if new_points.size == 0:
            return 0
        pts = np.asarray(new_points, dtype=np.float64)

        # 1) Basic validity gate.
        finite_mask = np.all(np.isfinite(pts), axis=1)
        if not np.any(finite_mask):
            return 0
        pts = pts[finite_mask]

        # 2) Building clip gate.
        if self.clip_to_building:
            in_building = np.all((pts >= self.build_min[None, :]) & (pts <= self.build_max[None, :]), axis=1)
            if not np.any(in_building):
                return 0
            pts = pts[in_building]

        if pts.size == 0:
            return 0

        added_or_updated = 0
        for p in pts.astype(np.float64):
            if self.voxel_size <= 0.0:
                if len(self.points_xyz) >= self.max_points:
                    continue
                self.points_xyz.append(p)
                self.obs_count.append(1)
                self.point_ids.append(int(self.next_point_id))
                self.next_point_id += 1
                self.point_revs.append(1)
                self.point_track_ids.append(-1)
                self.point_views_support.append(float(self.min_track_views))
                self.point_views_total.append(float(self.min_track_views))
                self.point_reproj_error_px.append(float(self.track_confirm_reproj_px))
                self.point_parallax_deg.append(float(max(self.min_parallax_deg, 1.0)))
                added_or_updated += 1
                continue

            key = tuple(np.floor(p / self.voxel_size).astype(np.int64).tolist())
            idx = self.voxel_to_idx.get(key, None)
            if idx is None:
                if len(self.points_xyz) >= self.max_points:
                    continue
                idx = len(self.points_xyz)
                self.voxel_to_idx[key] = idx
                self.points_xyz.append(p)
                self.obs_count.append(1)
                self.point_ids.append(int(self.next_point_id))
                self.next_point_id += 1
                self.point_revs.append(1)
                self.point_track_ids.append(-1)
                self.point_views_support.append(float(self.min_track_views))
                self.point_views_total.append(float(self.min_track_views))
                self.point_reproj_error_px.append(float(self.track_confirm_reproj_px))
                self.point_parallax_deg.append(float(max(self.min_parallax_deg, 1.0)))
            else:
                c = max(1, int(self.obs_count[idx]))
                self.points_xyz[idx] = (self.points_xyz[idx] * c + p) / float(c + 1)
                self.obs_count[idx] = c + 1
                if idx < len(self.point_revs):
                    self.point_revs[idx] = max(1, int(self.point_revs[idx]) + 1)
            added_or_updated += 1
        return added_or_updated

    def _process_pose_entry(self, entry: PoseEntry, seq_idx: int) -> bool:
        img_path = os.path.join(self.images_dir, entry.img_name)
        if not os.path.isfile(img_path):
            rospy.logwarn_throttle(2.0, "SFM waiting for image file: %s", img_path)
            return False

        # Always start from first image and keep every N-th pose by sequence order.
        if (seq_idx % self.stride) != 0:
            return True
        if not self._passes_gates(entry):
            return True

        img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_raw is None:
            rospy.logwarn("SFM failed to load image: %s", img_path)
            return True
        img = self._preprocess_feature_image(img_raw)
        stream_backend = str(self.feature_backend)
        if bool(getattr(self, "alternating_feature_backend", False)):
            stream_backend = self._next_alternating_feature_stream()
            self._current_feature_stream = stream_backend
            self._mark_alternating_feature_attempt_consumed()
        else:
            self._current_feature_stream = stream_backend

        p_mat, r_wc_cv, t_wc_cv = self._build_projection(entry)
        mask = self._make_building_roi_mask(
            r_wc_cv=r_wc_cv,
            t_wc_cv=t_wc_cv,
            img_h=int(img.shape[0]),
            img_w=int(img.shape[1]),
        )
        feature_sets, kpts_uv, per_backend_counts = self._extract_feature_sets_for_image(img, mask)
        if self._current_stream_uses_klt():
            klt_grid = self._make_klt_grid_points(
                mask=mask,
                image_h=int(img.shape[0]),
                image_w=int(img.shape[1]),
            )
            klt_offset = int(kpts_uv.shape[0])
            feature_sets["klt"] = {
                "kpts_uv": klt_grid,
                "desc": None,
                "offset": int(klt_offset),
                "count": int(klt_grid.shape[0]),
            }
            per_backend_counts["klt"] = int(klt_grid.shape[0])
            if klt_grid.shape[0] > 0:
                if kpts_uv.shape[0] > 0:
                    kpts_uv = np.vstack([kpts_uv, klt_grid]).astype(np.float32, copy=False)
                else:
                    kpts_uv = klt_grid.astype(np.float32, copy=False)
        total_keypoints = int(kpts_uv.shape[0])
        if total_keypoints < self.min_keypoints:
            rospy.logwarn(
                "[WARN] SFM low features on %s (kps=%d < %d, backend_counts=%s), skipping keyframe.",
                _blue_log_image_name(entry.img_name),
                total_keypoints,
                self.min_keypoints,
                str(per_backend_counts),
            )
            return True

        img_h, img_w = int(img.shape[0]), int(img.shape[1])
        kp_presence = self._build_kp_presence_mask(kpts_uv, image_h=img_h, image_w=img_w)
        if len(self.feature_detector_order) == 1 and not self.klt_backend_enabled:
            desc = next(iter(feature_sets.values()))["desc"]
        else:
            desc = np.zeros((total_keypoints, 1), dtype=np.uint8)

        cur_kf = Keyframe(
            kf_id=int(self.next_keyframe_uid),
            image_id=entry.image_id,
            img_name=entry.img_name,
            cam_pos_world=entry.cam_pos_world.copy(),
            cam_quat_xyzw_world=entry.cam_quat_xyzw_world.copy(),
            yaw_world=float(entry.yaw_world),
            kpts_uv=kpts_uv,
            desc=desc,
            p_mat=p_mat,
            r_wc_cv=r_wc_cv,
            t_wc_cv=t_wc_cv,
            image_h=img_h,
            image_w=img_w,
            kp_presence=kp_presence,
            feature_sets=feature_sets,
            image_gray=img.copy(),
            roi_mask=None if mask is None else mask.copy(),
            stream_backend=stream_backend,
        )
        self.next_keyframe_uid += 1
        self._register_keyframe_scaffold(cur_kf)

        candidate_track_ids = set()
        if bool(getattr(self, "alternating_feature_backend", False)):
            refs = [
                kf
                for kf in self.keyframes
                if str(getattr(kf, "stream_backend", "")).lower() == str(stream_backend).lower()
            ][-self.match_neighbors :]
        else:
            refs = self.keyframes[-self.match_neighbors :]
        for ref in refs:
            if rospy.is_shutdown():
                return True
            matched = self._match_pair_inliers(ref, cur_kf)
            if matched is None:
                continue
            ref_kp_idx_in, cur_kp_idx_in, pts_ref_in, pts_cur_in = matched
            self._register_pair_observations_scaffold(
                ref=ref,
                cur=cur_kf,
                ref_kp_idx=ref_kp_idx_in,
                cur_kp_idx=cur_kp_idx_in,
            )
            for kp_idx in cur_kp_idx_in.tolist():
                tid = self.obs_to_track.get(self._obs_key(cur_kf.kf_id, int(kp_idx)), None)
                if tid is not None:
                    candidate_track_ids.add(int(tid))

        total_keypoints = int(cur_kf.kpts_uv.shape[0])
        if self._current_stream_uses_klt() and isinstance(cur_kf.feature_sets, dict) and "klt" in cur_kf.feature_sets:
            per_backend_counts["klt_total"] = int(cur_kf.feature_sets["klt"].get("count", 0) or 0)

        total_used = 0
        (
            track_ids_used,
            track_points,
            track_reproj_error_px,
            track_parallax_deg,
            track_views_support,
            track_views_total,
        ) = self._triangulate_candidate_tracks(cur_kf, sorted(candidate_track_ids))
        if track_points.shape[0] > 0:
            total_used = self._integrate_track_points(
                track_ids=track_ids_used,
                new_points=track_points,
                reproj_error_px=track_reproj_error_px,
                parallax_deg=track_parallax_deg,
                views_support=track_views_support,
                views_total=track_views_total,
            )

        self.keyframes.append(cur_kf)
        if len(self.keyframes) > self.window_size:
            self.keyframes = self.keyframes[-self.window_size :]
        self._prune_tracks_to_active_keyframes([kf.kf_id for kf in self.keyframes])

        self.last_accepted_pose = entry
        self.accepted_keyframes += 1

        if (self.accepted_keyframes % self.publish_every_kf) == 0:
            self._publish_sparse_cloud()
        if (self.accepted_keyframes % self.save_every_kf) == 0:
            self._save_sparse_npz()

        rospy.loginfo(
            "SFM kf=%d img=%s stream=%s kp=%d backend_counts=%s tri_used=%d sparse_pts=%d tracks=%d",
            self.accepted_keyframes,
            _blue_log_image_name(entry.img_name),
            str(stream_backend),
            total_keypoints,
            str(per_backend_counts),
            total_used,
            len(self.points_xyz),
            len(self.tracks),
        )
        return True

    def spin(self) -> None:
        if not self.enabled:
            return
        rate = rospy.Rate(max(self.poll_hz, 0.2))
        while not rospy.is_shutdown():
            if not self._is_controller_running():
                self.depth_controller_gate_open = False
                rospy.loginfo_throttle(
                    1.0,
                    "SFM node gated; waiting for controller node %s...",
                    self.controller_node_name,
                )
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break
                continue

            if not self.depth_controller_gate_open:
                self.depth_controller_gate_open = True
                if not os.path.isfile(self.output_path):
                    rospy.logwarn(
                        "Geometry output npz was missing after controller gate opened; recreating empty snapshot at %s",
                        self.output_path,
                    )
                    self._save_sparse_npz()
                    self._publish_sparse_cloud()
            if self.geometry_point_source == "depth" and self.depth_source_mode == "ros_topic":
                try:
                    rate.sleep()
                except rospy.ROSInterruptException:
                    break
                continue

            entries = self._load_pose_entries()
            if len(entries) < self.next_pose_offset:
                rospy.logwarn("SFM pose list rewound/truncated. Restarting from beginning.")
                self.next_pose_offset = 0
                self.keyframes = []
                self.last_accepted_pose = None
                self.accepted_keyframes = 0
                self.next_keyframe_uid = 0
                self.next_track_id = 0
                self.alternating_feature_idx = 0
                self._current_feature_stream = self.feature_backend
                self.tracks = {}
                self.obs_to_track = {}
                self.track_last_triangulated_kf = {}
                self.track_to_point_idx = {}
                if self.point_track_ids:
                    self.point_track_ids = [-1 for _ in self.point_track_ids]
                if self.reset_on_pose_rewind:
                    self.points_xyz = []
                    self.obs_count = []
                    self.point_ids = []
                    self.point_revs = []
                    self.point_track_ids = []
                    self.point_views_support = []
                    self.point_views_total = []
                    self.point_reproj_error_px = []
                    self.point_parallax_deg = []
                    self.depth_voxels = {}
                    self.depth_mesh_accumulated_triangles = {}
                    self.depth_mesh_accumulated_triangle_cvi_raw = {}
                    self.latest_depth_patch = None
                    self.depth_point_view_counts = []
                    self.depth_point_sample_counts = []
                    self.depth_input_frames = 0
                    self.depth_processed_frames = 0
                    self.voxel_to_idx = {}
                    self.next_point_id = 1
                    self._save_sparse_npz()
                    self._publish_sparse_cloud()

            while self.next_pose_offset < len(entries) and not rospy.is_shutdown():
                seq_idx = self.next_pose_offset
                entry = entries[seq_idx]
                if self.geometry_point_source == "depth":
                    consumed = self._process_saved_depth_pose_entry(entry, seq_idx)
                else:
                    consumed = self._process_pose_entry(entry, seq_idx)
                if not consumed:
                    break
                self.next_pose_offset += 1

            rate.sleep()

        # Persist one last snapshot on shutdown.
        if self.enabled:
            self._save_sparse_npz()
            self._publish_sparse_cloud()


if __name__ == "__main__":
    try:
        node = RTSFMSparseNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
