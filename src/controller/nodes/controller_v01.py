#!/usr/bin/env python3

import rospy
import os
import random
import numpy as np
import math
import threading
import subprocess
import time
import json
import struct
import colorsys
import copy
import heapq
import shutil
from tf.transformations import quaternion_matrix, quaternion_from_euler, quaternion_multiply
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
import tf.transformations as tft
from message_filters import Subscriber, ApproximateTimeSynchronizer
import sys
from sensor_msgs import point_cloud2
# Make sure Python can see the AirSim package
# sys.path.append(os.path.expanduser("~/.local/lib/python3.8/site-packages"))
sys.path.append(os.path.expanduser("/home/thanos/miniconda3/envs/IROS_2026/lib/python3.8/site-packages"))
try:
    from scipy.interpolate import make_interp_spline
except Exception:
    make_interp_spline = None
import airsim
RT_MESH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "RT_meshing"))
if RT_MESH_DIR not in sys.path:
    sys.path.append(RT_MESH_DIR)
try:
    from rt_mesher import RTBoxMesh
except Exception:
    RTBoxMesh = None

from quadrotor_msgs.msg import PositionCommand
from airsim_ros_pkgs.msg import GimbalAngleEulerCmd


_LOG_BLUE = "\033[94m"
_LOG_RESET = "\033[0m"


def _blue_log_text(value):
    return f"{_LOG_BLUE}{value}{_LOG_RESET}"


def _blue_log_image_id(image_id):
    try:
        return _blue_log_text("%06d" % int(image_id))
    except Exception:
        return _blue_log_text(str(image_id))


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


class ControllerNode:
    def __init__(self):
        rospy.init_node("controller_node")

        # Frame used for all published messages (must exist in TF)
        self.frame_id = rospy.get_param("~frame_id", "odom_local_ned")

        # Wait for one odom message (blocks until received)
        odom_msg = rospy.wait_for_message(
            "/airsim_node/drone_1/odom_local_ned",
            Odometry
        )

        # Extract quaternion
        q = odom_msg.pose.pose.orientation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w

        # Convert to Euler if needed
        self.init_roll, self.init_pitch, self.init_yaw = tft.euler_from_quaternion([qx, qy, qz, qw])
        self.home_pos = np.array(
            [
                float(odom_msg.pose.pose.position.x),
                float(odom_msg.pose.pose.position.y),
                float(odom_msg.pose.pose.position.z),
            ],
            dtype=np.float64,
        )
        self.rot_factor = 0.0275/(math.pi/2)

        rospy.loginfo("Initial orientation offset (world frame): roll=%.3f pitch=%.3f yaw=%.3f",
                    self.init_roll, self.init_pitch, self.init_yaw)
        
        # AirSim RPC client (for camera pose control)
        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client.confirmConnection()

        # Parameters (loaded from YAML / launch)
        # Paths
        self.img_dir_param     = rospy.get_param("~file_paths/img_dir", "/home/thanos/Documents/IROS_2026/")
        self.poses_dir_param   = rospy.get_param("~file_paths/poses_dir", "/home/thanos/Documents/IROS_2026/")
        self.guassians_dir_name = rospy.get_param("~file_paths/gaussians_path", "/home/thanos/Documents/IROS_2026/GaussianSpace/default_loc/")
        self.mesh_dir_name = rospy.get_param("~file_paths/mesh_path", "/home/thanos/Documents/IROS_2026/RT_meshing/Mesh_Space/RTmesh_test0/")
        self.settings_path = rospy.get_param("~file_paths/settings_path", "/home/thanos/Documents/AirSim")
        # Flight Params
        self.viewpoint_period = rospy.get_param("~flight_params/viewpoint_period", 5.0)
        self.radius           = rospy.get_param("~flight_params/radius", 5.0)
        self.altitude_z       = rospy.get_param("~flight_params/altitude_z", 5.0)
        self.cmd_rate_hz      = rospy.get_param("~flight_params/cmd_rate_hz", 50.0)
        self.camera_pitch_smooth_enabled = bool(
            rospy.get_param("~flight_params/camera_pitch_smooth_enabled", True)
        )
        self.camera_pitch_rate_rps = max(
            1e-3, float(rospy.get_param("~flight_params/camera_pitch_rate_rps", 0.8))
        )
        # Building shape
        (
            self.build_center_x,
            self.build_center_y,
            self.build_center_z,
            self.build_width,
            self.build_length,
            self.build_height,
        ) = _get_building_params_from_profile()
        # Side surface sampling
        self.side_points_total = rospy.get_param("~side_points/total", 50)
        self.side_points_margin = rospy.get_param("~side_points/margin", 0.0)
        # Flight Params
        self.cam_fx = rospy.get_param("~calibration/fx", 381.361145)
        self.cam_fy = rospy.get_param("~calibration/fy", 381.361145)
        self.cam_cx = rospy.get_param("~calibration/cx", 320.0)
        self.cam_cy = rospy.get_param("~calibration/cy", 240.0)
        self.cam_k1 = rospy.get_param("~calibration/k1", 0.0)
        self.cam_k2 = rospy.get_param("~calibration/k2", 0.0)
        self.cam_p1 = rospy.get_param("~calibration/p1", 0.0)
        self.cam_p2 = rospy.get_param("~calibration/p2", 0.0)
        self.cam_k3 = rospy.get_param("~calibration/k3", 0.0)
        self.cam_width = rospy.get_param("~calibration/width", 640.0)
        self.cam_height = rospy.get_param("~calibration/height", 320.0)
        self.viz_flip_viewpoint_z = bool(rospy.get_param("~viz_flip_viewpoint_z", True))
        # RT meshing params
        self.use_mesh = rospy.get_param("~use_mesh", False)
        self.rt_mesh_target_edge_m = float(rospy.get_param("~rt_meshing/mesh_target_edge_m", 0.5))
        self.rt_mesh_include_bottom = rospy.get_param("~rt_meshing/include_bottom", True)
        self.rt_mesh_point_stride = rospy.get_param("~rt_meshing/point_stride", 1)
        self.rt_mesh_viz_period = rospy.get_param("~rt_meshing/mesh_viz_period", 1.0)
        self.rt_mesh_color = rospy.get_param("~rt_meshing/mesh_color", [0.7, 0.7, 0.7, 0.4])
        self.rt_mesh_center_z_offset = rospy.get_param("~rt_meshing/center_z_offset", 0.0)
        self.mesh_view_ground_clearance_m = float(
            rospy.get_param("~rt_meshing/mesh_view_ground_clearance_m", 0.1)
        )
        self.mesh_view_candidate_ground_clearance_m = float(
            rospy.get_param("~rt_meshing/mesh_view_candidate_ground_clearance_m", 0.5)
        )
        self.mesh_view_sample_hz = float(rospy.get_param("~rt_meshing/mesh_view_sample_hz", 0.1))
        self.mesh_view_plan_min_cycle_s = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_plan_min_cycle_s", 20.0))
        )
        self.skip_stationary_pose_entries = bool(
            rospy.get_param("~rt_meshing/skip_stationary_pose_entries", False)
        )
        self.stationary_speed_thresh_mps = max(
            0.0, float(rospy.get_param("~rt_meshing/stationary_speed_thresh_mps", 0.05))
        )
        self.pathing_mode = str(
            rospy.get_param("~rt_meshing/pathing_mode", "initial_round")
        ).strip().lower()
        if self.pathing_mode not in ("initial_round", "baseline_circle", "atsp"):
            self.pathing_mode = "initial_round"
        self.baseline_circle_points = max(
            3, int(rospy.get_param("~rt_meshing/baseline_circle_points", 24))
        )
        self.baseline_circle_continuous = bool(
            rospy.get_param("~rt_meshing/baseline_circle_continuous", True)
        )
        self.baseline_circle_speed_mps = max(
            1.0e-3, float(rospy.get_param("~rt_meshing/baseline_circle_speed_mps", 1.0))
        )
        self.baseline_circle_start_angle_rad = math.radians(
            float(rospy.get_param("~rt_meshing/baseline_circle_start_angle_deg", 0.0))
        )
        self.baseline_circle_start_angle_mode = str(
            rospy.get_param("~rt_meshing/baseline_circle_start_angle_mode", "nearest")
        ).strip().lower()
        if self.baseline_circle_start_angle_mode not in ("nearest", "fixed"):
            self.baseline_circle_start_angle_mode = "nearest"
        self.baseline_circle_start_tol_m = max(
            1.0e-3, float(rospy.get_param("~rt_meshing/baseline_circle_start_tol_m", 0.8))
        )
        self.baseline_circle_climb_timeout_s = max(
            0.0, float(rospy.get_param("~rt_meshing/baseline_circle_climb_timeout_s", 10.0))
        )
        self.baseline_circle_stop_after_one_loop = bool(
            rospy.get_param("~rt_meshing/baseline_circle_stop_after_one_loop", True)
        )
        self.baseline_circle_land_after_one_loop = bool(
            rospy.get_param("~rt_meshing/baseline_circle_land_after_one_loop", True)
        )
        self.mesh_view_debug_logs = bool(
            rospy.get_param("~rt_meshing/mesh_view_debug_logs", False)
        )
        self.mesh_view_start_leg_debug_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_start_leg_debug_enabled", False)
        )
        self.mission_completion_progress_logs = bool(
            rospy.get_param("~rt_meshing/mission_completion_progress_logs", False)
        )
        self.mesh_view_follow_reach_only = bool(
            rospy.get_param("~rt_meshing/mesh_view_follow_reach_only", True)
        )
        self.mesh_view_reach_pos_tol_m = float(
            rospy.get_param("~rt_meshing/mesh_view_reach_pos_tol_m", 0.6)
        )
        self.mesh_view_velocity_handoff_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_velocity_handoff_enabled", True)
        )
        self.mesh_view_velocity_handoff_min_speed_mps = max(
            0.0,
            float(
                rospy.get_param(
                    "~rt_meshing/mesh_view_velocity_handoff_min_speed_mps",
                    rospy.get_param("~rt_meshing/stationary_speed_thresh_mps", 0.05),
                )
            ),
        )
        self.mesh_view_sampling_mode = str(
            rospy.get_param("~rt_meshing/mesh_view_sampling_mode", "triangles")
        ).strip().lower()
        if self.mesh_view_sampling_mode not in ("triangles", "regions"):
            self.mesh_view_sampling_mode = "triangles"
        self.mesh_view_region_count = max(1, int(rospy.get_param("~rt_meshing/mesh_view_region_count", 5)))
        self.mesh_view_region_quantize_eps = float(
            rospy.get_param("~rt_meshing/mesh_view_region_quantize_eps", 1.0e-6)
        )
        self.mesh_view_region_samples_per_region = max(
            1, int(rospy.get_param("~rt_meshing/mesh_view_region_samples_per_region", 5))
        )
        self.mesh_view_region_fan_half_angle_deg = float(
            rospy.get_param("~rt_meshing/mesh_view_region_fan_half_angle_deg", 18.0)
        )
        self.mesh_view_region_fan_d_max_scale = float(
            rospy.get_param("~rt_meshing/mesh_view_region_fan_d_max_scale", 1.5)
        )
        self.mesh_view_region_avoid_inside_building = bool(
            rospy.get_param("~rt_meshing/mesh_view_region_avoid_inside_building", True)
        )
        self.mesh_view_region_building_avoid_margin = float(
            rospy.get_param("~rt_meshing/mesh_view_region_building_avoid_margin", 0.0)
        )
        self.mesh_view_region_boundary_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_region_boundary_enabled", True)
        )
        self.mesh_view_region_boundary_width = float(
            rospy.get_param("~rt_meshing/mesh_view_region_boundary_width", 0.06)
        )
        self.mesh_view_region_boundary_alpha = float(
            rospy.get_param("~rt_meshing/mesh_view_region_boundary_alpha", 0.95)
        )
        self.mesh_view_region_drop_warn_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_region_drop_warn_enabled", False)
        )
        self.mesh_view_termination_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_termination_enabled", True)
        )
        self.mesh_view_termination_cvi_threshold = float(
            rospy.get_param("~rt_meshing/mesh_view_termination_cvi_threshold", 1.4)
        )
        self.mesh_view_termination_active_fraction_epsilon = max(
            0.0,
            float(rospy.get_param("~rt_meshing/mesh_view_termination_active_fraction_epsilon", 0.1)),
        )
        self.mesh_view_subsample = max(1, int(rospy.get_param("~rt_meshing/mesh_view_subsample", 1)))
        self.mesh_view_subsample_random = bool(rospy.get_param("~rt_meshing/mesh_view_subsample_random", False))
        self.mesh_view_d_safe_m = float(rospy.get_param("~rt_meshing/mesh_view_d_safe_m", 1.0))
        self.mesh_view_fit_margin = float(rospy.get_param("~rt_meshing/mesh_view_fit_margin", 1.05))
        self.mesh_view_max_samples = max(0, int(rospy.get_param("~rt_meshing/mesh_view_max_samples", 0)))
        self.mesh_view_vector_color = rospy.get_param("~rt_meshing/mesh_view_vector_color", [1.0, 0.2, 0.9, 0.9])
        self.mesh_view_vector_width = float(rospy.get_param("~rt_meshing/mesh_view_vector_width", 0.03))
        self.mesh_view_plan_enabled = bool(rospy.get_param("~rt_meshing/mesh_view_plan_enabled", True))
        self.mesh_view_plan_backend = str(
            rospy.get_param("~rt_meshing/mesh_view_plan_backend", "msgnn")
        ).strip().lower()
        if self.mesh_view_plan_backend == "heuristic":
            self.mesh_view_plan_backend = "msgnn"
        if self.mesh_view_plan_backend not in ("msgnn", "lkh"):
            self.mesh_view_plan_backend = "msgnn"
        self.mesh_view_plan_trans_speed_mps = max(
            1e-3, float(rospy.get_param("~rt_meshing/mesh_view_plan_trans_speed_mps", 5.0))
        )
        self.mesh_view_plan_rot_speed_rps = max(
            1e-3, float(rospy.get_param("~rt_meshing/mesh_view_plan_rot_speed_rps", 1.0))
        )
        self.mesh_view_plan_pitch_speed_rps = max(
            1e-3, float(rospy.get_param("~rt_meshing/mesh_view_plan_pitch_speed_rps", 0.8))
        )
        self.mesh_view_plan_cvi_weight = float(
            rospy.get_param("~rt_meshing/mesh_view_plan_cvi_weight", 1.0)
        )
        self.mesh_view_plan_gain_weight = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_plan_gain_weight", 1.0))
        )
        self.mesh_view_plan_base_cost = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_plan_base_cost", 1.0))
        )
        self.mesh_conf_min = float(rospy.get_param("~rt_meshing/mesh_conf_min", 1.0))
        self.mesh_conf_max = float(rospy.get_param("~rt_meshing/mesh_conf_max", 2.0))
        self.mesh_view_plan_distance_weight = max(
            0.0,
            float(rospy.get_param("~rt_meshing/mesh_view_plan_distance_weight", 0.9)),
        )
        self.mesh_view_plan_yaw_weight = max(
            0.0,
            float(rospy.get_param("~rt_meshing/mesh_view_plan_yaw_weight", 0.1)),
        )
        self.mesh_view_local_planning_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_local_planning_enabled", True)
        )
        self.mesh_view_local_orientation_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_local_orientation_enabled", True)
        )
        self.mesh_view_local_orientation_yaw_candidates = max(
            1, int(rospy.get_param("~rt_meshing/mesh_view_local_orientation_yaw_candidates", 3))
        )
        self.mesh_view_local_orientation_pitch_candidates = max(
            1, int(rospy.get_param("~rt_meshing/mesh_view_local_orientation_pitch_candidates", 3))
        )
        self.mesh_view_local_orientation_distance_weight = max(
            0.0,
            float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_distance_weight", 0.1)),
        )
        self.mesh_view_local_orientation_yaw_weight = max(
            0.0,
            float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_yaw_weight", 0.63)),
        )
        self.mesh_view_local_orientation_pitch_weight = max(
            0.0,
            float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_pitch_weight", 0.27)),
        )
        self.mesh_view_local_orientation_base_cost = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_base_cost", 1.0))
        )
        self.mesh_view_local_orientation_cvi_weight = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_cvi_weight", 0.0))
        )
        self.mesh_view_local_orientation_yaw_half_span_deg = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_yaw_half_span_deg", 90.0))
        )
        self.mesh_view_local_orientation_pitch_half_span_deg = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_pitch_half_span_deg", 30.0))
        )
        self.mesh_view_local_orientation_gain_weight = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_local_orientation_gain_weight", 1.0))
        )
        self.mesh_view_plan_atsp_multi_start = max(
            1, int(rospy.get_param("~rt_meshing/mesh_view_plan_atsp_multi_start", 8))
        )
        self.mesh_view_plan_2opt_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_plan_2opt_enabled", True)
        )
        self.mesh_view_plan_2opt_max_nodes = max(
            0, int(rospy.get_param("~rt_meshing/mesh_view_plan_2opt_max_nodes", 120))
        )
        self.mesh_view_plan_2opt_max_iters = max(
            0, int(rospy.get_param("~rt_meshing/mesh_view_plan_2opt_max_iters", 2))
        )
        self.mesh_view_plan_include_start = bool(
            rospy.get_param("~rt_meshing/mesh_view_plan_include_start", True)
        )
        self.mesh_view_plan_color = rospy.get_param(
            "~rt_meshing/mesh_view_plan_color", [1.0, 0.55, 0.1, 0.9]
        )
        self.mesh_view_plan_width = float(rospy.get_param("~rt_meshing/mesh_view_plan_width", 0.05))
        self.mesh_view_plan_point_color = rospy.get_param(
            "~rt_meshing/mesh_view_plan_point_color", [0.15, 0.95, 1.0, 0.95]
        )
        self.mesh_view_plan_point_scale = float(
            rospy.get_param("~rt_meshing/mesh_view_plan_point_scale", 0.32)
        )
        self.mesh_view_plan_lkh_binary = str(
            rospy.get_param("~rt_meshing/mesh_view_plan_lkh_binary", "/home/thanos/Documents/LKH-3.0.13/LKH")
        )
        self.mesh_view_plan_lkh_workdir = str(
            rospy.get_param(
                "~rt_meshing/mesh_view_plan_lkh_workdir",
                os.path.join(self.mesh_dir_name, "lkh_plan"),
            )
        )
        self.mesh_view_plan_lkh_scale = max(
            1, int(rospy.get_param("~rt_meshing/mesh_view_plan_lkh_scale", 1000))
        )
        self.mesh_view_plan_lkh_max_candidates = max(
            2, int(rospy.get_param("~rt_meshing/mesh_view_plan_lkh_max_candidates", 12))
        )
        self.mesh_view_plan_lkh_runs = max(
            1, int(rospy.get_param("~rt_meshing/mesh_view_plan_lkh_runs", 10))
        )
        self.mesh_view_plan_lkh_max_trials = max(
            1, int(rospy.get_param("~rt_meshing/mesh_view_plan_lkh_max_trials", 10000))
        )
        self.mesh_view_plan_lkh_timeout_s = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_plan_lkh_timeout_s", 15.0))
        )
        self.mesh_view_plan_lkh_trace_level = max(
            0, int(rospy.get_param("~rt_meshing/mesh_view_plan_lkh_trace_level", 0))
        )
        self.mesh_view_plan_log_timing = bool(
            rospy.get_param("~rt_meshing/mesh_view_plan_log_timing", True)
        )
        obstacle_mode_raw = str(
            rospy.get_param("~rt_meshing/mesh_view_obstacle_mode", "surface_graph")
        ).strip().lower()
        legacy_surface_graph_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_intersection_enabled", True)
        )
        legacy_voxel_enabled = bool(
            rospy.get_param("~rt_meshing/mesh_view_voxel_enabled", False)
        )
        if obstacle_mode_raw in ("", "auto"):
            if legacy_voxel_enabled and not legacy_surface_graph_enabled:
                obstacle_mode_raw = "voxel_grid"
            else:
                obstacle_mode_raw = "surface_graph"
        if obstacle_mode_raw not in ("surface_graph", "voxel_grid"):
            rospy.logwarn(
                "Unknown mesh_view_obstacle_mode='%s'; defaulting to surface_graph.",
                obstacle_mode_raw,
            )
            obstacle_mode_raw = "surface_graph"
        self.mesh_view_obstacle_mode = obstacle_mode_raw
        self.mesh_view_surface_graph_enabled = self.mesh_view_obstacle_mode == "surface_graph"
        self.mesh_view_intersection_quantize_eps = float(
            rospy.get_param(
                "~rt_meshing/mesh_view_intersection_quantize_eps",
                rospy.get_param("~rt_meshing/mesh_view_region_quantize_eps", 1.0e-6),
            )
        )
        self.mesh_view_intersection_bvh_leaf_size = max(
            1,
            int(rospy.get_param("~rt_meshing/mesh_view_intersection_bvh_leaf_size", 8)),
        )
        self.mesh_view_surface_graph_astar_alpha_start = max(
            0.1, float(rospy.get_param("~rt_meshing/mesh_view_surface_graph_astar_alpha_start", 1.0))
        )
        self.mesh_view_surface_graph_astar_alpha_step = max(
            1.0e-3, float(rospy.get_param("~rt_meshing/mesh_view_surface_graph_astar_alpha_step", 0.5))
        )
        self.mesh_view_surface_graph_clearance_m = max(
            0.0,
            float(
                rospy.get_param(
                    "~rt_meshing/mesh_view_surface_graph_clearance_m",
                    rospy.get_param("~rt_meshing/mesh_view_voxel_clearance_m", 0.5),
                )
            ),
        )
        self.mesh_view_surface_graph_path_spacing_m = max(
            0.05,
            float(rospy.get_param("~rt_meshing/mesh_view_surface_graph_path_spacing_m", 0.5)),
        )
        self.mesh_view_voxel_enabled = self.mesh_view_obstacle_mode == "voxel_grid"
        self.mesh_view_voxel_size_m = max(
            1e-3, float(rospy.get_param("~rt_meshing/mesh_view_voxel_size_m", 0.25))
        )
        self.mesh_view_voxel_aabb_margin_m = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_voxel_aabb_margin_m", 3.0))
        )
        self.mesh_view_voxel_clearance_m = max(
            0.0, float(rospy.get_param("~rt_meshing/mesh_view_voxel_clearance_m", 0.5))
        )
        self.mesh_view_voxel_log_timing = bool(
            rospy.get_param("~rt_meshing/mesh_view_voxel_log_timing", True)
        )
        if self.mesh_view_obstacle_mode == "surface_graph" and make_interp_spline is None:
            rospy.logwarn(
                "SciPy spline interpolation unavailable; surface-graph detours will fall back to sampled polylines."
            )
        self.rt_mesh_reset_on_start = rospy.get_param("~rt_meshing/reset_mesh_on_start", False)
        self.sfm_reset_on_start = rospy.get_param("~rt_meshing/sfm_reset_on_start", True)
        self.rt_mesh_sparse_path = rospy.get_param(
            "~rt_meshing/mesh_update_sparse_path",
            os.path.join(self.mesh_dir_name, "sparse_latest.npz"),
        )
        self.rt_mesh_depth_path = rospy.get_param(
            "~rt_meshing/mesh_update_depth_path",
            os.path.join(self.mesh_dir_name, "depth_latest.npz"),
        )
        self.geometry_point_source = str(rospy.get_param("~rt_meshing/geometry_point_source", "sfm")).strip().lower()
        self.depth_source_mode = str(rospy.get_param("~rt_meshing/depth_source_mode", "ros_topic")).strip().lower()
        self.depth_topic = str(
            rospy.get_param("~rt_meshing/depth_topic", "/airsim_node/drone_1/front_center/DepthPerspective")
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
        self.depth_frame_stride = max(1, int(rospy.get_param("~rt_meshing/depth_frame_stride", 1)))
        self.depth_dir = rospy.get_param(
            "~rt_meshing/depth_dir",
            os.path.join(self.img_dir_param, "depth"),
        )
        self.save_depth_frames = self.geometry_point_source == "depth" and self.depth_source_mode == "saved_frames"
        self.publish_depth_captures = self.geometry_point_source == "depth" and self.depth_source_mode == "ros_topic"
        self.pair_depth_with_rgb = bool(self.save_depth_frames or self.publish_depth_captures)
        self.rt_mesh_sfm_output_path = rospy.get_param(
            "~rt_meshing/sfm_output_npz",
            os.path.join(self.mesh_dir_name, "sparse_latest.npz"),
        )
        self.rt_mesh_depth_output_path = rospy.get_param(
            "~rt_meshing/depth_output_npz",
            os.path.join(self.mesh_dir_name, "depth_latest.npz"),
        )
        self.align_image_pose_logs_on_shutdown = bool(
            rospy.get_param("~rt_meshing/align_image_pose_logs_on_shutdown", True)
        )
        self._image_pose_alignment_done = False
        self.convert_poses_file_on_finalize = bool(
            rospy.get_param("~rt_meshing/convert_poses_file_on_finalize", True)
        )
        self.convert_poses_file_script = os.path.abspath(
            os.path.expanduser(
                rospy.get_param(
                    "~rt_meshing/convert_poses_file_script",
                    "/home/thanos/Documents/IROS_2026/testing/convert_poses_file.py",
                )
            )
        )
        self._pose_file_conversion_done = False

        # Directories for logging
        self.image_dir = os.path.join(self.img_dir_param, "images")
        self.pose_dir  = os.path.join(self.poses_dir_param, "poses")
        self.rviz_replay_dir = os.path.join(
            self.image_dir,
            str(rospy.get_param("~rt_meshing/rviz_replay_dir_name", "rviz_replay")),
        )
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.pose_dir,  exist_ok=True)
        if self.save_depth_frames:
            os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.mesh_dir_name, exist_ok=True)
        try:
            os.makedirs(self.mesh_view_plan_lkh_workdir, exist_ok=True)
        except Exception as e:
            rospy.logwarn(
                "Failed to create LKH workdir '%s': %s. Falling back to MSGNN planner.",
                self.mesh_view_plan_lkh_workdir,
                str(e),
            )
            self.mesh_view_plan_backend = "msgnn"
        if self.mesh_view_plan_backend == "lkh" and not os.path.isfile(self.mesh_view_plan_lkh_binary):
            rospy.logwarn(
                "LKH binary '%s' not found. Falling back to MSGNN planner.",
                self.mesh_view_plan_lkh_binary,
            )
            self.mesh_view_plan_backend = "msgnn"
        self.pose_filename = os.path.join(self.pose_dir, f"poses.txt")
        self.pose_filename_degrees = os.path.join(self.pose_dir, f"poses_degrees.txt")
        self.sparse_cameras = os.path.join(self.pose_dir, f"cameras.txt")
        self.actual_path_filename = os.path.join(self.pose_dir, "actual_flight_path.txt")
        self.mission_metrics_filename = os.path.join(self.pose_dir, "mission_metrics.txt")
        self.mission_metrics_json_filename = os.path.join(self.pose_dir, "mission_metrics.json")
        self.gaussians_filename = os.path.join(self.guassians_dir_name, f"gaussians_latest.txt")
        self.mesh_filename = os.path.join(self.mesh_dir_name, "mesh_latest.npz")
        self.settings_filename = os.path.join(self.settings_path, f"settings.json")
        rospy.on_shutdown(self._align_image_pose_logs_on_shutdown)
        self._rviz_replay_saved = False
        self._rviz_replay_cache = {
            "sampled_view_vectors_points": np.zeros((0, 3), dtype=np.float32),
            "trajectory_points": np.zeros((0, 3), dtype=np.float32),
            "trajectory_skipped_points": np.zeros((0, 3), dtype=np.float32),
            "viewpoint_positions": np.zeros((0, 3), dtype=np.float32),
            "global_viewpoint_positions": np.zeros((0, 3), dtype=np.float32),
            "current_pose_point": np.zeros((0, 3), dtype=np.float32),
            "region_boundaries_points": np.zeros((0, 3), dtype=np.float32),
            "region_boundaries_colors": np.zeros((0, 4), dtype=np.float32),
        }

        # Optional: remove old run artifacts in images/ and poses/ when reset flags indicate fresh run.
        if self.rt_mesh_reset_on_start or self.sfm_reset_on_start:
            self._clear_image_pose_files_on_reset()

        # Gaussian data cache
        self.gauss_pos = None
        self.gauss_opacity = None
        self.gauss_scale = None
        self.gauss_cvi = None
        self.gaussians_mtime = None
        self.gaussian_viz_period = rospy.get_param("~gaussians_viz_period", 1.0)
        self.mesh_triangles = None
        self.mesh_points = None
        self.mesh_tri_ids = None
        self.mesh_tri_cvi = None
        self.mesh_tri_cvi_raw = None
        self.mesh_tri_cvi_effective = None
        self.mesh_tri_conf = None
        self.mesh_tri_vertex_conf = None
        self.mesh_tri_inactive_bottom = None
        self.mesh_next_tri_id = None
        self.mesh_l0 = None
        self.mesh_mtime = None
        self.mesh_view_voxel_grid = None
        self.mesh_view_voxel_origin = None
        self.mesh_view_voxel_bounds_min = None
        self.mesh_view_voxel_bounds_max = None
        self.mesh_view_voxel_shape = None
        self.mesh_view_voxel_size = None
        self.mesh_view_voxel_occupied_count = 0
        self.mesh_view_voxel_mesh_mtime = None
        self.mesh_view_intersection_cache = None
        self.mesh_view_intersection_mesh_mtime = None
        self.mesh_view_intersection_pairs = {}

        # Optional: reset mesh (and CVI/state stored in mesh_latest.npz) on controller start.
        if self.use_mesh and self.rt_mesh_reset_on_start:
            rospy.logwarn("reset_mesh_on_start=true: re-initializing mesh (overwriting %s)", self.mesh_filename)
            self._init_mesh()
            self._clear_stale_sparse_on_mesh_reset()
            try:
                self.mesh_mtime = os.path.getmtime(self.mesh_filename)
            except Exception:
                self.mesh_mtime = None
        
        # Import from files and also initialize some things
        # --- Load camera-to-body extrinsics from AirSim settings.json ---
        # settings.json: Vehicles -> drone_1 -> Cameras -> front_center :contentReference[oaicite:1]{index=1}
        try:
            with open(self.settings_filename, "r") as f:
                settings = json.load(f)

            cam_cfg = settings["Vehicles"]["drone_1"]["Cameras"]["front_center"]

            # Position of camera in body frame (NED coordinates)
            self.cam_offset_body = np.array(
                [cam_cfg.get("X", 0.0),
                 cam_cfg.get("Y", 0.0),
                 cam_cfg.get("Z", 0.0)],
                dtype=np.float64,
            )

            # Orientation of camera in body frame (degrees)
            self.cam_rpy_deg = np.array(
                [cam_cfg.get("Roll", 0.0),
                 cam_cfg.get("Pitch", 0.0),
                 cam_cfg.get("Yaw", 0.0)],
                dtype=np.float64,
            )

            rospy.loginfo(
                "Loaded camera extrinsics from settings.json: "
                "offset_body = %s, rpy_deg = %s",
                self.cam_offset_body,
                self.cam_rpy_deg,
            )

        except Exception as e:
            rospy.logwarn(
                "Failed to load camera extrinsics from %s: %s. "
                "Using fallback (0.125, 0.0, 0.0) and zero RPY.",
                self.settings_filename,
                str(e),
            )
            # Default values otherwise
            self.cam_offset_body = np.array([0.125, 0.0, 0.0], dtype=np.float64)
            self.cam_rpy_deg = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Precompute camera orientation (body -> camera) as quaternion
        roll_rad  = math.radians(self.cam_rpy_deg[0])
        pitch_rad = math.radians(self.cam_rpy_deg[1])
        yaw_rad   = math.radians(self.cam_rpy_deg[2])

        # TF uses [x, y, z, w] order
        self.q_body_cam = quaternion_from_euler(
            roll_rad, pitch_rad, yaw_rad, axes="sxyz"
        )

        try:
            with open(self.pose_filename, "w") as f:
                # positive y, happens to be the starting heading of the drone.
                f.write("# Probably lef-handed UE4 system with -z. I can't really know handeness because I can't interpret the quaternions. They have already handeness encoded in them and converting them to degrees assumes handeness.\n")
                f.write("# image_id qw qx qy qz tx ty tz 1 image_name\n")
            with open(self.pose_filename_degrees, "w") as f:
                f.write("# Probably lef-handed UE4 system with -z. I can't really know handeness because I can't interpret the quaternions. They have already handeness encoded in them and converting them to degrees assumes handeness.\n")
                f.write("# image_id roll_deg pitch_deg yaw_deg tx ty tz image_name\n")
        except Exception as e:
                rospy.logwarn("Failed to create sparse images file")

        try:
            with open(self.sparse_cameras, "w") as f:
                f.write("# image_id PINHOLE width height fx fy cx cy\n")
        except Exception as e:
                rospy.logwarn("Failed to create sparse cameras file")
        

        self.bridge = CvBridge()
        self.img_cnt = 0
        self.vp_cnt = 0
        self.last_img_cnt = 0
        self.drone_updated = True
        self.camera_updated = True
        self.last_gaussian_update_time = rospy.Time.now()

        # Publisher for position commands (to px4ctrl)
        self.pos_cmd_pub = rospy.Publisher(
            "/planning/pos_cmd",
            PositionCommand,
            queue_size=10
        )

        # Publishers for visualization and downstream nodes
        self.viewpoints_pub = rospy.Publisher(
            "controller/viewpoints",
            PoseArray,
            queue_size=1,
        )
        self.best_view_pub = rospy.Publisher(
            "controller/best_viewpoint",
            PoseStamped,
            queue_size=1,
        )
        self.camera_pose_pub = rospy.Publisher(
            "controller/camera_pose",
            PoseStamped,
            queue_size=1,
        )
        self.depth_capture_pub = rospy.Publisher(
            self.depth_capture_topic,
            Image,
            queue_size=2,
        )
        self.depth_capture_pose_pub = rospy.Publisher(
            self.depth_capture_pose_topic,
            PoseStamped,
            queue_size=2,
        )
        self.gaussian_cloud_pub = rospy.Publisher(
            "controller/gaussian_centers",
            PointCloud2,
            queue_size=1,
            latch=True,
        )
        self.mesh_points_pub = rospy.Publisher(
            "controller/mesh_points",
            PointCloud2,
            queue_size=1,
            latch=True,
        )
        self.mesh_tri_pub = rospy.Publisher(
            "controller/mesh_triangles",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_vectors_pub = rospy.Publisher(
            "controller/sampled_view_vectors",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_trajectory_pub = rospy.Publisher(
            "controller/trajectory",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_skipped_trajectory_pub = rospy.Publisher(
            "controller/trajectory_skipped",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_viewpoint_positions_pub = rospy.Publisher(
            "controller/viewpoint_positions",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_global_viewpoint_positions_pub = rospy.Publisher(
            "controller/global_viewpoint_positions",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_current_pose_point_pub = rospy.Publisher(
            "controller/current_pose_point",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_actual_flight_path_pub = rospy.Publisher(
            "controller/actual_flight_path",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_start_leg_points_pub = rospy.Publisher(
            "controller/start_leg_checked_points",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.mesh_view_region_boundaries_pub = rospy.Publisher(
            "controller/region_boundaries",
            Marker,
            queue_size=1,
            latch=True,
        )
        self.side_points_pub = rospy.Publisher(
            "controller/side_points",
            PoseArray,
            queue_size=1,
            latch=True,
        )

        # RGB capture stays independent; depth uses a side synchronizer with the same RGB stream.
        self.latest_odom = None
        self._capture_images_enabled = True
        self.arrival_cnt = 0
        self.block_start_time = rospy.Time.now()
        self._depth_capture_lock = threading.Lock()
        self._depth_capture_records = {}
        self._depth_capture_cache_limit = 128

        self.image_sub = None
        self.depth_rgb_sub = None
        self.depth_sub = None
        self.image_depth_sync = None
        if self.pair_depth_with_rgb:
            self.image_sub = Subscriber(
                "/airsim_node/drone_1/front_center/Scene",
                Image,
                queue_size=1,
            )
            self.image_sub.registerCallback(self.image_callback)
            self.depth_rgb_sub = self.image_sub
            self.depth_sub = Subscriber(
                self.depth_topic,
                Image,
                queue_size=1,
            )
            self.image_depth_sync = ApproximateTimeSynchronizer(
                [self.depth_rgb_sub, self.depth_sub],
                queue_size=20,
                slop=float(self.depth_capture_sync_slop_s),
            )
            self.image_depth_sync.registerCallback(self.depth_rgb_callback)
        else:
            self.image_sub = rospy.Subscriber(
                "/airsim_node/drone_1/front_center/Scene",
                Image,
                self.image_callback,
                queue_size=1
            )

        rospy.Subscriber(
            "/airsim_node/drone_1/odom_local_ned",
            Odometry,
            self.odom_callback,
            queue_size=10
        )

        # Target state
        self.target_pos = (0.0, 0.0, self.altitude_z)  # x, y, z
        self.target_velocity = (0.0, 0.0, 0.0)  # x, y, z
        self.target_angles = (0.0, 0.0, 0.0)  # roll pitch yaw
        self.camera_pitch_cmd_current = 0.0
        self._camera_pitch_cmd_last_t = None

        # Timing for viewpoint changes
        self.last_drone_update_time = rospy.Time.now()
        self._planned_samples_lock = threading.Lock()
        self._planned_samples = []
        self._planned_idx = 0
        self._planned_exec_waypoints = []
        self._planned_exec_idx = 0
        self._planned_exec_direction = 1
        self._planned_version = 0
        self._active_plan_version = -1
        self._atsp_active_target = False
        self._atsp_active_sample = None
        self._atsp_active_nbv_sample = None
        self._mesh_view_planner_thread = None
        self._mesh_view_first_success_done = False
        self._mesh_view_last_success_cycle_t = None
        self._start_leg_diag_logged_version = -1
        self._planner_timing_num_updates = 0
        self._planner_timing_stats_pct = {}
        self._planner_timing_stats_ms = {}
        self._planner_mode_timing_stats_ms = {}
        self._mesh_view_local_orientation_last_ms = 0.0
        self._mesh_view_local_orientation_last_segments = 0
        self._planner_start_last_source = "unknown"
        self._planner_start_last_point = None
        self._nearest_exec_diag_logged_version = -1
        self._baseline_circle_phase = "climb"
        self._baseline_circle_target_initialized = False
        self._baseline_circle_sweep_started_t = None
        self._baseline_circle_radius_m = 0.0
        self._baseline_circle_start_angle_rad_effective = None
        self._baseline_circle_climb_target = None
        self._baseline_circle_phase_started_t = time.monotonic()
        self._mesh_view_planner_started_t = time.monotonic()
        self._mission_started_t = float(self._mesh_view_planner_started_t)
        self._mesh_view_termination_reached = False
        self._planner_triangle_activity_last = {}
        self._mission_completion_active = False
        self._mission_completion_reason = ""
        self._mission_completion_logged = False
        self._mission_completion_stage = "idle"
        self._mission_completion_hold_z = float(self.home_pos[2])
        self._mission_completion_activity_snapshot = {}
        self._mission_actual_path_length_m = 0.0
        self._mission_actual_path_points = [self.home_pos.copy()]
        self._mission_actual_last_odom_point = self.home_pos.copy()
        self._mission_actual_last_saved_point = self.home_pos.copy()
        self._mission_path_record_step_m = 0.05
        self._clear_actual_flight_path_marker()

        # Side sampling disabled in favor of gaussian CVI visualization.
        # self.side_points = self.sample_side_points(
        #     self.side_points_total,
        #     self.side_points_margin,
        # )
        # if self.side_points:
        #     self.publish_side_points(self.side_points)

        rospy.loginfo(
            "ControllerNode initialized. pathing_mode=%s planner_backend=%s local_orient=%s",
            self.pathing_mode,
            self.mesh_view_plan_backend,
            self._local_planning_mode_label(),
        )
        if (not self.use_mesh) and self.gaussian_viz_period > 0.0:
            self._gaussian_viz_timer = rospy.Timer(
                rospy.Duration(self.gaussian_viz_period),
                self._gaussian_viz_timer_cb,
            )
        else:
            self._gaussian_viz_timer = None

        if self.use_mesh and self.rt_mesh_viz_period > 0.0:
            self._mesh_viz_timer = rospy.Timer(
                rospy.Duration(self.rt_mesh_viz_period),
                self._mesh_viz_timer_cb,
            )
        else:
            self._mesh_viz_timer = None

        if self.use_mesh and self.mesh_view_sample_hz > 0.0:
            self._mesh_view_planner_thread = threading.Thread(
                target=self._mesh_view_planner_loop,
                daemon=True,
            )
            self._mesh_view_planner_thread.start()
            self._mesh_view_sampling_timer = None
        else:
            self._mesh_view_sampling_timer = None

    def load_gaussians_from_txt(self, path):
        """
        Load gaussians from MonoGS TXT file, if present.
        Expected column layout per line (current exporter):
        x y z sx sy sz r g b opacity   (10 floats)

        If a legacy 14+ column file is detected, we fall back to the
        old interpretation (opacity pre-sigmoid, scales in log-space).
        """
        print("Loading gaussians txt from controller_v01")
        if not os.path.isfile(path):
            # File does not exist yet → no gaussians
            return

        mtime = os.path.getmtime(path)
        # Only reload if file changed since last time
        if self.gaussians_mtime is not None and mtime == self.gaussians_mtime:
            return

        try:
            # np.loadtxt will ignore lines starting with '#'
            data = np.loadtxt(path, comments="#", dtype=np.float32)
        except Exception as e:
            rospy.logwarn("Failed to load gaussians TXT '%s': %s", path, str(e))
            return

        if data.ndim == 1:
            data = data[None, :]

        num_cols = data.shape[1]

        if num_cols >= 14:
            # Legacy MonoGS format (pre-sigmoid opacity, log-scale)
            pos = data[:, 0:3]  # x,y,z
            opacity_param = data[:, 9]  # column 10
            opacity = 1.0 / (1.0 + np.exp(-opacity_param))
            scale_log = data[:, 10:13]  # cols 11,12,13
            scale = np.exp(scale_log)
            cvi = None
        elif num_cols >= 10:
            # Current exporter: x y z sx sy sz r g b opacity
            pos = data[:, 0:3]
            scale = data[:, 3:6]
            opacity_raw = data[:, 9]
            cvi = data[:, 10] if num_cols >= 11 else None

            # If opacity already in [0,1], keep it; otherwise sigmoid.
            if np.any(opacity_raw < 0.0) or np.any(opacity_raw > 1.0):
                opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
            else:
                opacity = opacity_raw
        else:
            rospy.logwarn(
                "Gaussians TXT has %d columns, expected at least 10. Shape: %s",
                num_cols,
                data.shape,
            )
            return

        self.gauss_pos = pos
        self.gauss_opacity = opacity
        self.gauss_scale = scale
        self.gauss_cvi = cvi
        self.gaussians_mtime = mtime

        rospy.loginfo(
            "Loaded %d gaussians from '%s'",
            self.gauss_pos.shape[0],
            path,
        )

        self.publish_gaussian_cloud()

    def _gaussian_viz_timer_cb(self, _event):
        self.load_gaussians_from_txt(self.gaussians_filename)

    def _mesh_viz_timer_cb(self, _event):
        if self._load_mesh_from_file():
            pass
        elif self.mesh_triangles is None or self.mesh_points is None:
            self._init_mesh()
        self.publish_mesh()

    def _mesh_view_planner_loop(self):
        idle_sleep_s = 1.0 / max(1e-6, float(self.mesh_view_sample_hz))
        while not rospy.is_shutdown():
            t_loop_start = time.perf_counter()
            ran_cycle = False
            try:
                ran_cycle = bool(self._mesh_view_planning_cycle())
            except Exception as e:
                rospy.logwarn("Mesh-view planning cycle failed: %s", str(e))
            if ran_cycle:
                if not bool(self._mesh_view_first_success_done):
                    self._mesh_view_first_success_done = True
                else:
                    elapsed_s = float(time.perf_counter() - t_loop_start)
                    wait_s = max(0.0, float(self.mesh_view_plan_min_cycle_s) - elapsed_s)
                    if wait_s > 0.0:
                        rospy.sleep(wait_s)
                    else:
                        rospy.sleep(1.0e-3)
            else:
                rospy.sleep(idle_sleep_s)

    def _mesh_view_sampling_timer_cb(self, _event):
        self._mesh_view_planning_cycle()

    def _mesh_view_planning_cycle(self):
        if bool(self._mission_completion_active):
            return False
        now_t = time.monotonic()
        if (
            self._mesh_view_last_success_cycle_t is not None
            and float(self.mesh_view_plan_min_cycle_s) > 0.0
        ):
            elapsed_since_success = float(now_t - float(self._mesh_view_last_success_cycle_t))
            if elapsed_since_success + 1.0e-9 < float(self.mesh_view_plan_min_cycle_s):
                remaining = float(self.mesh_view_plan_min_cycle_s - elapsed_since_success)
                if self.mesh_view_debug_logs:
                    rospy.loginfo_throttle(
                        1.0,
                        "Mesh-view planner gated by min cycle: remaining=%.2fs min_cycle=%.2fs",
                        remaining,
                        float(self.mesh_view_plan_min_cycle_s),
                    )
                return False

        t_cycle_start = time.perf_counter()
        t_load_mesh_ms = 0.0
        t_voxelize_ms = 0.0
        t_intersection_cache_ms = 0.0
        t_sample_ms = 0.0
        t_publish_vectors_ms = 0.0
        t_intersection_pairs_ms = 0.0
        t_route_costs_ms = 0.0
        t_route_solver_ms = 0.0
        t_exec_paths_ms = 0.0
        t_local_orientation_ms = 0.0
        t_publish_traj_ms = 0.0
        t_commit_ms = 0.0
        voxel_stats = None
        intersection_stats = None

        t0 = time.perf_counter()
        if self._load_mesh_from_file():
            pass
        elif self.mesh_triangles is None:
            return False
        t_load_mesh_ms = (time.perf_counter() - t0) * 1000.0

        if self.mesh_triangles is not None:
            activity = self._planner_triangle_activity(self.mesh_triangles)
            self._planner_triangle_activity_last = dict(activity)
            eligible_count = int(activity.get("eligible_count", 0))
            active_count = int(activity.get("active_count", 0))
            active_fraction = float(activity.get("active_fraction", 0.0))
            if eligible_count <= 0:
                self._mesh_view_termination_reached = False
            if not self.mesh_view_termination_enabled:
                self._mesh_view_termination_reached = False
            if self.mesh_view_termination_enabled and eligible_count > 0:
                terminate_now = active_fraction <= float(self.mesh_view_termination_active_fraction_epsilon)
                if terminate_now:
                    if not bool(self._mesh_view_termination_reached):
                        rospy.loginfo(
                            "Mesh-view termination reached: active=%d eligible=%d fraction=%.4f threshold=%.4f cvi_threshold=%.4f mean=%.4f std=%.4f",
                            int(active_count),
                            int(eligible_count),
                            float(active_fraction),
                            float(self.mesh_view_termination_active_fraction_epsilon),
                            float(self.mesh_view_termination_cvi_threshold),
                            float(activity.get("eligible_cvi_mean", float("nan"))),
                            float(activity.get("eligible_cvi_std", float("nan"))),
                        )
                    self._mesh_view_termination_reached = True
                    self._begin_mission_completion(reason="termination", activity=activity)
                    return False
                if bool(self._mesh_view_termination_reached):
                    rospy.loginfo(
                        "Mesh-view termination cleared: active=%d eligible=%d fraction=%.4f threshold=%.4f",
                        int(active_count),
                        int(eligible_count),
                        float(active_fraction),
                        float(self.mesh_view_termination_active_fraction_epsilon),
                    )
                self._mesh_view_termination_reached = False

        t0 = time.perf_counter()
        if self.mesh_view_obstacle_mode == "voxel_grid":
            try:
                voxel_stats = self._build_mesh_view_voxel_grid()
            except Exception as e:
                self._clear_mesh_view_voxel_grid()
                rospy.logwarn("Mesh-view voxelization failed: %s", str(e))
        else:
            self._clear_mesh_view_voxel_grid()
        t_voxelize_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if self.mesh_view_obstacle_mode == "surface_graph":
            try:
                intersection_stats = self._build_mesh_view_intersection_cache()
            except Exception as e:
                self._clear_mesh_view_intersection_cache()
                rospy.logwarn("Mesh-view intersection prep failed: %s", str(e))
        else:
            self._clear_mesh_view_intersection_cache()
        t_intersection_cache_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if self.mesh_view_sampling_mode == "regions":
            samples = self.sample_region_viewpoints()
        else:
            self._clear_region_boundaries_marker()
            samples = self.sample_triangle_viewpoints()
        samples_for_plan = self._inject_active_sample(samples)
        for plan_idx, sample in enumerate(list(samples_for_plan) if samples_for_plan is not None else []):
            try:
                sample["_plan_idx"] = int(plan_idx)
            except Exception:
                pass
        t_sample_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        self.publish_sampled_view_vectors(samples_for_plan)
        t_publish_vectors_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if self.mesh_view_obstacle_mode == "surface_graph":
            try:
                pair_stats = self._measure_sample_pair_intersections(samples_for_plan)
                if intersection_stats is None:
                    intersection_stats = {}
                intersection_stats.update(pair_stats)
            except Exception as e:
                rospy.logwarn("Mesh-view sample-pair intersections failed: %s", str(e))
        t_intersection_pairs_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        cost_data = None
        if self.mesh_view_plan_enabled:
            t_cost = time.perf_counter()
            cost_data = self._build_viewpoint_plan_costs(samples_for_plan)
            if cost_data is None:
                return False
            t_route_costs_ms = (time.perf_counter() - t_cost) * 1000.0
            t_solver = time.perf_counter()
            if self.mesh_view_plan_backend == "lkh":
                ordered = self._plan_viewpoint_path_lkh(samples_for_plan, cost_data=cost_data)
            else:
                ordered = self.plan_viewpoint_path(samples_for_plan, cost_data=cost_data)
            t_route_solver_ms = (time.perf_counter() - t_solver) * 1000.0
        else:
            ordered = list(samples_for_plan)
        if self.mesh_view_debug_logs and isinstance(cost_data, dict) and len(samples_for_plan) > 0:
            try:
                start_cost = np.asarray(cost_data.get("start_cost", []), dtype=np.float64).reshape(-1)
                start_pt = np.asarray(self._planner_start_last_point, dtype=np.float64).reshape(3) if self._planner_start_last_point is not None else None
                sample0_pt = np.asarray(samples_for_plan[0].get("origin", None), dtype=np.float64).reshape(3)
                ordered0_pt = (
                    np.asarray(ordered[0].get("origin", None), dtype=np.float64).reshape(3)
                    if len(ordered) > 0
                    else None
                )
                top_k = min(3, int(start_cost.size))
                top_parts = []
                if top_k > 0:
                    for idx in np.argsort(start_cost)[:top_k].tolist():
                        top_parts.append("%d:%.6f" % (int(idx), float(start_cost[int(idx)])))
                ordered0_plan_idx = -1
                ordered0_start_cost = float("nan")
                if len(ordered) > 0:
                    try:
                        ordered0_plan_idx = int(ordered[0].get("_plan_idx", 0))
                    except Exception:
                        ordered0_plan_idx = 0
                    if 0 <= int(ordered0_plan_idx) < int(start_cost.size):
                        ordered0_start_cost = float(start_cost[int(ordered0_plan_idx)])
                rospy.loginfo(
                    "Planner start diagnostics: source=%s start_mesh=%s sample0_mesh=%s ordered0_mesh=%s ordered0_plan_idx=%d ordered0_start_cost=%.6f top_start_costs=%s",
                    str(self._planner_start_last_source),
                    self._fmt_point_xyz(start_pt),
                    self._fmt_point_xyz(sample0_pt),
                    self._fmt_point_xyz(ordered0_pt),
                    int(ordered0_plan_idx),
                    float(ordered0_start_cost) if np.isfinite(ordered0_start_cost) else float("nan"),
                    "[" + ", ".join(top_parts) + "]",
                )
            except Exception as diag_exc:
                rospy.logwarn("Planner start diagnostics failed: %s", str(diag_exc))
        t_exec = time.perf_counter()
        exec_waypoints = self._build_execution_waypoints(ordered, cost_data=cost_data)
        t_exec_total_ms = (time.perf_counter() - t_exec) * 1000.0
        t_local_orientation_ms = float(max(0.0, self._mesh_view_local_orientation_last_ms))
        t_exec_paths_ms = float(max(0.0, t_exec_total_ms - t_local_orientation_ms))

        t0 = time.perf_counter()
        if self.mesh_view_plan_enabled:
            self.publish_sampled_view_trajectory(ordered, exec_waypoints=exec_waypoints)
        else:
            self.publish_sampled_view_trajectory(ordered, exec_waypoints=exec_waypoints)
        t_publish_traj_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if self.mesh_view_plan_enabled:
            self._commit_planned_samples(ordered, exec_waypoints=exec_waypoints)
        else:
            self._commit_planned_samples(ordered, exec_waypoints=exec_waypoints)
        self._mesh_view_last_success_cycle_t = time.monotonic()
        t_commit_ms = (time.perf_counter() - t0) * 1000.0

        if self.mesh_view_plan_log_timing:
            t_cycle_total_ms = (time.perf_counter() - t_cycle_start) * 1000.0
            self._log_mesh_view_planning_timing(
                n_samples=int(len(samples_for_plan)),
                cycle_total_ms=float(t_cycle_total_ms),
                load_mesh_ms=float(t_load_mesh_ms),
                voxelize_ms=float(t_voxelize_ms),
                intersection_cache_ms=float(t_intersection_cache_ms),
                sample_ms=float(t_sample_ms),
                publish_vectors_ms=float(t_publish_vectors_ms),
                intersection_pairs_ms=float(t_intersection_pairs_ms),
                route_costs_ms=float(t_route_costs_ms),
                route_solver_ms=float(t_route_solver_ms),
                exec_paths_ms=float(t_exec_paths_ms),
                local_orientation_ms=float(t_local_orientation_ms),
                publish_traj_ms=float(t_publish_traj_ms),
                commit_ms=float(t_commit_ms),
                voxel_stats=voxel_stats,
                intersection_stats=intersection_stats,
            )
        return True

    def _clear_mesh_view_voxel_grid(self):
        self.mesh_view_voxel_grid = None
        self.mesh_view_voxel_origin = None
        self.mesh_view_voxel_bounds_min = None
        self.mesh_view_voxel_bounds_max = None
        self.mesh_view_voxel_shape = None
        self.mesh_view_voxel_size = None
        self.mesh_view_voxel_occupied_count = 0
        self.mesh_view_voxel_mesh_mtime = None

    def _clear_mesh_view_intersection_cache(self):
        self.mesh_view_intersection_cache = None
        self.mesh_view_intersection_mesh_mtime = None
        self.mesh_view_intersection_pairs = {}

    def _update_running_timing_pct_stats(self, timing_items, cycle_total_ms):
        denom = max(float(cycle_total_ms), 1e-9)
        self._planner_timing_num_updates += 1
        for key, val_ms in timing_items:
            pct = 100.0 * float(val_ms) / denom
            stat_pct = self._planner_timing_stats_pct.get(key)
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
            self._planner_timing_stats_pct[key] = stat_pct

            stat_ms = self._planner_timing_stats_ms.get(key)
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
            self._planner_timing_stats_ms[key] = stat_ms

        out = {}
        for key, stat_pct in self._planner_timing_stats_pct.items():
            n = float(stat_pct["n"])
            var_pct = float(stat_pct["m2"]) / (n - 1.0) if n > 1.0 else 0.0
            std_pct = float(np.sqrt(max(var_pct, 0.0)))
            stat_ms = self._planner_timing_stats_ms[key]
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
    def _update_running_mean_stats(stats_store, timing_items):
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

    def _build_mesh_view_voxel_grid(self):
        if self.mesh_triangles is None:
            self._clear_mesh_view_voxel_grid()
            return None

        triangles = np.asarray(self.mesh_triangles, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[0] <= 0 or triangles.shape[1:] != (3, 3):
            self._clear_mesh_view_voxel_grid()
            return {
                "voxel_total_ms": 0.0,
                "voxel_mesh_aabb_ms": 0.0,
                "voxel_grid_alloc_ms": 0.0,
                "voxel_mark_occupied_ms": 0.0,
                "num_triangles": int(triangles.shape[0]) if triangles.ndim >= 1 else 0,
                "grid_dims": (0, 0, 0),
                "num_cells": 0,
                "num_occupied": 0,
                "occupancy_ratio_pct": 0.0,
                "marked_candidate_voxels": 0,
            }

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        tri_min = np.min(triangles, axis=1)
        tri_max = np.max(triangles, axis=1)
        mesh_min = np.min(tri_min, axis=0)
        mesh_max = np.max(tri_max, axis=0)
        outer_pad = float(self.mesh_view_voxel_clearance_m + self.mesh_view_voxel_aabb_margin_m)
        grid_min = mesh_min - outer_pad
        grid_max = mesh_max + outer_pad
        t_mesh_aabb_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        voxel_size = float(self.mesh_view_voxel_size_m)
        spans = np.maximum(grid_max - grid_min, 1.0e-9)
        grid_shape = np.maximum(1, np.ceil(spans / voxel_size).astype(np.int64))
        occupied = np.zeros(tuple(int(v) for v in grid_shape.tolist()), dtype=np.bool_)
        t_grid_alloc_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        inflate = float(self.mesh_view_voxel_clearance_m)
        grid_shape_max = grid_shape - 1
        marked_candidate_voxels = 0
        for tri_lo, tri_hi in zip(tri_min, tri_max):
            box_min = tri_lo - inflate
            box_max = tri_hi + inflate
            idx_lo = np.floor((box_min - grid_min) / voxel_size).astype(np.int64)
            idx_hi = np.ceil((box_max - grid_min) / voxel_size).astype(np.int64) - 1
            idx_lo = np.clip(idx_lo, 0, grid_shape_max)
            idx_hi = np.clip(idx_hi, 0, grid_shape_max)
            if np.any(idx_hi < idx_lo):
                continue
            occupied[
                idx_lo[0] : idx_hi[0] + 1,
                idx_lo[1] : idx_hi[1] + 1,
                idx_lo[2] : idx_hi[2] + 1,
            ] = True
            marked_candidate_voxels += int(
                (idx_hi[0] - idx_lo[0] + 1)
                * (idx_hi[1] - idx_lo[1] + 1)
                * (idx_hi[2] - idx_lo[2] + 1)
            )
        t_mark_occupied_ms = (time.perf_counter() - t0) * 1000.0

        num_cells = int(occupied.size)
        num_occupied = int(np.count_nonzero(occupied))
        occupancy_ratio_pct = 100.0 * float(num_occupied) / max(1.0, float(num_cells))
        grid_bounds_max = grid_min + grid_shape.astype(np.float64) * voxel_size

        self.mesh_view_voxel_grid = occupied
        self.mesh_view_voxel_origin = np.asarray(grid_min, dtype=np.float32)
        self.mesh_view_voxel_bounds_min = np.asarray(grid_min, dtype=np.float32)
        self.mesh_view_voxel_bounds_max = np.asarray(grid_bounds_max, dtype=np.float32)
        self.mesh_view_voxel_shape = tuple(int(v) for v in grid_shape.tolist())
        self.mesh_view_voxel_size = float(voxel_size)
        self.mesh_view_voxel_occupied_count = int(num_occupied)
        self.mesh_view_voxel_mesh_mtime = self.mesh_mtime

        return {
            "voxel_total_ms": float((time.perf_counter() - t_total) * 1000.0),
            "voxel_mesh_aabb_ms": float(t_mesh_aabb_ms),
            "voxel_grid_alloc_ms": float(t_grid_alloc_ms),
            "voxel_mark_occupied_ms": float(t_mark_occupied_ms),
            "num_triangles": int(triangles.shape[0]),
            "grid_dims": tuple(int(v) for v in grid_shape.tolist()),
            "num_cells": int(num_cells),
            "num_occupied": int(num_occupied),
            "occupancy_ratio_pct": float(occupancy_ratio_pct),
            "marked_candidate_voxels": int(marked_candidate_voxels),
        }

    def _build_mesh_view_intersection_cache(self):
        if self.mesh_triangles is None:
            self._clear_mesh_view_intersection_cache()
            return None

        triangles = np.asarray(self.mesh_triangles, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[0] <= 0 or triangles.shape[1:] != (3, 3):
            self._clear_mesh_view_intersection_cache()
            return {
                "intersection_triangles": 0,
                "intersection_vertices": 0,
                "intersection_edges": 0,
                "intersection_tri_aabb_ms": 0.0,
                "intersection_bvh_ms": 0.0,
                "intersection_graph_ms": 0.0,
            }

        t0 = time.perf_counter()
        tri_aabb_min = np.min(triangles, axis=1)
        tri_aabb_max = np.max(triangles, axis=1)
        t_tri_aabb_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        (
            tri_bvh_root,
            tri_bvh_order,
            tri_bvh_min,
            tri_bvh_max,
            tri_bvh_left,
            tri_bvh_right,
            tri_bvh_start,
            tri_bvh_count,
        ) = self._build_triangle_bvh(
            tri_aabb_min=tri_aabb_min,
            tri_aabb_max=tri_aabb_max,
            leaf_size=int(self.mesh_view_intersection_bvh_leaf_size),
        )
        t_bvh_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        eps = max(float(self.mesh_view_intersection_quantize_eps), 1.0e-12)
        key_to_vid = {}
        vertices = []
        tri_vidx = np.zeros((triangles.shape[0], 3), dtype=np.int32)
        next_vid = 0
        for tid in range(int(triangles.shape[0])):
            for j in range(3):
                p = triangles[tid, j]
                key = tuple(np.round(p / eps).astype(np.int64).tolist())
                vid = key_to_vid.get(key, None)
                if vid is None:
                    vid = next_vid
                    key_to_vid[key] = vid
                    vertices.append(np.asarray(p, dtype=np.float64))
                    next_vid += 1
                tri_vidx[tid, j] = int(vid)

        if len(vertices) > 0:
            vertices_arr = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        else:
            vertices_arr = np.zeros((0, 3), dtype=np.float64)

        edge_to_length = {}
        adjacency = [set() for _ in range(int(vertices_arr.shape[0]))]
        adjacency_weighted = [[] for _ in range(int(vertices_arr.shape[0]))]
        for tid in range(int(tri_vidx.shape[0])):
            a, b, c = [int(v) for v in tri_vidx[tid]]
            for va, vb in ((a, b), (b, c), (c, a)):
                key = (va, vb) if va < vb else (vb, va)
                if key not in edge_to_length:
                    pa = vertices_arr[int(key[0])]
                    pb = vertices_arr[int(key[1])]
                    edge_to_length[key] = float(np.linalg.norm(pb - pa))
                adjacency[int(va)].add(int(vb))
                adjacency[int(vb)].add(int(va))
        for (va, vb), length in edge_to_length.items():
            adjacency_weighted[int(va)].append((int(vb), float(length)))
            adjacency_weighted[int(vb)].append((int(va), float(length)))
        edge_list = np.asarray(list(edge_to_length.keys()), dtype=np.int32).reshape(-1, 2) if edge_to_length else np.zeros((0, 2), dtype=np.int32)
        edge_lengths = np.asarray(list(edge_to_length.values()), dtype=np.float64).reshape(-1) if edge_to_length else np.zeros((0,), dtype=np.float64)
        adjacency = [sorted(list(nbs)) for nbs in adjacency]
        adjacency_weighted = [sorted(nbs, key=lambda x: x[0]) for nbs in adjacency_weighted]
        t_graph_ms = (time.perf_counter() - t0) * 1000.0

        self.mesh_view_intersection_cache = {
            "triangles": triangles,
            "tri_aabb_min": tri_aabb_min,
            "tri_aabb_max": tri_aabb_max,
            "tri_bvh_root": int(tri_bvh_root),
            "tri_bvh_order": tri_bvh_order,
            "tri_bvh_min": tri_bvh_min,
            "tri_bvh_max": tri_bvh_max,
            "tri_bvh_left": tri_bvh_left,
            "tri_bvh_right": tri_bvh_right,
            "tri_bvh_start": tri_bvh_start,
            "tri_bvh_count": tri_bvh_count,
            "tri_vidx": tri_vidx,
            "vertices": vertices_arr,
            "edges": edge_list,
            "edge_lengths": edge_lengths,
            "adjacency": adjacency,
            "adjacency_weighted": adjacency_weighted,
        }
        self.mesh_view_intersection_mesh_mtime = self.mesh_mtime
        self.mesh_view_intersection_pairs = {}
        return {
            "intersection_triangles": int(triangles.shape[0]),
            "intersection_vertices": int(vertices_arr.shape[0]),
            "intersection_edges": int(edge_list.shape[0]),
            "intersection_tri_aabb_ms": float(t_tri_aabb_ms),
            "intersection_bvh_ms": float(t_bvh_ms),
            "intersection_graph_ms": float(t_graph_ms),
        }

    @staticmethod
    def _segment_triangle_intersection_moller(p0, p1, tri, eps=1.0e-9):
        orig = np.asarray(p0, dtype=np.float64).reshape(3)
        end = np.asarray(p1, dtype=np.float64).reshape(3)
        tri3 = np.asarray(tri, dtype=np.float64).reshape(3, 3)
        direction = end - orig
        v0, v1, v2 = tri3[0], tri3[1], tri3[2]
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(direction, edge2)
        a = float(np.dot(edge1, h))
        if abs(a) <= float(eps):
            return None
        f = 1.0 / a
        s = orig - v0
        u = f * float(np.dot(s, h))
        if u < -float(eps) or u > 1.0 + float(eps):
            return None
        q = np.cross(s, edge1)
        v = f * float(np.dot(direction, q))
        if v < -float(eps) or (u + v) > 1.0 + float(eps):
            return None
        t = f * float(np.dot(edge2, q))
        if t < -float(eps) or t > 1.0 + float(eps):
            return None
        point = orig + t * direction
        return float(t), np.asarray(point, dtype=np.float64)

    @staticmethod
    def _build_triangle_bvh(tri_aabb_min, tri_aabb_max, leaf_size=8):
        tri_min = np.asarray(tri_aabb_min, dtype=np.float64).reshape(-1, 3)
        tri_max = np.asarray(tri_aabb_max, dtype=np.float64).reshape(-1, 3)
        n_tri = int(tri_min.shape[0])
        if n_tri <= 0:
            return (
                -1,
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
            )

        leaf_size = max(1, int(leaf_size))
        centers = 0.5 * (tri_min + tri_max)
        order = np.arange(n_tri, dtype=np.int32)
        node_min = []
        node_max = []
        node_left = []
        node_right = []
        node_start = []
        node_count = []

        def build(lo, hi):
            idx = len(node_left)
            subset = order[lo:hi]
            bmin = np.min(tri_min[subset], axis=0)
            bmax = np.max(tri_max[subset], axis=0)
            node_min.append(np.asarray(bmin, dtype=np.float64))
            node_max.append(np.asarray(bmax, dtype=np.float64))
            node_left.append(-1)
            node_right.append(-1)
            node_start.append(int(lo))
            count = int(hi - lo)
            node_count.append(int(count))
            if count <= leaf_size:
                return idx

            axis = int(np.argmax(bmax - bmin))
            perm = np.argsort(centers[subset, axis], kind="mergesort")
            order[lo:hi] = subset[perm]
            mid = int((lo + hi) // 2)
            left_idx = build(lo, mid)
            right_idx = build(mid, hi)
            node_left[idx] = int(left_idx)
            node_right[idx] = int(right_idx)
            node_start[idx] = -1
            node_count[idx] = 0
            return idx

        root = int(build(0, n_tri))
        return (
            root,
            np.asarray(order, dtype=np.int32),
            np.asarray(node_min, dtype=np.float64).reshape(-1, 3),
            np.asarray(node_max, dtype=np.float64).reshape(-1, 3),
            np.asarray(node_left, dtype=np.int32).reshape(-1),
            np.asarray(node_right, dtype=np.int32).reshape(-1),
            np.asarray(node_start, dtype=np.int32).reshape(-1),
            np.asarray(node_count, dtype=np.int32).reshape(-1),
        )

    @staticmethod
    def _segment_aabb_t_interval(p0, p1, box_min, box_max, eps=1.0e-12):
        a = np.asarray(p0, dtype=np.float64).reshape(3)
        b = np.asarray(p1, dtype=np.float64).reshape(3)
        bmin = np.asarray(box_min, dtype=np.float64).reshape(3)
        bmax = np.asarray(box_max, dtype=np.float64).reshape(3)
        d = b - a
        tmin = 0.0
        tmax = 1.0
        for axis in range(3):
            da = float(d[axis])
            if abs(da) <= float(eps):
                if float(a[axis]) < float(bmin[axis]) - float(eps) or float(a[axis]) > float(bmax[axis]) + float(eps):
                    return None
                continue
            inv = 1.0 / da
            t1 = (float(bmin[axis]) - float(a[axis])) * inv
            t2 = (float(bmax[axis]) - float(a[axis])) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, float(t1))
            tmax = min(tmax, float(t2))
            if tmin > tmax + float(eps):
                return None
        return float(tmin), float(tmax)

    def _segment_mesh_first_hit_bvh(self, p0, p1):
        cache = self.mesh_view_intersection_cache
        if cache is None:
            return {
                "hit": False,
                "tested_triangles": 0,
                "visited_nodes": 0,
            }

        root = int(cache.get("tri_bvh_root", -1))
        node_min = np.asarray(cache.get("tri_bvh_min", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64).reshape(-1, 3)
        node_max = np.asarray(cache.get("tri_bvh_max", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64).reshape(-1, 3)
        node_left = np.asarray(cache.get("tri_bvh_left", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        node_right = np.asarray(cache.get("tri_bvh_right", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        node_start = np.asarray(cache.get("tri_bvh_start", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        node_count = np.asarray(cache.get("tri_bvh_count", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        tri_order = np.asarray(cache.get("tri_bvh_order", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        triangles = np.asarray(cache.get("triangles", np.zeros((0, 3, 3), dtype=np.float64)), dtype=np.float64).reshape(-1, 3, 3)
        if root < 0 or node_min.shape[0] <= 0 or triangles.shape[0] <= 0:
            return {
                "hit": False,
                "tested_triangles": 0,
                "visited_nodes": 0,
            }

        root_interval = self._segment_aabb_t_interval(p0, p1, node_min[root], node_max[root])
        if root_interval is None:
            return {
                "hit": False,
                "tested_triangles": 0,
                "visited_nodes": 0,
            }

        tested_triangles = 0
        visited_nodes = 0
        best_t = np.inf
        best_tid = None
        best_point = None
        heap = [(float(root_interval[0]), int(root))]

        while heap:
            node_tmin, node_idx = heapq.heappop(heap)
            if float(node_tmin) > float(best_t) + 1.0e-12:
                break
            visited_nodes += 1
            left_idx = int(node_left[node_idx]) if node_idx < int(node_left.shape[0]) else -1
            right_idx = int(node_right[node_idx]) if node_idx < int(node_right.shape[0]) else -1
            if left_idx < 0 and right_idx < 0:
                lo = int(node_start[node_idx])
                count = int(node_count[node_idx])
                hi = lo + count
                for order_idx in range(lo, hi):
                    tid = int(tri_order[order_idx])
                    tested_triangles += 1
                    res = self._segment_triangle_intersection_moller(p0, p1, triangles[tid])
                    if res is None:
                        continue
                    t_hit, point = res
                    if float(t_hit) + 1.0e-12 < float(best_t):
                        best_t = float(t_hit)
                        best_tid = int(tid)
                        best_point = np.asarray(point, dtype=np.float64)
                continue

            for child_idx in (left_idx, right_idx):
                if child_idx < 0:
                    continue
                child_interval = self._segment_aabb_t_interval(p0, p1, node_min[child_idx], node_max[child_idx])
                if child_interval is None:
                    continue
                if float(child_interval[0]) > float(best_t) + 1.0e-12:
                    continue
                heapq.heappush(heap, (float(child_interval[0]), int(child_idx)))

        if best_tid is None or best_point is None or not np.isfinite(best_t):
            return {
                "hit": False,
                "tested_triangles": int(tested_triangles),
                "visited_nodes": int(visited_nodes),
            }
        return {
            "hit": True,
            "tid": int(best_tid),
            "t": float(best_t),
            "point": np.asarray(best_point, dtype=np.float64),
            "tested_triangles": int(tested_triangles),
            "visited_nodes": int(visited_nodes),
        }

    def _segment_mesh_intersections(self, p0, p1):
        cache = self.mesh_view_intersection_cache
        if cache is None:
            return {
                "candidate_count": 0,
                "hit_count": 0,
                "tri_ids": [],
                "entry_point": None,
                "exit_point": None,
                "entry_t": None,
                "exit_t": None,
            }

        a = np.asarray(p0, dtype=np.float64).reshape(3)
        b = np.asarray(p1, dtype=np.float64).reshape(3)
        first_hit = self._segment_mesh_first_hit_bvh(a, b)
        if not bool(first_hit.get("hit", False)):
            return {
                "candidate_count": int(first_hit.get("tested_triangles", 0)),
                "hit_count": 0,
                "tri_ids": [],
                "entry_point": None,
                "exit_point": None,
                "entry_t": None,
                "exit_t": None,
            }

        reverse_hit = self._segment_mesh_first_hit_bvh(b, a)
        entry_t = float(first_hit["t"])
        entry_tid = int(first_hit["tid"])
        entry_point = np.asarray(first_hit["point"], dtype=np.float64)
        exit_t = float(entry_t)
        exit_tid = int(entry_tid)
        exit_point = np.asarray(entry_point, dtype=np.float64)
        hit_count = 1
        tri_ids = [int(entry_tid)]
        candidate_count = int(first_hit.get("tested_triangles", 0))
        if bool(reverse_hit.get("hit", False)):
            candidate_count += int(reverse_hit.get("tested_triangles", 0))
            reverse_t = float(reverse_hit["t"])
            reverse_tid = int(reverse_hit["tid"])
            reverse_point = np.asarray(reverse_hit["point"], dtype=np.float64)
            mapped_t = float(1.0 - reverse_t)
            same_hit = (
                abs(mapped_t - entry_t) <= 1.0e-7
                or float(np.linalg.norm(reverse_point - entry_point)) <= 1.0e-7
            )
            if not same_hit:
                exit_t = float(mapped_t)
                exit_tid = int(reverse_tid)
                exit_point = np.asarray(reverse_point, dtype=np.float64)
                hit_count = 2
                if int(reverse_tid) != int(entry_tid):
                    tri_ids.append(int(reverse_tid))
        return {
            "candidate_count": int(candidate_count),
            "hit_count": int(hit_count),
            "tri_ids": tri_ids,
            "entry_point": np.asarray(entry_point, dtype=np.float64),
            "exit_point": np.asarray(exit_point, dtype=np.float64),
            "entry_t": float(entry_t),
            "exit_t": float(exit_t),
            "entry_tid": int(entry_tid),
            "exit_tid": int(exit_tid),
        }

    def _measure_sample_pair_intersections(self, samples):
        if self.mesh_view_intersection_cache is None:
            self.mesh_view_intersection_pairs = {}
            return {
                "intersection_pair_count": 0,
                "intersection_blocked_pairs": 0,
                "intersection_candidate_triangles": 0,
                "intersection_hits_total": 0,
                "intersection_pairs_stored": 0,
                "intersection_astar_ms": 0.0,
                "intersection_astar_pairs_attempted": 0,
                "intersection_astar_pairs_solved": 0,
                "intersection_astar_total_nodes": 0,
            }

        origins = []
        for s in list(samples) if samples is not None else []:
            try:
                origins.append(np.asarray(s.get("origin", None), dtype=np.float64).reshape(3))
            except Exception:
                origins.append(None)

        pair_cache = {}
        blocked_pairs = 0
        candidate_triangles = 0
        hits_total = 0
        pair_count = 0
        astar_ms = 0.0
        astar_pairs_attempted = 0
        astar_pairs_solved = 0
        astar_total_nodes = 0
        astar_total_attempts = 0
        astar_max_alpha = 0.0
        n = len(origins)
        for i in range(n):
            oi = origins[i]
            if oi is None:
                continue
            for j in range(i + 1, n):
                oj = origins[j]
                if oj is None:
                    continue
                pair_count += 1
                hit = self._segment_mesh_intersections(oi, oj)
                candidate_triangles += int(hit.get("candidate_count", 0))
                hits_total += int(hit.get("hit_count", 0))
                if int(hit.get("hit_count", 0)) > 0:
                    blocked_pairs += 1
                    entry_point = hit.get("entry_point", None)
                    exit_point = hit.get("exit_point", None)
                    if entry_point is not None and exit_point is not None:
                        t0 = time.perf_counter()
                        route = self._surface_graph_route_between_points(
                            entry_point,
                            exit_point,
                            entry_tid=hit.get("entry_tid", None),
                            exit_tid=hit.get("exit_tid", None),
                        )
                        astar_ms += (time.perf_counter() - t0) * 1000.0
                        astar_pairs_attempted += 1
                        astar_total_attempts += int(route.get("attempts", 0))
                        astar_max_alpha = max(astar_max_alpha, float(route.get("alpha", 0.0)))
                        if bool(route.get("success", False)):
                            astar_pairs_solved += 1
                            astar_total_nodes += int(route.get("num_nodes", 0))
                        hit["surface_graph_route"] = route
                pair_cache[(int(i), int(j))] = hit

        self.mesh_view_intersection_pairs = pair_cache
        return {
            "intersection_pair_count": int(pair_count),
            "intersection_blocked_pairs": int(blocked_pairs),
            "intersection_candidate_triangles": int(candidate_triangles),
            "intersection_hits_total": int(hits_total),
            "intersection_pairs_stored": int(len(pair_cache)),
            "intersection_astar_ms": float(astar_ms),
            "intersection_astar_pairs_attempted": int(astar_pairs_attempted),
            "intersection_astar_pairs_solved": int(astar_pairs_solved),
            "intersection_astar_total_nodes": int(astar_total_nodes),
            "intersection_astar_total_attempts": int(astar_total_attempts),
            "intersection_astar_max_alpha": float(astar_max_alpha),
        }

    def _surface_graph_route_between_points(self, p_start, p_goal, entry_tid=None, exit_tid=None):
        cache = self.mesh_view_intersection_cache
        if cache is None:
            return {"success": False, "reason": "no_cache"}

        vertices = np.asarray(cache.get("vertices", np.zeros((0, 3), dtype=np.float64)), dtype=np.float64).reshape(-1, 3)
        adjacency_weighted = cache.get("adjacency_weighted", [])
        tri_vidx = np.asarray(cache.get("tri_vidx", np.zeros((0, 3), dtype=np.int32)), dtype=np.int32).reshape(-1, 3)
        if vertices.shape[0] <= 0 or not adjacency_weighted or tri_vidx.shape[0] <= 0:
            return {"success": False, "reason": "empty_graph"}

        start = np.asarray(p_start, dtype=np.float64).reshape(3)
        goal = np.asarray(p_goal, dtype=np.float64).reshape(3)
        try:
            entry_tid = int(entry_tid)
            exit_tid = int(exit_tid)
        except Exception:
            return {"success": False, "reason": "bad_hit_triangle"}
        if (
            entry_tid < 0
            or exit_tid < 0
            or entry_tid >= int(tri_vidx.shape[0])
            or exit_tid >= int(tri_vidx.shape[0])
        ):
            return {"success": False, "reason": "bad_hit_triangle"}

        start_attach = sorted({int(v) for v in tri_vidx[int(entry_tid)].tolist()})
        goal_attach = sorted({int(v) for v in tri_vidx[int(exit_tid)].tolist()})
        if len(start_attach) <= 0 or len(goal_attach) <= 0:
            return {"success": False, "reason": "empty_attachment"}

        seg_len = float(np.linalg.norm(goal - start))
        d_start = np.linalg.norm(vertices - start.reshape(1, 3), axis=1)
        d_goal = np.linalg.norm(vertices - goal.reshape(1, 3), axis=1)
        union_cover_radius = float(np.max(np.minimum(d_start, d_goal))) if vertices.shape[0] > 0 else 0.0
        alpha = float(self.mesh_view_surface_graph_astar_alpha_start)
        alpha_step = float(self.mesh_view_surface_graph_astar_alpha_step)
        if seg_len <= 1.0e-9:
            alpha = 1.0
            required_alpha = 1.0
        else:
            required_alpha = max(alpha, union_cover_radius / seg_len)
        start_id = int(vertices.shape[0])
        goal_id = int(vertices.shape[0] + 1)

        if entry_tid == exit_tid:
            return {
                "success": True,
                "start_vid": int(start_id),
                "goal_vid": int(goal_id),
                "num_nodes": 2,
                "path_length": float(seg_len),
                "path_vids": [int(start_id), int(goal_id)],
                "path_points": [start.tolist(), goal.tolist()],
                "alpha": float(alpha),
                "attempts": 1,
                "kept_vertices": int(len(set(start_attach + goal_attach))),
            }

        start_attach_set = set(start_attach)
        goal_attach_set = set(goal_attach)

        def node_position(node_id):
            if int(node_id) == int(start_id):
                return start
            if int(node_id) == int(goal_id):
                return goal
            return vertices[int(node_id)]

        def heuristic(node_id):
            return float(np.linalg.norm(node_position(int(node_id)) - goal))

        def reconstruct_path(parent_map, total_cost, used_alpha, kept_mask):
            path_vids = []
            node = int(goal_id)
            while node >= 0:
                path_vids.append(int(node))
                node = int(parent_map.get(int(node), -1))
            path_vids.reverse()
            path_points = [node_position(v).tolist() for v in path_vids]
            return {
                "success": True,
                "start_vid": int(start_id),
                "goal_vid": int(goal_id),
                "num_nodes": int(len(path_vids)),
                "path_length": float(total_cost),
                "path_vids": path_vids,
                "path_points": path_points,
                "alpha": float(used_alpha),
                "kept_vertices": int(np.count_nonzero(kept_mask)),
            }

        attempts = 0
        while True:
            attempts += 1
            radius = float(alpha * max(seg_len, 1.0e-9))
            keep_mask = (d_start <= radius) | (d_goal <= radius)
            if start_attach:
                keep_mask[np.asarray(start_attach, dtype=np.int64)] = True
            if goal_attach:
                keep_mask[np.asarray(goal_attach, dtype=np.int64)] = True

            g_cost = {int(start_id): 0.0}
            parent = {int(start_id): -1}
            open_heap = [(heuristic(start_id), 0.0, int(start_id))]

            while open_heap:
                _f_cost, cur_g, u = heapq.heappop(open_heap)
                if cur_g > float(g_cost.get(int(u), np.inf)) + 1.0e-12:
                    continue
                if int(u) == int(goal_id):
                    route = reconstruct_path(parent, cur_g, alpha, keep_mask)
                    route["attempts"] = int(attempts)
                    return route

                if int(u) == int(start_id):
                    neighbors = [
                        (int(v), float(np.linalg.norm(vertices[int(v)] - start)))
                        for v in start_attach
                        if 0 <= int(v) < int(vertices.shape[0]) and bool(keep_mask[int(v)])
                    ]
                elif int(u) == int(goal_id):
                    neighbors = []
                else:
                    if not bool(keep_mask[int(u)]):
                        continue
                    neighbors = [
                        (int(v), float(w))
                        for v, w in adjacency_weighted[int(u)]
                        if 0 <= int(v) < int(vertices.shape[0]) and bool(keep_mask[int(v)])
                    ]
                    if int(u) in goal_attach_set:
                        neighbors.append((int(goal_id), float(np.linalg.norm(goal - vertices[int(u)]))))

                for v, w in neighbors:
                    new_g = float(cur_g + float(w))
                    if new_g + 1.0e-12 < float(g_cost.get(int(v), np.inf)):
                        g_cost[int(v)] = new_g
                        parent[int(v)] = int(u)
                        heapq.heappush(open_heap, (new_g + heuristic(v), new_g, int(v)))

            if alpha >= required_alpha - 1.0e-12:
                break
            alpha = min(required_alpha, alpha + alpha_step)

        return {
            "success": False,
            "reason": "no_path",
            "start_vid": int(start_id),
            "goal_vid": int(goal_id),
            "alpha": float(alpha),
            "attempts": int(attempts),
            "kept_vertices": int(np.count_nonzero(keep_mask)) if "keep_mask" in locals() else 0,
        }

    def _log_mesh_view_planning_timing(
        self,
        n_samples,
        cycle_total_ms,
        load_mesh_ms,
        voxelize_ms,
        intersection_cache_ms,
        sample_ms,
        publish_vectors_ms,
        intersection_pairs_ms,
        route_costs_ms,
        route_solver_ms,
        exec_paths_ms,
        local_orientation_ms,
        publish_traj_ms,
        commit_ms,
        voxel_stats,
        intersection_stats,
    ):
        label_map = {
            "cycle_total": "cycle_total",
            "load_mesh": "load_mesh",
            "sample_views": "sample_views",
            "publish_vectors": "publish_vectors",
            "route_costs": "route_costs",
            "route_solver": "route_solver",
            "exec_paths": "exec_path_build",
            "local_orientation": "local_orientation_plan",
            "publish_trajectory": "publish_trajectory",
            "commit_plan": "commit_plan",
            "surface_graph_prep": "surface_graph_build",
            "surface_graph_pairs": "pair_collision_stage",
            "voxelize_total": "voxelize_total",
            "other_untracked": "other_untracked",
            "tri_aabb_build": "tri_aabb_build",
            "tri_bvh_build": "tri_bvh_build",
            "mesh_graph_build": "mesh_graph_build",
            "pair_intersections": "pair_collision_check",
            "pair_astar": "pair_surface_astar",
            "mesh_aabb": "mesh_aabb",
            "grid_alloc": "grid_alloc",
            "mark_occupied": "mark_occupied",
        }

        def display_key(key):
            return label_map.get(str(key), str(key))

        planner_elapsed_s = max(
            0.0,
            float(time.monotonic() - float(getattr(self, "_mesh_view_planner_started_t", time.monotonic()))),
        )
        planner_elapsed_hms = self._format_elapsed_hms(planner_elapsed_s)

        top_level_items = [
            ("cycle_total", float(cycle_total_ms)),
            ("load_mesh", float(load_mesh_ms)),
            ("sample_views", float(sample_ms)),
            ("publish_vectors", float(publish_vectors_ms)),
            ("route_costs", float(route_costs_ms)),
            ("route_solver", float(route_solver_ms)),
            ("exec_paths", float(exec_paths_ms)),
            ("local_orientation", float(local_orientation_ms)),
            ("publish_trajectory", float(publish_traj_ms)),
            ("commit_plan", float(commit_ms)),
        ]
        if self.mesh_view_obstacle_mode == "surface_graph":
            top_level_items.append(("surface_graph_prep", float(intersection_cache_ms)))
            top_level_items.append(("surface_graph_pairs", float(intersection_pairs_ms)))
        elif self.mesh_view_obstacle_mode == "voxel_grid":
            top_level_items.append(("voxelize_total", float(voxelize_ms)))
        top_level_sum_ms = float(sum(v for k, v in top_level_items if k != "cycle_total"))
        top_level_other_ms = max(0.0, float(cycle_total_ms) - top_level_sum_ms)
        top_level_items.append(("other_untracked", top_level_other_ms))

        denom = max(float(cycle_total_ms), 1e-9)
        pct_stats = self._update_running_timing_pct_stats(
            timing_items=top_level_items,
            cycle_total_ms=float(cycle_total_ms),
        )
        mean_cycle_total_ms = max(float(pct_stats.get("cycle_total", {}).get("mean_ms", cycle_total_ms)), 1e-9)
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
                f"  {display_key(k)}: {100.0 * (v / mean_cycle_total_ms):.2f}% ({v:.2f} ms)"
                for k, v in [("cycle_total", mean_cycle_total_ms)] + running_top_items
            ]
        )

        mode_block_title = "Obstacle-mode substeps"
        mode_block_total_ms = 0.0
        mode_block_lines = "  obstacle-mode prep disabled or unavailable"
        mode_block_summary = ""
        if self.mesh_view_obstacle_mode == "surface_graph" and intersection_stats is not None:
            mode_block_title = "Surface-graph substeps"
            mode_block_total_ms = float(intersection_cache_ms + intersection_pairs_ms)
            mode_items = [
                ("tri_aabb_build", float(intersection_stats.get("intersection_tri_aabb_ms", 0.0))),
                ("tri_bvh_build", float(intersection_stats.get("intersection_bvh_ms", 0.0))),
                ("mesh_graph_build", float(intersection_stats.get("intersection_graph_ms", 0.0))),
                (
                    "pair_intersections",
                    max(
                        0.0,
                        float(intersection_pairs_ms) - float(intersection_stats.get("intersection_astar_ms", 0.0)),
                    ),
                ),
                ("pair_astar", float(intersection_stats.get("intersection_astar_ms", 0.0))),
            ]
            mode_means = self._update_running_mean_stats(
                self._planner_mode_timing_stats_ms,
                [("mode_total", float(mode_block_total_ms))] + mode_items,
            )
            mean_mode_total_ms = max(float(mode_means.get("mode_total", mode_block_total_ms)), 1e-9)
            mode_sorted = sorted(
                [(k, float(mode_means.get(k, 0.0))) for k, _v in mode_items],
                key=lambda kv: kv[1],
                reverse=True,
            )
            mode_block_lines = "\n".join(
                [
                    f"  {display_key(k)}: {100.0 * (v / mean_mode_total_ms):.2f}% ({v:.2f} ms)"
                    for k, v in mode_sorted
                ]
            )
            mode_block_summary = (
                "\nSurface-graph stats: tris=%d verts=%d edges=%d pairs=%d blocked=%d tested_tris=%d hits=%d astar_attempted=%d astar_solved=%d astar_nodes=%d astar_tries=%d astar_max_alpha=%.2f"
            ) % (
                int(intersection_stats.get("intersection_triangles", 0)),
                int(intersection_stats.get("intersection_vertices", 0)),
                int(intersection_stats.get("intersection_edges", 0)),
                int(intersection_stats.get("intersection_pair_count", 0)),
                int(intersection_stats.get("intersection_blocked_pairs", 0)),
                int(intersection_stats.get("intersection_candidate_triangles", 0)),
                int(intersection_stats.get("intersection_hits_total", 0)),
                int(intersection_stats.get("intersection_astar_pairs_attempted", 0)),
                int(intersection_stats.get("intersection_astar_pairs_solved", 0)),
                int(intersection_stats.get("intersection_astar_total_nodes", 0)),
                int(intersection_stats.get("intersection_astar_total_attempts", 0)),
                float(intersection_stats.get("intersection_astar_max_alpha", 0.0)),
            )
        elif self.mesh_view_obstacle_mode == "voxel_grid" and voxel_stats is not None:
            mode_block_title = "Voxel-grid substeps"
            mode_block_total_ms = float(voxelize_ms)
            mode_items = [
                ("mesh_aabb", float(voxel_stats.get("voxel_mesh_aabb_ms", 0.0))),
                ("grid_alloc", float(voxel_stats.get("voxel_grid_alloc_ms", 0.0))),
                ("mark_occupied", float(voxel_stats.get("voxel_mark_occupied_ms", 0.0))),
            ]
            mode_means = self._update_running_mean_stats(
                self._planner_mode_timing_stats_ms,
                [("mode_total", float(mode_block_total_ms))] + mode_items,
            )
            mean_mode_total_ms = max(float(mode_means.get("mode_total", mode_block_total_ms)), 1e-9)
            mode_sorted = sorted(
                [(k, float(mode_means.get(k, 0.0))) for k, _v in mode_items],
                key=lambda kv: kv[1],
                reverse=True,
            )
            mode_block_lines = "\n".join(
                [
                    f"  {display_key(k)}: {100.0 * (v / mean_mode_total_ms):.2f}% ({v:.2f} ms)"
                    for k, v in mode_sorted
                ]
            )
            mode_block_summary = (
                "\nVoxel-grid stats: tris=%d grid=%dx%dx%d cells=%d occupied=%d ratio=%.2f%%"
            ) % (
                int(voxel_stats.get("num_triangles", 0)),
                int(voxel_stats.get("grid_dims", (0, 0, 0))[0]),
                int(voxel_stats.get("grid_dims", (0, 0, 0))[1]),
                int(voxel_stats.get("grid_dims", (0, 0, 0))[2]),
                int(voxel_stats.get("num_cells", 0)),
                int(voxel_stats.get("num_occupied", 0)),
                float(voxel_stats.get("occupancy_ratio_pct", 0.0)),
            )

        rospy.loginfo(
            "Mesh-view planner timing averages: planner_elapsed=%.2fs (%s) samples=%d backend=%s obstacle_mode=%s local_orient=%s updates=%d\nTop-level running means (share from mean_ms / mean_cycle_total):\n%s\n%s running means:\n%s%s",
            float(planner_elapsed_s),
            planner_elapsed_hms,
            int(n_samples),
            self.mesh_view_plan_backend,
            self.mesh_view_obstacle_mode,
            self._local_planning_mode_label(),
            int(self._planner_timing_num_updates),
            timing_so_far_lines,
            mode_block_title,
            mode_block_lines,
            mode_block_summary,
        )

    def _commit_planned_samples(self, ordered, exec_waypoints=None):
        with self._planned_samples_lock:
            new_samples = list(ordered) if ordered is not None else []
            new_exec_waypoints = list(exec_waypoints) if exec_waypoints is not None else list(new_samples)
            if len(new_exec_waypoints) <= 0:
                new_exec_waypoints = list(new_samples)
            if len(new_samples) > 0:
                self._planned_samples = new_samples
                self._planned_exec_waypoints = new_exec_waypoints
                self._planned_version += 1
                self._nearest_exec_diag_logged_version = -1
                # Defer the nearest-local-waypoint selection until the final handoff
                # in follow_atsp_coverage(), immediately before the next command is sent.
                self._planned_exec_idx = -1
                self._planned_exec_direction = 1
                self._planned_idx = 0
            elif len(self._planned_samples) <= 0:
                self._planned_idx = 0
                self._planned_exec_waypoints = []
                self._planned_exec_idx = 0
                self._planned_exec_direction = 1
                self._atsp_active_target = False
                self._atsp_active_sample = None
                self._atsp_active_nbv_sample = None

    def _clear_planned_trajectory(self, hold_current_pose=False):
        hold_pos = None
        hold_yaw = None
        hold_pitch = float(self.camera_pitch_cmd_current)
        if hold_current_pose and self.latest_odom is not None:
            try:
                p = self.latest_odom.pose.pose.position
                hold_pos = (float(p.x), float(p.y), float(p.z))
                hold_yaw = self._get_current_yaw()
            except Exception:
                hold_pos = None
                hold_yaw = None
        with self._planned_samples_lock:
            had_plan = bool(self._planned_samples) or bool(self._planned_exec_waypoints) or bool(self._atsp_active_target)
            self._planned_samples = []
            self._planned_idx = 0
            self._planned_exec_waypoints = []
            self._planned_exec_idx = 0
            self._planned_exec_direction = 1
            if had_plan:
                self._planned_version += 1
            self._active_plan_version = -1
            self._nearest_exec_diag_logged_version = -1
            self._start_leg_diag_logged_version = -1
            self._atsp_active_target = False
            self._atsp_active_sample = None
            self._atsp_active_nbv_sample = None
        if hold_pos is not None:
            self.target_pos = tuple(hold_pos)
        if hold_yaw is not None:
            self.target_angles = (0.0, float(hold_pitch), float(hold_yaw))

    def _inject_active_sample(self, samples):
        out = list(samples) if samples is not None else []
        active = self._atsp_active_nbv_sample
        if active is None:
            active = self._atsp_active_sample
        if active is None:
            return out
        if self.mesh_view_termination_enabled:
            active_mask = np.asarray(
                self._planner_triangle_activity_last.get("active_mask", np.zeros((0,), dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            if active_mask.size > 0:
                try:
                    tri_id = active.get("tri_id", None)
                    if tri_id is not None:
                        tri_id = int(tri_id)
                        if tri_id < 0 or tri_id >= int(active_mask.size) or (not bool(active_mask[tri_id])):
                            return out
                    else:
                        tri_ids = np.asarray(active.get("tri_ids", []), dtype=np.int64).reshape(-1)
                        if tri_ids.size > 0:
                            tri_ids = tri_ids[(tri_ids >= 0) & (tri_ids < int(active_mask.size))]
                            if tri_ids.size <= 0 or (not bool(np.any(active_mask[tri_ids]))):
                                return out
                except Exception:
                    return out

        try:
            a_origin = np.asarray(active.get("origin", None), dtype=np.float64).reshape(3)
        except Exception:
            return out

        dedup_tol = float(max(1e-3, 0.5 * self.mesh_view_reach_pos_tol_m))
        for s in out:
            try:
                o = np.asarray(s.get("origin", None), dtype=np.float64).reshape(3)
            except Exception:
                continue
            if float(np.linalg.norm(o - a_origin)) <= dedup_tol:
                return out

        out.append(copy.deepcopy(active))
        return out

    def _init_mesh(self):
        if RTBoxMesh is None:
            rospy.logwarn("RTBoxMesh not available; cannot build mesh.")
            return
        center = np.array(
            [self.build_center_x, self.build_center_y, -self.build_center_z + self.rt_mesh_center_z_offset],
            dtype=np.float32,
        )
        size = np.array(
            [self.build_width, self.build_length, self.build_height],
            dtype=np.float32,
        )
        mesh = RTBoxMesh(
            center=center,
            size=size,
            target_edge_m=self.rt_mesh_target_edge_m,
            include_bottom=self.rt_mesh_include_bottom,
        )
        self.mesh_triangles, self.mesh_points = mesh.build()
        num_tris = int(self.mesh_triangles.shape[0])
        if num_tris > 0:
            tri0 = self.mesh_triangles[0].astype(np.float64)
            e01 = float(np.linalg.norm(tri0[1] - tri0[0]))
            e12 = float(np.linalg.norm(tri0[2] - tri0[1]))
            e20 = float(np.linalg.norm(tri0[0] - tri0[2]))
            self.mesh_l0 = float((e01 + e12 + e20) / 3.0)
        else:
            self.mesh_l0 = None
        self.mesh_tri_ids = np.arange(num_tris, dtype=np.int64)
        self.mesh_tri_cvi_raw = np.zeros(num_tris, dtype=np.float32)
        self.mesh_tri_vertex_conf = np.ones((num_tris, 3), dtype=np.float32)
        self.mesh_tri_conf = np.ones(num_tris, dtype=np.float32)
        self.mesh_tri_cvi_effective = np.zeros(num_tris, dtype=np.float32)
        self.mesh_tri_inactive_bottom = self._infer_bottom_inactive_triangles(
            self.mesh_triangles,
            mesh_l0=self.mesh_l0,
        ).astype(bool)
        self.mesh_tri_cvi = self.mesh_tri_cvi_effective
        self.mesh_next_tri_id = np.int64(num_tris)
        try:
            tmp_path = f"{self.mesh_filename}.tmp.{os.getpid()}.npz"
            np.savez(
                tmp_path,
                triangles=self.mesh_triangles,
                points=self.mesh_points,
                tri_ids=self.mesh_tri_ids,
                tri_vertex_conf=self.mesh_tri_vertex_conf,
                tri_conf=self.mesh_tri_conf,
                tri_inactive_bottom=self.mesh_tri_inactive_bottom.astype(bool),
                tri_cvi_raw=self.mesh_tri_cvi_raw,
                tri_cvi_effective=self.mesh_tri_cvi_effective,
                tri_cvi=self.mesh_tri_cvi_raw,  # backward-compatible alias
                next_tri_id=self.mesh_next_tri_id,
                cvi_state_pose_offset=np.int64(0),
                center=center,
                size=size,
                mesh_target_edge_m=np.float32(self.rt_mesh_target_edge_m),
                mesh_l0=np.float32(self.mesh_l0 if self.mesh_l0 is not None else 0.0),
                include_bottom=self.rt_mesh_include_bottom,
            )
            os.replace(tmp_path, self.mesh_filename)
            if self.mesh_l0 is not None:
                rospy.loginfo(
                    "Saved RT mesh to %s (mesh_l0=%.4f m; runtime max_corr=mesh_update_max_corr_l0_weight*L0, split_corr=mesh_split_corr_l0_weight*L0)",
                    self.mesh_filename,
                    float(self.mesh_l0),
                )
            else:
                rospy.loginfo("Saved RT mesh to %s", self.mesh_filename)
        except Exception as e:
            rospy.logwarn("Failed to save RT mesh to %s: %s", self.mesh_filename, str(e))

    def _clear_stale_sparse_on_mesh_reset(self):
        sparse_paths = []
        for p in [
            self.rt_mesh_sparse_path,
            self.rt_mesh_sfm_output_path,
            self.rt_mesh_depth_path,
            self.rt_mesh_depth_output_path,
        ]:
            if p and p not in sparse_paths:
                sparse_paths.append(p)
        for p in sparse_paths:
            try:
                if os.path.isfile(p):
                    os.remove(p)
                    rospy.logwarn("reset_mesh_on_start=true: removed stale sparse file %s", p)
            except Exception as e:
                rospy.logwarn("Failed to remove stale sparse file %s: %s", p, str(e))

    def _clear_image_pose_files_on_reset(self):
        def _clear_dir_files(dir_path, label):
            removed = 0
            try:
                if not os.path.isdir(dir_path):
                    return 0
                for entry in os.scandir(dir_path):
                    try:
                        if entry.is_file(follow_symlinks=False):
                            os.remove(entry.path)
                            removed += 1
                    except Exception as e:
                        rospy.logwarn("Failed to remove %s file %s: %s", label, entry.path, str(e))
            except Exception as e:
                rospy.logwarn("Failed to scan %s directory %s: %s", label, dir_path, str(e))
            return removed

        img_removed = _clear_dir_files(self.image_dir, "image")
        pose_removed = _clear_dir_files(self.pose_dir, "pose")
        depth_removed = _clear_dir_files(self.depth_dir, "depth") if self.save_depth_frames else 0
        try:
            if os.path.isdir(self.rviz_replay_dir):
                shutil.rmtree(self.rviz_replay_dir)
        except Exception as e:
            rospy.logwarn("Failed to remove stale RViz replay directory %s: %s", self.rviz_replay_dir, str(e))
        rospy.logwarn(
            "Startup reset: removed %d image files from %s, %d pose files from %s, and %d depth files from %s",
            img_removed,
            self.image_dir,
            pose_removed,
            self.pose_dir,
            depth_removed,
            self.depth_dir,
        )

    def _load_mesh_from_file(self):
        if not os.path.isfile(self.mesh_filename):
            return False
        mtime = os.path.getmtime(self.mesh_filename)
        if self.mesh_mtime is not None and mtime == self.mesh_mtime:
            return False
        try:
            data = np.load(self.mesh_filename, allow_pickle=False)
            triangles = data.get("triangles")
            points = data.get("points")
            if triangles is None or points is None:
                return False
            self.mesh_triangles = np.array(triangles, dtype=np.float32)
            self.mesh_points = np.array(points, dtype=np.float32)
            tri_ids = data.get("tri_ids")
            tri_cvi_raw = data.get("tri_cvi_raw")
            if tri_cvi_raw is None:
                tri_cvi_raw = data.get("tri_cvi")
            tri_cvi_effective = data.get("tri_cvi_effective")
            tri_conf = data.get("tri_conf")
            tri_vertex_conf = data.get("tri_vertex_conf")
            tri_inactive_bottom = data.get("tri_inactive_bottom")
            next_tri_id = data.get("next_tri_id")
            mesh_l0 = data.get("mesh_l0")
            if tri_ids is not None:
                self.mesh_tri_ids = np.array(tri_ids, dtype=np.int64).reshape(-1)
            else:
                self.mesh_tri_ids = None
            if tri_cvi_raw is not None:
                self.mesh_tri_cvi_raw = np.array(tri_cvi_raw, dtype=np.float32).reshape(-1)
            else:
                self.mesh_tri_cvi_raw = None
            if tri_cvi_effective is not None:
                self.mesh_tri_cvi_effective = np.array(tri_cvi_effective, dtype=np.float32).reshape(-1)
            elif self.mesh_tri_cvi_raw is not None:
                self.mesh_tri_cvi_effective = self.mesh_tri_cvi_raw.copy()
            else:
                self.mesh_tri_cvi_effective = None
            if tri_conf is not None:
                self.mesh_tri_conf = np.array(tri_conf, dtype=np.float32).reshape(-1)
            else:
                self.mesh_tri_conf = None
            if tri_vertex_conf is not None:
                self.mesh_tri_vertex_conf = np.array(tri_vertex_conf, dtype=np.float32)
            else:
                self.mesh_tri_vertex_conf = None
            # Colorization uses effective CVI by default.
            self.mesh_tri_cvi = self.mesh_tri_cvi_effective
            if next_tri_id is not None:
                self.mesh_next_tri_id = np.array(next_tri_id, dtype=np.int64).reshape(()).item()
            else:
                self.mesh_next_tri_id = None
            if mesh_l0 is not None:
                try:
                    self.mesh_l0 = float(np.array(mesh_l0, dtype=np.float64).reshape(()).item())
                except Exception:
                    self.mesh_l0 = None
            else:
                self.mesh_l0 = None
            if tri_inactive_bottom is not None and np.asarray(tri_inactive_bottom).reshape(-1).shape[0] == int(self.mesh_triangles.shape[0]):
                self.mesh_tri_inactive_bottom = np.asarray(tri_inactive_bottom, dtype=bool).reshape(-1)
            else:
                self.mesh_tri_inactive_bottom = self._infer_bottom_inactive_triangles(
                    self.mesh_triangles,
                    mesh_l0=self.mesh_l0,
                ).astype(bool)
            self.mesh_mtime = mtime
            if not bool(getattr(self, "_mission_completion_active", False)):
                rospy.loginfo("Loaded RT mesh from %s", self.mesh_filename)
            return True
        except Exception as e:
            rospy.logwarn("Failed to load RT mesh from %s: %s", self.mesh_filename, str(e))
            return False

    def publish_mesh(self):
        if self.mesh_triangles is None or self.mesh_points is None:
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        # Publish triangle mesh marker
        marker = Marker()
        marker.header = header
        marker.ns = "rt_mesh"
        marker.id = 1
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        r, g, b, a = (self.rt_mesh_color + [0.4])[:4]
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

        marker.points = []
        marker.colors = []

        use_cvi_colors = (
            self.mesh_tri_cvi is not None
            and self.mesh_tri_cvi.shape[0] == self.mesh_triangles.shape[0]
        )
        if use_cvi_colors:
            cvi = self.mesh_tri_cvi.astype(np.float32)
            vmin = float(np.min(cvi)) if cvi.size else 0.0
            vmax = float(np.max(cvi)) if cvi.size else 0.0
            if abs(vmax - vmin) < 1e-8:
                norm = np.zeros_like(cvi)
            else:
                norm = (cvi - vmin) / (vmax - vmin)
        else:
            norm = None

        for tri_idx, tri in enumerate(self.mesh_triangles):
            if norm is not None:
                cr, cg, cb = self._cvi_temperature_color(float(norm[tri_idx]))
                col = ColorRGBA(r=float(cr), g=float(cg), b=float(cb), a=float(a))
            else:
                col = ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(a))

            for p in tri:
                pt = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                marker.points.append(pt)
                marker.colors.append(col)
        self.mesh_tri_pub.publish(marker)

        # Publish mesh points (cell centers)
        stride = max(1, int(self.rt_mesh_point_stride))
        points = self.mesh_points[::stride]
        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("intensity", 12, PointField.FLOAT32, 1),
        ]
        cloud_points = [
            (float(x), float(y), float(z), 1.0) for x, y, z in points
        ]
        cloud = point_cloud2.create_cloud(header, fields, cloud_points)
        self.mesh_points_pub.publish(cloud)

    def sample_triangle_viewpoints(self):
        if self.mesh_triangles is None:
            return []

        triangles = np.asarray(self.mesh_triangles, dtype=np.float64)
        n_tri = int(triangles.shape[0])
        if n_tri <= 0:
            return []
        activity = self._planner_triangle_activity(triangles)
        self._planner_triangle_activity_last = dict(activity)
        tri_cvi = np.asarray(activity.get("tri_cvi", np.zeros((n_tri,), dtype=np.float64)), dtype=np.float64)
        tri_idx = np.where(np.asarray(activity.get("active_mask", np.zeros((n_tri,), dtype=bool)), dtype=bool))[0].astype(np.int64)
        if self.mesh_view_subsample > 1:
            if self.mesh_view_subsample_random:
                keep_prob = 1.0 / float(self.mesh_view_subsample)
                mask = np.random.rand(tri_idx.shape[0]) < keep_prob
                tri_idx = tri_idx[mask]
            else:
                tri_idx = tri_idx[:: self.mesh_view_subsample]

        if self.mesh_view_max_samples > 0 and tri_idx.size > self.mesh_view_max_samples:
            tri_idx = tri_idx[: self.mesh_view_max_samples]

        if tri_idx.size == 0:
            return []

        tan_h = math.tan(math.atan2(float(self.cam_width), 2.0 * max(1e-9, float(self.cam_fx))))
        tan_v = math.tan(math.atan2(float(self.cam_height), 2.0 * max(1e-9, float(self.cam_fy))))
        tan_h = max(1e-6, tan_h)
        tan_v = max(1e-6, tan_v)

        build_center = np.array(
            [self.build_center_x, self.build_center_y, -self.build_center_z + self.rt_mesh_center_z_offset],
            dtype=np.float64,
        )

        samples = []
        for tid in tri_idx.tolist():
            tri = triangles[int(tid)]
            p0, p1, p2 = tri[0], tri[1], tri[2]
            center = (p0 + p1 + p2) / 3.0

            n = np.cross(p1 - p0, p2 - p0)
            nn = float(np.linalg.norm(n))
            if nn <= 1e-10:
                continue
            n = n / nn

            if float(np.dot(n, center - build_center)) < 0.0:
                n = -n

            sample = self._sample_view_from_patch(
                patch_vertices=tri,
                center=center,
                normal=n,
                tan_h=tan_h,
                tan_v=tan_v,
            )
            if sample is None:
                continue
            if not self._planner_viewpoint_origin_allowed(sample["origin"]):
                continue

            sample["tri_id"] = int(tid)
            sample["cvi"] = float(tri_cvi[int(tid)])
            samples.append(sample)

        return samples

    def sample_region_viewpoints(self):
        if self.mesh_triangles is None:
            self._clear_region_boundaries_marker()
            return []

        triangles = np.asarray(self.mesh_triangles, dtype=np.float64)
        n_tri = int(triangles.shape[0])
        if n_tri <= 0:
            self._clear_region_boundaries_marker()
            return []
        activity = self._planner_triangle_activity(triangles)
        self._planner_triangle_activity_last = dict(activity)
        tri_cvi = np.asarray(activity.get("tri_cvi", np.zeros((n_tri,), dtype=np.float64)), dtype=np.float64)

        active_idx = np.where(np.asarray(activity.get("active_mask", np.zeros((n_tri,), dtype=bool)), dtype=bool))[0].astype(np.int64)
        if active_idx.size <= 0:
            self._clear_region_boundaries_marker()
            return []

        triangles = triangles[active_idx]
        tri_cvi = tri_cvi[active_idx]
        n_tri = int(triangles.shape[0])

        tri_vidx, tri_neighbors = self._build_triangle_topology(
            triangles=triangles,
            quantize_eps=float(self.mesh_view_region_quantize_eps),
        )
        if tri_vidx is None or tri_neighbors is None:
            self._clear_region_boundaries_marker()
            return []

        centroids = np.mean(triangles, axis=1)
        n_regions = int(min(max(1, self.mesh_view_region_count), n_tri))
        seeds = self._farthest_point_seeds(centroids, n_regions)
        if len(seeds) == 0:
            self._clear_region_boundaries_marker()
            return []

        region_ids = self._grow_regions_from_seeds(
            tri_neighbors=tri_neighbors,
            seeds=seeds,
            centroids=centroids,
        )

        tan_h = math.tan(math.atan2(float(self.cam_width), 2.0 * max(1e-9, float(self.cam_fx))))
        tan_v = math.tan(math.atan2(float(self.cam_height), 2.0 * max(1e-9, float(self.cam_fy))))
        tan_h = max(1e-6, tan_h)
        tan_v = max(1e-6, tan_v)
        build_center = np.array(
            [self.build_center_x, self.build_center_y, -self.build_center_z + self.rt_mesh_center_z_offset],
            dtype=np.float64,
        )

        region_unique = np.unique(region_ids[region_ids >= 0]).astype(np.int64)
        if region_unique.size == 0:
            self._clear_region_boundaries_marker()
            return []

        self.publish_region_boundaries(
            triangles=triangles,
            tri_vidx=tri_vidx,
            region_ids=region_ids,
        )

        if self.mesh_view_subsample > 1:
            if self.mesh_view_subsample_random:
                keep_prob = 1.0 / float(self.mesh_view_subsample)
                mask = np.random.rand(region_unique.shape[0]) < keep_prob
                region_unique = region_unique[mask]
            else:
                region_unique = region_unique[:: self.mesh_view_subsample]

        if self.mesh_view_max_samples > 0 and region_unique.size > self.mesh_view_max_samples:
            region_unique = region_unique[: self.mesh_view_max_samples]

        samples = []
        fan_half_angle_rad = math.radians(
            max(0.0, min(89.0, float(self.mesh_view_region_fan_half_angle_deg)))
        )
        fan_tan = math.tan(fan_half_angle_rad)
        d_max_scale = max(1.0, float(self.mesh_view_region_fan_d_max_scale))

        def region_warn(rid, tri_count, reason, **kwargs):
            if not bool(self.mesh_view_region_drop_warn_enabled):
                return
            parts = [
                "region_id=%d" % int(rid),
                "n_tri=%d" % int(tri_count),
                "reason=%s" % str(reason),
            ]
            for key, val in kwargs.items():
                if isinstance(val, float):
                    if np.isfinite(float(val)):
                        parts.append("%s=%.6f" % (str(key), float(val)))
                    else:
                        parts.append("%s=%s" % (str(key), str(val)))
                else:
                    parts.append("%s=%s" % (str(key), str(val)))
            rospy.logwarn("Region viewpoint dropped: %s", " ".join(parts))

        for rid in region_unique.tolist():
            tri_ids = np.where(region_ids == int(rid))[0]
            if tri_ids.size == 0:
                continue

            tris = triangles[tri_ids]
            c = np.mean(tris, axis=1)
            n_raw = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
            n_len = np.linalg.norm(n_raw, axis=1)
            valid = n_len > 1e-10
            if not np.any(valid):
                region_warn(rid, tri_ids.size, "all_triangle_normals_degenerate")
                continue

            c = c[valid]
            n_raw = n_raw[valid]
            n_len = n_len[valid]
            n_unit = n_raw / n_len[:, None]
            out_sign = np.sign(np.einsum("ij,ij->i", n_unit, c - build_center.reshape(1, 3)))
            out_sign[out_sign == 0.0] = 1.0
            n_out = n_unit * out_sign[:, None]

            w = n_len
            wsum = float(np.sum(w))
            if wsum <= 1e-12:
                region_warn(rid, tri_ids.size, "zero_area_weight_sum")
                continue
            center = np.sum(c * w[:, None], axis=0) / wsum
            normal = np.sum(n_out * w[:, None], axis=0) / wsum
            nn = float(np.linalg.norm(normal))
            if nn <= 1e-10:
                region_warn(rid, tri_ids.size, "region_normal_collapsed")
                continue
            normal = normal / nn

            patch_vertices = tris.reshape(-1, 3)
            d_base, axis_n, right_base, up_base = self._compute_patch_view_geometry(
                patch_vertices=patch_vertices,
                center=center,
                normal=normal,
                tan_h=tan_h,
                tan_v=tan_v,
            )
            if d_base is None:
                region_warn(rid, tri_ids.size, "patch_view_geometry_failed")
                continue

            d_min = float(d_base)
            d_max = float(max(d_min, d_max_scale * d_min))

            tri_cent = np.mean(tris, axis=1)
            tri_area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
            tri_area = np.asarray(tri_area, dtype=np.float64)
            mean_area = float(np.mean(tri_area)) if tri_area.size > 0 else 1.0
            if mean_area <= 1e-12:
                mean_area = 1.0
            area_w = tri_area / mean_area

            tri_unseen = np.zeros((tri_ids.size,), dtype=np.float64)
            if tri_ids.size > 0:
                tri_region_cvi = np.asarray(tri_cvi[tri_ids], dtype=np.float64)
                cmin = float(np.min(tri_region_cvi))
                cmax = float(np.max(tri_region_cvi))
                if abs(cmax - cmin) < 1e-9:
                    tri_unseen = np.ones_like(tri_region_cvi, dtype=np.float64)
                else:
                    tri_unseen = 1.0 - ((tri_region_cvi - cmin) / (cmax - cmin))

            # Keep all per-triangle scoring arrays aligned with the valid-normal subset.
            tri_cent = tri_cent[valid]
            tri_area = tri_area[valid]
            area_w = area_w[valid]
            tri_unseen = tri_unseen[valid]

            best_sample = None
            best_score = -np.inf
            attempts = max(
                int(self.mesh_view_region_samples_per_region),
                int(self.mesh_view_region_samples_per_region) * 3,
            )
            accepted = 0
            rejected_ground = 0
            rejected_inside = 0
            rejected_pose = 0
            pushed_outside = 0
            for _ in range(attempts):
                if accepted >= int(self.mesh_view_region_samples_per_region):
                    break

                d = float(np.random.uniform(d_min, d_max))
                lateral_max = float(max(0.0, fan_tan * d))
                rho = lateral_max * math.sqrt(float(np.random.rand()))
                phi = float(np.random.uniform(0.0, 2.0 * math.pi))
                lateral = rho * (math.cos(phi) * right_base + math.sin(phi) * up_base)
                viewpoint = center + d * axis_n + lateral
                viewpoint = self._pull_candidate_above_ground(viewpoint=viewpoint, target=center)
                if viewpoint is None:
                    rejected_ground += 1
                    continue

                pushed_viewpoint = self._push_candidate_outside_building(viewpoint=viewpoint, target=center)
                if pushed_viewpoint is None:
                    rejected_inside += 1
                    continue
                if float(np.linalg.norm(np.asarray(pushed_viewpoint, dtype=np.float64).reshape(3) - np.asarray(viewpoint, dtype=np.float64).reshape(3))) > 1.0e-9:
                    pushed_outside += 1
                viewpoint = np.asarray(pushed_viewpoint, dtype=np.float64).reshape(3)
                if not self._planner_viewpoint_origin_allowed(viewpoint):
                    rejected_inside += 1
                    continue

                pose = self._look_at_pose(origin=viewpoint, target=center)
                if pose is None:
                    rejected_pose += 1
                    continue
                accepted += 1

                # Cheap CVI-aware proxy score (no rasterization):
                # score = sum_t [ I_fov(t) * unseen_t * front_t * area_w_t / (dist_t^2 + eps) ]
                # where:
                #   I_fov(t): centroid in camera frustum of this candidate
                #   unseen_t: 1-normalized-CVI for triangle t
                #   front_t: max(0, dot(n_t, dir_tri_to_cam))
                score = 0.0
                view_dir = center - viewpoint
                vd_n = float(np.linalg.norm(view_dir))
                if vd_n <= 1e-9:
                    continue
                forward = view_dir / vd_n
                up_hint_c = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                if abs(float(np.dot(up_hint_c, forward))) > 0.95:
                    up_hint_c = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                right_c = np.cross(forward, up_hint_c)
                rcn = float(np.linalg.norm(right_c))
                if rcn <= 1e-9:
                    continue
                right_c = right_c / rcn
                up_c = np.cross(right_c, forward)
                ucn = float(np.linalg.norm(up_c))
                if ucn <= 1e-9:
                    continue
                up_c = up_c / ucn

                rel_tri = tri_cent - viewpoint.reshape(1, 3)
                x_cam = rel_tri @ forward
                y_cam = rel_tri @ right_c
                z_cam = rel_tri @ up_c
                in_fov = (
                    (x_cam > 1e-6)
                    & (np.abs(y_cam) <= (x_cam * tan_h))
                    & (np.abs(z_cam) <= (x_cam * tan_v))
                )
                if np.any(in_fov):
                    dir_tri_to_cam = viewpoint.reshape(1, 3) - tri_cent
                    dtc_n = np.linalg.norm(dir_tri_to_cam, axis=1)
                    dtc_n = np.maximum(dtc_n, 1e-9)
                    dir_tri_to_cam = dir_tri_to_cam / dtc_n[:, None]
                    front = np.einsum("ij,ij->i", n_out, dir_tri_to_cam)
                    front = np.maximum(front, 0.0)
                    contrib = (
                        tri_unseen
                        * front
                        * area_w
                        / (dtc_n * dtc_n + 1e-6)
                    )
                    score = float(np.sum(contrib[in_fov]))

                if score > best_score:
                    best_score = score
                    best_sample = {
                        "origin": viewpoint,
                        "target": center,
                        "pose": pose,
                        "score": float(score),
                    }

            if best_sample is None:
                # Fallback to deterministic center-normal viewpoint if random candidates fail.
                fallback = self._sample_view_from_patch(
                    patch_vertices=patch_vertices,
                    center=center,
                    normal=normal,
                    tan_h=tan_h,
                    tan_v=tan_v,
                )
                if fallback is None:
                    region_warn(
                        rid,
                        tri_ids.size,
                        "fallback_sampling_failed",
                        valid_triangles=int(np.count_nonzero(valid)),
                        accepted_random=int(accepted),
                        rejected_ground=int(rejected_ground),
                        rejected_inside=int(rejected_inside),
                        rejected_pose=int(rejected_pose),
                        pushed_outside=int(pushed_outside),
                    )
                    continue
                fallback_origin = self._pull_candidate_above_ground(
                    viewpoint=fallback["origin"],
                    target=fallback["target"],
                )
                if fallback_origin is None:
                    region_warn(
                        rid,
                        tri_ids.size,
                        "fallback_ground_clearance_failed",
                        valid_triangles=int(np.count_nonzero(valid)),
                        accepted_random=int(accepted),
                        rejected_ground=int(rejected_ground),
                        rejected_inside=int(rejected_inside),
                        rejected_pose=int(rejected_pose),
                        pushed_outside=int(pushed_outside),
                    )
                    continue
                pushed_fallback_origin = self._push_candidate_outside_building(
                    viewpoint=fallback_origin,
                    target=fallback["target"],
                )
                if pushed_fallback_origin is None:
                    region_warn(
                        rid,
                        tri_ids.size,
                        "fallback_inside_building_failed",
                        valid_triangles=int(np.count_nonzero(valid)),
                        accepted_random=int(accepted),
                        rejected_ground=int(rejected_ground),
                        rejected_inside=int(rejected_inside),
                        rejected_pose=int(rejected_pose),
                        pushed_outside=int(pushed_outside),
                    )
                    continue
                fallback_origin = np.asarray(pushed_fallback_origin, dtype=np.float64).reshape(3)
                if not self._planner_viewpoint_origin_allowed(fallback_origin):
                    region_warn(
                        rid,
                        tri_ids.size,
                        "fallback_origin_still_inside_building",
                        valid_triangles=int(np.count_nonzero(valid)),
                        accepted_random=int(accepted),
                        rejected_ground=int(rejected_ground),
                        rejected_inside=int(rejected_inside),
                        rejected_pose=int(rejected_pose),
                        pushed_outside=int(pushed_outside),
                    )
                    continue
                fallback["origin"] = fallback_origin
                fallback["pose"] = self._look_at_pose(origin=fallback_origin, target=fallback["target"])
                if fallback["pose"] is None:
                    region_warn(
                        rid,
                        tri_ids.size,
                        "fallback_pose_failed",
                        valid_triangles=int(np.count_nonzero(valid)),
                        accepted_random=int(accepted),
                        rejected_ground=int(rejected_ground),
                        rejected_inside=int(rejected_inside),
                        rejected_pose=int(rejected_pose),
                        pushed_outside=int(pushed_outside),
                    )
                    continue
                best_sample = fallback
            if not self._planner_viewpoint_origin_allowed(best_sample["origin"]):
                region_warn(
                    rid,
                    tri_ids.size,
                    "best_sample_origin_inside_building",
                    valid_triangles=int(np.count_nonzero(valid)),
                    accepted_random=int(accepted),
                    rejected_ground=int(rejected_ground),
                    rejected_inside=int(rejected_inside),
                    rejected_pose=int(rejected_pose),
                    pushed_outside=int(pushed_outside),
                )
                continue
            best_sample["score"] = 0.0

            best_sample["region_id"] = int(rid)
            best_sample["tri_ids"] = active_idx[tri_ids.astype(np.int64)]
            best_sample["cvi"] = float(np.mean(tri_cvi[tri_ids])) if tri_ids.size > 0 else 0.0
            samples.append(best_sample)

        return samples

    def _planner_max_region_z(self):
        build_center_mesh_z = -float(self.build_center_z) + float(self.rt_mesh_center_z_offset)
        # RTBoxMesh uses a NED-like mesh convention: -Z is top, +Z is bottom.
        bottom_z = build_center_mesh_z + 0.5 * float(self.build_height)
        return float(bottom_z - float(self.mesh_view_ground_clearance_m))

    def _planner_candidate_max_z(self):
        build_center_mesh_z = -float(self.build_center_z) + float(self.rt_mesh_center_z_offset)
        bottom_z = build_center_mesh_z + 0.5 * float(self.build_height)
        return float(bottom_z - float(self.mesh_view_candidate_ground_clearance_m))

    def _infer_bottom_inactive_triangles(self, triangles, mesh_l0=None):
        tri = np.asarray(triangles, dtype=np.float64)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            return np.zeros((0,), dtype=bool)
        n_tri = int(tri.shape[0])
        if n_tri <= 0 or (not bool(self.rt_mesh_include_bottom)):
            return np.zeros((n_tri,), dtype=bool)
        centers = np.mean(tri, axis=1)
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        cross_norm = np.linalg.norm(cross, axis=1)
        valid = cross_norm > 1.0e-10
        normals = np.zeros_like(cross, dtype=np.float64)
        normals[valid] = cross[valid] / cross_norm[valid][:, None]
        build_center = np.array(
            [self.build_center_x, self.build_center_y, -self.build_center_z + self.rt_mesh_center_z_offset],
            dtype=np.float64,
        )
        outward = np.einsum("ij,ij->i", normals, centers - build_center.reshape(1, 3))
        outward_sign = np.where(outward < 0.0, -1.0, 1.0)
        normals = normals * outward_sign[:, None]
        bottom_z = float(build_center[2] + 0.5 * float(self.build_height))
        l0 = None
        try:
            if mesh_l0 is not None:
                l0 = float(mesh_l0)
        except Exception:
            l0 = None
        if l0 is None or (not np.isfinite(l0)) or l0 <= 0.0:
            l0 = float(max(1.0e-3, self.rt_mesh_target_edge_m))
        tol = float(max(5.0e-2, 1.5 * l0))
        vertex_near_bottom = np.all(np.abs(tri[:, :, 2] - bottom_z) <= tol, axis=1)
        centroid_near_bottom = np.abs(centers[:, 2] - bottom_z) <= tol
        downward_outward = normals[:, 2] >= 0.7
        return np.asarray(valid & vertex_near_bottom & centroid_near_bottom & downward_outward, dtype=bool)

    def _pull_candidate_above_ground(self, viewpoint, target):
        p = np.asarray(viewpoint, dtype=np.float64).reshape(3)
        t = np.asarray(target, dtype=np.float64).reshape(3)
        max_z = float(self._planner_candidate_max_z())
        if float(p[2]) <= max_z:
            return p
        if float(t[2]) > max_z:
            return None
        dz = float(t[2] - p[2])
        if abs(dz) <= 1e-12:
            return None
        alpha = float((max_z - float(p[2])) / dz)
        alpha = max(0.0, min(1.0, alpha))
        return p + alpha * (t - p)

    def _planner_triangle_mask(self, triangles):
        tri = np.asarray(triangles, dtype=np.float64)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            return np.zeros((0,), dtype=bool)
        centroids = np.mean(tri, axis=1)
        return np.asarray(
            centroids[:, 2] <= float(self._planner_max_region_z()),
            dtype=bool,
        )

    def _planner_triangle_activity(self, triangles):
        tri = np.asarray(triangles, dtype=np.float64)
        n_tri = int(tri.shape[0]) if tri.ndim == 3 and tri.shape[1:] == (3, 3) else 0
        eligible_mask = self._planner_triangle_mask(tri)
        bottom_mask = np.asarray(self.mesh_tri_inactive_bottom, dtype=bool).reshape(-1) if self.mesh_tri_inactive_bottom is not None else np.zeros((0,), dtype=bool)
        if bottom_mask.shape[0] == n_tri:
            eligible_mask = np.asarray(eligible_mask & (~bottom_mask), dtype=bool)
        tri_cvi = self._get_triangle_cvi_values(n_tri)
        if tri_cvi.shape[0] != n_tri:
            tri_cvi = np.zeros((n_tri,), dtype=np.float64)
        tri_cvi = np.asarray(tri_cvi, dtype=np.float64)
        finite_cvi = np.isfinite(tri_cvi)
        if self.mesh_view_termination_enabled:
            active_mask = np.asarray(
                eligible_mask & finite_cvi & (tri_cvi < float(self.mesh_view_termination_cvi_threshold)),
                dtype=bool,
            )
        else:
            active_mask = np.asarray(eligible_mask, dtype=bool)
        eligible_count = int(np.count_nonzero(eligible_mask))
        active_count = int(np.count_nonzero(active_mask))
        active_fraction = (
            float(active_count) / float(eligible_count)
            if eligible_count > 0
            else 0.0
        )
        eligible_cvi = tri_cvi[np.asarray(eligible_mask, dtype=bool)]
        return {
            "n_tri": int(n_tri),
            "tri_cvi": tri_cvi,
            "eligible_mask": np.asarray(eligible_mask, dtype=bool),
            "active_mask": np.asarray(active_mask, dtype=bool),
            "eligible_count": int(eligible_count),
            "active_count": int(active_count),
            "active_fraction": float(active_fraction),
            "threshold": float(self.mesh_view_termination_cvi_threshold),
            "eligible_cvi_mean": float(np.mean(eligible_cvi)) if eligible_cvi.size > 0 else float("nan"),
            "eligible_cvi_std": float(np.std(eligible_cvi)) if eligible_cvi.size > 0 else float("nan"),
        }

    @staticmethod
    def _region_color_rgba(region_id, alpha):
        hue = float((0.61803398875 * int(region_id)) % 1.0)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        return ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(alpha))

    def _clear_region_boundaries_marker(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "region_boundaries"
        marker.id = 0
        marker.action = Marker.DELETE
        self.mesh_view_region_boundaries_pub.publish(marker)

    def publish_region_boundaries(self, triangles, tri_vidx, region_ids):
        if not self.mesh_view_region_boundary_enabled:
            self._clear_region_boundaries_marker()
            return

        tri = np.asarray(triangles, dtype=np.float64)
        vdx = np.asarray(tri_vidx, dtype=np.int64)
        rid = np.asarray(region_ids, dtype=np.int32).reshape(-1)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3) or vdx.shape != (tri.shape[0], 3) or rid.shape[0] != tri.shape[0]:
            self._clear_region_boundaries_marker()
            return

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "region_boundaries"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(max(1e-4, self.mesh_view_region_boundary_width))
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = float(np.clip(self.mesh_view_region_boundary_alpha, 0.0, 1.0))

        edge_to_entries = {}
        for tid in range(int(tri.shape[0])):
            region = int(rid[tid])
            if region < 0:
                continue
            vids = vdx[tid]
            for ia, ib in ((0, 1), (1, 2), (2, 0)):
                va = int(vids[ia])
                vb = int(vids[ib])
                key = (va, vb) if va < vb else (vb, va)
                edge_to_entries.setdefault(key, []).append((int(tid), int(ia), int(ib), region))

        points = []
        colors = []
        alpha = float(np.clip(self.mesh_view_region_boundary_alpha, 0.0, 1.0))
        for entries in edge_to_entries.values():
            if len(entries) <= 0:
                continue
            unique_regions = sorted({int(e[3]) for e in entries if int(e[3]) >= 0})
            if len(entries) == 1:
                draw = True
            else:
                draw = len(unique_regions) > 1
            if not draw:
                continue

            tid, ia, ib, _ = entries[0]
            p0 = tri[int(tid), int(ia)]
            p1 = tri[int(tid), int(ib)]
            region_for_color = int(unique_regions[0]) if unique_regions else 0
            col = self._region_color_rgba(region_for_color, alpha)
            points.append(Point(x=float(p0[0]), y=float(p0[1]), z=float(p0[2])))
            points.append(Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2])))
            colors.append(col)
            colors.append(col)

        if not points:
            self._clear_region_boundaries_marker()
            return

        marker.points = points
        marker.colors = colors
        self._rviz_cache_points("region_boundaries_points", marker.points)
        color_arr = self._marker_colors_to_np(marker.colors)
        if color_arr.shape[0] > 0:
            self._rviz_replay_cache["region_boundaries_colors"] = color_arr
        self.mesh_view_region_boundaries_pub.publish(marker)

    @staticmethod
    def _build_triangle_topology(triangles, quantize_eps=1.0e-6):
        tri = np.asarray(triangles, dtype=np.float64)
        if tri.ndim != 3 or tri.shape[1:] != (3, 3):
            return None, None
        n_tri = int(tri.shape[0])
        if n_tri <= 0:
            return None, None

        eps = max(float(quantize_eps), 1.0e-12)
        key_to_vid = {}
        tri_vidx = np.zeros((n_tri, 3), dtype=np.int32)
        next_vid = 0

        for tid in range(n_tri):
            for j in range(3):
                p = tri[tid, j]
                key = tuple(np.round(p / eps).astype(np.int64).tolist())
                vid = key_to_vid.get(key, None)
                if vid is None:
                    vid = next_vid
                    key_to_vid[key] = vid
                    next_vid += 1
                tri_vidx[tid, j] = int(vid)

        edge_to_tris = {}
        for tid in range(n_tri):
            a, b, c = [int(v) for v in tri_vidx[tid]]
            for ea, eb in ((a, b), (b, c), (c, a)):
                key = (ea, eb) if ea < eb else (eb, ea)
                edge_to_tris.setdefault(key, []).append(int(tid))

        neighbors = [set() for _ in range(n_tri)]
        for _, tids in edge_to_tris.items():
            if len(tids) < 2:
                continue
            for i in range(len(tids)):
                ti = int(tids[i])
                for j in range(i + 1, len(tids)):
                    tj = int(tids[j])
                    neighbors[ti].add(tj)
                    neighbors[tj].add(ti)

        return tri_vidx, neighbors

    @staticmethod
    def _farthest_point_seeds(points, n_seeds):
        pts = np.asarray(points, dtype=np.float64)
        n = int(pts.shape[0])
        if n <= 0 or n_seeds <= 0:
            return []
        n_seeds = min(int(n_seeds), n)
        seeds = [0]
        min_d2 = np.sum((pts - pts[0]) ** 2, axis=1)
        min_d2[0] = 0.0
        while len(seeds) < n_seeds:
            idx = int(np.argmax(min_d2))
            if idx in seeds:
                break
            seeds.append(idx)
            d2 = np.sum((pts - pts[idx]) ** 2, axis=1)
            min_d2 = np.minimum(min_d2, d2)
            min_d2[seeds] = 0.0
        return seeds

    @staticmethod
    def _grow_regions_from_seeds(tri_neighbors, seeds, centroids):
        n_tri = len(tri_neighbors)
        region = -np.ones((n_tri,), dtype=np.int32)
        queue = []
        head = 0
        for rid, sid in enumerate(seeds):
            s = int(sid)
            if s < 0 or s >= n_tri or region[s] >= 0:
                continue
            region[s] = int(rid)
            queue.append((s, int(rid)))

        while head < len(queue):
            tid, rid = queue[head]
            head += 1
            for nb in tri_neighbors[int(tid)]:
                if region[int(nb)] >= 0:
                    continue
                region[int(nb)] = int(rid)
                queue.append((int(nb), int(rid)))

        unassigned = np.where(region < 0)[0]
        if unassigned.size > 0:
            seed_pts = np.asarray(centroids, dtype=np.float64)[np.asarray(seeds, dtype=np.int64)]
            for tid in unassigned.tolist():
                p = np.asarray(centroids[int(tid)], dtype=np.float64)
                d2 = np.sum((seed_pts - p[None, :]) ** 2, axis=1)
                region[int(tid)] = int(np.argmin(d2))

        return region

    def _compute_patch_view_geometry(self, patch_vertices, center, normal, tan_h, tan_v):
        center = np.asarray(center, dtype=np.float64).reshape(3)
        normal = np.asarray(normal, dtype=np.float64).reshape(3)
        patch = np.asarray(patch_vertices, dtype=np.float64).reshape(-1, 3)
        if patch.shape[0] <= 0:
            return None, None, None, None

        nn = float(np.linalg.norm(normal))
        if nn <= 1e-10:
            return None, None, None, None
        n = normal / nn

        forward = -n
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(up_hint, forward))) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        right = np.cross(up_hint, forward)
        rn = float(np.linalg.norm(right))
        if rn <= 1e-10:
            return None, None, None, None
        right = right / rn
        up = np.cross(forward, right)
        un = float(np.linalg.norm(up))
        if un <= 1e-10:
            return None, None, None, None
        up = up / un

        rel = patch - center.reshape(1, 3)
        xs = rel @ right
        ys = rel @ up
        d_fit_h = float(np.max(np.abs(xs))) / max(1e-6, float(tan_h))
        d_fit_v = float(np.max(np.abs(ys))) / max(1e-6, float(tan_v))
        d_fit = max(d_fit_h, d_fit_v)
        d = max(float(self.mesh_view_d_safe_m), float(self.mesh_view_fit_margin) * d_fit)
        return d, n, right, up

    def _look_at_pose(self, origin, target):
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        t = np.asarray(target, dtype=np.float64).reshape(3)
        look_vec = t - o
        lv = float(np.linalg.norm(look_vec))
        if lv <= 1e-9:
            return None
        dx, dy, dz = float(look_vec[0]), float(look_vec[1]), float(look_vec[2])
        yaw = math.atan2(dx, dy)
        dist_xy = math.sqrt(dx * dx + dy * dy)
        pitch = math.atan2(dz, dist_xy)
        roll = 0.0
        return (float(o[0]), float(o[1]), float(o[2]), roll, pitch, yaw)

    def _look_at_pose_viz(self, origin, target):
        """
        Visualization look-at convention (standard yaw around z with atan2(dy, dx)).
        Keep separate from control look-at convention used for command generation.
        """
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        t = np.asarray(target, dtype=np.float64).reshape(3)
        look_vec = t - o
        lv = float(np.linalg.norm(look_vec))
        if lv <= 1e-9:
            return None
        dx, dy, dz = float(look_vec[0]), float(look_vec[1]), float(look_vec[2])
        yaw = math.atan2(dy, dx)
        dist_xy = math.sqrt(dx * dx + dy * dy)
        pitch = math.atan2(dz, dist_xy)
        roll = 0.0
        return (float(o[0]), float(o[1]), float(o[2]), roll, pitch, yaw)

    def _is_point_inside_building(self, point_xyz, margin=0.0):
        p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        cx = float(self.build_center_x)
        cy = float(self.build_center_y)
        cz = float(-self.build_center_z + self.rt_mesh_center_z_offset)
        hx = 0.5 * float(self.build_width) + float(margin)
        hy = 0.5 * float(self.build_length) + float(margin)
        hz = 0.5 * float(self.build_height) + float(margin)
        return (
            (cx - hx) <= float(p[0]) <= (cx + hx)
            and (cy - hy) <= float(p[1]) <= (cy + hy)
            and (cz - hz) <= float(p[2]) <= (cz + hz)
        )

    def _push_candidate_outside_building(self, viewpoint, target, margin=None):
        p = np.asarray(viewpoint, dtype=np.float64).reshape(3)
        if margin is None:
            margin = float(self.mesh_view_region_building_avoid_margin)
        margin = float(max(0.0, margin))
        if not self._is_point_inside_building(p, margin=margin):
            return p

        t = np.asarray(target, dtype=np.float64).reshape(3)
        ray = p - t
        ray_norm = float(np.linalg.norm(ray))
        if ray_norm <= 1.0e-12:
            build_center = np.array(
                [self.build_center_x, self.build_center_y, -self.build_center_z + self.rt_mesh_center_z_offset],
                dtype=np.float64,
            )
            ray = p - build_center
            ray_norm = float(np.linalg.norm(ray))
            if ray_norm <= 1.0e-12:
                return None
        ray = ray / ray_norm

        cx = float(self.build_center_x)
        cy = float(self.build_center_y)
        cz = float(-self.build_center_z + self.rt_mesh_center_z_offset)
        hx = 0.5 * float(self.build_width) + margin
        hy = 0.5 * float(self.build_length) + margin
        hz = 0.5 * float(self.build_height) + margin
        bmin = np.array([cx - hx, cy - hy, cz - hz], dtype=np.float64)
        bmax = np.array([cx + hx, cy + hy, cz + hz], dtype=np.float64)

        tmin = -np.inf
        tmax = np.inf
        for axis in range(3):
            da = float(ray[axis])
            pa = float(p[axis])
            if abs(da) <= 1.0e-12:
                if pa < float(bmin[axis]) or pa > float(bmax[axis]):
                    return p
                continue
            ta = float((float(bmin[axis]) - pa) / da)
            tb = float((float(bmax[axis]) - pa) / da)
            t_enter = min(ta, tb)
            t_exit = max(ta, tb)
            tmin = max(float(tmin), float(t_enter))
            tmax = min(float(tmax), float(t_exit))
            if tmax < tmin:
                return None

        if (not np.isfinite(tmax)) or float(tmax) < 0.0:
            return None

        push_eps = float(max(1.0e-3, 0.05 * float(self.rt_mesh_target_edge_m)))
        out = p + (float(tmax) + push_eps) * ray
        if self._is_point_inside_building(out, margin=margin):
            out = p + (float(tmax) + 5.0 * push_eps) * ray
            if self._is_point_inside_building(out, margin=margin):
                return None
        return np.asarray(out, dtype=np.float64).reshape(3)

    def _planner_viewpoint_origin_allowed(self, point_xyz):
        if not self.mesh_view_region_avoid_inside_building:
            return True
        return not self._is_point_inside_building(
            point_xyz,
            margin=float(self.mesh_view_region_building_avoid_margin),
        )

    def _sample_view_from_patch(self, patch_vertices, center, normal, tan_h, tan_v):
        d, n, _, _ = self._compute_patch_view_geometry(
            patch_vertices=patch_vertices,
            center=center,
            normal=normal,
            tan_h=tan_h,
            tan_v=tan_v,
        )
        if d is None:
            return None

        viewpoint = center + d * n
        pose = self._look_at_pose(origin=viewpoint, target=center)
        if pose is None:
            return None
        return {
            "origin": viewpoint,
            "target": center,
            "pose": pose,
        }

    def _get_triangle_cvi_values(self, n_tri):
        if (
            self.mesh_tri_cvi_effective is not None
            and int(self.mesh_tri_cvi_effective.shape[0]) == int(n_tri)
        ):
            return np.asarray(self.mesh_tri_cvi_effective, dtype=np.float64)
        if (
            self.mesh_tri_cvi_raw is not None
            and int(self.mesh_tri_cvi_raw.shape[0]) == int(n_tri)
        ):
            return np.asarray(self.mesh_tri_cvi_raw, dtype=np.float64)
        return np.zeros((int(n_tri),), dtype=np.float64)

    def _get_triangle_cvi_raw_values(self, n_tri):
        if self.mesh_tri_cvi_raw is not None and int(self.mesh_tri_cvi_raw.shape[0]) == int(n_tri):
            return np.asarray(self.mesh_tri_cvi_raw, dtype=np.float64)
        return np.zeros((int(n_tri),), dtype=np.float64)

    def _get_triangle_conf_values(self, n_tri):
        if self.mesh_tri_conf is not None and int(self.mesh_tri_conf.shape[0]) == int(n_tri):
            return np.asarray(self.mesh_tri_conf, dtype=np.float64)
        return np.ones((int(n_tri),), dtype=np.float64)

    @staticmethod
    def _normalize_minmax_to_range(values, out_min, out_max):
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size <= 0:
            return np.zeros((0,), dtype=np.float64)
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1.0e-12:
            return np.full(arr.shape, float(out_min), dtype=np.float64)
        norm = (arr - vmin) / (vmax - vmin)
        return float(out_min) + (float(out_max) - float(out_min)) * norm

    def _effective_cvi_bounds(self):
        cmin = max(0.0, float(self.mesh_conf_min))
        cmax = max(cmin, float(self.mesh_conf_max))
        eff_min = 1.0 * cmin
        eff_max = 2.0 * cmax
        if eff_max <= eff_min + 1.0e-12:
            eff_max = eff_min + 1.0
        return float(eff_min), float(eff_max)

    def _local_planning_mode_label(self):
        if not self.mesh_view_local_planning_enabled:
            return "off"
        if self.mesh_view_local_orientation_enabled:
            return "force_field"
        return "path_only"

    @staticmethod
    def _angle_diff(a, b):
        return (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _unit_vector_or_none(vec, eps=1.0e-9):
        try:
            v = np.asarray(vec, dtype=np.float64).reshape(3)
        except Exception:
            return None
        n = float(np.linalg.norm(v))
        if (not np.isfinite(n)) or n <= float(eps):
            return None
        return v / n

    @staticmethod
    def _format_elapsed_hms(seconds):
        total_s = max(0, int(round(float(seconds))))
        hours = total_s // 3600
        minutes = (total_s % 3600) // 60
        secs = total_s % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _format_elapsed_min_sec(seconds):
        total_s = max(0.0, float(seconds))
        minutes = int(total_s // 60.0)
        secs = total_s - 60.0 * float(minutes)
        return minutes, secs

    def _current_drone_world_point(self):
        if self.latest_odom is None:
            return None
        try:
            p = self.latest_odom.pose.pose.position
            return np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
        except Exception:
            return None

    def _look_at_building_center_angles_from_world_point(self, world_point):
        origin = np.asarray(world_point, dtype=np.float64).reshape(3)
        target = self._mesh_to_flight_point(self._building_center_mesh_point())
        pose = self._look_at_pose(origin=origin, target=target)
        if pose is None:
            return None
        return (float(pose[3]), float(pose[4]), float(pose[5]))

    def _append_actual_path_point(self, point_xyz):
        p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        if self._mission_actual_last_odom_point is not None:
            delta = p - np.asarray(self._mission_actual_last_odom_point, dtype=np.float64).reshape(3)
            step = float(np.linalg.norm(delta))
            if np.isfinite(step) and step > 0.0:
                self._mission_actual_path_length_m += float(step)
        self._mission_actual_last_odom_point = p.copy()
        if self._mission_actual_last_saved_point is None:
            self._mission_actual_last_saved_point = p.copy()
            self._mission_actual_path_points.append(p.copy())
            return
        save_delta = p - np.asarray(self._mission_actual_last_saved_point, dtype=np.float64).reshape(3)
        if float(np.linalg.norm(save_delta)) >= float(self._mission_path_record_step_m):
            self._mission_actual_path_points.append(p.copy())
            self._mission_actual_last_saved_point = p.copy()

    def _save_actual_path_file(self, elapsed_s):
        pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in list(self._mission_actual_path_points)]
        cur = self._current_drone_world_point()
        if cur is not None:
            if len(pts) <= 0 or float(np.linalg.norm(cur - pts[-1])) > 1.0e-6:
                pts.append(cur.copy())
        try:
            with open(self.actual_path_filename, "w") as f:
                f.write(
                    "# elapsed_s=%.6f elapsed_hms=%s path_length_m=%.6f num_points=%d\n"
                    % (
                        float(elapsed_s),
                        str(self._format_elapsed_hms(elapsed_s)),
                        float(self._mission_actual_path_length_m),
                        int(len(pts)),
                    )
                )
                f.write("# x y z\n")
                for p in pts:
                    f.write("%.6f %.6f %.6f\n" % (float(p[0]), float(p[1]), float(p[2])))
            return True
        except Exception as exc:
            rospy.logwarn("Failed to save actual flight path to %s: %s", self.actual_path_filename, str(exc))
            return False

    def _count_saved_images(self):
        try:
            return int(
                sum(
                    1
                    for name in os.listdir(self.image_dir)
                    if str(name).lower().endswith((".jpg", ".jpeg", ".png"))
                )
            )
        except Exception:
            return 0

    def _count_saved_pose_entries(self):
        try:
            count = 0
            with open(self.pose_filename, "r") as f:
                for line in f:
                    s = line.strip()
                    if s and not s.startswith("#"):
                        count += 1
            return int(count)
        except Exception:
            return 0

    @staticmethod
    def _is_mission_image_name(name):
        base = os.path.basename(str(name))
        lower = base.lower()
        if not lower.endswith((".jpg", ".jpeg", ".png")):
            return False
        if not lower.startswith("image_"):
            return False
        stem = os.path.splitext(base)[0]
        suffix = stem[len("image_") :]
        return bool(suffix.isdigit())

    @staticmethod
    def _split_pose_file(path):
        headers = []
        rows = []
        if not os.path.isfile(path):
            return headers, rows
        with open(path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    headers.append(line.rstrip("\n"))
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                rows.append((parts[-1], line.rstrip("\n")))
        return headers, rows

    @staticmethod
    def _rewrite_pose_file(path, headers, rows):
        with open(path, "w") as f:
            for line in headers:
                f.write(str(line).rstrip("\n") + "\n")
            for _image_name, line in rows:
                f.write(str(line).rstrip("\n") + "\n")

    def _align_image_pose_logs(self, reason="shutdown", force=False):
        if not bool(getattr(self, "align_image_pose_logs_on_shutdown", True)):
            return None
        if bool(getattr(self, "_image_pose_alignment_done", False)) and not bool(force):
            return None
        self._image_pose_alignment_done = True

        try:
            image_names = {
                str(name)
                for name in os.listdir(self.image_dir)
                if self._is_mission_image_name(name)
            }
        except Exception as exc:
            rospy.logwarn("[WARN] Failed to scan image directory for pose alignment: %s", str(exc))
            return None

        try:
            pose_headers, pose_rows = self._split_pose_file(self.pose_filename)
            degree_headers, degree_rows = self._split_pose_file(self.pose_filename_degrees)

            kept_pose_rows = [(img, line) for img, line in pose_rows if img in image_names]
            dropped_pose_rows = int(len(pose_rows) - len(kept_pose_rows))
            keep_images = {img for img, _line in kept_pose_rows}

            deleted_images = 0
            for image_name in sorted(image_names - keep_images):
                path = os.path.join(self.image_dir, image_name)
                try:
                    os.remove(path)
                    deleted_images += 1
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    rospy.logwarn("[WARN] Failed to delete unmatched image %s: %s", path, str(exc))

            kept_degree_rows = [(img, line) for img, line in degree_rows if img in keep_images]
            dropped_degree_rows = int(len(degree_rows) - len(kept_degree_rows))

            self._rewrite_pose_file(self.pose_filename, pose_headers, kept_pose_rows)
            self._rewrite_pose_file(self.pose_filename_degrees, degree_headers, kept_degree_rows)

            rospy.loginfo(
                "Image/pose alignment complete (%s): kept=%d deleted_images=%d dropped_poses=%d dropped_degree_poses=%d",
                str(reason),
                int(len(keep_images)),
                int(deleted_images),
                int(dropped_pose_rows),
                int(dropped_degree_rows),
            )
            self._convert_pose_files_after_alignment(reason=reason)
            return {
                "kept": int(len(keep_images)),
                "deleted_images": int(deleted_images),
                "dropped_poses": int(dropped_pose_rows),
                "dropped_degree_poses": int(dropped_degree_rows),
            }
        except Exception as exc:
            rospy.logwarn("[WARN] Failed to align image/pose logs: %s", str(exc))
            return None

    def _convert_pose_files_after_alignment(self, reason="finalize", force=False):
        if not bool(getattr(self, "convert_poses_file_on_finalize", True)):
            return False
        if bool(getattr(self, "_pose_file_conversion_done", False)) and not bool(force):
            return False
        if self._count_saved_pose_entries() <= 0:
            rospy.logwarn("[WARN] Skipping pose conversion (%s): no pose rows found.", str(reason))
            return False
        script = str(getattr(self, "convert_poses_file_script", "")).strip()
        if not script or not os.path.isfile(script):
            rospy.logwarn("[WARN] Skipping pose conversion (%s): script not found: %s", str(reason), script)
            return False
        output_csv = os.path.join(self.pose_dir, "poses_metashape_ypr.csv")
        cmd = [
            sys.executable or "python3",
            script,
            "--poses",
            self.pose_filename,
            "--image-dir",
            self.image_dir,
            "--output",
            output_csv,
        ]
        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self._pose_file_conversion_done = True
            stdout = (proc.stdout or "").strip()
            if stdout:
                rospy.loginfo("Pose conversion complete (%s):\n%s", str(reason), stdout)
            else:
                rospy.loginfo(
                    "Pose conversion complete (%s): csv=%s geo=%s",
                    str(reason),
                    output_csv,
                    os.path.join(self.image_dir, "geo.txt"),
                )
            return True
        except Exception as exc:
            stderr = getattr(exc, "stderr", None)
            if stderr:
                rospy.logwarn("[WARN] Pose conversion failed (%s): %s\n%s", str(reason), str(exc), str(stderr))
            else:
                rospy.logwarn("[WARN] Pose conversion failed (%s): %s", str(reason), str(exc))
            return False

    def _align_image_pose_logs_on_shutdown(self):
        self._align_image_pose_logs(reason="shutdown")
        if not bool(getattr(self, "_rviz_replay_saved", False)):
            try:
                if int(self._count_saved_images()) > 0 or int(self._count_saved_pose_entries()) > 0:
                    elapsed_s = max(0.0, float(time.monotonic() - float(self._mission_started_t)))
                    self._save_actual_path_file(elapsed_s)
                    self._save_rviz_replay_bundle(reason="shutdown")
            except Exception as exc:
                rospy.logwarn("Failed to save RViz replay bundle during shutdown: %s", str(exc))

    def _build_mission_metrics(self, elapsed_s, status="complete"):
        minutes, secs = self._format_elapsed_min_sec(elapsed_s)
        activity = {}
        try:
            for k, v in dict(self._mission_completion_activity_snapshot).items():
                if isinstance(v, np.generic):
                    v = v.item()
                if isinstance(v, (bool, int, float, str)) or v is None:
                    activity[str(k)] = v
        except Exception:
            activity = {}
        return {
            "keyword": "MISSION_METRICS",
            "status": str(status),
            "reason": str(self._mission_completion_reason),
            "elapsed_s": float(elapsed_s),
            "elapsed_hms": str(self._format_elapsed_hms(elapsed_s)),
            "elapsed_minutes": int(minutes),
            "elapsed_remaining_seconds": float(secs),
            "actual_path_length_m": float(self._mission_actual_path_length_m),
            "actual_path_points": int(len(self._mission_actual_path_points)),
            "actual_path_file": str(self.actual_path_filename),
            "metrics_file": str(self.mission_metrics_filename),
            "metrics_json_file": str(self.mission_metrics_json_filename),
            "image_count": int(self._count_saved_images()),
            "pose_count": int(self._count_saved_pose_entries()),
            "completion_stage": str(self._mission_completion_stage),
            "activity": activity,
        }

    def _save_mission_metrics_file(self, elapsed_s, status="complete"):
        metrics = self._build_mission_metrics(elapsed_s, status=status)
        try:
            with open(self.mission_metrics_filename, "w") as f:
                f.write("MISSION_METRICS\n")
                for key in sorted(k for k in metrics.keys() if k != "activity"):
                    f.write("%s=%s\n" % (str(key), str(metrics[key])))
                if metrics.get("activity"):
                    f.write("[activity]\n")
                    for key in sorted(metrics["activity"].keys()):
                        f.write("%s=%s\n" % (str(key), str(metrics["activity"][key])))
            with open(self.mission_metrics_json_filename, "w") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as exc:
            rospy.logwarn(
                "Failed to save mission metrics to %s: %s",
                self.mission_metrics_filename,
                str(exc),
            )
        return metrics

    @staticmethod
    def _marker_points_to_np(points):
        arr = []
        for p in list(points) if points is not None else []:
            try:
                arr.append((float(p.x), float(p.y), float(p.z)))
            except Exception:
                continue
        if not arr:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(arr, dtype=np.float32).reshape(-1, 3)

    @staticmethod
    def _marker_colors_to_np(colors):
        arr = []
        for c in list(colors) if colors is not None else []:
            try:
                arr.append((float(c.r), float(c.g), float(c.b), float(c.a)))
            except Exception:
                continue
        if not arr:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray(arr, dtype=np.float32).reshape(-1, 4)

    def _rviz_cache_points(self, key, points):
        try:
            arr = self._marker_points_to_np(points)
            if arr.shape[0] > 0:
                self._rviz_replay_cache[str(key)] = arr
        except Exception as exc:
            rospy.logwarn("Failed to cache RViz replay points for %s: %s", str(key), str(exc))

    def _rviz_cache_array(self, key, values, width=3):
        try:
            arr = np.asarray(values, dtype=np.float32).reshape(-1, int(width))
            if arr.shape[0] > 0:
                self._rviz_replay_cache[str(key)] = arr
        except Exception as exc:
            rospy.logwarn("Failed to cache RViz replay array for %s: %s", str(key), str(exc))

    def _copy_rviz_replay_file(self, src, dst_name, manifest):
        try:
            src = os.path.abspath(os.path.expanduser(str(src)))
            if not os.path.isfile(src):
                return None
            dst = os.path.join(self.rviz_replay_dir, str(dst_name))
            shutil.copy2(src, dst)
            manifest[str(dst_name)] = {
                "source": src,
                "bytes": int(os.path.getsize(dst)),
            }
            return dst
        except Exception as exc:
            rospy.logwarn("Failed to copy RViz replay artifact %s: %s", str(src), str(exc))
            return None

    def _save_rviz_replay_bundle(self, reason="final"):
        try:
            os.makedirs(self.rviz_replay_dir, exist_ok=True)
        except Exception as exc:
            rospy.logwarn("Failed to create RViz replay directory %s: %s", self.rviz_replay_dir, str(exc))
            return False

        manifest = {
            "reason": str(reason),
            "saved_unix": float(time.time()),
            "frame_id": str(self.frame_id),
            "image_dir": str(self.image_dir),
            "pose_dir": str(self.pose_dir),
            "geometry_point_source": str(self.geometry_point_source),
            "files": {},
        }

        self._copy_rviz_replay_file(self.mesh_filename, "mesh_latest.npz", manifest["files"])

        sparse_src = self.rt_mesh_sfm_output_path
        depth_src = self.rt_mesh_depth_output_path
        copied_sparse = self._copy_rviz_replay_file(sparse_src, "sparse_latest.npz", manifest["files"])
        copied_depth = self._copy_rviz_replay_file(depth_src, "depth_latest.npz", manifest["files"])
        if str(self.geometry_point_source) == "depth" and copied_depth is not None:
            self._copy_rviz_replay_file(depth_src, "geometry_points_latest.npz", manifest["files"])
        elif copied_sparse is not None:
            self._copy_rviz_replay_file(sparse_src, "geometry_points_latest.npz", manifest["files"])
        elif copied_depth is not None:
            self._copy_rviz_replay_file(depth_src, "geometry_points_latest.npz", manifest["files"])

        arrays = {}
        for key, value in dict(self._rviz_replay_cache).items():
            try:
                arrays[str(key)] = np.asarray(value, dtype=np.float32)
            except Exception:
                pass
        arrays["frame_id"] = np.asarray([str(self.frame_id)])
        arrays["geometry_point_source"] = np.asarray([str(self.geometry_point_source)])
        arrays["saved_unix"] = np.asarray([time.time()], dtype=np.float64)
        markers_path = os.path.join(self.rviz_replay_dir, "rviz_replay_markers.npz")
        try:
            np.savez(markers_path, **arrays)
            manifest["files"]["rviz_replay_markers.npz"] = {
                "source": "controller_cache",
                "bytes": int(os.path.getsize(markers_path)),
            }
        except Exception as exc:
            rospy.logwarn("Failed to save RViz replay markers to %s: %s", markers_path, str(exc))

        manifest_path = os.path.join(self.rviz_replay_dir, "manifest.json")
        try:
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception as exc:
            rospy.logwarn("Failed to save RViz replay manifest to %s: %s", manifest_path, str(exc))

        try:
            total_bytes = 0
            for root, _dirs, files in os.walk(self.rviz_replay_dir):
                for name in files:
                    try:
                        total_bytes += int(os.path.getsize(os.path.join(root, name)))
                    except Exception:
                        pass
            rospy.logwarn(
                "RVIZ_REPLAY_SAVED dir=%s size=%.2f MB files=%d",
                str(self.rviz_replay_dir),
                float(total_bytes) / (1024.0 * 1024.0),
                int(sum(len(files) for _root, _dirs, files in os.walk(self.rviz_replay_dir))),
            )
        except Exception:
            rospy.logwarn("RVIZ_REPLAY_SAVED dir=%s", str(self.rviz_replay_dir))
        self._rviz_replay_saved = True
        return True

    def _stop_image_capture(self):
        if not bool(getattr(self, "_capture_images_enabled", True)):
            return
        self._capture_images_enabled = False
        stopped_sub_ids = set()
        for name in ("image_sub", "depth_rgb_sub", "depth_sub"):
            sub = getattr(self, name, None)
            if sub is None or id(sub) in stopped_sub_ids:
                continue
            stopped_sub_ids.add(id(sub))
            try:
                sub.unregister()
            except Exception as exc:
                rospy.logwarn("Failed to unregister %s after mission completion: %s", str(name), str(exc))
        with self._depth_capture_lock:
            self._depth_capture_records.clear()
        rospy.logwarn(
            "MISSION_IMAGE_CAPTURE_STOPPED saved_images=%d reason=%s",
            int(self._count_saved_images()),
            str(self._mission_completion_reason),
        )

    def _begin_mission_completion(self, reason="termination", activity=None):
        if bool(self._mission_completion_active):
            return
        self._mission_completion_active = True
        self._mission_completion_reason = str(reason)
        self._mission_completion_logged = False
        self._mission_completion_stage = "return_xy"
        self._mission_completion_activity_snapshot = dict(activity) if activity is not None else {}
        self._clear_region_boundaries_marker()
        self._clear_start_leg_checked_points()
        self.publish_sampled_view_vectors([])
        self.publish_sampled_view_trajectory([], exec_waypoints=[])
        self._clear_actual_flight_path_marker()
        self._clear_planned_trajectory(hold_current_pose=False)
        self._mesh_view_local_orientation_last_ms = 0.0
        self._mesh_view_local_orientation_last_segments = 0
        cur = self._current_drone_world_point()
        if cur is not None:
            self._mission_completion_hold_z = float(cur[2])
        else:
            try:
                self._mission_completion_hold_z = float(self.target_pos[2])
            except Exception:
                self._mission_completion_hold_z = float(self.home_pos[2])
        self.target_pos = (
            float(self.home_pos[0]),
            float(self.home_pos[1]),
            float(self._mission_completion_hold_z),
        )
        self.target_velocity = (0.0, 0.0, 0.0)
        look_angles = self._look_at_building_center_angles_from_world_point(self.home_pos)
        if look_angles is None:
            look_angles = (0.0, float(self.camera_pitch_cmd_current), float(self.init_yaw))
        self.target_angles = tuple(float(v) for v in look_angles[:3])
        self.drone_updated = True
        self.camera_updated = True
        self.last_drone_update_time = rospy.Time.now()
        rospy.loginfo(
            "Mission completion started: reason=%s returning to home position=(%.2f, %.2f, %.2f)",
            str(self._mission_completion_reason),
            float(self.home_pos[0]),
            float(self.home_pos[1]),
            float(self.home_pos[2]),
        )
        try:
            active_count = int(activity.get("active_count", 0)) if activity is not None else 0
            eligible_count = int(activity.get("eligible_count", 0)) if activity is not None else 0
            active_fraction = float(activity.get("active_fraction", 0.0)) if activity is not None else 0.0
        except Exception:
            active_count = 0
            eligible_count = 0
            active_fraction = 0.0
        elapsed_s = max(0.0, float(time.monotonic() - float(self._mission_started_t)))
        self._stop_image_capture()
        self._align_image_pose_logs(reason="mission_completion_condition_met", force=True)
        self._save_actual_path_file(elapsed_s)
        metrics = self._save_mission_metrics_file(elapsed_s, status="condition_met")
        self._save_rviz_replay_bundle(reason="condition_met")
        self._publish_actual_flight_path_marker()
        rospy.sleep(0.10)
        rospy.logwarn(
            "MISSION_METRICS_FINAL status=condition_met elapsed=%.2f s (%d min %.2f s, %s) actual_path_length=%.2f m active=%d eligible=%d fraction=%.4f threshold=%.4f images=%d poses=%d metrics_file=%s path_file=%s",
            float(metrics.get("elapsed_s", elapsed_s)),
            int(metrics.get("elapsed_minutes", 0)),
            float(metrics.get("elapsed_remaining_seconds", 0.0)),
            str(metrics.get("elapsed_hms", self._format_elapsed_hms(elapsed_s))),
            float(metrics.get("actual_path_length_m", self._mission_actual_path_length_m)),
            int(active_count),
            int(eligible_count),
            float(active_fraction),
            float(self.mesh_view_termination_active_fraction_epsilon),
            int(metrics.get("image_count", 0)),
            int(metrics.get("pose_count", 0)),
            str(self.mission_metrics_filename),
            str(self.actual_path_filename),
        )
        self._mission_completion_logged = True

    def _complete_baseline_circle(self, shutdown=True):
        if bool(self._mission_completion_logged):
            if bool(shutdown):
                rospy.signal_shutdown("Baseline circle completed")
            return
        self._mission_completion_reason = "baseline_circle_complete"
        self._mission_completion_stage = "baseline_circle_complete"
        self._mission_completion_activity_snapshot = {
            "pathing_mode": str(self.pathing_mode),
            "baseline_circle_continuous": bool(self.baseline_circle_continuous),
            "baseline_circle_points": int(self.baseline_circle_points),
            "baseline_circle_speed_mps": float(self.baseline_circle_speed_mps),
            "baseline_circle_radius_m": float(self._baseline_circle_radius_m),
            "baseline_circle_phase": str(self._baseline_circle_phase),
            "baseline_viewpoints_issued": int(self.vp_cnt),
        }
        self.target_velocity = (0.0, 0.0, 0.0)
        self._stop_image_capture()
        self._align_image_pose_logs(reason="baseline_circle_final", force=True)
        elapsed_s = max(0.0, float(time.monotonic() - float(self._mission_started_t)))
        self._save_actual_path_file(elapsed_s)
        metrics = self._save_mission_metrics_file(elapsed_s, status="baseline_circle_complete")
        self._save_rviz_replay_bundle(reason="baseline_circle_complete")
        self._publish_actual_flight_path_marker()
        rospy.sleep(0.10)
        rospy.logwarn(
            "MISSION_METRICS_FINAL mode=baseline_circle elapsed=%.2f s (%d min %.2f s, %s) actual_path_length=%.2f m images=%d poses=%d metrics_file=%s path_file=%s",
            float(metrics.get("elapsed_s", elapsed_s)),
            int(metrics.get("elapsed_minutes", 0)),
            float(metrics.get("elapsed_remaining_seconds", 0.0)),
            str(metrics.get("elapsed_hms", self._format_elapsed_hms(elapsed_s))),
            float(metrics.get("actual_path_length_m", self._mission_actual_path_length_m)),
            int(metrics.get("image_count", 0)),
            int(metrics.get("pose_count", 0)),
            str(self.mission_metrics_filename),
            str(self.actual_path_filename),
        )
        self._mission_completion_logged = True
        if bool(shutdown):
            rospy.signal_shutdown("Baseline circle completed")

    def _update_mission_completion(self):
        if not bool(self._mission_completion_active):
            return
        home = np.asarray(self.home_pos, dtype=np.float64).reshape(3)
        cur = self._current_drone_world_point()
        look_origin = cur if cur is not None else home
        look_angles = self._look_at_building_center_angles_from_world_point(look_origin)
        if look_angles is None:
            look_angles = (0.0, float(self.camera_pitch_cmd_current), float(self.init_yaw))
        self.target_angles = tuple(float(v) for v in look_angles[:3])
        if cur is None:
            rospy.logwarn_throttle(
                2.0,
                "Mission completion waiting for odometry before confirming return-home.",
            )
            return
        tol = float(max(1.0e-3, self.mesh_view_reach_pos_tol_m))
        xy_dist = float(np.linalg.norm(cur[:2] - home[:2]))
        z_err = float(abs(float(cur[2]) - float(home[2])))

        if str(self._mission_completion_stage) == "return_xy":
            self.target_pos = (
                float(home[0]),
                float(home[1]),
                float(self._mission_completion_hold_z),
            )
            if xy_dist > tol:
                if bool(self.mission_completion_progress_logs):
                    rospy.loginfo_throttle(
                        10.0,
                        "Mission completion: return_xy dist_xy=%.3f tol=%.3f",
                        float(xy_dist),
                        float(tol),
                    )
                return
            if not bool(self._mission_completion_logged):
                elapsed_s = max(0.0, float(time.monotonic() - float(self._mission_started_t)))
                self._stop_image_capture()
                self._align_image_pose_logs(reason="mission_completion_before_landing", force=True)
                self._save_actual_path_file(elapsed_s)
                metrics = self._save_mission_metrics_file(elapsed_s, status="before_landing")
                self._save_rviz_replay_bundle(reason="before_landing")
                self._publish_actual_flight_path_marker()
                rospy.sleep(0.10)
                rospy.logwarn(
                    "MISSION_METRICS_FINAL status=before_landing elapsed=%.2f s (%d min %.2f s, %s) actual_path_length=%.2f m images=%d poses=%d metrics_file=%s path_file=%s",
                    float(metrics.get("elapsed_s", elapsed_s)),
                    int(metrics.get("elapsed_minutes", 0)),
                    float(metrics.get("elapsed_remaining_seconds", 0.0)),
                    str(metrics.get("elapsed_hms", self._format_elapsed_hms(elapsed_s))),
                    float(metrics.get("actual_path_length_m", self._mission_actual_path_length_m)),
                    int(metrics.get("image_count", 0)),
                    int(metrics.get("pose_count", 0)),
                    str(self.mission_metrics_filename),
                    str(self.actual_path_filename),
                )
                self._mission_completion_logged = True
            self._mission_completion_stage = "land"
            self.drone_updated = True
            self.camera_updated = True
            if bool(self.mission_completion_progress_logs):
                rospy.loginfo(
                    "Mission completion: horizontal return complete, starting landing at home xy=(%.2f, %.2f) current_z=%.2f target_z=%.2f",
                    float(home[0]),
                    float(home[1]),
                    float(cur[2]),
                    float(home[2]),
                )

        self.target_pos = (float(home[0]), float(home[1]), float(home[2]))
        if xy_dist > tol or z_err > tol:
            if bool(self.mission_completion_progress_logs):
                rospy.loginfo_throttle(
                    10.0,
                    "Mission completion: landing dist_xy=%.3f dz=%.3f tol=%.3f",
                    float(xy_dist),
                    float(z_err),
                    float(tol),
                )
            return
        if bool(self._mission_completion_logged):
            rospy.signal_shutdown("Mesh-view mission completed")
            return
        elapsed_s = max(0.0, float(time.monotonic() - float(self._mission_started_t)))
        self._stop_image_capture()
        self._align_image_pose_logs(reason="mission_completion_final", force=True)
        self._save_actual_path_file(elapsed_s)
        metrics = self._save_mission_metrics_file(elapsed_s, status="complete")
        self._save_rviz_replay_bundle(reason="complete")
        self._publish_actual_flight_path_marker()
        rospy.sleep(0.10)
        rospy.logwarn(
            "MISSION_METRICS_FINAL elapsed=%.2f s (%d min %.2f s, %s) actual_path_length=%.2f m images=%d poses=%d metrics_file=%s path_file=%s",
            float(metrics.get("elapsed_s", elapsed_s)),
            int(metrics.get("elapsed_minutes", 0)),
            float(metrics.get("elapsed_remaining_seconds", 0.0)),
            str(metrics.get("elapsed_hms", self._format_elapsed_hms(elapsed_s))),
            float(metrics.get("actual_path_length_m", self._mission_actual_path_length_m)),
            int(metrics.get("image_count", 0)),
            int(metrics.get("pose_count", 0)),
            str(self.mission_metrics_filename),
            str(self.actual_path_filename),
        )
        self._mission_completion_logged = True
        rospy.signal_shutdown("Mesh-view mission completed")

    def _step_camera_pitch_command(self):
        target_pitch = float(self.target_angles[1])
        now = rospy.Time.now().to_sec()
        if self._camera_pitch_cmd_last_t is None:
            self._camera_pitch_cmd_last_t = now
            self.camera_pitch_cmd_current = target_pitch
            return float(self.camera_pitch_cmd_current)

        dt = max(0.0, float(now - self._camera_pitch_cmd_last_t))
        self._camera_pitch_cmd_last_t = now
        if (not self.camera_pitch_smooth_enabled) or dt <= 0.0:
            self.camera_pitch_cmd_current = target_pitch
            return float(self.camera_pitch_cmd_current)

        max_step = float(self.camera_pitch_rate_rps) * dt
        err = float(target_pitch - self.camera_pitch_cmd_current)
        step = float(np.clip(err, -max_step, max_step))
        self.camera_pitch_cmd_current = float(self.camera_pitch_cmd_current + step)
        return float(self.camera_pitch_cmd_current)

    def _estimate_camera_pose_from_latest_odom(self):
        """
        Estimate current camera pose in world frame from latest body odom + camera mount + commanded gimbal pitch.
        """
        if self.latest_odom is None:
            return None, None
        p = self.latest_odom.pose.pose.position
        q = self.latest_odom.pose.pose.orientation
        p_world_body = np.array([p.x, p.y, p.z], dtype=np.float64)
        q_world_body = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        R_world_body = quaternion_matrix(q_world_body)[:3, :3]
        p_world_cam = p_world_body + R_world_body.dot(self.cam_offset_body.astype(np.float64))

        pitch_cmd = float(self.camera_pitch_cmd_current)
        q_dyn = quaternion_from_euler(0.0, float(pitch_cmd), 0.0, axes="sxyz")
        q_body_cam_total = quaternion_multiply(self.q_body_cam, q_dyn)
        q_world_cam = quaternion_multiply(q_world_body, q_body_cam_total)
        return p_world_cam, q_world_cam

    def _current_camera_pose_mesh_state(self):
        cam_pos_world, cam_q_world = self._estimate_camera_pose_from_latest_odom()
        if cam_pos_world is None or cam_q_world is None:
            return None, None, None
        _, pitch, yaw = tft.euler_from_quaternion(
            [cam_q_world[0], cam_q_world[1], cam_q_world[2], cam_q_world[3]],
            axes="sxyz",
        )
        pos_mesh = self._flight_to_mesh_point(
            np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
        )
        return pos_mesh, float(yaw), float(-pitch)

    def _current_drone_pose_mesh_point(self):
        if self.latest_odom is None:
            return None
        try:
            p = self.latest_odom.pose.pose.position
            body_world = np.array([p.x, p.y, p.z], dtype=np.float64)
        except Exception:
            return None
        return self._flight_to_mesh_point(body_world)

    def _planner_start_state(self, samples):
        pos_mesh, yaw_mesh, pitch_mesh = self._current_camera_pose_mesh_state()
        if pos_mesh is not None and yaw_mesh is not None and pitch_mesh is not None:
            start_pos = np.asarray(pos_mesh, dtype=np.float64).reshape(3)
            self._planner_start_last_source = "odom_camera"
            self._planner_start_last_point = np.asarray(start_pos, dtype=np.float64).reshape(3)
            return start_pos, float(yaw_mesh), float(pitch_mesh)
        self._planner_start_last_source = "unavailable"
        self._planner_start_last_point = None
        rospy.logwarn_throttle(
            2.0,
            "Planner start unavailable: waiting for current camera pose from odometry before planning.",
        )
        return None, None, None

    def _make_start_planning_node(self, start_pos, start_yaw, start_pitch):
        pos = np.asarray(start_pos, dtype=np.float64).reshape(3)
        center = self._building_center_mesh_point().reshape(3)
        ray_dist = float(np.linalg.norm(center - pos))
        ray_dist = max(1.0, ray_dist)
        target = self._target_from_yaw_pitch(pos, float(start_yaw), float(start_pitch), ray_dist)
        return {
            "origin": np.asarray(pos, dtype=np.float64).reshape(3),
            "target": np.asarray(target, dtype=np.float64).reshape(3),
            "pose": (
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                0.0,
                float(start_pitch),
                float(start_yaw),
            ),
            "_plan_idx": -1,
            "_is_start_node": True,
        }

    def _nearest_exec_waypoint_info(self, exec_waypoints):
        pts = list(exec_waypoints) if exec_waypoints is not None else []
        if len(pts) <= 0:
            return None
        cur_mesh, _yaw, _pitch = self._current_camera_pose_mesh_state()
        cur_viz = self._current_camera_display_viz_point()
        if cur_mesh is None or cur_viz is None:
            return None
        cur_mesh = np.asarray(cur_mesh, dtype=np.float64).reshape(3)
        cur_viz = np.asarray(cur_viz, dtype=np.float64).reshape(3)
        best_idx = None
        best_dist = float("inf")
        all_dists = []
        for idx, wp in enumerate(pts):
            try:
                o_mesh = np.asarray(wp.get("origin", None), dtype=np.float64).reshape(3)
            except Exception:
                continue
            o_viz = self._mesh_to_traj_viz_point(o_mesh)
            d_mesh = float(np.linalg.norm(o_mesh - cur_mesh))
            d_viz = float(np.linalg.norm(o_viz - cur_viz))
            all_dists.append((int(idx), float(d_viz), float(d_mesh), o_mesh, o_viz))
            if d_viz + 1.0e-12 < best_dist:
                best_dist = d_viz
                best_idx = int(idx)
        if best_idx is None:
            return None
        top = sorted(all_dists, key=lambda item: float(item[1]))[:3]
        return {
            "idx": int(best_idx),
            "dist": float(best_dist),
            "current_mesh": np.asarray(cur_mesh, dtype=np.float64).reshape(3),
            "current_viz": np.asarray(cur_viz, dtype=np.float64).reshape(3),
            "top": [
                {
                    "idx": int(idx),
                    "viz_dist": float(viz_dist),
                    "mesh_dist": float(mesh_dist),
                    "mesh_point": np.asarray(mesh_pt, dtype=np.float64).reshape(3),
                    "viz_point": np.asarray(viz_pt, dtype=np.float64).reshape(3),
                }
                for idx, viz_dist, mesh_dist, mesh_pt, viz_pt in top
            ],
        }

    def _current_velocity_mesh_vector(self):
        if self.latest_odom is None:
            return None, 0.0, "no_odom"
        try:
            v = self.latest_odom.twist.twist.linear
            vx = float(v.x)
            vy = float(v.y)
            vz = float(v.z)
        except Exception:
            return None, 0.0, "invalid_odom_twist"
        # Mesh point conversion maps flight/control z to -z, so apply the
        # same linear transform to velocity before comparing path directions.
        v_mesh = np.array([vx, vy, -vz], dtype=np.float64)
        speed = float(np.linalg.norm(v_mesh))
        if (not np.isfinite(speed)) or speed <= 1.0e-9:
            return None, 0.0, "zero_velocity"
        min_speed = float(max(0.0, self.mesh_view_velocity_handoff_min_speed_mps))
        if speed < min_speed:
            return None, speed, "below_min_speed"
        return v_mesh, speed, "ok"

    def _exec_waypoint_origin_mesh(self, exec_waypoints, idx):
        pts = list(exec_waypoints) if exec_waypoints is not None else []
        if len(pts) <= 0:
            return None
        try:
            wp = pts[int(idx) % len(pts)]
            return np.asarray(wp.get("origin", None), dtype=np.float64).reshape(3)
        except Exception:
            return None

    def _exec_direction_tangent_unit(self, exec_waypoints, idx, direction):
        pts = list(exec_waypoints) if exec_waypoints is not None else []
        n = int(len(pts))
        if n <= 1:
            return None
        idx = int(idx) % n
        step = 1 if int(direction) >= 0 else -1
        p0 = self._exec_waypoint_origin_mesh(pts, idx)
        p1 = self._exec_waypoint_origin_mesh(pts, (idx + step) % n)
        if p0 is None or p1 is None:
            return None
        return self._unit_vector_or_none(p1 - p0)

    def _collect_exec_waypoints_to_next_nbv(self, exec_waypoints, nearest_idx, direction):
        pts = list(exec_waypoints) if exec_waypoints is not None else []
        n = int(len(pts))
        if n <= 0:
            return []
        step_dir = 1 if int(direction) >= 0 else -1
        idx = int(nearest_idx) % n
        out = []
        for step_count in range(n):
            out.append((int(idx), int(step_dir), int(step_count)))
            if step_count > 0:
                try:
                    if bool(pts[int(idx)].get("is_nbv", False)):
                        break
                except Exception:
                    pass
            idx = (int(idx) + int(step_dir)) % n
            if int(idx) == int(nearest_idx) % n:
                break
        return out

    def _velocity_aware_exec_handoff_info(self, exec_waypoints, nearest_exec):
        pts = list(exec_waypoints) if exec_waypoints is not None else []
        if len(pts) <= 0 or not isinstance(nearest_exec, dict):
            return None
        n = int(len(pts))
        nearest_idx = max(0, min(int(nearest_exec.get("idx", 0)), n - 1))
        fallback = {
            "idx": int(nearest_idx),
            "nearest_idx": int(nearest_idx),
            "direction": 1,
            "reason": "nearest_fallback",
            "speed": 0.0,
            "score": float("nan"),
            "tangent_score": float("nan"),
            "candidate_count": 0,
            "top": [],
        }
        if not bool(self.mesh_view_velocity_handoff_enabled):
            fallback["reason"] = "disabled"
            return fallback

        try:
            current_mesh = np.asarray(nearest_exec.get("current_mesh", None), dtype=np.float64).reshape(3)
        except Exception:
            current_mesh = None
        if current_mesh is None:
            current_mesh, _yaw, _pitch = self._current_camera_pose_mesh_state()
            if current_mesh is not None:
                current_mesh = np.asarray(current_mesh, dtype=np.float64).reshape(3)
        if current_mesh is None:
            fallback["reason"] = "no_current_pose"
            return fallback

        velocity_mesh, speed, vel_reason = self._current_velocity_mesh_vector()
        fallback["speed"] = float(speed)
        if velocity_mesh is None:
            fallback["reason"] = str(vel_reason)
            return fallback
        vel_unit = self._unit_vector_or_none(velocity_mesh)
        if vel_unit is None:
            fallback["reason"] = "invalid_velocity"
            return fallback

        candidates = (
            self._collect_exec_waypoints_to_next_nbv(pts, nearest_idx, 1)
            + self._collect_exec_waypoints_to_next_nbv(pts, nearest_idx, -1)
        )
        best = None
        evaluated = []
        for cand_idx, cand_dir, steps_from_nearest in candidates:
            origin = self._exec_waypoint_origin_mesh(pts, cand_idx)
            if origin is None:
                continue
            vec = origin - current_mesh
            dist = float(np.linalg.norm(vec))
            approach_unit = self._unit_vector_or_none(vec, eps=1.0e-4)
            tangent_unit = self._exec_direction_tangent_unit(pts, cand_idx, cand_dir)
            tangent_score = (
                float(np.dot(vel_unit, tangent_unit))
                if tangent_unit is not None
                else float("-inf")
            )
            if approach_unit is None:
                score = float(tangent_score)
                basis = "tangent"
            else:
                score = float(np.dot(vel_unit, approach_unit))
                basis = "approach"
            item = {
                "idx": int(cand_idx),
                "direction": int(cand_dir),
                "steps": int(steps_from_nearest),
                "dist": float(dist),
                "score": float(score),
                "tangent_score": float(tangent_score),
                "basis": str(basis),
                "is_nbv": bool(pts[int(cand_idx)].get("is_nbv", False)),
            }
            evaluated.append(item)
            if best is None:
                best = item
                continue
            if score > float(best["score"]) + 1.0e-9:
                best = item
            elif abs(score - float(best["score"])) <= 1.0e-9:
                if tangent_score > float(best["tangent_score"]) + 1.0e-9:
                    best = item
                elif abs(tangent_score - float(best["tangent_score"])) <= 1.0e-9 and dist < float(best["dist"]):
                    best = item

        if best is None:
            fallback["reason"] = "no_valid_candidates"
            return fallback

        top = sorted(
            evaluated,
            key=lambda item: (
                float(item.get("score", float("-inf"))),
                float(item.get("tangent_score", float("-inf"))),
                -float(item.get("dist", float("inf"))),
            ),
            reverse=True,
        )[:5]
        return {
            "idx": int(best["idx"]),
            "nearest_idx": int(nearest_idx),
            "direction": int(best["direction"]),
            "reason": "velocity_alignment",
            "speed": float(speed),
            "score": float(best["score"]),
            "tangent_score": float(best["tangent_score"]),
            "basis": str(best["basis"]),
            "dist": float(best["dist"]),
            "steps": int(best["steps"]),
            "candidate_count": int(len(evaluated)),
            "top": top,
        }

    def _publish_current_pose_point(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "current_pose_point"
        marker.id = 3
        pos_mesh, _yaw, _pitch = self._current_camera_pose_mesh_state()
        if pos_mesh is None:
            marker.action = Marker.DELETE
            self.mesh_view_current_pose_point_pub.publish(marker)
            return

        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        scale = float(max(1e-3, self.mesh_view_plan_point_scale))
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 0.98
        p = np.asarray(pos_mesh, dtype=np.float64).reshape(3)
        marker.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))]
        self._rviz_cache_points("current_pose_point", marker.points)
        self.mesh_view_current_pose_point_pub.publish(marker)

    def _clear_actual_flight_path_marker(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "actual_flight_path"
        marker.id = 5
        marker.action = Marker.DELETE
        self.mesh_view_actual_flight_path_pub.publish(marker)

    def _publish_actual_flight_path_marker(self):
        pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in list(self._mission_actual_path_points)]
        cur = self._current_drone_world_point()
        if cur is not None:
            if len(pts) <= 0 or float(np.linalg.norm(cur - pts[-1])) > 1.0e-6:
                pts.append(cur.copy())

        if len(pts) < 2:
            self._clear_actual_flight_path_marker()
            return

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "actual_flight_path"
        marker.id = 5
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(max(1e-4, self.mesh_view_plan_width))
        marker.color.r = 0.05
        marker.color.g = 0.15
        marker.color.b = 0.65
        marker.color.a = 0.95
        marker.points = [
            Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts
        ]
        self.mesh_view_actual_flight_path_pub.publish(marker)

    def _building_center_mesh_point(self):
        return np.array(
            [
                float(self.build_center_x),
                float(self.build_center_y),
                float(-self.build_center_z + self.rt_mesh_center_z_offset),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _fmt_point_xyz(point_xyz):
        try:
            p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
            return "(%.2f, %.2f, %.2f)" % (float(p[0]), float(p[1]), float(p[2]))
        except Exception:
            return "(nan, nan, nan)"

    @staticmethod
    def _mesh_to_traj_viz_point(point_xyz):
        p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        return np.array([-float(p[0]), float(p[1]), -float(p[2])], dtype=np.float64)

    def _current_camera_display_viz_point(self):
        p = self._camera_pose_viz_position_from_latest_odom()
        if p is None:
            return None
        q = np.asarray(p, dtype=np.float64).reshape(3)
        return np.array([-float(q[0]), float(q[1]), float(q[2])], dtype=np.float64)

    def _fmt_point_list(self, points, limit=5, viz=False, first_override=None):
        pts = list(points) if points is not None else []
        out = []
        first_override_arr = None
        if first_override is not None:
            first_override_arr = np.asarray(first_override, dtype=np.float64).reshape(3)
        for idx, p in enumerate(pts[: max(0, int(limit))]):
            if viz and idx == 0 and first_override_arr is not None:
                q = first_override_arr
            else:
                q = self._mesh_to_traj_viz_point(p) if viz else p
            out.append(self._fmt_point_xyz(q))
        return "[" + ", ".join(out) + "]"

    @staticmethod
    def _first_segment_lengths(points, limit=4):
        pts = [
            np.asarray(p, dtype=np.float64).reshape(3)
            for p in (list(points) if points is not None else [])
        ]
        vals = []
        for i in range(min(max(0, int(limit)), max(0, len(pts) - 1))):
            vals.append(float(np.linalg.norm(pts[i + 1] - pts[i])))
        return vals

    def _clear_start_leg_checked_points(self):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "start_leg_checked_points"
        marker.id = 0
        marker.action = Marker.DELETE
        self.mesh_view_start_leg_points_pub.publish(marker)

    def publish_start_leg_checked_points(self, start_mesh, goal_mesh):
        if not self.mesh_view_start_leg_debug_enabled:
            self._clear_start_leg_checked_points()
            return
        if start_mesh is None or goal_mesh is None:
            self._clear_start_leg_checked_points()
            return
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "start_leg_checked_points"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.42
        marker.scale.y = 0.42
        marker.scale.z = 0.42
        marker.color.r = 1.0
        marker.color.g = 0.15
        marker.color.b = 0.15
        marker.color.a = 0.98
        pts = []
        for p in (start_mesh, goal_mesh):
            q = np.asarray(p, dtype=np.float64).reshape(3)
            pts.append(Point(x=float(q[0]), y=float(q[1]), z=float(q[2])))
        marker.points = pts
        self.mesh_view_start_leg_points_pub.publish(marker)

    @staticmethod
    def _dedup_consecutive_points(points, eps=1.0e-6):
        pts = []
        for p in list(points) if points is not None else []:
            try:
                q = np.asarray(p, dtype=np.float64).reshape(3)
            except Exception:
                continue
            if pts and float(np.linalg.norm(q - pts[-1])) <= float(eps):
                continue
            pts.append(q)
        return pts

    @staticmethod
    def _polyline_length(points):
        pts = ControllerNode._dedup_consecutive_points(points)
        if len(pts) <= 1:
            return 0.0
        return float(
            np.sum(
                np.linalg.norm(
                    np.diff(np.asarray(pts, dtype=np.float64).reshape(-1, 3), axis=0),
                    axis=1,
                )
            )
        )

    def _sample_polyline(self, points, spacing=None):
        pts = self._dedup_consecutive_points(points)
        if len(pts) <= 1:
            return pts
        if spacing is None:
            spacing = float(self.mesh_view_surface_graph_path_spacing_m)
        spacing = max(1.0e-3, float(spacing))
        out = [pts[0]]
        for i in range(len(pts) - 1):
            a = np.asarray(pts[i], dtype=np.float64).reshape(3)
            b = np.asarray(pts[i + 1], dtype=np.float64).reshape(3)
            seg = b - a
            seg_len = float(np.linalg.norm(seg))
            if seg_len <= 1.0e-9:
                continue
            n_sub = max(1, int(np.ceil(seg_len / spacing)))
            for s in range(1, n_sub + 1):
                alpha = float(s) / float(n_sub)
                out.append(a + alpha * seg)
        return self._dedup_consecutive_points(out)

    def _push_points_outward_from_building_center(self, points, clearance):
        pts = self._dedup_consecutive_points(points)
        if len(pts) <= 0:
            return []
        clearance = max(0.0, float(clearance))
        if clearance <= 0.0:
            return [np.asarray(p, dtype=np.float64).reshape(3) for p in pts]
        center = self._building_center_mesh_point().reshape(3)
        out = []
        for p in pts:
            v = np.asarray(p, dtype=np.float64).reshape(3) - center
            vn = float(np.linalg.norm(v))
            if vn <= 1.0e-9:
                out.append(np.asarray(p, dtype=np.float64).reshape(3))
            else:
                out.append(np.asarray(p, dtype=np.float64).reshape(3) + (clearance / vn) * v)
        return out

    def _sample_cubic_bspline(self, control_points, spacing=None):
        pts = self._dedup_consecutive_points(control_points)
        if len(pts) <= 1:
            return pts
        if len(pts) == 2 or make_interp_spline is None:
            return self._sample_polyline(pts, spacing=spacing)
        spacing = max(
            1.0e-3,
            float(self.mesh_view_surface_graph_path_spacing_m if spacing is None else spacing),
        )
        ctrl = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        ctrl_len = float(self._polyline_length(ctrl))
        sample_count = max(int(ctrl.shape[0]), int(np.ceil(ctrl_len / spacing)) + 1)
        t = np.linspace(0.0, 1.0, int(ctrl.shape[0]))
        u = np.linspace(0.0, 1.0, int(max(2, sample_count)))
        try:
            spline = make_interp_spline(t, ctrl, k=int(min(3, ctrl.shape[0] - 1)), axis=0)
            sampled = np.asarray(spline(u), dtype=np.float64).reshape(-1, 3)
            sampled[0] = ctrl[0]
            sampled[-1] = ctrl[-1]
            return self._dedup_consecutive_points(sampled)
        except Exception:
            return self._sample_polyline(ctrl, spacing=spacing)

    def _sample_execution_segment_path(self, control_points, preserve_endpoints=True):
        ctrl = self._dedup_consecutive_points(control_points)
        if len(ctrl) <= 1:
            return ctrl
        sampled = self._sample_cubic_bspline(
            ctrl,
            spacing=float(self.mesh_view_surface_graph_path_spacing_m),
        )
        sampled = self._clamp_path_points_above_ground(
            sampled,
            preserve_endpoints=bool(preserve_endpoints),
        )
        return self._dedup_consecutive_points(sampled)

    def _sample_execution_control_span(
        self,
        control_points,
        span_start_idx,
        span_end_idx,
        preserve_endpoints=True,
    ):
        ctrl = self._dedup_consecutive_points(control_points)
        if len(ctrl) <= 1:
            return ctrl
        i0 = max(0, min(int(span_start_idx), len(ctrl) - 1))
        i1 = max(i0, min(int(span_end_idx), len(ctrl) - 1))
        if i0 == i1:
            return [np.asarray(ctrl[i0], dtype=np.float64).reshape(3)]
        if len(ctrl) == 2 or make_interp_spline is None:
            sampled = self._sample_polyline(ctrl[i0 : i1 + 1], spacing=float(self.mesh_view_surface_graph_path_spacing_m))
            sampled = self._clamp_path_points_above_ground(sampled, preserve_endpoints=bool(preserve_endpoints))
            return self._dedup_consecutive_points(sampled)

        spacing = max(1.0e-3, float(self.mesh_view_surface_graph_path_spacing_m))
        ctrl_arr = np.asarray(ctrl, dtype=np.float64).reshape(-1, 3)
        t = np.linspace(0.0, 1.0, int(ctrl_arr.shape[0]))
        u0 = float(t[int(i0)])
        u1 = float(t[int(i1)])
        approx_len = float(self._polyline_length(ctrl_arr[i0 : i1 + 1]))
        sample_count = max(2, int(np.ceil(approx_len / spacing)) + 1)
        u = np.linspace(u0, u1, int(sample_count))
        try:
            spline = make_interp_spline(t, ctrl_arr, k=int(min(3, ctrl_arr.shape[0] - 1)), axis=0)
            sampled = np.asarray(spline(u), dtype=np.float64).reshape(-1, 3)
            sampled[0] = ctrl_arr[int(i0)]
            sampled[-1] = ctrl_arr[int(i1)]
        except Exception:
            sampled = self._sample_polyline(ctrl_arr[i0 : i1 + 1], spacing=spacing)
        sampled = self._clamp_path_points_above_ground(sampled, preserve_endpoints=bool(preserve_endpoints))
        return self._dedup_consecutive_points(sampled)

    def _clamp_path_points_above_ground(self, points, preserve_endpoints=True):
        pts = self._dedup_consecutive_points(points)
        if len(pts) <= 0:
            return []
        max_z = float(self._planner_candidate_max_z())
        out = []
        last_idx = len(pts) - 1
        for idx, p in enumerate(pts):
            q = np.asarray(p, dtype=np.float64).reshape(3).copy()
            if bool(preserve_endpoints) and (idx == 0 or idx == last_idx):
                out.append(q)
                continue
            if float(q[2]) > max_z:
                q[2] = max_z
            out.append(q)
        return self._dedup_consecutive_points(out)

    @staticmethod
    def _look_dir_from_yaw_pitch(yaw, pitch):
        cp = math.cos(float(pitch))
        return np.array(
            [
                math.sin(float(yaw)) * cp,
                math.cos(float(yaw)) * cp,
                math.sin(float(pitch)),
            ],
            dtype=np.float64,
        )

    def _target_from_yaw_pitch(self, origin, yaw, pitch, distance):
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        ray = self._look_dir_from_yaw_pitch(yaw, pitch)
        return o + max(1.0e-3, float(distance)) * ray

    def _build_local_orientation_context(self):
        if self.mesh_triangles is None:
            return None
        triangles_all = np.asarray(self.mesh_triangles, dtype=np.float64)
        n_tri = int(triangles_all.shape[0])
        if n_tri <= 0:
            return None

        activity = self._planner_triangle_activity(triangles_all)
        self._planner_triangle_activity_last = dict(activity)
        active_mask = np.asarray(activity.get("active_mask", np.zeros((n_tri,), dtype=bool)), dtype=bool)
        tri_idx = np.where(active_mask)[0].astype(np.int64)
        if tri_idx.size <= 0:
            return None

        triangles = triangles_all[tri_idx]
        tri_cent = np.mean(triangles, axis=1)
        n_raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        n_len = np.linalg.norm(n_raw, axis=1)
        valid = n_len > 1.0e-10
        if not np.any(valid):
            return None

        triangles = triangles[valid]
        tri_cent = tri_cent[valid]
        n_raw = n_raw[valid]
        n_len = n_len[valid]
        n_unit = n_raw / n_len[:, None]
        build_center = self._building_center_mesh_point().reshape(3)
        out_sign = np.sign(np.einsum("ij,ij->i", n_unit, tri_cent - build_center.reshape(1, 3)))
        out_sign[out_sign == 0.0] = 1.0
        n_out = n_unit * out_sign[:, None]

        tri_area = 0.5 * n_len
        tri_area = np.asarray(tri_area, dtype=np.float64)

        tri_raw_all = self._get_triangle_cvi_raw_values(n_tri)
        tri_raw = np.asarray(tri_raw_all[tri_idx], dtype=np.float64)[valid]
        tri_conf_all = self._get_triangle_conf_values(n_tri)
        tri_conf = np.asarray(tri_conf_all[tri_idx], dtype=np.float64)[valid]
        tri_public = self._normalize_minmax_to_range(tri_raw, 1.0, 2.0)
        tri_eff = np.asarray(tri_public * tri_conf, dtype=np.float64)

        tan_h = math.tan(math.atan2(float(self.cam_width), 2.0 * max(1e-9, float(self.cam_fx))))
        tan_v = math.tan(math.atan2(float(self.cam_height), 2.0 * max(1e-9, float(self.cam_fy))))
        tan_h = max(1.0e-6, tan_h)
        tan_v = max(1.0e-6, tan_v)

        return {
            "build_center": build_center,
            "tri_cent": np.asarray(tri_cent, dtype=np.float64),
            "tri_normals": np.asarray(n_out, dtype=np.float64),
            "tri_area": np.asarray(tri_area, dtype=np.float64),
            "tri_raw": np.asarray(tri_raw, dtype=np.float64),
            "tri_conf": np.asarray(tri_conf, dtype=np.float64),
            "tri_eff": np.asarray(tri_eff, dtype=np.float64),
            "tan_h": float(tan_h),
            "tan_v": float(tan_v),
        }

    def _evaluate_viewpoint_effective_metrics(self, viewpoint, yaw, pitch, orient_ctx):
        if not isinstance(orient_ctx, dict):
            return {"c_eff": 0.0, "gain_eff": 0.0}
        tri_cent = np.asarray(orient_ctx.get("tri_cent", []), dtype=np.float64)
        if tri_cent.ndim != 2 or tri_cent.shape[0] <= 0:
            return {"c_eff": 0.0, "gain_eff": 0.0}
        tri_normals = np.asarray(orient_ctx.get("tri_normals", []), dtype=np.float64)
        tri_area = np.asarray(orient_ctx.get("tri_area", []), dtype=np.float64).reshape(-1)
        tri_raw = np.asarray(orient_ctx.get("tri_raw", []), dtype=np.float64).reshape(-1)
        tri_conf = np.asarray(orient_ctx.get("tri_conf", []), dtype=np.float64).reshape(-1)
        tri_eff = np.asarray(orient_ctx.get("tri_eff", []), dtype=np.float64).reshape(-1)
        tan_h = float(orient_ctx.get("tan_h", 1.0))
        tan_v = float(orient_ctx.get("tan_v", 1.0))

        viewpoint = np.asarray(viewpoint, dtype=np.float64).reshape(3)
        forward = self._look_dir_from_yaw_pitch(yaw, pitch)
        fn = float(np.linalg.norm(forward))
        if fn <= 1.0e-9:
            return {"c_eff": 0.0, "gain_eff": 0.0}
        forward = forward / fn

        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(up_hint, forward))) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_hint)
        rn = float(np.linalg.norm(right))
        if rn <= 1.0e-9:
            return {"c_eff": 0.0, "gain_eff": 0.0}
        right = right / rn
        up = np.cross(right, forward)
        un = float(np.linalg.norm(up))
        if un <= 1.0e-9:
            return {"c_eff": 0.0, "gain_eff": 0.0}
        up = up / un

        rel_tri = tri_cent - viewpoint.reshape(1, 3)
        x_cam = rel_tri @ forward
        y_cam = rel_tri @ right
        z_cam = rel_tri @ up
        in_fov = (
            (x_cam > 1.0e-6)
            & (np.abs(y_cam) <= (x_cam * tan_h))
            & (np.abs(z_cam) <= (x_cam * tan_v))
        )
        if not np.any(in_fov):
            return {"c_eff": 0.0, "gain_eff": 0.0}

        dir_tri_to_cam = viewpoint.reshape(1, 3) - tri_cent
        dtc_n = np.linalg.norm(dir_tri_to_cam, axis=1)
        dtc_n = np.maximum(dtc_n, 1.0e-9)
        dir_tri_to_cam = dir_tri_to_cam / dtc_n[:, None]
        front = np.einsum("ij,ij->i", tri_normals, dir_tri_to_cam)
        front = np.maximum(front, 0.0)
        visible_mask = in_fov & (front > 0.0)
        if not np.any(visible_mask):
            return {"c_eff": 0.0, "gain_eff": 0.0}

        raw_gain = np.zeros_like(tri_raw, dtype=np.float64)
        raw_gain[visible_mask] = tri_area[visible_mask] * front[visible_mask] / (dtc_n[visible_mask] * dtc_n[visible_mask] + 1.0e-6)
        visible_eff = np.asarray(tri_eff[visible_mask], dtype=np.float64)
        current_eff_mean = float(np.mean(visible_eff)) if visible_eff.size > 0 else 0.0
        next_raw = tri_raw + raw_gain
        next_public = self._normalize_minmax_to_range(next_raw, 1.0, 2.0)
        next_eff = next_public * tri_conf
        visible_gain = np.asarray((next_eff - tri_eff)[visible_mask], dtype=np.float64)
        eff_gain_mean = float(np.mean(visible_gain)) if visible_gain.size > 0 else 0.0

        eff_min, eff_max = self._effective_cvi_bounds()
        eff_range = max(1.0e-12, float(eff_max - eff_min))
        current_eff_norm = float(np.clip((current_eff_mean - eff_min) / eff_range, 0.0, 1.0))
        gain_eff_norm = float(np.clip(eff_gain_mean / eff_range, 0.0, 1.0))
        return {"c_eff": current_eff_norm, "gain_eff": gain_eff_norm}

    def _score_local_orientation_candidate(self, viewpoint, yaw, pitch, orient_ctx):
        return float(self._evaluate_viewpoint_effective_metrics(viewpoint, yaw, pitch, orient_ctx).get("gain_eff", 0.0))

    def _local_force_field_orientation(self, position, orient_ctx, fallback_target=None):
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        build_center = np.asarray(orient_ctx.get("build_center", pos), dtype=np.float64).reshape(3)
        center_vec = build_center - pos
        center_dist = float(np.linalg.norm(center_vec))
        if center_dist <= 1.0e-9 and fallback_target is not None:
            center_vec = np.asarray(fallback_target, dtype=np.float64).reshape(3) - pos
            center_dist = float(np.linalg.norm(center_vec))
        if center_dist <= 1.0e-9:
            center_vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            center_dist = 1.0
        center_dir = center_vec / center_dist
        ray_dist = max(1.0, center_dist)

        tri_cent = np.asarray(orient_ctx.get("tri_cent", []), dtype=np.float64)
        tri_normals = np.asarray(orient_ctx.get("tri_normals", []), dtype=np.float64)
        tri_area = np.asarray(orient_ctx.get("tri_area", []), dtype=np.float64).reshape(-1)
        tri_eff = np.asarray(orient_ctx.get("tri_eff", []), dtype=np.float64).reshape(-1)
        if tri_cent.ndim == 2 and tri_cent.shape[0] > 0:
            rel = tri_cent - pos.reshape(1, 3)
            dist = np.linalg.norm(rel, axis=1)
            valid = dist > 1.0e-9
            if np.any(valid):
                rel_valid = rel[valid]
                dist_valid = np.maximum(dist[valid], 1.0e-9)
                dir_point_to_tri = rel_valid / dist_valid[:, None]
                dir_tri_to_cam = -dir_point_to_tri
                normals_valid = tri_normals[valid]
                area_valid = tri_area[valid]
                eff_valid = np.maximum(tri_eff[valid], 1.0e-6)

                center_side_mask = np.einsum("ij,j->i", normals_valid, center_dir) < 0.0
                front = np.einsum("ij,ij->i", normals_valid, dir_tri_to_cam)
                front = np.maximum(front, 0.0)
                keep_mask = center_side_mask & (front > 0.0)
                if np.any(keep_mask):
                    gain = np.zeros_like(front, dtype=np.float64)
                    gain[keep_mask] = area_valid[keep_mask] * front[keep_mask] / (dist_valid[keep_mask] * dist_valid[keep_mask] + 1.0e-6)
                    gain_scale = float(self.mesh_view_local_orientation_gain_weight) if float(self.mesh_view_local_orientation_gain_weight) > 0.0 else 1.0
                    cvi_scale = float(self.mesh_view_local_orientation_cvi_weight) if float(self.mesh_view_local_orientation_cvi_weight) > 0.0 else 1.0
                    gain_max = float(np.max(gain[keep_mask])) if np.any(keep_mask) else 0.0
                    gain_norm = np.zeros_like(gain, dtype=np.float64)
                    if gain_max > 1.0e-12:
                        gain_norm[keep_mask] = gain[keep_mask] / gain_max
                    eff_min, eff_max = self._effective_cvi_bounds()
                    eff_range = max(1.0e-12, float(eff_max - eff_min))
                    eff_norm = np.clip((eff_valid - float(eff_min)) / eff_range, 0.0, 1.0)
                    weight = np.zeros_like(gain, dtype=np.float64)
                    weight[keep_mask] = (gain_scale * gain_norm[keep_mask]) / (cvi_scale * eff_norm[keep_mask] + 1.0e-6)
                    force_vec = np.sum(weight[:, None] * dir_point_to_tri, axis=0)
                    force_norm = float(np.linalg.norm(force_vec))
                    if force_norm > 1.0e-9:
                        look_dir = force_vec / force_norm
                        target = pos + ray_dist * look_dir
                        pose = self._look_at_pose(origin=pos, target=target)
                        if pose is not None:
                            return {
                                "yaw": float(pose[5]),
                                "pitch": float(pose[4]),
                                "pose": pose,
                                "target": np.asarray(target, dtype=np.float64).reshape(3),
                                "force_norm": float(force_norm),
                                "tri_kept": int(np.count_nonzero(keep_mask)),
                                "gain_sum": float(np.sum(gain[keep_mask])) if np.any(keep_mask) else 0.0,
                                "eff_mean": float(np.mean(eff_valid[keep_mask])) if np.any(keep_mask) else 0.0,
                                "fallback": False,
                            }

        target = (
            np.asarray(fallback_target, dtype=np.float64).reshape(3)
            if fallback_target is not None
            else pos + ray_dist * center_dir
        )
        pose = self._look_at_pose(origin=pos, target=target)
        if pose is None:
            return None
        return {
            "yaw": float(pose[5]),
            "pitch": float(pose[4]),
            "pose": pose,
            "target": np.asarray(target, dtype=np.float64).reshape(3),
            "force_norm": 0.0,
            "tri_kept": 0,
            "gain_sum": 0.0,
            "eff_mean": 0.0,
            "fallback": True,
        }

    def _plan_local_orientation_chain(self, path_points, start_pos, start_yaw, start_pitch, orient_ctx, fallback_target=None):
        pts = self._dedup_consecutive_points(path_points)
        if len(pts) <= 0:
            return {"points": [], "shift": 0.0}

        chosen_points = []
        for pt in pts:
            cand = self._local_force_field_orientation(
                pt,
                orient_ctx=orient_ctx,
                fallback_target=fallback_target,
            )
            if cand is None:
                return {"points": [], "shift": 0.0}
            chosen_points.append(cand)
        if self.mesh_view_debug_logs:
            try:
                rospy.loginfo(
                    "Local orient diagnostics: layers=%d start_yaw_deg=%.2f start_pitch_deg=%.2f",
                    int(len(chosen_points)),
                    math.degrees(float(start_yaw)),
                    math.degrees(float(start_pitch)),
                )
                for layer_idx, chosen in enumerate(chosen_points):
                    rospy.loginfo(
                        "Local orient layer %d: yaw_deg=%.2f pitch_deg=%.2f tri_kept=%d gain_sum=%.6f eff_mean=%.6f force_norm=%.6f fallback=%s",
                        int(layer_idx),
                        math.degrees(float(chosen.get("yaw", 0.0))),
                        math.degrees(float(chosen.get("pitch", 0.0))),
                        int(chosen.get("tri_kept", 0)),
                        float(chosen.get("gain_sum", 0.0)),
                        float(chosen.get("eff_mean", 0.0)),
                        float(chosen.get("force_norm", 0.0)),
                        "yes" if bool(chosen.get("fallback", False)) else "no",
                    )
            except Exception as diag_exc:
                rospy.logwarn("Local orient diagnostics failed: %s", str(diag_exc))
        return {
            "points": chosen_points,
            "shift": 0.0,
        }

    @staticmethod
    def _monotonic_nearest_indices(path_points, anchor_points):
        pts = [
            np.asarray(p, dtype=np.float64).reshape(3)
            for p in (list(path_points) if path_points is not None else [])
        ]
        anchors = [
            np.asarray(p, dtype=np.float64).reshape(3)
            for p in (list(anchor_points) if anchor_points is not None else [])
        ]
        if len(pts) <= 0 or len(anchors) <= 0:
            return []
        out = []
        start_idx = 0
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        for anchor in anchors:
            if start_idx >= arr.shape[0]:
                out.append(int(arr.shape[0] - 1))
                continue
            search = arr[start_idx:]
            d = np.linalg.norm(search - anchor.reshape(1, 3), axis=1)
            idx = int(start_idx + int(np.argmin(d)))
            out.append(int(idx))
            start_idx = int(idx)
        return out

    def _pair_cache_hit(self, plan_idx_a, plan_idx_b):
        try:
            ia = int(plan_idx_a)
            ib = int(plan_idx_b)
        except Exception:
            return None, False
        if ia == ib:
            return None, False
        key = (int(min(ia, ib)), int(max(ia, ib)))
        hit = self.mesh_view_intersection_pairs.get(key, None)
        return hit, bool(ia > ib)

    def _pair_path_details_mesh(self, sample_a, sample_b):
        details = {
            "points": [],
            "kind": "straight",
            "status": "straight_invalid",
            "reason": "invalid_pair",
            "collided": False,
            "fallback": False,
        }
        try:
            origin_a = np.asarray(sample_a.get("origin", None), dtype=np.float64).reshape(3)
            origin_b = np.asarray(sample_b.get("origin", None), dtype=np.float64).reshape(3)
        except Exception:
            return details

        a_is_start = bool(sample_a.get("_is_start_node", False))
        b_is_start = bool(sample_b.get("_is_start_node", False))
        if a_is_start and b_is_start:
            details["points"] = [origin_a, origin_b]
            details["status"] = "straight_no_collision"
            details["reason"] = "same_start_node_pair"
            return details
        if a_is_start and not b_is_start:
            return self._start_path_details_mesh(sample_b, start_pos=origin_a)
        if b_is_start and not a_is_start:
            rev = self._start_path_details_mesh(sample_a, start_pos=origin_b)
            rev_pts = [
                np.asarray(p, dtype=np.float64).reshape(3)
                for p in list(reversed(rev.get("points", [])))
            ]
            rev["points"] = rev_pts
            return rev

        hit, reverse = self._pair_cache_hit(sample_a.get("_plan_idx", None), sample_b.get("_plan_idx", None))
        forward_origin = np.asarray(origin_b if reverse else origin_a, dtype=np.float64).reshape(3)
        forward_goal = np.asarray(origin_a if reverse else origin_b, dtype=np.float64).reshape(3)
        if hit is None or int(hit.get("hit_count", 0)) <= 0:
            details["points"] = [origin_a, origin_b]
            details["status"] = "straight_no_collision"
            details["reason"] = "no_collision"
            return details

        cached = hit.get("exec_path_points_forward", None)
        if cached is not None:
            cached_pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in list(cached)]
            points = list(reversed(cached_pts)) if reverse else cached_pts
            status = str(hit.get("exec_path_status", "surface_graph_detour"))
            reason = str(hit.get("exec_path_reason", status))
            details["points"] = points
            details["status"] = status
            details["reason"] = reason
            details["collided"] = True
            details["fallback"] = status.startswith("collision_fallback:")
            details["kind"] = "straight" if details["fallback"] else "surface_graph_detour"
            return details

        route = hit.get("surface_graph_route", None)
        if not isinstance(route, dict) or not bool(route.get("success", False)):
            reason = str(route.get("reason", "no_surface_route")) if isinstance(route, dict) else "no_surface_route"
            hit["exec_path_points_forward"] = [forward_origin, forward_goal]
            hit["exec_path_status"] = "collision_fallback:%s" % reason
            hit["exec_path_reason"] = reason
            cached_pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in list(hit["exec_path_points_forward"])]
            details["points"] = list(reversed(cached_pts)) if reverse else cached_pts
            details["status"] = str(hit["exec_path_status"])
            details["reason"] = str(hit["exec_path_reason"])
            details["collided"] = True
            details["fallback"] = True
            details["kind"] = "straight"
            return details

        base_points = [
            np.asarray(p, dtype=np.float64).reshape(3)
            for p in list(route.get("path_points", []))
        ]
        if reverse:
            base_points = list(reversed(base_points))
        pushed = self._push_points_outward_from_building_center(
            base_points,
            clearance=float(self.mesh_view_surface_graph_clearance_m),
        )
        if reverse:
            pushed = list(reversed(pushed))
        control = [forward_origin] + pushed + [forward_goal]
        control = self._dedup_consecutive_points(control)
        if len(control) <= 1:
            hit["exec_path_points_forward"] = [forward_origin, forward_goal]
            hit["exec_path_status"] = "collision_fallback:degenerate_control"
            hit["exec_path_reason"] = "degenerate_control"
        else:
            sampled = self._sample_execution_segment_path(control)
            if len(sampled) <= 1:
                hit["exec_path_points_forward"] = [forward_origin, forward_goal]
                hit["exec_path_status"] = "collision_fallback:degenerate_sampled"
                hit["exec_path_reason"] = "degenerate_sampled"
            else:
                hit["exec_path_points_forward"] = sampled
                hit["exec_path_status"] = (
                    "surface_graph_bspline" if make_interp_spline is not None and len(control) > 2 else "surface_graph_polyline"
                )
                hit["exec_path_reason"] = str(hit["exec_path_status"])
        cached_pts = [np.asarray(p, dtype=np.float64).reshape(3) for p in list(hit["exec_path_points_forward"])]
        details["points"] = list(reversed(cached_pts)) if reverse else cached_pts
        details["status"] = str(hit.get("exec_path_status", "surface_graph_detour"))
        details["reason"] = str(hit.get("exec_path_reason", details["status"]))
        details["collided"] = True
        details["fallback"] = details["status"].startswith("collision_fallback:")
        details["kind"] = "straight" if details["fallback"] else "surface_graph_detour"
        return details

    def _start_path_details_mesh(self, sample_b, start_pos=None):
        details = {
            "points": [],
            "kind": "straight",
            "status": "straight_invalid",
            "reason": "invalid_start",
            "collided": False,
            "fallback": False,
            "hit_count": 0,
            "entry_point": None,
            "exit_point": None,
            "entry_tid": None,
            "exit_tid": None,
            "start_pos_mesh": None,
            "goal_pos_mesh": None,
            "route_reason": None,
            "route_attempts": 0,
            "route_alpha": 0.0,
        }
        try:
            origin_b = np.asarray(sample_b.get("origin", None), dtype=np.float64).reshape(3)
        except Exception:
            return details

        if start_pos is None:
            start_pos, _start_yaw, _start_pitch = self._planner_start_state([sample_b])
        try:
            origin_a = np.asarray(start_pos, dtype=np.float64).reshape(3)
        except Exception:
            return details
        details["start_pos_mesh"] = np.asarray(origin_a, dtype=np.float64).reshape(3)
        details["goal_pos_mesh"] = np.asarray(origin_b, dtype=np.float64).reshape(3)

        if self.mesh_view_obstacle_mode != "surface_graph" or self.mesh_view_intersection_cache is None:
            details["points"] = [origin_a, origin_b]
            details["status"] = "straight_no_collision"
            details["reason"] = "obstacle_mode_disabled"
            return details

        hit = self._segment_mesh_intersections(origin_a, origin_b)
        details["hit_count"] = int(hit.get("hit_count", 0))
        details["entry_point"] = hit.get("entry_point", None)
        details["exit_point"] = hit.get("exit_point", None)
        details["entry_tid"] = hit.get("entry_tid", None)
        details["exit_tid"] = hit.get("exit_tid", None)
        if int(hit.get("hit_count", 0)) <= 0:
            details["points"] = [origin_a, origin_b]
            details["status"] = "straight_no_collision"
            details["reason"] = "no_collision"
            return details

        details["collided"] = True
        route = None
        entry_point = hit.get("entry_point", None)
        exit_point = hit.get("exit_point", None)
        if entry_point is not None and exit_point is not None:
            route = self._surface_graph_route_between_points(
                entry_point,
                exit_point,
                entry_tid=hit.get("entry_tid", None),
                exit_tid=hit.get("exit_tid", None),
            )
        if isinstance(route, dict):
            details["route_reason"] = route.get("reason", None)
            details["route_attempts"] = int(route.get("attempts", 0))
            details["route_alpha"] = float(route.get("alpha", 0.0))
        if not isinstance(route, dict) or not bool(route.get("success", False)):
            reason = str(route.get("reason", "no_surface_route")) if isinstance(route, dict) else "no_surface_route"
            details["points"] = [origin_a, origin_b]
            details["status"] = "collision_fallback:%s" % reason
            details["reason"] = reason
            details["fallback"] = True
            return details

        base_points = [
            np.asarray(p, dtype=np.float64).reshape(3)
            for p in list(route.get("path_points", []))
        ]
        pushed = self._push_points_outward_from_building_center(
            base_points,
            clearance=float(self.mesh_view_surface_graph_clearance_m),
        )
        control = [origin_a] + pushed + [origin_b]
        control = self._dedup_consecutive_points(control)
        if len(control) <= 1:
            details["points"] = [origin_a, origin_b]
            details["status"] = "collision_fallback:degenerate_control"
            details["reason"] = "degenerate_control"
            details["fallback"] = True
            return details

        sampled = self._sample_execution_segment_path(control)
        if len(sampled) <= 1:
            details["points"] = [origin_a, origin_b]
            details["status"] = "collision_fallback:degenerate_sampled"
            details["reason"] = "degenerate_sampled"
            details["fallback"] = True
            return details

        details["points"] = sampled
        details["status"] = (
            "surface_graph_bspline" if make_interp_spline is not None and len(control) > 2 else "surface_graph_polyline"
        )
        details["reason"] = str(details["status"])
        details["kind"] = "surface_graph_detour"
        return details

    def _pair_path_points_mesh(self, sample_a, sample_b):
        return list(self._pair_path_details_mesh(sample_a, sample_b).get("points", []))

    def _pair_path_kind(self, sample_a, sample_b):
        return str(self._pair_path_details_mesh(sample_a, sample_b).get("status", "straight_invalid"))

    def _pair_path_length_mesh(self, sample_a, sample_b):
        pts = self._pair_path_details_mesh(sample_a, sample_b).get("points", [])
        if len(pts) <= 1:
            try:
                oa = np.asarray(sample_a.get("origin", None), dtype=np.float64).reshape(3)
                ob = np.asarray(sample_b.get("origin", None), dtype=np.float64).reshape(3)
                return float(np.linalg.norm(ob - oa))
            except Exception:
                return 0.0
        return float(self._polyline_length(pts))

    def _build_execution_waypoints(self, ordered_samples, cost_data=None):
        ordered = list(ordered_samples) if ordered_samples is not None else []
        self._mesh_view_local_orientation_last_ms = 0.0
        self._mesh_view_local_orientation_last_segments = 0
        if len(ordered) <= 0:
            self._clear_start_leg_checked_points()
            return []

        exec_waypoints = []
        fallback_diags = []
        status_counts = {}
        self._clear_start_leg_checked_points()
        start_pos = None
        start_yaw = 0.0
        start_pitch = 0.0
        if isinstance(cost_data, dict):
            try:
                start_pos = np.asarray(cost_data.get("start_pos", None), dtype=np.float64).reshape(3)
            except Exception:
                start_pos = None
            start_yaw = float(cost_data.get("start_yaw", 0.0))
            start_pitch = float(cost_data.get("start_pitch", 0.0))
        if start_pos is None and len(ordered) > 0:
            try:
                start_pos, start_yaw, start_pitch = self._planner_start_state([ordered[0]])
            except Exception:
                start_pos = None
        local_orientation_ms = 0.0
        local_orientation_segments = 0
        orient_ctx = None
        if bool(self.mesh_view_local_planning_enabled and self.mesh_view_local_orientation_enabled):
            t_local_ctx = time.perf_counter()
            orient_ctx = self._build_local_orientation_context()
            local_orientation_ms += (time.perf_counter() - t_local_ctx) * 1000.0

        segment_defs = []
        if len(ordered) == 1:
            first_wp = copy.deepcopy(ordered[0])
            first_wp["is_nbv"] = True
            first_wp["nbv_index"] = 0
            first_wp["edge_kind"] = "cycle_single"
            self._mesh_view_local_orientation_last_ms = float(local_orientation_ms)
            self._mesh_view_local_orientation_last_segments = int(local_orientation_segments)
            return [first_wp]

        for nbv_idx in range(len(ordered)):
            src = ordered[nbv_idx]
            dst = ordered[(nbv_idx + 1) % len(ordered)]
            edge_details = self._pair_path_details_mesh(src, dst)
            edge_pts = list(edge_details.get("points", []))
            if len(edge_pts) <= 0:
                try:
                    edge_pts = [
                        np.asarray(src.get("origin", None), dtype=np.float64).reshape(3),
                        np.asarray(dst.get("origin", None), dtype=np.float64).reshape(3),
                    ]
                except Exception:
                    edge_pts = []
            edge_pts = self._dedup_consecutive_points(edge_pts)
            if len(edge_pts) <= 0:
                continue
            edge_kind = str(edge_details.get("status", "straight_invalid"))
            status_counts[edge_kind] = int(status_counts.get(edge_kind, 0)) + 1
            if bool(edge_details.get("fallback", False)):
                fallback_diags.append(
                    "edge %d->%d used straight fallback after collision: %s"
                    % (int(nbv_idx), int(nbv_idx + 1), str(edge_details.get("reason", "unknown")))
                )
            try:
                dst_target = np.asarray(dst.get("target", None), dtype=np.float64).reshape(3)
            except Exception:
                continue
            segment_defs.append(
                {
                    "points": list(edge_pts),
                    "target": np.asarray(dst_target, dtype=np.float64).reshape(3),
                    "goal_origin": np.asarray(dst.get("origin", None), dtype=np.float64).reshape(3),
                    "nbv_index": int((nbv_idx + 1) % len(ordered)),
                    "edge_kind": edge_kind,
                    "sample_template": copy.deepcopy(dst),
                }
            )

        if len(segment_defs) <= 0:
            first_wp = copy.deepcopy(ordered[0])
            first_wp["is_nbv"] = True
            first_wp["nbv_index"] = 0
            first_wp["edge_kind"] = "cycle_fallback"
            self._mesh_view_local_orientation_last_ms = float(local_orientation_ms)
            self._mesh_view_local_orientation_last_segments = int(local_orientation_segments)
            return [first_wp]

        if not bool(self.mesh_view_local_planning_enabled):
            for seg in segment_defs:
                seg_pts = self._dedup_consecutive_points(seg.get("points", []))
                if len(seg_pts) <= 0:
                    continue
                cmd_pts = list(seg_pts[1:]) if len(seg_pts) > 1 else list(seg_pts)
                if len(cmd_pts) <= 0:
                    continue
                target = np.asarray(seg["target"], dtype=np.float64).reshape(3)
                for pt_idx, pt in enumerate(cmd_pts):
                    wp = copy.deepcopy(seg["sample_template"])
                    pt = np.asarray(pt, dtype=np.float64).reshape(3)
                    pose = self._look_at_pose(origin=pt, target=target)
                    if pose is None:
                        continue
                    wp["origin"] = pt
                    wp["target"] = np.asarray(target, dtype=np.float64).reshape(3)
                    wp["pose"] = tuple(float(v) for v in pose[:6])
                    wp["is_nbv"] = bool(int(pt_idx) == int(len(cmd_pts) - 1))
                    wp["nbv_index"] = int(seg["nbv_index"])
                    wp["edge_kind"] = str(seg["edge_kind"])
                    exec_waypoints.append(wp)

            if len(exec_waypoints) <= 0:
                first_wp = copy.deepcopy(ordered[0])
                first_wp["is_nbv"] = True
                first_wp["nbv_index"] = 0
                first_wp["edge_kind"] = "cycle_fallback"
                exec_waypoints.append(first_wp)

            if self.mesh_view_start_leg_debug_enabled:
                for msg in fallback_diags:
                    rospy.logwarn("Trajectory fallback: %s", msg)
                if status_counts:
                    parts = ["%s=%d" % (str(k), int(status_counts[k])) for k in sorted(status_counts.keys())]
                    rospy.loginfo("Trajectory edge status summary: %s", ", ".join(parts))
                if len(exec_waypoints) > 0:
                    exec_origins = [np.asarray(wp.get("origin", None), dtype=np.float64).reshape(3) for wp in exec_waypoints[:5]]
                    rospy.loginfo(
                        "Execution trajectory diagnostics: total_exec_pts=%d local_orient_segments=%d first_exec_mesh=%s first_exec_viz=%s",
                        int(len(exec_waypoints)),
                        int(local_orientation_segments),
                        self._fmt_point_list(exec_origins, limit=5, viz=False),
                        self._fmt_point_list(exec_origins, limit=5, viz=True),
                    )

            self._mesh_view_local_orientation_last_ms = float(local_orientation_ms)
            self._mesh_view_local_orientation_last_segments = int(local_orientation_segments)
            return exec_waypoints

        cmd_pts = []
        cmd_seg_indices = []
        n_ordered = int(len(ordered))
        for seg_idx, seg in enumerate(segment_defs):
            seg_pts = self._dedup_consecutive_points(seg.get("points", []))
            if len(seg_pts) <= 0:
                continue
            edge_kind = str(seg.get("edge_kind", ""))
            use_triplet_bspline = (
                n_ordered >= 3
                and edge_kind == "straight_no_collision"
            )
            if use_triplet_bspline:
                prev_idx = int((seg_idx - 1) % n_ordered)
                src_idx = int(seg_idx % n_ordered)
                dst_idx = int((seg_idx + 1) % n_ordered)
                try:
                    ctrl = [
                        np.asarray(ordered[prev_idx].get("origin", None), dtype=np.float64).reshape(3),
                        np.asarray(ordered[src_idx].get("origin", None), dtype=np.float64).reshape(3),
                        np.asarray(ordered[dst_idx].get("origin", None), dtype=np.float64).reshape(3),
                    ]
                    seg_cmd_pts = self._sample_execution_control_span(ctrl, 1, 2, preserve_endpoints=True)
                except Exception:
                    seg_cmd_pts = list(seg_pts)
            else:
                seg_cmd_pts = list(seg_pts)

            seg_cmd_pts = self._dedup_consecutive_points(seg_cmd_pts)
            if len(seg_cmd_pts) <= 0:
                continue
            seg_cmd_pts = list(seg_cmd_pts[1:]) if len(seg_cmd_pts) > 1 else list(seg_cmd_pts)
            for pt in seg_cmd_pts:
                cmd_pts.append(np.asarray(pt, dtype=np.float64).reshape(3))
                cmd_seg_indices.append(int(seg_idx))

        planned_points = []
        if bool(self.mesh_view_local_planning_enabled and self.mesh_view_local_orientation_enabled) and orient_ctx is not None and len(cmd_pts) > 0:
            t_local = time.perf_counter()
            orient_plan = self._plan_local_orientation_chain(
                path_points=cmd_pts,
                start_pos=start_pos,
                start_yaw=float(start_yaw),
                start_pitch=float(start_pitch),
                orient_ctx=orient_ctx,
                fallback_target=np.asarray(segment_defs[-1]["target"], dtype=np.float64).reshape(3),
            )
            local_orientation_ms += (time.perf_counter() - t_local) * 1000.0
            local_orientation_segments += 1
            planned_points = list(orient_plan.get("points", []))

        for cmd_idx, pt in enumerate(cmd_pts, start=1):
            seg_idx = int(cmd_seg_indices[int(cmd_idx - 1)]) if int(cmd_idx - 1) < len(cmd_seg_indices) else 0
            seg = segment_defs[min(max(seg_idx, 0), len(segment_defs) - 1)]
            wp = copy.deepcopy(seg["sample_template"])
            wp["origin"] = np.asarray(pt, dtype=np.float64).reshape(3)
            if len(planned_points) == len(cmd_pts):
                cand = planned_points[int(cmd_idx - 1)]
                target = np.asarray(cand.get("target", seg["target"]), dtype=np.float64).reshape(3)
                pose = cand.get("pose", None)
                if pose is None:
                    pose = self._look_at_pose(origin=pt, target=target)
            else:
                target = np.asarray(seg["target"], dtype=np.float64).reshape(3)
                pose = self._look_at_pose(origin=pt, target=target)
            if pose is None:
                continue
            wp["target"] = np.asarray(target, dtype=np.float64).reshape(3)
            wp["pose"] = tuple(float(v) for v in pose[:6])
            next_seg_idx = (
                int(cmd_seg_indices[int(cmd_idx)]) if int(cmd_idx) < len(cmd_seg_indices) else None
            )
            wp["is_nbv"] = bool(next_seg_idx is None or int(next_seg_idx) != int(seg_idx))
            wp["nbv_index"] = int(seg["nbv_index"])
            wp["edge_kind"] = str(seg["edge_kind"])
            exec_waypoints.append(wp)

        if len(exec_waypoints) <= 0:
            first_wp = copy.deepcopy(ordered[0])
            first_wp["is_nbv"] = True
            first_wp["nbv_index"] = 0
            first_wp["edge_kind"] = "cycle_fallback"
            exec_waypoints.append(first_wp)

        if self.mesh_view_start_leg_debug_enabled:
            for msg in fallback_diags:
                rospy.logwarn("Trajectory fallback: %s", msg)
            if status_counts:
                parts = ["%s=%d" % (str(k), int(status_counts[k])) for k in sorted(status_counts.keys())]
                rospy.loginfo("Trajectory edge status summary: %s", ", ".join(parts))
            if len(exec_waypoints) > 0:
                exec_origins = [np.asarray(wp.get("origin", None), dtype=np.float64).reshape(3) for wp in exec_waypoints[:5]]
                rospy.loginfo(
                    "Execution trajectory diagnostics: total_exec_pts=%d local_orient_segments=%d first_exec_mesh=%s first_exec_viz=%s",
                    int(len(exec_waypoints)),
                    int(local_orientation_segments),
                    self._fmt_point_list(exec_origins, limit=5, viz=False),
                    self._fmt_point_list(exec_origins, limit=5, viz=True),
                )

        self._mesh_view_local_orientation_last_ms = float(local_orientation_ms)
        self._mesh_view_local_orientation_last_segments = int(local_orientation_segments)
        return exec_waypoints

    def _global_path_edge_cost(
        self,
        pos_i,
        yaw_i,
        pitch_i,
        pos_j,
        yaw_j,
        pitch_j,
        c_eff_i,
        gain_eff_i,
        c_eff_j,
        gain_eff_j,
        trans_dist=None,
    ):
        if trans_dist is None:
            if pos_i is None or pos_j is None:
                dist = 0.0
            else:
                dist = float(np.linalg.norm(np.asarray(pos_j) - np.asarray(pos_i)))
        else:
            dist = float(max(0.0, trans_dist))
        dyaw = abs(self._angle_diff(yaw_j, yaw_i))
        motion = (
            float(self.mesh_view_plan_base_cost)
            + float(self.mesh_view_plan_distance_weight) * (dist / float(self.mesh_view_plan_trans_speed_mps))
            + float(self.mesh_view_plan_yaw_weight) * (dyaw / float(self.mesh_view_plan_rot_speed_rps))
        )
        current_eff_sum = max(0.0, float(c_eff_i)) + max(0.0, float(c_eff_j))
        gain_eff_sum = max(0.0, float(gain_eff_i)) + max(0.0, float(gain_eff_j))
        return motion * (
            (1.0 + max(0.0, float(self.mesh_view_plan_cvi_weight)) * current_eff_sum)
            / (1.0 + max(0.0, float(self.mesh_view_plan_gain_weight)) * gain_eff_sum)
        )

    def _local_path_edge_cost(
        self,
        pos_i,
        yaw_i,
        pitch_i,
        pos_j,
        yaw_j,
        pitch_j,
        c_eff_i,
        gain_eff_i,
        c_eff_j,
        gain_eff_j,
        trans_dist=None,
    ):
        if trans_dist is None:
            if pos_i is None or pos_j is None:
                dist = 0.0
            else:
                dist = float(np.linalg.norm(np.asarray(pos_j) - np.asarray(pos_i)))
        else:
            dist = float(max(0.0, trans_dist))
        dyaw = abs(self._angle_diff(yaw_j, yaw_i))
        dpitch = abs(float(pitch_j) - float(pitch_i))
        motion = (
            float(self.mesh_view_local_orientation_base_cost)
            + float(self.mesh_view_local_orientation_distance_weight) * (dist / float(self.mesh_view_plan_trans_speed_mps))
            + float(self.mesh_view_local_orientation_yaw_weight) * (dyaw / float(self.mesh_view_plan_rot_speed_rps))
            + float(self.mesh_view_local_orientation_pitch_weight) * (dpitch / float(self.mesh_view_plan_pitch_speed_rps))
        )
        current_eff_sum = max(0.0, float(c_eff_i)) + max(0.0, float(c_eff_j))
        gain_eff_sum = max(0.0, float(gain_eff_i)) + max(0.0, float(gain_eff_j))
        return motion * (
            (1.0 + max(0.0, float(self.mesh_view_local_orientation_cvi_weight)) * current_eff_sum)
            / (1.0 + max(0.0, float(self.mesh_view_local_orientation_gain_weight)) * gain_eff_sum)
        )

    def _build_viewpoint_plan_costs(self, samples):
        n = len(samples)
        if n <= 0:
            return None

        start_pos, start_yaw, start_pitch = self._planner_start_state(samples)
        if start_pos is None or start_yaw is None or start_pitch is None:
            return None

        planning_nodes = list(samples)
        m = int(len(planning_nodes))
        origins = np.asarray([s["origin"] for s in planning_nodes], dtype=np.float64)
        yaws = np.asarray([float(s["pose"][5]) for s in planning_nodes], dtype=np.float64)
        pitches = np.asarray([float(s["pose"][4]) for s in planning_nodes], dtype=np.float64)

        metric_ctx = self._build_local_orientation_context()
        sample_metrics = []
        for i in range(m):
            if metric_ctx is None:
                sample_metrics.append({"c_eff": 0.0, "gain_eff": 0.0})
            else:
                sample_metrics.append(
                    self._evaluate_viewpoint_effective_metrics(origins[i], yaws[i], pitches[i], metric_ctx)
                )

        C = np.full((m, m), np.inf, dtype=np.float64)
        for i in range(m):
            C[i, i] = 0.0
        for i in range(m):
            for j in range(i + 1, m):
                pair_details = self._pair_path_details_mesh(planning_nodes[i], planning_nodes[j])
                pair_pts = pair_details.get("points", [])
                trans_dist = float(self._polyline_length(pair_pts))
                if trans_dist <= 1.0e-9:
                    trans_dist = float(np.linalg.norm(origins[j] - origins[i]))
                edge_cost = self._global_path_edge_cost(
                    origins[i],
                    yaws[i],
                    pitches[i],
                    origins[j],
                    yaws[j],
                    pitches[j],
                    float(sample_metrics[i].get("c_eff", 0.0)),
                    float(sample_metrics[i].get("gain_eff", 0.0)),
                    float(sample_metrics[j].get("c_eff", 0.0)),
                    float(sample_metrics[j].get("gain_eff", 0.0)),
                    trans_dist=trans_dist,
                )
                C[i, j] = float(edge_cost)
                C[j, i] = float(edge_cost)

        start_metrics = {"c_eff": 0.0, "gain_eff": 0.0}
        if metric_ctx is not None:
            start_metrics = self._evaluate_viewpoint_effective_metrics(
                np.asarray(start_pos, dtype=np.float64).reshape(3),
                float(start_yaw),
                float(start_pitch),
                metric_ctx,
            )

        start_cost = np.full((m,), np.inf, dtype=np.float64)
        for j in range(m):
            start_details = self._start_path_details_mesh(planning_nodes[j], start_pos=start_pos)
            start_pts = start_details.get("points", [])
            trans_dist = float(self._polyline_length(start_pts))
            if trans_dist <= 1.0e-9:
                trans_dist = float(np.linalg.norm(origins[j] - np.asarray(start_pos, dtype=np.float64).reshape(3)))
            start_cost[j] = float(
                self._global_path_edge_cost(
                    np.asarray(start_pos, dtype=np.float64).reshape(3),
                    float(start_yaw),
                    float(start_pitch),
                    origins[j],
                    yaws[j],
                    pitches[j],
                    float(start_metrics.get("c_eff", 0.0)),
                    float(start_metrics.get("gain_eff", 0.0)),
                    float(sample_metrics[j].get("c_eff", 0.0)),
                    float(sample_metrics[j].get("gain_eff", 0.0)),
                    trans_dist=trans_dist,
                )
            )

        return {
            "n": int(n),
            "planning_nodes": planning_nodes,
            "full_C": C,
            "origins": origins,
            "yaws": yaws,
            "pitches": pitches,
            "sample_c_eff": np.asarray([float(m.get("c_eff", 0.0)) for m in sample_metrics], dtype=np.float64),
            "sample_gain_eff": np.asarray([float(m.get("gain_eff", 0.0)) for m in sample_metrics], dtype=np.float64),
            "start_pos": start_pos,
            "start_yaw": float(start_yaw),
            "start_pitch": float(start_pitch),
            "start_c_eff": float(start_metrics.get("c_eff", 0.0)),
            "start_gain_eff": float(start_metrics.get("gain_eff", 0.0)),
            "C": C,
            "start_cost": start_cost,
        }

    def _plan_viewpoint_path_msgnn(self, samples, cost_data=None):
        n = len(samples)
        if n <= 1:
            return list(samples)
        if cost_data is None:
            cost_data = self._build_viewpoint_plan_costs(samples)
        if cost_data is None:
            return list(samples)

        C = np.asarray(cost_data.get("C", []), dtype=np.float64)
        m = int(len(samples))
        if m <= 1 or C.shape != (m, m):
            return list(samples)

        def seq_cost(node_ord):
            if not node_ord:
                return 0.0
            total = 0.0
            if len(node_ord) == 1:
                return 0.0
            for k in range(len(node_ord)):
                total += float(C[int(node_ord[k]), int(node_ord[(k + 1) % len(node_ord)])])
            return float(total)

        row_sums = np.sum(C, axis=1)
        sorted_first = sorted(range(m), key=lambda idx: float(row_sums[int(idx)]))
        n_starts = min(len(sorted_first), int(max(1, self.mesh_view_plan_atsp_multi_start)))
        candidate_first = [int(x) for x in sorted_first[:n_starts]]

        best_order = None
        best_cost = np.inf
        for first in candidate_first:
            unvisited = set(range(m))
            order = [int(first)]
            unvisited.remove(int(first))

            while unvisited:
                cur = int(order[-1])
                nxt = min(unvisited, key=lambda j: float(C[cur, int(j)]))
                order.append(int(nxt))
                unvisited.remove(int(nxt))

            if (
                self.mesh_view_plan_2opt_enabled
                and m >= 4
                and (self.mesh_view_plan_2opt_max_nodes <= 0 or m <= self.mesh_view_plan_2opt_max_nodes)
                and self.mesh_view_plan_2opt_max_iters > 0
            ):
                local_best = order[:]
                local_best_cost = seq_cost(local_best)
                for _ in range(int(self.mesh_view_plan_2opt_max_iters)):
                    improved = False
                    for i in range(0, m - 2):
                        for j in range(i + 1, m - 1):
                            cand = local_best[:i] + local_best[i : j + 1][::-1] + local_best[j + 1 :]
                            cand_cost = seq_cost(cand)
                            if cand_cost + 1e-9 < local_best_cost:
                                local_best = cand
                                local_best_cost = cand_cost
                                improved = True
                    if not improved:
                        break
                order = local_best

            cst = seq_cost(order)
            if cst + 1e-9 < best_cost:
                best_cost = cst
                best_order = order

        if best_order is None:
            return list(samples)
        if len(best_order) > 1:
            pivot = best_order.index(min(best_order))
            best_order = best_order[pivot:] + best_order[:pivot]
        return [samples[idx] for idx in best_order if 0 <= int(idx) < n]

    def _build_lkh_problem_files(self, samples, cost_data, prefix):
        planning_nodes = list(cost_data.get("planning_nodes", []))
        n = int(len(planning_nodes))
        scale = int(self.mesh_view_plan_lkh_scale)
        C = np.asarray(cost_data["C"], dtype=np.float64)
        finite_vals = C[np.isfinite(C) & (~np.eye(n, dtype=bool))]
        if finite_vals.size <= 0:
            large_cost = scale
        else:
            large_cost = max(scale, int(np.ceil(float(np.max(finite_vals)) * float(scale) * 10.0)))

        C_int = np.zeros((n, n), dtype=np.int64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = float(C[i, j])
                if not np.isfinite(val):
                    C_int[i, j] = int(large_cost)
                else:
                    C_int[i, j] = int(max(0, round(float(scale) * val)))
        tsp_path = os.path.join(self.mesh_view_plan_lkh_workdir, prefix + ".tsp")
        par_path = os.path.join(self.mesh_view_plan_lkh_workdir, prefix + ".par")
        tour_path = os.path.join(self.mesh_view_plan_lkh_workdir, prefix + ".tour")

        with open(tsp_path, "w") as f:
            f.write("NAME : %s\n" % prefix)
            f.write("COMMENT : Controller symmetric cyclic TSP with %d nodes.\n" % n)
            f.write("TYPE : TSP\n")
            f.write("DIMENSION : %d\n" % n)
            f.write("EDGE_WEIGHT_TYPE : EXPLICIT\n")
            f.write("EDGE_WEIGHT_FORMAT : FULL_MATRIX\n")
            f.write("EDGE_WEIGHT_SECTION\n")
            for i in range(n):
                row = [str(int(C_int[i, j])) for j in range(n)]
                f.write(" ".join(row) + "\n")
            f.write("EOF\n")

        with open(par_path, "w") as f:
            f.write("PROBLEM_FILE = %s\n" % tsp_path)
            f.write("OUTPUT_TOUR_FILE = %s\n" % tour_path)
            f.write("MAX_CANDIDATES = %d\n" % int(self.mesh_view_plan_lkh_max_candidates))
            f.write("MAX_TRIALS = %d\n" % int(self.mesh_view_plan_lkh_max_trials))
            f.write("RUNS = %d\n" % int(self.mesh_view_plan_lkh_runs))
            f.write("RESTRICTED_SEARCH = NO\n")
            f.write("NONSEQUENTIAL_MOVE_TYPE = 9\n")
            f.write("PATCHING_A = 3 EXTENDED\n")
            f.write("PATCHING_C = 3 EXTENDED\n")
            f.write("POPULATION_SIZE = 30\n")
            f.write("KICKS = 4\n")
            f.write("SEED = 1\n")
            f.write("TRACE_LEVEL = %d\n" % int(self.mesh_view_plan_lkh_trace_level))
            f.write("SPECIAL\n")

        return tsp_path, par_path, tour_path

    def _parse_lkh_tour(self, tour_path, n):
        if not os.path.isfile(tour_path):
            raise RuntimeError("LKH tour file not found: %s" % tour_path)

        length_value = None
        nodes = []
        in_section = False
        with open(tour_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("COMMENT") and "Length =" in line:
                    try:
                        length_value = float(line.split("Length =", 1)[1].strip())
                    except Exception:
                        length_value = None
                if line == "TOUR_SECTION":
                    in_section = True
                    continue
                if not in_section:
                    continue
                if line in ("EOF", "-1"):
                    break
                nodes.append(int(line))

        expected = set(range(1, int(n) + 1))
        if len(nodes) != int(n) or set(nodes) != expected:
            raise RuntimeError("Unexpected LKH tour content for %d viewpoints" % int(n))
        order = [int(node) - 1 for node in nodes]
        if len(order) > 1:
            pivot = order.index(min(order))
            order = order[pivot:] + order[:pivot]
        return order, length_value

    def _plan_viewpoint_path_lkh(self, samples, cost_data=None):
        n = len(samples)
        if n <= 1:
            return list(samples)
        if cost_data is None:
            cost_data = self._build_viewpoint_plan_costs(samples)
        if cost_data is None:
            return list(samples)
        planning_nodes = list(cost_data.get("planning_nodes", []))

        t0 = time.monotonic()
        try:
            prefix = "nbv_plan_current"
            _tsp_path, par_path, tour_path = self._build_lkh_problem_files(samples, cost_data, prefix)
            run_kwargs = dict(
                cwd=self.mesh_view_plan_lkh_workdir,
                capture_output=True,
                text=True,
                check=False,
            )
            if float(self.mesh_view_plan_lkh_timeout_s) > 0.0:
                run_kwargs["timeout"] = float(self.mesh_view_plan_lkh_timeout_s)
            proc = subprocess.run(
                [self.mesh_view_plan_lkh_binary, par_path],
                **run_kwargs,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "LKH returned non-zero")
            order, total_cost = self._parse_lkh_tour(
                tour_path,
                int(len(planning_nodes)),
            )
            ordered = [samples[idx] for idx in order if 0 <= int(idx) < n]
            if self.mesh_view_debug_logs:
                rospy.loginfo(
                    "Planner cycle: backend=lkh n=%d cost=%s time=%.2fs",
                    len(ordered),
                    str(total_cost),
                    float(time.monotonic() - t0),
                )
            return ordered
        except Exception as e:
            rospy.logwarn("LKH planning failed: %s. Falling back to MSGNN ordering.", str(e))
            ordered = self._plan_viewpoint_path_msgnn(samples, cost_data=cost_data)
            if self.mesh_view_debug_logs:
                rospy.loginfo(
                    "Planner cycle: backend=msgnn_fallback n=%d time=%.2fs",
                    len(ordered),
                    float(time.monotonic() - t0),
                )
            return ordered

    def plan_viewpoint_path(self, samples, cost_data=None):
        return self._plan_viewpoint_path_msgnn(samples, cost_data=cost_data)

    def publish_sampled_view_trajectory(self, ordered_samples, exec_waypoints=None):
        ordered = list(ordered_samples) if ordered_samples is not None else []
        exec_pts_src = list(exec_waypoints) if exec_waypoints is not None else []
        self._publish_current_pose_point()

        if not ordered and not exec_pts_src:
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = self.frame_id
            marker.ns = "trajectory"
            marker.id = 0
            marker.action = Marker.DELETE
            self.mesh_view_trajectory_pub.publish(marker)

            skipped_marker = Marker()
            skipped_marker.header.stamp = rospy.Time.now()
            skipped_marker.header.frame_id = self.frame_id
            skipped_marker.ns = "trajectory_skipped"
            skipped_marker.id = 4
            skipped_marker.action = Marker.DELETE
            self.mesh_view_skipped_trajectory_pub.publish(skipped_marker)

            points_marker = Marker()
            points_marker.header.stamp = rospy.Time.now()
            points_marker.header.frame_id = self.frame_id
            points_marker.ns = "viewpoint_positions"
            points_marker.id = 1
            points_marker.action = Marker.DELETE
            self.mesh_view_viewpoint_positions_pub.publish(points_marker)

            global_points_marker = Marker()
            global_points_marker.header.stamp = rospy.Time.now()
            global_points_marker.header.frame_id = self.frame_id
            global_points_marker.ns = "global_viewpoint_positions"
            global_points_marker.id = 2
            global_points_marker.action = Marker.DELETE
            self.mesh_view_global_viewpoint_positions_pub.publish(global_points_marker)

            current_pose_marker = Marker()
            current_pose_marker.header.stamp = rospy.Time.now()
            current_pose_marker.header.frame_id = self.frame_id
            current_pose_marker.ns = "current_pose_point"
            current_pose_marker.id = 3
            current_pose_marker.action = Marker.DELETE
            self.mesh_view_current_pose_point_pub.publish(current_pose_marker)
            return

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "trajectory"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(max(1e-4, self.mesh_view_plan_width))
        color_vals = list(self.mesh_view_plan_color) + [1.0, 0.55, 0.1, 0.9]
        marker.color.r = float(color_vals[0])
        marker.color.g = float(color_vals[1])
        marker.color.b = float(color_vals[2])
        marker.color.a = float(color_vals[3])

        skipped_marker = Marker()
        skipped_marker.header.stamp = marker.header.stamp
        skipped_marker.header.frame_id = self.frame_id
        skipped_marker.ns = "trajectory_skipped"
        skipped_marker.id = 4
        skipped_marker.type = Marker.LINE_STRIP
        skipped_marker.action = Marker.ADD
        skipped_marker.pose.orientation.w = 1.0
        skipped_marker.scale.x = float(max(1e-4, self.mesh_view_plan_width))
        skipped_marker.color.r = 1.0
        skipped_marker.color.g = 0.1
        skipped_marker.color.b = 0.1
        skipped_marker.color.a = float(color_vals[3])

        pts = []
        skipped_pts = []
        if self.mesh_view_plan_include_start:
            start_pos, _, _ = self._planner_start_state(ordered if ordered else exec_pts_src)
            if start_pos is not None:
                start_pt = Point(x=float(start_pos[0]), y=float(start_pos[1]), z=float(start_pos[2]))
                pts.append(start_pt)
        path_samples = exec_pts_src if exec_pts_src else ordered
        path_sample_offset = 0
        skipped_samples = []
        if exec_pts_src:
            nearest_exec = self._nearest_exec_waypoint_info(exec_pts_src)
            if nearest_exec is not None:
                path_sample_offset = max(0, min(int(nearest_exec.get("idx", 0)), len(exec_pts_src) - 1))
                path_samples = list(exec_pts_src[path_sample_offset:])
                skipped_samples = list(exec_pts_src[:path_sample_offset])
        path_origins = []
        skipped_origins = []
        global_origins = []
        for s in ordered:
            o = np.asarray(s["origin"], dtype=np.float64).reshape(3)
            global_origins.append(o)
        if len(path_samples) > 0 and len(skipped_samples) > 0:
            try:
                o = np.asarray(path_samples[-1]["origin"], dtype=np.float64).reshape(3)
                skipped_pts.append(Point(x=float(o[0]), y=float(o[1]), z=float(o[2])))
            except Exception:
                pass
        for s in skipped_samples:
            o = np.asarray(s["origin"], dtype=np.float64).reshape(3)
            skipped_origins.append(o)
            skipped_pts.append(Point(x=float(o[0]), y=float(o[1]), z=float(o[2])))
        for s in path_samples:
            o = np.asarray(s["origin"], dtype=np.float64).reshape(3)
            path_origins.append(o)
            pts.append(Point(x=float(o[0]), y=float(o[1]), z=float(o[2])))

        if self.mesh_view_start_leg_debug_enabled:
            rospy.loginfo(
                "Trajectory publish diagnostics: using_exec=%s ordered=%d exec=%d published_path_pts=%d first_pub_mesh=%s first_pub_viz=%s",
                "yes" if bool(exec_pts_src) else "no",
                int(len(ordered)),
                int(len(exec_pts_src)),
                int(len(path_origins)),
                self._fmt_point_list(path_origins, limit=5, viz=False),
                self._fmt_point_list(path_origins, limit=5, viz=True),
            )

        marker.points = pts
        self._rviz_cache_points("trajectory_points", marker.points)
        self.mesh_view_trajectory_pub.publish(marker)
        if len(skipped_pts) >= 2:
            skipped_marker.points = skipped_pts
            self._rviz_cache_points("trajectory_skipped_points", skipped_marker.points)
            self.mesh_view_skipped_trajectory_pub.publish(skipped_marker)
        else:
            skipped_marker.action = Marker.DELETE
            self.mesh_view_skipped_trajectory_pub.publish(skipped_marker)

        points_marker = Marker()
        points_marker.header.stamp = marker.header.stamp
        points_marker.header.frame_id = self.frame_id
        points_marker.ns = "viewpoint_positions"
        points_marker.id = 1
        points_marker.type = Marker.SPHERE_LIST
        points_marker.action = Marker.ADD
        points_marker.pose.orientation.w = 1.0
        point_scale = float(max(1e-3, self.mesh_view_plan_point_scale))
        points_marker.scale.x = point_scale
        points_marker.scale.y = point_scale
        points_marker.scale.z = point_scale
        point_color_vals = list(self.mesh_view_plan_point_color) + [0.15, 0.95, 1.0, 0.95]
        points_marker.color.r = float(point_color_vals[0])
        points_marker.color.g = float(point_color_vals[1])
        points_marker.color.b = float(point_color_vals[2])
        points_marker.color.a = float(point_color_vals[3])
        points_marker.points = [
            Point(x=float(o[0]), y=float(o[1]), z=float(o[2])) for o in path_origins
        ]
        self._rviz_cache_points("viewpoint_positions", points_marker.points)
        self.mesh_view_viewpoint_positions_pub.publish(points_marker)

        global_points_marker = Marker()
        global_points_marker.header.stamp = marker.header.stamp
        global_points_marker.header.frame_id = self.frame_id
        global_points_marker.ns = "global_viewpoint_positions"
        global_points_marker.id = 2
        global_points_marker.type = Marker.SPHERE_LIST
        global_points_marker.action = Marker.ADD
        global_points_marker.pose.orientation.w = 1.0
        global_points_marker.scale.x = point_scale
        global_points_marker.scale.y = point_scale
        global_points_marker.scale.z = point_scale
        global_points_marker.color.r = 1.0
        global_points_marker.color.g = 0.1
        global_points_marker.color.b = 0.1
        global_points_marker.color.a = float(point_color_vals[3])
        global_points_marker.points = [
            Point(x=float(o[0]), y=float(o[1]), z=float(o[2])) for o in global_origins
        ]
        self._rviz_cache_points("global_viewpoint_positions", global_points_marker.points)
        self.mesh_view_global_viewpoint_positions_pub.publish(global_points_marker)

    def publish_sampled_view_vectors(self, samples):
        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = self.frame_id
        marker.ns = "sampled_view_vectors"
        marker.id = 0

        if not samples:
            marker.action = Marker.DELETE
            self.mesh_view_vectors_pub.publish(marker)
            return

        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = float(max(1e-4, self.mesh_view_vector_width))
        color_vals = list(self.mesh_view_vector_color) + [1.0, 0.2, 0.9, 0.9]
        r, g, b, a = color_vals[:4]
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)

        pts = []
        line_w = float(max(1e-4, self.mesh_view_vector_width))
        for s in samples:
            o = s["origin"]
            t = s["target"]
            o3 = np.asarray(o, dtype=np.float64).reshape(3)
            t3 = np.asarray(t, dtype=np.float64).reshape(3)
            d = t3 - o3
            dn = float(np.linalg.norm(d))
            if dn <= 1e-9:
                continue

            # Main shaft: viewpoint origin -> look target.
            pts.append(Point(x=float(o3[0]), y=float(o3[1]), z=float(o3[2])))
            pts.append(Point(x=float(t3[0]), y=float(t3[1]), z=float(t3[2])))

            # Arrow head at target so direction is visually explicit.
            fwd = d / dn
            up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(fwd, up_hint))) > 0.95:
                up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            side = np.cross(fwd, up_hint)
            sn = float(np.linalg.norm(side))
            if sn <= 1e-9:
                continue
            side = side / sn

            head_len = min(max(6.0 * line_w, 0.12), 0.25 * dn)
            head_w = 0.45 * head_len
            left = t3 - head_len * fwd + head_w * side
            right = t3 - head_len * fwd - head_w * side

            pts.append(Point(x=float(t3[0]), y=float(t3[1]), z=float(t3[2])))
            pts.append(Point(x=float(left[0]), y=float(left[1]), z=float(left[2])))
            pts.append(Point(x=float(t3[0]), y=float(t3[1]), z=float(t3[2])))
            pts.append(Point(x=float(right[0]), y=float(right[1]), z=float(right[2])))
        marker.points = pts
        self._rviz_cache_points("sampled_view_vectors_points", marker.points)
        self.mesh_view_vectors_pub.publish(marker)

    @staticmethod
    def _cvi_temperature_color(t):
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

    def publish_gaussian_cloud(self):
        if self.gauss_pos is None:
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        fields = [
            PointField("x", 0, PointField.FLOAT32, 1),
            PointField("y", 4, PointField.FLOAT32, 1),
            PointField("z", 8, PointField.FLOAT32, 1),
            PointField("rgb", 12, PointField.FLOAT32, 1),
        ]

        if self.gauss_cvi is not None:
            cvi = self.gauss_cvi
        else:
            cvi = self.gauss_opacity if self.gauss_opacity is not None else np.zeros(self.gauss_pos.shape[0])

        if cvi.size > 0:
            vmin = float(np.min(cvi))
            vmax = float(np.max(cvi))
            if abs(vmax - vmin) < 1e-8:
                norm = np.zeros_like(cvi)
            else:
                norm = (cvi - vmin) / (vmax - vmin)
        else:
            norm = cvi

        points = []
        for (x, y, z), t in zip(self.gauss_pos, norm):
            r, g, b = self._cvi_temperature_color(t)
            r_i = int(max(0, min(255, round(r * 255))))
            g_i = int(max(0, min(255, round(g * 255))))
            b_i = int(max(0, min(255, round(b * 255))))
            rgb_uint32 = (r_i << 16) | (g_i << 8) | b_i
            rgb_float = struct.unpack("f", struct.pack("I", rgb_uint32))[0]
            points.append((float(x), float(y), float(z), rgb_float))

        cloud = point_cloud2.create_cloud(header, fields, points)
        self.gaussian_cloud_pub.publish(cloud)
    
    def drone2global_angles(self, roll=0.0, pitch=0.0, yaw=0.0):
        rot_factor = self.rot_factor
        gr = -rot_factor*(pitch - self.init_pitch)
        gp = -rot_factor*(roll - self.init_roll)
        gy = -rot_factor*(yaw - self.init_yaw)
        return gr, gp, gy

    def _drone_command_angles(self, roll=None, pitch=None, yaw=None):
        """
        Convert internal planner/control roll-pitch-yaw radians to the simulator/body-command
        convention immediately before publishing a drone orientation command.
        """
        troll, tpitch, tyaw = self.target_angles
        if roll is None:
            roll = troll
        if pitch is None:
            pitch = tpitch
        if yaw is None:
            yaw = tyaw
        return self.drone2global_angles(roll, pitch, yaw)

    def odom_callback(self, odom_msg):
        """Store latest odometry (NED)."""
        self.latest_odom = odom_msg
        try:
            p = odom_msg.pose.pose.position
            self._append_actual_path_point(
                np.array([float(p.x), float(p.y), float(p.z)], dtype=np.float64)
            )
        except Exception:
            pass

    def _latest_trans_speed_mps(self, odom_msg=None):
        odom_msg = self.latest_odom if odom_msg is None else odom_msg
        if odom_msg is None:
            return None
        try:
            v = odom_msg.twist.twist.linear
            vx = float(v.x)
            vy = float(v.y)
            vz = float(v.z)
        except Exception:
            return None
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        return float(speed)

    @staticmethod
    def _message_stamp_key(msg):
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is None or stamp == rospy.Time():
            return None
        try:
            return int(stamp.to_nsec())
        except Exception:
            return None

    def _prune_depth_capture_cache_locked(self):
        while len(self._depth_capture_records) > int(self._depth_capture_cache_limit):
            self._depth_capture_records.pop(next(iter(self._depth_capture_records)))

    def _depth_capture_selected_for_rgb(self, image_id):
        if bool(getattr(self, "save_depth_frames", False)):
            return True
        stride = max(1, int(getattr(self, "depth_frame_stride", 1)))
        return ((int(image_id) - 1) % stride) == 0

    def _emit_paired_depth_capture(self, record, depth_msg):
        image_id = int(record["image_id"])
        pose_msg = record["pose_msg"]
        if bool(getattr(self, "save_depth_frames", False)):
            try:
                depth = np.asarray(
                    self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"),
                    dtype=np.float32,
                )
                np.save(os.path.join(self.depth_dir, f"depth_{int(record['pose_image_id']):06d}.npy"), depth)
            except Exception as e:
                rospy.logwarn("[WARN] Failed to save paired depth frame %s: %s", _blue_log_image_id(image_id), str(e))

        if bool(getattr(self, "publish_depth_captures", False)):
            # rospy owns Header.seq on published messages. Keep the paired RGB id
            # in this internal depth topic's frame_id so the geometry node can log it.
            depth_msg.header.seq = image_id
            depth_msg.header.stamp = pose_msg.header.stamp
            depth_msg.header.frame_id = "image_%06d" % image_id
            self.depth_capture_pub.publish(depth_msg)
            self.depth_capture_pose_pub.publish(pose_msg)

    def _register_rgb_depth_capture_candidate(self, img_msg, image_id, pose_image_id, tx, ty, tz, qx, qy, qz, qw):
        if not bool(getattr(self, "pair_depth_with_rgb", False)):
            return
        if not self._depth_capture_selected_for_rgb(image_id):
            return
        key = self._message_stamp_key(img_msg)
        if key is None:
            rospy.logwarn_throttle(
                2.0,
                "[WARN] Skipping depth association for RGB image %s because it has no usable timestamp.",
                _blue_log_image_id(image_id),
            )
            return

        capture_stamp = img_msg.header.stamp
        if capture_stamp == rospy.Time():
            capture_stamp = rospy.Time.now()
        pose_msg = PoseStamped()
        pose_msg.header.seq = int(image_id)
        pose_msg.header.stamp = capture_stamp
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose.position.x = float(tx)
        pose_msg.pose.position.y = float(ty)
        pose_msg.pose.position.z = float(tz)
        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)
        record = {
            "image_id": int(image_id),
            "pose_image_id": int(pose_image_id),
            "pose_msg": pose_msg,
        }

        with self._depth_capture_lock:
            self._depth_capture_records[key] = record
            self._prune_depth_capture_cache_locked()

    def depth_rgb_callback(self, img_msg, depth_msg):
        if not bool(getattr(self, "_capture_images_enabled", True)):
            return
        key = self._message_stamp_key(img_msg)
        if key is None:
            return

        record = None
        with self._depth_capture_lock:
            record = self._depth_capture_records.pop(key, None)
        if record is not None:
            self._emit_paired_depth_capture(record, depth_msg)

    def image_callback(self, img_msg):
        """Save an image and its capture pose when odometry is available."""
        if not bool(getattr(self, "_capture_images_enabled", True)):
            return
        self.arrival_cnt += 1
        idx = self.arrival_cnt
        now = rospy.Time.now()

        # Check if frame rate is consistent
        # Every CMD_RATE_HZ arrivals, compute time for this block
        if self.arrival_cnt % self.cmd_rate_hz == 0:
            dt = (now - self.block_start_time).to_sec()
            hz = self.cmd_rate_hz / dt if dt > 0 else 0.0
            rospy.loginfo("Captured %d images in %.3f s (%.2f Hz)\n",
                        self.cmd_rate_hz, dt, hz)
            self.block_start_time = now

        # ---- Save image ----
        try:
            cv_img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            img_filename = os.path.join(self.image_dir, f"image_{idx:06d}.jpg")
            cv2.imwrite(img_filename, cv_img)
            self.img_cnt += 1
        except Exception as e:
            rospy.logwarn("Failed to save image %s: %s", _blue_log_image_id(idx), str(e))
            return
        capture_odom = self.latest_odom
        speed_mps = self._latest_trans_speed_mps(capture_odom)
        skip_pose_due_to_stationary = (
            bool(self.skip_stationary_pose_entries)
            and speed_mps is not None
            and float(speed_mps) <= float(self.stationary_speed_thresh_mps)
        )

        # ---- Save pose if we have odom and this frame is eligible for downstream processing ----
        if capture_odom is not None and (not skip_pose_due_to_stationary):
            # Drone pose in world (AirSim NED) from /odom_local_ned
            p = capture_odom.pose.pose.position
            q = capture_odom.pose.pose.orientation

            # Position of drone in world (NED)
            p_world_body = np.array([p.x, p.y, p.z], dtype=np.float64)

            # Orientation of drone in world as quaternion [x,y,z,w]
            q_world_body = np.array(
                [q.x, q.y, q.z, q.w], dtype=np.float64
            )

            # Rotation matrix world -> body or body -> world?
            # Here we assume Pose orientation rotates from body to world,
            # so R_world_body can be obtained directly from q_world_body.
            R_world_body = quaternion_matrix(q_world_body)[:3, :3]

            # Camera position in world:
            # C_world = P_world_body + R_world_body * cam_offset_body
            cam_offset_body = self.cam_offset_body.astype(np.float64)
            p_world_cam = p_world_body + R_world_body.dot(cam_offset_body)

            # Camera orientation in world:
            # q_world_cam = q_world_body ⊗ q_body_cam (fixed extrinsic) ⊗ q_dyn (gimbal cmd)
            # Camera gimbal we actually command is pitch-only (roll=0,yaw=0) via simSetCameraPose.
            pitch_cmd = float(self.camera_pitch_cmd_current)
            q_dyn = quaternion_from_euler(0.0, pitch_cmd, 0.0, axes="sxyz")
            q_body_cam_total = quaternion_multiply(self.q_body_cam, q_dyn)
            q_world_cam = quaternion_multiply(q_world_body, q_body_cam_total)

            tx, ty, tz = p_world_cam
            qx, qy, qz, qw = q_world_cam
            depth_capture_id = int(idx)

            # Here I use idx+1 because there was a 1 image mismatch between images-pose pairs. Now they are aligned
            try:
                with open(self.pose_filename, "a") as f:
                    f.write(f"{idx+1} {qw:.5f} {qx:.5f} {qy:.5f} {qz:.5f} {tx:.5f} {ty:.5f} {tz:.5f} 1 image_{idx+1:06d}.jpg\n")
            except Exception as e:
                rospy.logwarn("Failed to save pose %s: %s for sparse_images", _blue_log_image_id(idx), str(e))
            try:
                # Convert to roll/pitch/yaw in degrees (matching sxyz order)
                roll_rad, pitch_rad, yaw_rad = tft.euler_from_quaternion([qx, qy, qz, qw], axes="sxyz")
                roll_deg = math.degrees(roll_rad)
                pitch_deg = math.degrees(pitch_rad)
                yaw_deg = math.degrees(yaw_rad)
                with open(self.pose_filename_degrees, "a") as f:
                    f.write(f"{idx+1} {roll_deg:.3f} {pitch_deg:.3f} {yaw_deg:.3f} {tx:.5f} {ty:.5f} {tz:.5f} image_{idx+1:06d}.jpg\n")
            except Exception as e:
                rospy.logwarn("Failed to save pose %s: %s for poses_degrees", _blue_log_image_id(idx), str(e))
            try:
                with open(self.sparse_cameras, "a") as f:
                    f.write(f"{idx} PINHOLE {self.cam_width} {self.cam_height} {self.cam_fx} {self.cam_fy} {self.cam_cx} {self.cam_cy}\n")
            except Exception as e:
                rospy.logwarn("Failed to save pose %s: %s for sparse cameras", _blue_log_image_id(idx), str(e))
            self._register_rgb_depth_capture_candidate(
                img_msg,
                depth_capture_id,
                idx + 1,
                tx,
                ty,
                tz,
                qx,
                qy,
                qz,
                qw,
            )
        elif skip_pose_due_to_stationary:
            rospy.loginfo_throttle(
                2.0,
                "Skipping downstream pose entry for image %s because translational speed is near zero (speed=%.3f m/s <= %.3f m/s).",
                _blue_log_image_id(idx),
                float(speed_mps) if speed_mps is not None else -1.0,
                float(self.stationary_speed_thresh_mps),
            )

        rospy.loginfo_throttle(
            2.0,
            "Saved image %s (pose %s, speed %s)\n",
            _blue_log_image_id(idx),
            (
                "yes"
                if (capture_odom is not None and (not skip_pose_due_to_stationary))
                else ("skipped_stationary" if skip_pose_due_to_stationary else "no")
            ),
            ("%.3f m/s" % float(speed_mps)) if speed_mps is not None else "unknown",
        )

    def _baseline_circle_pose_from_theta(self, theta):
        cx = self.build_center_x
        cy = self.build_center_y
        cz = self.build_center_z
        w = self.build_width
        l = self.build_length
        h = self.build_height

        base_radius = 0.5 * math.sqrt(w * w + l * l)
        radius = 1.0 * base_radius
        x = cx + radius * math.cos(float(theta))
        y = cy + radius * math.sin(float(theta))
        z = 1.5 * h

        target_x = cx
        target_y = cy
        target_z = cz - 1
        dx = target_x - x
        dy = target_y - y
        dz = target_z - z
        yaw = math.atan2(dx, dy)
        dist_xy = math.sqrt(dx * dx + dy * dy)
        pitch = math.atan2(dz, dist_xy)
        return x, y, z, 0.0, pitch, yaw, radius

    def _set_baseline_circle_target(self, theta, speed_mps=0.0):
        x, y, z, roll, pitch, yaw, radius = self._baseline_circle_pose_from_theta(theta)
        self.target_pos = (float(x), float(y), float(z))
        self.target_angles = (float(roll), float(pitch), float(yaw))
        speed = float(speed_mps)
        if abs(speed) > 1.0e-9:
            self.target_velocity = (
                float(-speed * math.sin(float(theta))),
                float(speed * math.cos(float(theta))),
                0.0,
            )
        else:
            self.target_velocity = (0.0, 0.0, 0.0)
        self._baseline_circle_radius_m = float(radius)
        return np.array([float(x), float(y), float(z)], dtype=np.float64), float(radius)

    def _set_baseline_point_target(self, point_xyz):
        p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        self.target_pos = (float(p[0]), float(p[1]), float(p[2]))
        look_angles = self._look_at_building_center_angles_from_world_point(p)
        if look_angles is not None:
            self.target_angles = tuple(float(v) for v in look_angles[:3])
        self.target_velocity = (0.0, 0.0, 0.0)
        return p

    def _baseline_circle_start_theta(self, cur=None):
        if self._baseline_circle_start_angle_rad_effective is not None:
            return float(self._baseline_circle_start_angle_rad_effective)
        theta = float(self.baseline_circle_start_angle_rad)
        if str(self.baseline_circle_start_angle_mode) == "nearest":
            ref = cur
            if ref is None:
                ref = self.home_pos
            try:
                ref = np.asarray(ref, dtype=np.float64).reshape(3)
                dx = float(ref[0] - float(self.build_center_x))
                dy = float(ref[1] - float(self.build_center_y))
                if math.hypot(dx, dy) > 1.0e-6:
                    theta = math.atan2(dy, dx)
            except Exception:
                theta = float(self.baseline_circle_start_angle_rad)
        self._baseline_circle_start_angle_rad_effective = float(theta)
        return float(theta)

    def _set_baseline_circle_phase(self, phase):
        self._baseline_circle_phase = str(phase)
        self._baseline_circle_target_initialized = False
        self._baseline_circle_phase_started_t = time.monotonic()

    def _continuous_baseline_circle_coverage(self):
        cur = self._current_drone_world_point()
        theta0 = self._baseline_circle_start_theta(cur=cur)
        sx, sy, sz, _sroll, _spitch, _syaw, radius = self._baseline_circle_pose_from_theta(theta0)
        start_point = np.array([float(sx), float(sy), float(sz)], dtype=np.float64)
        tol = float(self.baseline_circle_start_tol_m)
        if self._baseline_circle_climb_target is None:
            ref = cur if cur is not None else self.home_pos
            ref = np.asarray(ref, dtype=np.float64).reshape(3)
            self._baseline_circle_climb_target = np.array(
                [float(ref[0]), float(ref[1]), float(start_point[2])],
                dtype=np.float64,
            )
        climb_point = np.asarray(self._baseline_circle_climb_target, dtype=np.float64).reshape(3)

        if str(self._baseline_circle_phase) == "climb":
            self._set_baseline_point_target(climb_point)
            if not bool(self._baseline_circle_target_initialized):
                self.drone_updated = True
                self.camera_updated = True
                self._baseline_circle_target_initialized = True
                rospy.loginfo(
                    "Baseline circle climb target: target=(%.2f, %.2f, %.2f)",
                    float(climb_point[0]),
                    float(climb_point[1]),
                    float(climb_point[2]),
                )
            if cur is None:
                return (*self.target_pos, *self.target_angles)
            climb_dz = abs(float(cur[2] - climb_point[2]))
            climb_elapsed_s = max(
                0.0,
                float(time.monotonic() - float(getattr(self, "_baseline_circle_phase_started_t", time.monotonic()))),
            )
            climb_timed_out = (
                float(self.baseline_circle_climb_timeout_s) > 0.0
                and climb_elapsed_s >= float(self.baseline_circle_climb_timeout_s)
            )
            if climb_dz > tol and not climb_timed_out:
                rospy.loginfo_throttle(
                    2.0,
                    "Baseline circle climb waiting: dz=%.3f tol=%.3f elapsed=%.1f/%.1f",
                    float(climb_dz),
                    float(tol),
                    float(climb_elapsed_s),
                    float(self.baseline_circle_climb_timeout_s),
                )
                return (*self.target_pos, *self.target_angles)

            self._set_baseline_circle_phase("approach")
            self.drone_updated = True
            self.camera_updated = True
            rospy.loginfo(
                "Baseline circle climb complete; approaching circle start point. dz=%.3f timed_fallback=%s",
                float(climb_dz),
                str(bool(climb_timed_out)),
            )

        if str(self._baseline_circle_phase) == "approach":
            self._set_baseline_circle_target(theta0, speed_mps=0.0)
            if not bool(self._baseline_circle_target_initialized):
                self.drone_updated = True
                self.camera_updated = True
                self._baseline_circle_target_initialized = True
                rospy.loginfo(
                    "Baseline circle approach target: mode=%s theta=%.1f deg target=(%.2f, %.2f, %.2f)",
                    str(self.baseline_circle_start_angle_mode),
                    float(math.degrees(theta0)),
                    float(start_point[0]),
                    float(start_point[1]),
                    float(start_point[2]),
                )
            if cur is None:
                return (*self.target_pos, *self.target_angles)
            if float(np.linalg.norm(cur[:2] - start_point[:2])) > tol:
                return (*self.target_pos, *self.target_angles)

            self._set_baseline_circle_phase("circle")
            self._baseline_circle_sweep_started_t = time.monotonic()
            self.vp_cnt = 0
            self.last_drone_update_time = rospy.Time.now()
            rospy.loginfo(
                "Baseline circle sweep started: radius=%.2f speed=%.2f m/s expected_loop_time=%.2f s",
                float(radius),
                float(self.baseline_circle_speed_mps),
                float((2.0 * math.pi * radius) / max(1.0e-6, float(self.baseline_circle_speed_mps))),
            )

        if str(self._baseline_circle_phase) == "land":
            land_point = np.array(
                [float(start_point[0]), float(start_point[1]), float(self.home_pos[2])],
                dtype=np.float64,
            )
            self._set_baseline_point_target(land_point)
            if not bool(self._baseline_circle_target_initialized):
                self.drone_updated = True
                self.camera_updated = True
                self._baseline_circle_target_initialized = True
                rospy.loginfo(
                    "Baseline circle landing target: target=(%.2f, %.2f, %.2f)",
                    float(land_point[0]),
                    float(land_point[1]),
                    float(land_point[2]),
                )
            if cur is not None and float(np.linalg.norm(cur - land_point)) <= tol:
                self._complete_baseline_circle()
            return (*self.target_pos, *self.target_angles)

        if str(self._baseline_circle_phase) == "finish":
            if cur is not None and float(np.linalg.norm(cur[:2] - start_point[:2])) <= tol:
                self._complete_baseline_circle()
            return (*self.target_pos, *self.target_angles)

        started_t = self._baseline_circle_sweep_started_t
        if started_t is None:
            self._baseline_circle_sweep_started_t = time.monotonic()
            started_t = self._baseline_circle_sweep_started_t
        elapsed_s = max(0.0, float(time.monotonic() - float(started_t)))
        speed = float(self.baseline_circle_speed_mps)
        loop_time_s = float((2.0 * math.pi * radius) / max(1.0e-6, speed))

        if bool(self.baseline_circle_stop_after_one_loop) and elapsed_s >= loop_time_s:
            if bool(self.baseline_circle_land_after_one_loop) and not bool(self._mission_completion_logged):
                self._complete_baseline_circle(shutdown=False)
            self._set_baseline_circle_phase(
                "land" if bool(self.baseline_circle_land_after_one_loop) else "finish"
            )
            if bool(self.baseline_circle_land_after_one_loop):
                self._set_baseline_point_target(
                    np.array(
                        [float(start_point[0]), float(start_point[1]), float(self.home_pos[2])],
                        dtype=np.float64,
                    )
                )
            else:
                self._set_baseline_circle_target(theta0, speed_mps=0.0)
            self.drone_updated = True
            self.camera_updated = True
            if (
                not bool(self.baseline_circle_land_after_one_loop)
                and cur is not None
                and float(np.linalg.norm(cur[:2] - start_point[:2])) <= tol
            ):
                self._complete_baseline_circle()
            return (*self.target_pos, *self.target_angles)

        theta = theta0 + (speed * elapsed_s / max(1.0e-6, radius))
        if not bool(self.baseline_circle_stop_after_one_loop):
            theta = theta0 + ((theta - theta0) % (2.0 * math.pi))
        self.vp_cnt = int(math.floor(((theta - theta0) / (2.0 * math.pi)) * float(self.baseline_circle_points)))
        self._set_baseline_circle_target(theta, speed_mps=speed)
        return (*self.target_pos, *self.target_angles)

    def initial_round_coverage(self, nof_points=None):
        """
        Sample a viewpoint around the building and orient it
        so the camera looks towards the building center.
        Coordinates are in the same world frame as the drone's start:
        x forward from start, y sideways, z up.
        """
        if self.pathing_mode == "baseline_circle" and bool(self.baseline_circle_continuous):
            return self._continuous_baseline_circle_coverage()

        if nof_points is None:
            nof_points = int(self.baseline_circle_points)
        nof_points = max(3, int(nof_points))
        now = rospy.Time.now()

        # Pick series of around viewpoints
        theta = [p*2*math.pi/nof_points for p in range(nof_points)]

        if (now - self.last_drone_update_time).to_sec() >= self.viewpoint_period:
            # Close the loop by revisiting the first circle point once after all unique points.
            if bool(self.baseline_circle_stop_after_one_loop) and self.vp_cnt > nof_points:
                self._complete_baseline_circle()
                x, y, z = self.target_pos
                roll, pitch, yaw = self.target_angles
                return x, y, z, roll, pitch, yaw

            # Viewpoint position on a horizontal circle around the building
            x, y, z, roll, pitch, yaw, _radius = self._baseline_circle_pose_from_theta(theta[self.vp_cnt%nof_points])
            self.last_drone_update_time = now

            # Optional small jitter so views are not all identical
            # yaw   += random.uniform(-0.1, 0.1)
            # pitch += random.uniform(-0.1, 0.1)

            self.target_velocity = (0.0, 0.0, 0.0)

            # Increase counter
            self.vp_cnt += 1
            self.drone_updated = True


        # Change the camera continuously
        # elif self.latest_odom is not None:
        #     # Extract current position
        #     p = self.latest_odom.pose.pose.position
        #     x, y, z = p.x, p.x, p.y

        #     # Define the point we want to look at: building center (mid-height)
        #     target_x = cx
        #     target_y = cy
        #     target_z = cz - 1 # I give an offset so that we look a little bit down from the center

        #     # Vector from camera to building center
        #     dx = target_x - x
        #     dy = target_y - y
        #     dz = target_z - z


        #     # Pitch: angle up/down towards target
        #     dist_xy = math.sqrt(dx * dx + dy * dy)
        #     pitch = math.atan2(dz, dist_xy)

        #     # Optional small jitter so views are not all identical
        #     # yaw   += random.uniform(-0.1, 0.1)
        #     # pitch += random.uniform(-0.1, 0.1)

        #     # roll = 0 (no roll)
        #     roll = 0.0

        #     # Increase counter
        #     x, y, z = self.target_pos
        #     _, _, yaw = self.target_angles
        #     # self.camera_updated = True # This is for terminal output

        else:
            x, y, z = self.target_pos
            roll, pitch, yaw = self.target_angles


        self.target_pos = (x, y, z)
        self.target_angles = (roll, pitch, yaw)
    

        # Return position + camera angles
        return x, y, z, roll, pitch, yaw

    def _mesh_to_flight_point(self, point_xyz):
        """
        Convert mesh-frame point to flight/control frame.
        Mesh uses z = -z_flight + center_z_offset.
        """
        p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        x = float(p[0])
        y = float(p[1])
        z = float(-(p[2] - float(self.rt_mesh_center_z_offset)))
        return np.array([x, y, z], dtype=np.float64)

    def _flight_to_mesh_point(self, point_xyz):
        """
        Convert flight/control frame point to mesh frame.
        Inverse of _mesh_to_flight_point.
        """
        p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        x = float(p[0])
        y = float(p[1])
        z = float(-p[2] + float(self.rt_mesh_center_z_offset))
        return np.array([x, y, z], dtype=np.float64)

    def _camera_pose_viz_position_from_latest_odom(self):
        cam_pos_world, _ = self._estimate_camera_pose_from_latest_odom()
        if cam_pos_world is None:
            return None
        p = np.asarray(cam_pos_world, dtype=np.float64).reshape(3).copy()
        if self.viz_flip_viewpoint_z:
            p[2] = -float(p[2])
        return p

    def _best_view_viz_position(self, view):
        if view is None or len(view) < 3:
            return None
        p = np.asarray(view[:3], dtype=np.float64).reshape(3).copy()
        if self.viz_flip_viewpoint_z:
            p[2] = -float(p[2])
        return p

    def follow_atsp_coverage(self):
        now = rospy.Time.now()

        with self._planned_samples_lock:
            if len(self._planned_exec_waypoints) <= 0:
                self._atsp_active_target = False
                self._atsp_active_sample = None
                self._atsp_active_nbv_sample = None
                return
            planned_version = int(self._planned_version)
            total_exec = int(len(self._planned_exec_waypoints))
            need_resnap = (
                int(self._active_plan_version) != planned_version
                or int(self._planned_exec_idx) < 0
                or int(self._planned_exec_idx) >= total_exec
            )
            if need_resnap:
                nearest_exec = self._nearest_exec_waypoint_info(self._planned_exec_waypoints)
                if nearest_exec is None:
                    rospy.logwarn_throttle(
                        2.0,
                        "ATSP follow waiting for live pose before selecting nearest local waypoint on the new trajectory.",
                    )
                    return
                if int(planned_version) <= 1:
                    handoff = {
                        "idx": int(nearest_exec.get("idx", 0)),
                        "nearest_idx": int(nearest_exec.get("idx", 0)),
                        "direction": 1,
                        "reason": "initial_nearest_only",
                        "speed": 0.0,
                        "score": float("nan"),
                        "tangent_score": float("nan"),
                        "top": [],
                    }
                else:
                    handoff = self._velocity_aware_exec_handoff_info(
                        self._planned_exec_waypoints,
                        nearest_exec,
                    )
                if handoff is None:
                    handoff = {
                        "idx": int(nearest_exec.get("idx", 0)),
                        "nearest_idx": int(nearest_exec.get("idx", 0)),
                        "direction": 1,
                        "reason": "handoff_unavailable",
                    }
                nearest_idx = int(handoff.get("idx", nearest_exec.get("idx", 0)))
                nearest_idx = max(0, min(int(nearest_idx), total_exec - 1))
                exec_direction = 1 if int(handoff.get("direction", 1)) >= 0 else -1
                self._planned_exec_idx = int(nearest_idx)
                self._planned_exec_direction = int(exec_direction)
                try:
                    nearest_nbv_idx = int(self._planned_exec_waypoints[nearest_idx].get("nbv_index", 0))
                except Exception:
                    nearest_nbv_idx = 0
                if len(self._planned_samples) > 0:
                    self._planned_idx = max(0, min(nearest_nbv_idx, len(self._planned_samples) - 1))
                if int(self._nearest_exec_diag_logged_version) != planned_version:
                    top_parts = []
                    for top_item in list(nearest_exec.get("top", [])):
                        cand_idx = int(top_item.get("idx", -1))
                        cand_viz_dist = float(top_item.get("viz_dist", float("nan")))
                        cand_mesh_dist = float(top_item.get("mesh_dist", float("nan")))
                        cand_viz_pt = top_item.get("viz_point", None)
                        top_parts.append(
                            "%d:viz=%.3f mesh=%.3f@%s" % (
                                int(cand_idx),
                                float(cand_viz_dist),
                                float(cand_mesh_dist),
                                self._fmt_point_xyz(cand_viz_pt),
                            )
                        )
                    handoff_parts = []
                    for item in list(handoff.get("top", [])):
                        handoff_parts.append(
                            "%d:%s score=%.3f tan=%.3f dist=%.3f step=%d%s" % (
                                int(item.get("idx", -1)),
                                "fwd" if int(item.get("direction", 1)) >= 0 else "rev",
                                float(item.get("score", float("nan"))),
                                float(item.get("tangent_score", float("nan"))),
                                float(item.get("dist", float("nan"))),
                                int(item.get("steps", 0)),
                                "/nbv" if bool(item.get("is_nbv", False)) else "",
                            )
                        )
                    rospy.logwarn(
                        "[WARN] Velocity-aware local waypoint selection: plan_v=%d nearest_idx=%d chosen_idx=%d direction=%s reason=%s speed=%.3f score=%.3f tangent=%.3f current_mesh=%s current_viz=%s nearest_top=%s handoff_top=%s",
                        planned_version,
                        int(handoff.get("nearest_idx", nearest_exec.get("idx", 0))),
                        int(nearest_idx),
                        "forward" if int(exec_direction) >= 0 else "reverse",
                        str(handoff.get("reason", "unknown")),
                        float(handoff.get("speed", float("nan"))),
                        float(handoff.get("score", float("nan"))),
                        float(handoff.get("tangent_score", float("nan"))),
                        self._fmt_point_xyz(nearest_exec.get("current_mesh", None)),
                        self._fmt_point_xyz(nearest_exec.get("current_viz", None)),
                        "[" + ", ".join(top_parts) + "]",
                        "[" + ", ".join(handoff_parts) + "]",
                    )
                    self._nearest_exec_diag_logged_version = int(planned_version)
            idx = max(0, min(int(self._planned_exec_idx), total_exec - 1))
            sample = self._planned_exec_waypoints[idx]

        # Hold current goal until reached (optional), instead of advancing by time.
        if self._atsp_active_target and self.mesh_view_follow_reach_only and self.latest_odom is not None:
            same_trajectory = int(self._active_plan_version) == planned_version
            if same_trajectory:
                mesh_dist = float("inf")
                mesh_dist_xy = float("inf")
                if self._atsp_active_sample is not None:
                    try:
                        active_origin_mesh = np.asarray(
                            self._atsp_active_sample.get("origin", None), dtype=np.float64
                        ).reshape(3)
                        cam_pos_world, _ = self._estimate_camera_pose_from_latest_odom()
                        if cam_pos_world is not None:
                            cam_pos_mesh = self._flight_to_mesh_point(
                                np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
                            )
                            mesh_delta = cam_pos_mesh - active_origin_mesh
                            mesh_dist = float(np.linalg.norm(mesh_delta))
                            mesh_dist_xy = float(np.linalg.norm(mesh_delta[:2]))
                    except Exception:
                        mesh_dist = float("inf")
                        mesh_dist_xy = float("inf")

                tgt = np.array(self.target_pos, dtype=np.float64)
                p = self.latest_odom.pose.pose.position
                body_cur = np.array([p.x, p.y, p.z], dtype=np.float64)
                body_delta = body_cur - tgt
                body_dist = float(np.linalg.norm(body_delta))
                body_dist_xy = float(np.linalg.norm(body_delta[:2]))
                cam_pos_world, _ = self._estimate_camera_pose_from_latest_odom()
                if cam_pos_world is not None:
                    cam_cur = np.asarray(cam_pos_world, dtype=np.float64).reshape(3)
                    cam_delta = cam_cur - tgt
                    cam_dist = float(np.linalg.norm(cam_delta))
                    cam_dist_xy = float(np.linalg.norm(cam_delta[:2]))
                else:
                    cam_dist = float("inf")
                    cam_dist_xy = float("inf")
                active_origin_mesh = None
                active_target_mesh = None
                if self._atsp_active_sample is not None:
                    try:
                        active_origin_mesh = np.asarray(
                            self._atsp_active_sample.get("origin", None), dtype=np.float64
                        ).reshape(3)
                        active_target_mesh = np.asarray(
                            self._atsp_active_sample.get("target", None), dtype=np.float64
                        ).reshape(3)
                    except Exception:
                        active_origin_mesh = None
                        active_target_mesh = None
                if active_origin_mesh is not None and active_target_mesh is not None:
                    active_viz_pose = self._look_at_pose_viz(
                        origin=active_origin_mesh,
                        target=active_target_mesh,
                    )
                else:
                    active_viz_pose = None
                cur_viz_pos = self._camera_pose_viz_position_from_latest_odom()
                tgt_viz_pos = self._best_view_viz_position(active_viz_pose)
                if cur_viz_pos is not None and tgt_viz_pos is not None:
                    viz_dist = float(np.linalg.norm(cur_viz_pos - tgt_viz_pos))
                else:
                    viz_dist = float("inf")

                # Use Euclidean distance in the same position convention RViz uses for
                # the green current-view and red best-view frustums.
                if np.isfinite(viz_dist):
                    dist_to_target = viz_dist
                else:
                    dist_to_target = cam_dist if np.isfinite(cam_dist) else body_dist
                    rospy.logwarn_throttle(
                        1.0,
                        "ATSP reach gating fallback: viz_dist unavailable, using %s instead.",
                        "cam_dist" if np.isfinite(cam_dist) else "body_dist",
                    )
                tol = float(max(1e-3, self.mesh_view_reach_pos_tol_m))
                if dist_to_target > tol:
                    if self.mesh_view_debug_logs:
                        rospy.loginfo_throttle(
                            1.0,
                            "ATSP hold: plan_v=%d active_v=%d idx=%d/%d dist=%.3f tol=%.3f viz=%.3f body=%.3f cam=%.3f mesh=%.3f body_xy=%.3f cam_xy=%.3f mesh_xy=%.3f",
                            planned_version,
                            int(self._active_plan_version),
                            idx + 1,
                            total_exec,
                            dist_to_target,
                            tol,
                            viz_dist if np.isfinite(viz_dist) else -1.0,
                            body_dist,
                            cam_dist if np.isfinite(cam_dist) else -1.0,
                            mesh_dist if np.isfinite(mesh_dist) else -1.0,
                            body_dist_xy,
                            cam_dist_xy if np.isfinite(cam_dist_xy) else -1.0,
                            mesh_dist_xy if np.isfinite(mesh_dist_xy) else -1.0,
                        )
                    return
        elif self._atsp_active_target and (not self.mesh_view_follow_reach_only):
            if (now - self.last_drone_update_time).to_sec() < self.viewpoint_period:
                return

        pose = sample.get("pose", None)
        if pose is None or len(pose) < 6:
            return

        # Use ATSP sample geometry but compute command angles with the same logic as initial_round_coverage.
        origin_mesh = sample.get("origin", None)
        target_mesh = sample.get("target", None)
        if origin_mesh is None or target_mesh is None:
            # Fallback: keep old behavior if sample payload is incomplete.
            x, y, z, roll, pitch, yaw = [float(v) for v in pose[:6]]
            viz_pose = (x, y, z, roll, pitch, yaw)
        else:
            origin_flight = self._mesh_to_flight_point(origin_mesh)
            target_flight = self._mesh_to_flight_point(target_mesh)

            x = float(origin_flight[0])
            y = float(origin_flight[1])
            z = float(origin_flight[2])
            dx = float(target_flight[0] - origin_flight[0])
            dy = float(target_flight[1] - origin_flight[1])
            dz = float(target_flight[2] - origin_flight[2])
            yaw = math.atan2(dx, dy)
            dist_xy = math.sqrt(dx * dx + dy * dy)
            pitch = math.atan2(dz, dist_xy)
            roll = 0.0
            viz_origin = np.asarray(origin_mesh, dtype=np.float64).reshape(3)
            viz_target = np.asarray(target_mesh, dtype=np.float64).reshape(3)
            viz_computed = self._look_at_pose_viz(origin=viz_origin, target=viz_target)
            if viz_computed is not None:
                viz_pose = viz_computed
            elif pose is not None and len(pose) >= 6:
                viz_pose = tuple(float(v) for v in pose[:6])
            else:
                viz_pose = (x, y, z, roll, pitch, yaw)

        self.target_pos = (x, y, z)
        self.target_angles = (roll, pitch, yaw)
        self.last_drone_update_time = now
        self.drone_updated = True
        self.camera_updated = True
        if (
            self.mesh_view_start_leg_debug_enabled
            and int(self._active_plan_version) != planned_version
            and int(self._start_leg_diag_logged_version) != planned_version
        ):
            origin_mesh_dbg = None
            try:
                origin_mesh_dbg = np.asarray(sample.get("origin", None), dtype=np.float64).reshape(3)
            except Exception:
                origin_mesh_dbg = None
            rospy.logwarn(
                "New-plan first command: plan_v=%d exec_idx=%d/%d edge_kind=%s cmd_mesh=%s cmd_viz=%s cmd_flight=(%.2f, %.2f, %.2f) nbv_index=%d",
                planned_version,
                idx + 1,
                total_exec,
                str(sample.get("edge_kind", "unknown")),
                self._fmt_point_xyz(origin_mesh_dbg),
                self._fmt_point_xyz(self._mesh_to_traj_viz_point(origin_mesh_dbg)) if origin_mesh_dbg is not None else "(nan, nan, nan)",
                float(x),
                float(y),
                float(z),
                int(sample.get("nbv_index", 0)),
            )
            self._start_leg_diag_logged_version = int(planned_version)
        self._atsp_active_target = True
        self._active_plan_version = planned_version
        self._atsp_active_sample = copy.deepcopy(sample)
        nbv_index = int(sample.get("nbv_index", 0))
        with self._planned_samples_lock:
            if len(self._planned_samples) > 0:
                nbv_index = max(0, min(int(nbv_index), len(self._planned_samples) - 1))
                self._atsp_active_nbv_sample = copy.deepcopy(self._planned_samples[nbv_index])
            else:
                self._atsp_active_nbv_sample = copy.deepcopy(sample)
        with self._planned_samples_lock:
            if len(self._planned_exec_waypoints) > 0:
                exec_direction = 1 if int(getattr(self, "_planned_exec_direction", 1)) >= 0 else -1
                self._planned_exec_idx = (int(idx) + int(exec_direction)) % int(len(self._planned_exec_waypoints))
            self._planned_idx = int(nbv_index)
        # Publish best viewpoint in mesh frame for RViz consistency with sampled vectors/path.
        self.publish_best_viewpoint(viz_pose)
        if self.mesh_view_debug_logs:
            rospy.loginfo(
                "ATSP path target: plan_v=%d exec_idx=%d/%d nbv_idx=%d/%d x=%.2f y=%.2f z=%.2f pitch(deg)=%.2f yaw(deg)=%.2f kind=%s",
                planned_version,
                idx + 1,
                total_exec,
                nbv_index + 1,
                max(1, len(self._planned_samples)),
                x,
                y,
                z,
                math.degrees(pitch),
                math.degrees(yaw),
                str(sample.get("edge_kind", "unknown")),
            )

    def sample_viewpoints(self, num_samples=5):
        """
        Sample random viewpoints around the building, looking at its center.

        Returns:
            list of tuples (x, y, z, roll, pitch, yaw)
            in the *world* frame used by your controller.
        """
        print("Sampling Viewpoints")
        cx = self.build_center_x
        cy = self.build_center_y
        cz = self.build_center_z - 1.0  # slight downward look
        w  = self.build_width
        l  = self.build_length
        h  = self.build_height

        # Base radius around building (corner distance) and some margin
        base_radius = 0.5 * math.sqrt(w * w + l * l)
        r_min = 1.1 * base_radius
        r_max = 1.5 * base_radius

        z_min = 0.3 * h
        z_max = 1.2 * h

        viewpoints = []
        for _ in range(num_samples):
            theta = random.uniform(0.0, 2.0 * math.pi)
            radius = random.uniform(r_min, r_max)
            z = random.uniform(z_min, z_max)

            x = cx + radius * math.cos(theta)
            y = cy + radius * math.sin(theta)

            # Target point to look at
            target_x = cx
            target_y = cy
            target_z = cz

            dx = target_x - x
            dy = target_y - y
            dz = target_z - z

            yaw = math.atan2(dy, dx)
            yaw_weird = math.atan2(dx, dy)
            dist_xy = math.sqrt(dx * dx + dy * dy)
            pitch = math.atan2(dz, dist_xy)
            roll = 0.0

            viewpoints.append((x, y, z, roll, pitch, yaw, yaw_weird))

        return viewpoints

    @staticmethod
    def _allocate_point_counts(total_points, areas):
        if total_points <= 0 or not areas:
            return [0 for _ in areas]
        total_area = sum(areas)
        if total_area <= 0.0:
            return [0 for _ in areas]
        targets = [(total_points * area) / total_area for area in areas]
        counts = [int(math.floor(value)) for value in targets]
        remainder = total_points - sum(counts)
        if remainder > 0:
            fractions = [value - math.floor(value) for value in targets]
            order = sorted(range(len(fractions)), key=lambda i: fractions[i], reverse=True)
            for idx in order[:remainder]:
                counts[idx] += 1
        return counts

    @staticmethod
    def _grid_counts(num_points, dim_u, dim_v):
        if num_points <= 0:
            return 0, 0
        if dim_u <= 0.0 or dim_v <= 0.0:
            return num_points, 1
        aspect = dim_u / dim_v
        n_u = max(1, int(round(math.sqrt(num_points * aspect))))
        n_v = max(1, int(math.ceil(num_points / n_u)))
        return n_u, n_v

    @staticmethod
    def _grid_axis_values(dim, count):
        if count <= 1:
            return [0.0]
        return np.linspace(-0.5 * dim, 0.5 * dim, count).tolist()

    def sample_side_points(self, total_points, margin):
        """
        Sample points on the 5 faces of the building bounding box (no bottom),
        offset outward by the given margin. Returns list of (x, y, z).
        """
        total_points = int(total_points)
        if total_points <= 0:
            return []
        margin = float(margin)
        if margin < 0.0:
            rospy.logwarn("side_points margin < 0; clamping to 0.")
            margin = 0.0

        cx = self.build_center_x
        cy = self.build_center_y
        cz = self.build_center_z
        w = self.build_width
        l = self.build_length
        h = self.build_height

        half_w = 0.5 * w
        half_l = 0.5 * l
        half_h = 0.5 * h

        sides = [
            {"axis": "x", "sign": 1.0, "u_dim": l, "v_dim": h, "area": l * h},
            {"axis": "x", "sign": -1.0, "u_dim": l, "v_dim": h, "area": l * h},
            {"axis": "y", "sign": 1.0, "u_dim": w, "v_dim": h, "area": w * h},
            {"axis": "y", "sign": -1.0, "u_dim": w, "v_dim": h, "area": w * h},
            {"axis": "z", "sign": 1.0, "u_dim": w, "v_dim": l, "area": w * l},
        ]

        allocations = self._allocate_point_counts(total_points, [side["area"] for side in sides])
        points = []

        for side, count in zip(sides, allocations):
            if count <= 0 or side["area"] <= 0.0:
                continue

            n_u, n_v = self._grid_counts(count, side["u_dim"], side["v_dim"])
            u_vals = self._grid_axis_values(side["u_dim"], n_u)
            v_vals = self._grid_axis_values(side["v_dim"], n_v)

            grid = [(u, v) for v in v_vals for u in u_vals]
            if len(grid) > count:
                idxs = np.linspace(0, len(grid) - 1, count, dtype=int)
                grid = [grid[i] for i in idxs]

            for u, v in grid:
                if side["axis"] == "x":
                    x = cx + side["sign"] * (half_w + margin)
                    y = cy + u
                    z = cz + v
                elif side["axis"] == "y":
                    x = cx + u
                    y = cy + side["sign"] * (half_l + margin)
                    z = cz + v
                else:
                    x = cx + u
                    y = cy + v
                    z = cz + side["sign"] * (half_h + margin)
                points.append((x, y, z))

        return points

    def publish_side_points(self, points):
        if not points:
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        pose_array = PoseArray()
        pose_array.header = header

        for x, y, z in points:
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.w = 1.0
            pose_array.poses.append(pose)

        self.side_points_pub.publish(pose_array)

    def publish_viewpoints(self, viewpoints):
        print("Publishing viewpoints...")
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        pose_array = PoseArray()
        pose_array.header = header

        for x, y, z, roll, pitch, yaw, _ in viewpoints:
            qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw, axes="sxyz")
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            pose.orientation.x = qx
            pose.orientation.y = qy
            pose.orientation.z = qz
            pose.orientation.w = qw
            pose_array.poses.append(pose)

        self.viewpoints_pub.publish(pose_array)

    def publish_best_viewpoint(self, view):
        print("Publishing best viewpoints...")
        if view is None:
            return

        x, y, z, roll, pitch, yaw = view
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw, axes="sxyz")

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.best_view_pub.publish(msg)
    
    def viewpoint_evaluation(self, viewpoints):
        """
        Evaluate viewpoints using gaussians:
        - prefer high opacity
        - prefer high scale (variance)
        - prefer fewer gaussians inside the view (smaller count)

        We model the view as a cone around the direction from
        viewpoint to building center.
        """
        print("Evaluating viewpoints")
        if (
            self.gauss_pos is None
            or self.gauss_opacity is None
            or self.gauss_scale is None
        ):
            return None, None, None

        gauss_pos = self.gauss_pos      # (N,3)
        gauss_op = self.gauss_opacity   # (N,)
        gauss_sc = self.gauss_scale     # (N,3)

        # Precompute Gaussian "size" = mean scale across axes
        gauss_size = gauss_sc.mean(axis=1)  # (N,)

        # FOV cone parameters
        fov_half_angle_deg = 35.0
        cos_fov = math.cos(math.radians(fov_half_angle_deg))
        max_range = max(self.build_width, self.build_length) * 3.0

        best_idx = None
        best_score = -1e9
        best_view = None

        center = np.array(
            [self.build_center_x, self.build_center_y, self.build_center_z],
            dtype=np.float32,
        )

        for i, (x, y, z, roll, pitch, yaw, yaw_weird) in enumerate(viewpoints):
            vp = np.array([x, y, z], dtype=np.float32)

            # Forward direction: towards building center
            fwd = center - vp
            dist_fwd = np.linalg.norm(fwd)
            if dist_fwd < 1e-6:
                continue
            fwd /= dist_fwd

            # Vector from viewpoint to each gaussian
            vec = gauss_pos - vp[None, :]  # (N,3)
            dist = np.linalg.norm(vec, axis=1)  # (N,)

            # Avoid division by zero
            valid_dist = dist > 1e-6
            if not np.any(valid_dist):
                continue

            dir_norm = np.zeros_like(vec)
            dir_norm[valid_dist] = vec[valid_dist] / dist[valid_dist, None]

            # Cosine of angle with forward direction
            cosang = np.dot(dir_norm, fwd)  # (N,)

            # In front of camera, inside FOV and within max_range
            in_fov = (cosang >= cos_fov) & (dist <= max_range)

            idxs = np.where(in_fov)[0]
            N = idxs.shape[0]

            if N == 0:
                # No gaussians visible from here: low score
                score = -1.0
            else:
                mean_op = gauss_op[idxs].mean()
                mean_sz = gauss_size[idxs].mean()
                # Penalize many gaussians: denominator grows with count
                # (tune 0.1 factor as you like)
                score = (mean_op + mean_sz) / (1.0 + 0.1 * float(N))

            if score > best_score:
                best_score = score
                best_idx = i
                best_view = (x, y, z, roll, pitch, yaw)
                best_weird_yaw = yaw_weird

        return best_idx, best_view, best_weird_yaw, best_score

    def update_viewpoint_from_gaussians(self):
        """
        Periodically sample viewpoints and pick the best one using gaussian scores.
        Falls back to the first sampled viewpoint if evaluation is unavailable.
        """
        now = rospy.Time.now()

        if (now - self.last_gaussian_update_time).to_sec() < self.viewpoint_period:
            return

        # Refresh gaussian cache if available
        print("Loading gaussians txt")
        self.load_gaussians_from_txt(self.gaussians_filename)

        viewpoints = self.sample_viewpoints()
        _, best_view, best_weird_yaw, best_score = self.viewpoint_evaluation(viewpoints)

        # Publish sampled viewpoints regardless of scoring outcome
        if viewpoints:
            self.publish_viewpoints(viewpoints)

        if best_view is None and viewpoints:
            # If no gaussian data yet, pick the first sampled viewpoint
            best_view = viewpoints[0]
            rospy.loginfo("No gaussian-based score available; using first sampled viewpoint.")

        if best_view is None:
            return

        self.publish_best_viewpoint(best_view)

        x, y, z, roll, pitch, _ = best_view
        yaw = best_weird_yaw
        self.target_pos = (x, y, z)
        self.target_angles = (roll, pitch, yaw)
        self.last_drone_update_time = now
        self.last_gaussian_update_time = now
        self.drone_updated = True
        self.camera_updated = True

        pitch_deg = math.degrees(pitch)
        yaw_deg = math.degrees(yaw)
        rospy.loginfo(
            "Selected viewpoint with score %s: x=%.2f y=%.2f z=%.2f pitch(deg)=%.2f yaw(deg)=%.2f",
            "N/A" if best_score is None else f"{best_score:.4f}",
            x,
            y,
            z,
            pitch_deg,
            yaw_deg,
        )

    @staticmethod
    def _wrap_pi(angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _get_current_yaw(self):
        """Current drone yaw from latest odom (sxyz, radians)."""
        if self.latest_odom is None:
            return None
        q = self.latest_odom.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w], axes="sxyz")
        return yaw

    def publish_pos_yaw_cmd(self, x=None, y=None, z=None, yaw=None):
        """Publish PositionCommand based on current target (PredRecon / UAV_Simulator style)."""
        troll, tpitch, tyaw = self.target_angles  # radians
        tx, ty, tz = self.target_pos

        if x is None:
            x = tx
        if y is None:
            y = ty
        if z is None:
            z = tz
        if yaw is None:
            yaw = tyaw
        tvx, tvy, tvz = getattr(self, "target_velocity", (0.0, 0.0, 0.0))

        # Convert internal planner angles to the simulator/body-command convention
        # only at the drone-command boundary.
        roll, pitch, yaw = self._drone_command_angles(roll=troll, pitch=tpitch, yaw=yaw)
        
        cmd = PositionCommand()
        cmd.header.stamp = rospy.Time.now()
        cmd.header.frame_id = self.frame_id   # consistent with typical usage

        cmd.position.x = x
        cmd.position.y = y
        cmd.position.z = z

        cmd.velocity.x = float(tvx)
        cmd.velocity.y = float(tvy)
        cmd.velocity.z = float(tvz)

        cmd.acceleration.x = 0.0
        cmd.acceleration.y = 0.0
        cmd.acceleration.z = 0.0

        cmd.jerk.x = 0.0
        cmd.jerk.y = 0.0
        cmd.jerk.z = 0.0
        
        cmd.yaw = yaw
        cmd.yaw_dot = 0.0

        # Gains and trajectory metadata (often ignored or set inside controller)
        cmd.kx = [0.0, 0.0, 0.0]
        cmd.kv = [0.0, 0.0, 0.0]
        cmd.trajectory_id = 0
        cmd.trajectory_flag = 0

        # Log only when the update happens
        # Again in world system frame
        if self.drone_updated:
            if not bool(getattr(self, "_mission_completion_active", False)):
                rospy.loginfo(f"New Drone Position: x={cmd.position.x}, y={cmd.position.y}, z={cmd.position.z}. New Drone Yaw: yaw={yaw}\n")
            self.drone_updated = False
            
        
        self.pos_cmd_pub.publish(cmd)

    def set_camera_angles(self, roll=None, pitch=None, yaw=None):
        """Set camera orientation directly via AirSim RPC (simSetCameraPose)."""
        troll, tpitch, tyaw = self.target_angles  # radians
        if roll is None:
            roll = troll
        if pitch is None:
            pitch = tpitch
        if yaw is None:
            yaw = tyaw

        # Do not call drone2global_angles() here: this is a camera/gimbal-local AirSim command,
        # not a body-yaw command to the drone controller.
        
        current_info = self.airsim_client.simGetCameraInfo("0")
        current_pose = current_info.pose
        
        # AirSim's to_quaternion takes (pitch, roll, yaw) in radians
        q = airsim.to_quaternion(pitch, 0.0, 0.0)

        # Camera pose relative to vehicle body: only orientation, keep position from settings.json.
        pose = airsim.Pose(
            airsim.Vector3r(self.cam_offset_body[0], self.cam_offset_body[1], self.cam_offset_body[2]),
            q
        )

        try:
            self.airsim_client.simSetCameraPose(
                "front_center",   # camera_name from settings.json
                pose,
                vehicle_name="drone_1"
            )
            # Log only when the update happens
            if self.camera_updated:
                if not bool(getattr(self, "_mission_completion_active", False)):
                    rospy.loginfo(f"New Camera Rotation: roll=0.0, pitch={pitch}, yaw=0.0 \n")
                self.camera_updated = False
        except Exception as e:
            rospy.logwarn("simSetCameraPose failed: %s", str(e))

        if not bool(getattr(self, "_mission_completion_active", False)):
            rospy.loginfo_throttle(
                0.5,
                "Cam roll=%.2f pitch=%.2f yaw=%.2f\n",
                roll, pitch, yaw,
            )

    def publish_camera_pose(self):
        if self.latest_odom is None:
            return
        # Drone pose in world (AirSim NED) from /odom_local_ned
        p = self.latest_odom.pose.pose.position
        q = self.latest_odom.pose.pose.orientation
        p_world_body = np.array([p.x, p.y, p.z], dtype=np.float64)
        q_world_body = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)

        R_world_body = quaternion_matrix(q_world_body)[:3, :3]
        cam_offset_body = self.cam_offset_body.astype(np.float64)
        p_world_cam = p_world_body + R_world_body.dot(cam_offset_body)

        pitch_cmd = float(self.camera_pitch_cmd_current)
        q_dyn = quaternion_from_euler(0.0, pitch_cmd, 0.0, axes="sxyz")
        q_body_cam_total = quaternion_multiply(self.q_body_cam, q_dyn)
        q_world_cam = quaternion_multiply(q_world_body, q_body_cam_total)

        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = float(p_world_cam[0])
        msg.pose.position.y = float(p_world_cam[1])
        msg.pose.position.z = float(p_world_cam[2])
        msg.pose.orientation.x = float(q_world_cam[0])
        msg.pose.orientation.y = float(q_world_cam[1])
        msg.pose.orientation.z = float(q_world_cam[2])
        msg.pose.orientation.w = float(q_world_cam[3])
        self.camera_pose_pub.publish(msg)

    def spin(self):
        rate = rospy.Rate(self.cmd_rate_hz)
        while not rospy.is_shutdown():
            if bool(self._mission_completion_active):
                self._update_mission_completion()
            elif self.pathing_mode == "atsp" and self.use_mesh:
                self.follow_atsp_coverage()
            else:
                # Polygon run in a circle around the building
                self.initial_round_coverage()

            # Update target every VIEWPOINT_PERIOD seconds
            # self.update_viewpoint_from_gaussians()

            # Continuously publish PositionCommand at CMD_RATE_HZ. These are global coordinates
            # For some insane reason. It seems that 0.0275 is pi/2
            self.publish_pos_yaw_cmd()

            # Continuously publish Camera Angles at CMD_RATE_HZ
            pitch_cmd = self._step_camera_pitch_command()
            self.set_camera_angles(pitch=pitch_cmd)
            self.publish_camera_pose()

            rate.sleep()


if __name__ == "__main__":
    node = ControllerNode()
    node.spin()
