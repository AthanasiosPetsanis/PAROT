#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Point, PoseStamped, PoseArray
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, euler_matrix, quaternion_from_euler
from visualization_msgs.msg import Marker, MarkerArray


class ViewpointVisualizationNode:
    def __init__(self):
        rospy.init_node("viewpoint_visualization")

        # Parameters
        self.frame_id = rospy.get_param("~frame_id", "odom_local_ned")
        self.root_frame = rospy.get_param("~root_frame", "world")
        # If the incoming viewpoints are authored in a z-up convention while odom/gaussians use z-down (AirSim NED),
        # flip them for visualization only.
        self.flip_viewpoint_z = rospy.get_param("~viz_flip_viewpoint_z", True)
        self.build_center_x = rospy.get_param("~building/center_x", 0.0)
        self.build_center_y = rospy.get_param("~building/center_y", 0.0)
        self.build_center_z = rospy.get_param("~building/center_z", 0.0)
        self.build_width = rospy.get_param("~building/width", 20.0)
        self.build_length = rospy.get_param("~building/length", 30.0)
        self.build_height = rospy.get_param("~building/height", 10.0)
        self.use_intrinsics_frustum = rospy.get_param("~viewpoints/use_camera_intrinsics", True)
        self.cam_fx = float(rospy.get_param("~calibration/fx", 381.361145))
        self.cam_fy = float(rospy.get_param("~calibration/fy", 381.361145))
        self.cam_cx = float(rospy.get_param("~calibration/cx", 320.0))
        self.cam_cy = float(rospy.get_param("~calibration/cy", 240.0))
        self.cam_width = int(rospy.get_param("~calibration/width", 640))
        self.cam_height = int(rospy.get_param("~calibration/height", 480))
        self.fov_deg = rospy.get_param("~viewpoints/fov_deg", 70.0)
        self.near_clip = rospy.get_param("~viewpoints/near_clip", 0.5)
        self.far_clip = rospy.get_param("~viewpoints/far_clip", 5.0)

        self.current_odom: Optional[Odometry] = None
        self.current_camera_pose: Optional[PoseStamped] = None
        self.last_best_view: Optional[Tuple[float, float, float, float, float, float]] = None
        self.latest_viewpoints: List[Tuple[float, float, float, float, float, float]] = []
        self.viewpoints_frame_id: str = self.frame_id
        self.best_view_frame_id: str = self.frame_id

        # Static TF to ensure frames exist in RViz
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        if self.root_frame != self.frame_id:
            self.publish_identity_tf(self.root_frame, self.frame_id)

        # Publishers
        self.cloud_pub = rospy.Publisher(
            "viewpoint_viz/gaussian_cloud", PointCloud2, queue_size=1, latch=True
        )
        self.mesh_points_pub = rospy.Publisher(
            "viewpoint_viz/mesh_points", PointCloud2, queue_size=1, latch=True
        )
        self.sfm_points_pub = rospy.Publisher(
            "viewpoint_viz/sfm_sparse_points", PointCloud2, queue_size=1, latch=True
        )
        self.depth_raw_points_pub = rospy.Publisher(
            "viewpoint_viz/depth_raw_points", PointCloud2, queue_size=1, latch=True
        )
        self.depth_mesh_patch_pub = rospy.Publisher(
            "viewpoint_viz/depth_mesh_patch", Marker, queue_size=1, latch=True
        )
        self.depth_mesh_accumulated_patch_pub = rospy.Publisher(
            "viewpoint_viz/depth_mesh_accumulated_patch", Marker, queue_size=1, latch=True
        )
        self.depth_mesh_patch_boundary_pub = rospy.Publisher(
            "viewpoint_viz/depth_mesh_patch_boundary", Marker, queue_size=1, latch=True
        )
        self.depth_mesh_patch_intersections_pub = rospy.Publisher(
            "viewpoint_viz/depth_mesh_patch_intersections", Marker, queue_size=1, latch=True
        )
        self.depth_mesh_patch_cut_loop_pub = rospy.Publisher(
            "viewpoint_viz/depth_mesh_patch_cut_loop", Marker, queue_size=1, latch=True
        )
        self.depth_mesh_patch_failed_rays_pub = rospy.Publisher(
            "viewpoint_viz/depth_mesh_patch_failed_rays", Marker, queue_size=1, latch=True
        )
        self.mesh_tri_pub = rospy.Publisher(
            "viewpoint_viz/mesh_triangles", Marker, queue_size=1, latch=True
        )
        self.sampled_view_vectors_pub = rospy.Publisher(
            "viewpoint_viz/sampled_view_vectors", Marker, queue_size=1, latch=True
        )
        self.trajectory_pub = rospy.Publisher(
            "viewpoint_viz/trajectory", Marker, queue_size=1, latch=True
        )
        self.skipped_trajectory_pub = rospy.Publisher(
            "viewpoint_viz/trajectory_skipped", Marker, queue_size=1, latch=True
        )
        self.viewpoint_positions_pub = rospy.Publisher(
            "viewpoint_viz/viewpoint_positions", Marker, queue_size=1, latch=True
        )
        self.global_viewpoint_positions_pub = rospy.Publisher(
            "viewpoint_viz/global_viewpoint_positions", Marker, queue_size=1, latch=True
        )
        self.current_pose_point_pub = rospy.Publisher(
            "viewpoint_viz/current_pose_point", Marker, queue_size=1, latch=True
        )
        self.actual_flight_path_pub = rospy.Publisher(
            "viewpoint_viz/actual_flight_path", Marker, queue_size=1, latch=True
        )
        self.start_leg_checked_points_pub = rospy.Publisher(
            "viewpoint_viz/start_leg_checked_points", Marker, queue_size=1, latch=True
        )
        self.region_boundaries_pub = rospy.Publisher(
            "viewpoint_viz/region_boundaries", Marker, queue_size=1, latch=True
        )
        self.viewpoint_markers_pub = rospy.Publisher(
            "viewpoint_viz/viewpoints", MarkerArray, queue_size=1
        )
        self.side_points_markers_pub = rospy.Publisher(
            "viewpoint_viz/side_points", MarkerArray, queue_size=1, latch=True
        )
        self.path_pub = rospy.Publisher(
            "viewpoint_viz/selected_path", Path, queue_size=1
        )

        # Subscribers
        rospy.Subscriber(
            "/airsim_node/drone_1/odom_local_ned",
            Odometry,
            self.odom_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/viewpoints",
            PoseArray,
            self.viewpoints_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/best_viewpoint",
            PoseStamped,
            self.best_view_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/camera_pose",
            PoseStamped,
            self.camera_pose_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/gaussian_centers",
            PointCloud2,
            self.gaussian_cloud_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/mesh_points",
            PointCloud2,
            self.mesh_points_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/sfm_sparse_points",
            PointCloud2,
            self.sfm_points_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            rospy.get_param("~rt_meshing/depth_raw_points_topic", "controller/depth_raw_points"),
            PointCloud2,
            self.depth_raw_points_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~rt_meshing/depth_mesh_patch_topic", "controller/depth_mesh_patch"),
            Marker,
            self.depth_mesh_patch_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param(
                "~rt_meshing/depth_mesh_accumulated_patch_topic",
                "controller/depth_mesh_accumulated_patch",
            ),
            Marker,
            self.depth_mesh_accumulated_patch_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~rt_meshing/depth_mesh_patch_boundary_topic", "controller/depth_mesh_patch_boundary"),
            Marker,
            self.depth_mesh_patch_boundary_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~rt_meshing/depth_mesh_patch_intersections_topic", "controller/depth_mesh_patch_intersections"),
            Marker,
            self.depth_mesh_patch_intersections_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~rt_meshing/depth_mesh_patch_cut_loop_topic", "controller/depth_mesh_patch_cut_loop"),
            Marker,
            self.depth_mesh_patch_cut_loop_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            rospy.get_param("~rt_meshing/depth_mesh_patch_failed_rays_topic", "controller/depth_mesh_patch_failed_rays"),
            Marker,
            self.depth_mesh_patch_failed_rays_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            "controller/mesh_triangles",
            Marker,
            self.mesh_triangles_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/sampled_view_vectors",
            Marker,
            self.sampled_view_vectors_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/trajectory",
            Marker,
            self.trajectory_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/trajectory_skipped",
            Marker,
            self.skipped_trajectory_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/viewpoint_positions",
            Marker,
            self.viewpoint_positions_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/global_viewpoint_positions",
            Marker,
            self.global_viewpoint_positions_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/current_pose_point",
            Marker,
            self.current_pose_point_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/actual_flight_path",
            Marker,
            self.actual_flight_path_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/start_leg_checked_points",
            Marker,
            self.start_leg_checked_points_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/region_boundaries",
            Marker,
            self.region_boundaries_callback,
            queue_size=5,
        )
        rospy.Subscriber(
            "controller/side_points",
            PoseArray,
            self.side_points_callback,
            queue_size=5,
        )

        rospy.loginfo("viewpoint_visualization node ready and waiting for controller data.")

    def odom_callback(self, msg: Odometry):
        self.current_odom = msg

        self.publish_path(self.last_best_view)
        self.publish_drone_frustum()

    def camera_pose_callback(self, msg: PoseStamped):
        self.current_camera_pose = msg
        self.publish_drone_frustum()

    def gaussian_cloud_callback(self, msg: PointCloud2):
        # Keep visualization frame consistent
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            rospy.loginfo("Switching visualization frame to incoming gaussian frame: %s", msg.header.frame_id)
            self.frame_id = msg.header.frame_id
            self.viewpoints_frame_id = self.frame_id
            self.best_view_frame_id = self.frame_id
            self.publish_identity_tf(self.root_frame, self.frame_id)
        msg.header.frame_id = self.frame_id
        self.cloud_pub.publish(self._mirror_pointcloud_x(msg))

    def mesh_points_callback(self, msg: PointCloud2):
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            rospy.loginfo("Switching visualization frame to incoming mesh frame: %s", msg.header.frame_id)
            self.frame_id = msg.header.frame_id
            self.viewpoints_frame_id = self.frame_id
            self.best_view_frame_id = self.frame_id
            self.publish_identity_tf(self.root_frame, self.frame_id)
        msg.header.frame_id = self.frame_id
        self.mesh_points_pub.publish(self._mirror_pointcloud_x(msg))

    def sfm_points_callback(self, msg: PointCloud2):
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            rospy.loginfo("Switching visualization frame to incoming sfm frame: %s", msg.header.frame_id)
            self.frame_id = msg.header.frame_id
            self.viewpoints_frame_id = self.frame_id
            self.best_view_frame_id = self.frame_id
            self.publish_identity_tf(self.root_frame, self.frame_id)
        msg.header.frame_id = self.frame_id
        self.sfm_points_pub.publish(self._mirror_pointcloud_x(msg))

    def depth_raw_points_callback(self, msg: PointCloud2):
        if msg.header.frame_id and msg.header.frame_id != self.frame_id:
            rospy.loginfo("Switching visualization frame to incoming raw depth frame: %s", msg.header.frame_id)
            self.frame_id = msg.header.frame_id
            self.viewpoints_frame_id = self.frame_id
            self.best_view_frame_id = self.frame_id
            self.publish_identity_tf(self.root_frame, self.frame_id)
        msg.header.frame_id = self.frame_id
        self.depth_raw_points_pub.publish(self._mirror_pointcloud_x(msg))

    def depth_mesh_patch_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.depth_mesh_patch_pub.publish(self._mirror_marker_xz(msg))

    def depth_mesh_accumulated_patch_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.depth_mesh_accumulated_patch_pub.publish(self._mirror_marker_xz(msg))

    def depth_mesh_patch_boundary_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.depth_mesh_patch_boundary_pub.publish(self._mirror_marker_xz(msg))

    def depth_mesh_patch_intersections_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.depth_mesh_patch_intersections_pub.publish(self._mirror_marker_xz(msg))

    def depth_mesh_patch_cut_loop_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.depth_mesh_patch_cut_loop_pub.publish(self._mirror_marker_xz(msg))

    def depth_mesh_patch_failed_rays_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.depth_mesh_patch_failed_rays_pub.publish(self._mirror_marker_xz(msg))

    def mesh_triangles_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.mesh_tri_pub.publish(self._mirror_marker_xz(msg))

    def sampled_view_vectors_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.sampled_view_vectors_pub.publish(self._mirror_marker_xz(msg))

    def trajectory_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        marker = self._mirror_marker_xz(msg)
        marker = self._override_first_marker_point(marker, self._current_drone_display_position())
        self.trajectory_pub.publish(marker)

    def skipped_trajectory_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        marker = self._mirror_marker_xz(msg)
        self.skipped_trajectory_pub.publish(marker)

    def viewpoint_positions_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.viewpoint_positions_pub.publish(self._mirror_marker_xz(msg))

    def global_viewpoint_positions_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.global_viewpoint_positions_pub.publish(self._mirror_marker_xz(msg))

    def current_pose_point_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        marker = self._mirror_marker_xz(msg)
        marker = self._override_first_marker_point(marker, self._current_drone_display_position())
        self.current_pose_point_pub.publish(marker)

    def actual_flight_path_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.actual_flight_path_pub.publish(self._mirror_marker_xz(msg))

    def start_leg_checked_points_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        marker = self._mirror_marker_xz(msg)
        marker = self._override_first_marker_point(marker, self._current_drone_display_position())
        self.start_leg_checked_points_pub.publish(marker)

    def region_boundaries_callback(self, msg: Marker):
        msg.header.frame_id = self.frame_id
        self.region_boundaries_pub.publish(self._mirror_marker_xz(msg))

    def side_points_callback(self, msg: PoseArray):
        points: List[Point] = []
        for pose in msg.poses:
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            if self.flip_viewpoint_z:
                z = -z
            points.append(Point(x=float(x), y=float(y), z=float(z)))

        self.publish_side_point_markers(points)

    def viewpoints_callback(self, msg: PoseArray):
        self.viewpoints_frame_id = self.frame_id
        viewpoints: List[Tuple[float, float, float, float, float, float]] = []
        for pose in msg.poses:
            q = (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )
            roll, pitch, yaw = euler_from_quaternion(q, axes="sxyz")
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            if self.flip_viewpoint_z:
                z = -z

            viewpoints.append((x, y, z, roll, pitch, yaw))

        self.latest_viewpoints = viewpoints
        self.publish_viewpoint_markers(self.latest_viewpoints)

    def best_view_callback(self, msg: PoseStamped):
        self.best_view_frame_id = self.frame_id
        q = (
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        roll, pitch, yaw = euler_from_quaternion(q, axes="sxyz")
        pitch = -pitch
        x, y, z = (
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        )
        if self.flip_viewpoint_z:
            # Keep best-view frustum orientation consistent with current-view frustum convention.
            pitch = -pitch
            z = -z

        self.last_best_view = (
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
        )
        self.publish_path(self.last_best_view)
        self.publish_best_view_marker()

    def create_frustum_marker(
        self,
        view: Tuple[float, float, float, float, float, float],
        idx: int,
        frame_id: Optional[str] = None,
        namespace: str = "viewpoints",
        color: Tuple[float, float, float, float] = (0.1, 0.8, 1.0, 0.8),
    ) -> Marker:
        x, y, z, roll, pitch, yaw = view
        origin = np.array([x, y, z], dtype=np.float32)

        rot = euler_matrix(roll, pitch, yaw, axes="sxyz")[:3, :3]
        forward = rot[:, 0]
        right = rot[:, 1]
        cam_z = rot[:, 2]

        near_center = origin + forward * self.near_clip

        corners = []
        if (
            self.use_intrinsics_frustum
            and self.cam_fx > 1e-8
            and self.cam_fy > 1e-8
            and self.cam_width > 1
            and self.cam_height > 1
        ):
            u0 = 0.0
            v0 = 0.0
            u1 = float(self.cam_width - 1)
            v1 = float(self.cam_height - 1)
            pixel_corners = (
                (u0, v0),  # top-left
                (u1, v0),  # top-right
                (u1, v1),  # bottom-right
                (u0, v1),  # bottom-left
            )
            for depth in (self.near_clip, self.far_clip):
                center_point = origin + forward * depth
                for u_px, v_px in pixel_corners:
                    y_cam = (u_px - self.cam_cx) * depth / self.cam_fx
                    z_cam = (v_px - self.cam_cy) * depth / self.cam_fy
                    corners.append(center_point + right * y_cam + cam_z * z_cam)
        else:
            half_angle = math.radians(self.fov_deg * 0.5)
            near_h = math.tan(half_angle) * self.near_clip
            near_w = near_h
            far_h = math.tan(half_angle) * self.far_clip
            far_w = far_h
            far_center = origin + forward * self.far_clip
            for center_point, h, w in (
                (near_center, near_h, near_w),
                (far_center, far_h, far_w),
            ):
                corners.append(center_point + cam_z * h - right * w)
                corners.append(center_point + cam_z * h + right * w)
                corners.append(center_point - cam_z * h + right * w)
                corners.append(center_point - cam_z * h - right * w)

        marker = Marker()
        marker.header.frame_id = frame_id or self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = namespace
        marker.id = idx
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        marker.pose.orientation.w = 1.0

        pts: List[Point] = []
        # Connect near plane
        pts.extend(self._segment_points(corners[0], corners[1]))
        pts.extend(self._segment_points(corners[1], corners[2]))
        pts.extend(self._segment_points(corners[2], corners[3]))
        pts.extend(self._segment_points(corners[3], corners[0]))
        # Connect far plane
        pts.extend(self._segment_points(corners[4], corners[5]))
        pts.extend(self._segment_points(corners[5], corners[6]))
        pts.extend(self._segment_points(corners[6], corners[7]))
        pts.extend(self._segment_points(corners[7], corners[4]))
        # Connect near to far
        for i in range(4):
            pts.extend(self._segment_points(corners[i], corners[i + 4]))
        # Origin to near plane center
        pts.extend(self._segment_points(origin, near_center))

        marker.points = pts
        return marker

    @staticmethod
    def _segment_points(p1: np.ndarray, p2: np.ndarray) -> List[Point]:
        pt1 = Point(x=float(p1[0]), y=float(p1[1]), z=float(p1[2]))
        pt2 = Point(x=float(p2[0]), y=float(p2[1]), z=float(p2[2]))
        return [pt1, pt2]

    def _mirror_marker_array_x(self, marker_array: MarkerArray) -> MarkerArray:
        for marker in marker_array.markers:
            marker.pose.position.x = -marker.pose.position.x
            if marker.points:
                for pt in marker.points:
                    pt.x = -pt.x
        return marker_array

    def _mirror_marker_xz(self, marker: Marker) -> Marker:
        marker.pose.position.x = -marker.pose.position.x
        marker.pose.position.z = -marker.pose.position.z
        if marker.points:
            for pt in marker.points:
                pt.x = -pt.x
                pt.z = -pt.z
        return marker

    def _mirror_marker_z(self, marker: Marker) -> Marker:
        marker.pose.position.z = -marker.pose.position.z
        if marker.points:
            for pt in marker.points:
                pt.z = -pt.z
        return marker

    def _mirror_path_x(self, path: Path) -> Path:
        for pose in path.poses:
            pose.pose.position.x = -pose.pose.position.x
        return path

    def _current_drone_display_position(self) -> Optional[np.ndarray]:
        """
        Final RViz-displayed current camera position.
        Keep this exactly aligned with the green frustum transform path.
        """
        view = self._current_drone_view()
        if view is None:
            return None
        return np.array(
            [-float(view[0]), float(view[1]), float(view[2])],
            dtype=np.float64,
        )

    @staticmethod
    def _override_first_marker_point(marker: Marker, point_xyz: Optional[np.ndarray]) -> Marker:
        if marker is None or point_xyz is None or not marker.points:
            return marker
        p = np.asarray(point_xyz, dtype=np.float64).reshape(3)
        marker.points[0].x = float(p[0])
        marker.points[0].y = float(p[1])
        marker.points[0].z = float(p[2])
        return marker

    def _mirror_pointcloud_x(self, cloud: PointCloud2) -> PointCloud2:
        field_names = [f.name for f in cloud.fields]
        if "x" not in field_names:
            return cloud
        x_idx = field_names.index("x")
        z_idx = field_names.index("z") if "z" in field_names else None
        mirrored_points = []
        for pt in pc2.read_points(cloud, field_names=field_names, skip_nans=False):
            pt_list = list(pt)
            pt_list[x_idx] = -pt_list[x_idx]
            if z_idx is not None:
                pt_list[z_idx] = -pt_list[z_idx]
            mirrored_points.append(tuple(pt_list))

        mirrored = pc2.create_cloud(cloud.header, cloud.fields, mirrored_points)
        mirrored.height = cloud.height
        mirrored.width = cloud.width
        mirrored.is_bigendian = cloud.is_bigendian
        mirrored.is_dense = cloud.is_dense
        mirrored.point_step = cloud.point_step
        mirrored.row_step = mirrored.point_step * mirrored.width
        return mirrored

    def _mirror_pointcloud_z(self, cloud: PointCloud2) -> PointCloud2:
        field_names = [f.name for f in cloud.fields]
        if "z" not in field_names:
            return cloud
        z_idx = field_names.index("z")
        mirrored_points = []
        for pt in pc2.read_points(cloud, field_names=field_names, skip_nans=False):
            pt_list = list(pt)
            pt_list[z_idx] = -pt_list[z_idx]
            mirrored_points.append(tuple(pt_list))

        mirrored = pc2.create_cloud(cloud.header, cloud.fields, mirrored_points)
        mirrored.height = cloud.height
        mirrored.width = cloud.width
        mirrored.is_bigendian = cloud.is_bigendian
        mirrored.is_dense = cloud.is_dense
        mirrored.point_step = cloud.point_step
        mirrored.row_step = mirrored.point_step * mirrored.width
        return mirrored

    def publish_identity_tf(self, parent: str, child: str) -> None:
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.static_broadcaster.sendTransform(t)

    @staticmethod
    def _is_same_view(
        v1: Tuple[float, float, float, float, float, float],
        v2: Tuple[float, float, float, float, float, float],
        pos_tol: float = 1e-3,
        ang_tol: float = 1e-3,
    ) -> bool:
        for a, b in zip(v1[:3], v2[:3]):
            if abs(a - b) > pos_tol:
                return False
        for a, b in zip(v1[3:], v2[3:]):
            if abs(a - b) > ang_tol:
                return False
        return True

    def publish_viewpoint_markers(self, viewpoints: List[Tuple[float, float, float, float, float, float]]) -> None:
        markers = MarkerArray()
        for idx, view in enumerate(viewpoints):
            if self.last_best_view and self._is_same_view(view, self.last_best_view):
                continue
            markers.markers.append(
                self.create_frustum_marker(
                    view,
                    idx,
                    frame_id=self.viewpoints_frame_id,
                    namespace="viewpoints",
                    color=(0.1, 0.8, 1.0, 0.8),
                )
            )

        # Also overlay the best viewpoint if available
        if self.last_best_view is not None:
            markers.markers.append(
                self.create_frustum_marker(
                    self.last_best_view,
                    idx=9999,
                    frame_id=self.best_view_frame_id,
                    namespace="best_viewpoint",
                    color=(1.0, 0.2, 0.2, 0.95),
                )
            )
        self.viewpoint_markers_pub.publish(self._mirror_marker_array_x(markers))

    def publish_side_point_markers(self, points: List[Point]) -> None:
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = rospy.Time.now()
        marker.ns = "side_points"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.r = 1.0
        marker.color.g = 0.9
        marker.color.b = 0.1
        marker.color.a = 0.9
        marker.points = points
        marker_array.markers.append(marker)
        self.side_points_markers_pub.publish(self._mirror_marker_array_x(marker_array))

    def publish_path(self, best_view: Optional[Tuple[float, float, float, float, float, float]]) -> None:
        if best_view is None or self.current_odom is None:
            return

        odom_frame = self.frame_id

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = odom_frame

        current_view = self._current_drone_view()
        if current_view is None:
            return
        sx, sy, sz = current_view[0], current_view[1], current_view[2]

        if self.current_camera_pose is not None:
            sq = self.current_camera_pose.pose.orientation
        else:
            sq = self.current_odom.pose.pose.orientation
        sqx, sqy, sqz, sqw = float(sq.x), float(sq.y), float(sq.z), float(sq.w)

        start_pose = PoseStamped()
        start_pose.header = path.header
        start_pose.pose.position.x = sx
        start_pose.pose.position.y = sy
        start_pose.pose.position.z = sz
        start_pose.pose.orientation.x = sqx
        start_pose.pose.orientation.y = sqy
        start_pose.pose.orientation.z = sqz
        start_pose.pose.orientation.w = sqw

        x, y, z, _, _, yaw = best_view
        end_pose = PoseStamped()
        end_pose.header = path.header
        end_pose.pose.position.x = x
        end_pose.pose.position.y = y
        end_pose.pose.position.z = z
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, yaw, axes="sxyz")
        end_pose.pose.orientation.x = qx
        end_pose.pose.orientation.y = qy
        end_pose.pose.orientation.z = qz
        end_pose.pose.orientation.w = qw

        path.poses = [start_pose, end_pose]
        self.path_pub.publish(self._mirror_path_x(path))

    def publish_best_view_marker(self) -> None:
        """Publish a single red frustum for the current best viewpoint."""
        if self.last_best_view is None:
            return
        marker_array = MarkerArray()
        marker_array.markers.append(
            self.create_frustum_marker(
                self.last_best_view,
                idx=9999,
                frame_id=self.best_view_frame_id,
                namespace="best_viewpoint",
                color=(1.0, 0.2, 0.2, 0.95),
            )
        )
        self.viewpoint_markers_pub.publish(self._mirror_marker_array_x(marker_array))

    def _current_drone_view(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Current drone camera view in the same visualization convention used by frustum markers."""
        if self.current_camera_pose is not None:
            p = self.current_camera_pose.pose.position
            q = self.current_camera_pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion(
                (q.x, q.y, q.z, q.w), axes="sxyz"
            )
            if self.flip_viewpoint_z:
                pitch = -pitch
            z = -p.z if self.flip_viewpoint_z else p.z
            return (float(p.x), float(p.y), float(z), float(roll), float(pitch), float(yaw))

        if self.current_odom is None:
            return None

        p = self.current_odom.pose.pose.position
        q = self.current_odom.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(
            (q.x, q.y, q.z, q.w), axes="sxyz"
        )
        if self.flip_viewpoint_z:
            pitch = -pitch
        z = -p.z if self.flip_viewpoint_z else p.z
        return (float(p.x), float(p.y), float(z), float(roll), float(pitch), float(yaw))

    def publish_drone_frustum(self) -> None:
        """
        Publish a frustum representing the current drone pose/orientation.
        Uses the odometry pose (position + orientation) in the visualization frame.
        """
        view = self._current_drone_view()
        if view is None:
            return

        marker_array = MarkerArray()
        marker_array.markers.append(
            self.create_frustum_marker(
                view,
                idx=7777,
                frame_id=self.frame_id,
                namespace="drone",
                color=(0.2, 1.0, 0.2, 0.95),
            )
        )
        self.viewpoint_markers_pub.publish(self._mirror_marker_array_x(marker_array))

if __name__ == "__main__":
    node = ViewpointVisualizationNode()
    rospy.spin()
