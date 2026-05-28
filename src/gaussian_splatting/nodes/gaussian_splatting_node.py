#!/usr/bin/env python3
import os
import time
import math
import struct
import rospy
import numpy as np

import sys

# add projectB to import paths
sys.path.append("/home/thanos/Documents/IROS_2026/src/gaussian_splatting/gs_trainer")
from gs_trainer import GSTrainer
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2


class GaussianModel(object):
    """
    Gaussian model that wraps the PyTorch GSTrainer.

    - GSTrainer owns the GaussianField (3D GS in PyTorch, random init in RoI).
    - GaussianModel keeps a lightweight list-of-dicts view for saving / RViz.
    """

    def __init__(self, center, extent, num_init_gaussians,
                 fx, fy, cx, cy, width, height,
                 image_dir,
                 renders_dir,
                 device="cuda"):
        self.center = np.array(center, dtype=np.float32)
        self.extent = np.array(extent, dtype=np.float32)
        self.num_init_gaussians = num_init_gaussians
        self.device = device

        self.gaussians = []  # list of dicts used for saving + publishing

        self.image_dir = image_dir
        self.renders_dir = renders_dir

        # Create PyTorch trainer (this also randomly initializes Gaussians
        # uniformly inside the RoI).
        self.trainer = GSTrainer(
            roi_center=self.center,
            roi_extent=self.extent,
            num_gaussians=self.num_init_gaussians,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            image_dir=image_dir,
            device=device,
            render_dir=renders_dir,
        )

    # ---------- internal helpers ----------

    def _quat_to_rot(self, qw, qx, qy, qz):
        """
        Convert quaternion (qw,qx,qy,qz) to 3x3 rotation matrix.
        Assumes (qw,qx,qy,qz) is camera orientation in the *world* frame.
        """
        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        # normalize
        norm = np.linalg.norm(q)
        if norm < 1e-8:
            return np.eye(3, dtype=np.float32)
        q = q / norm
        w, x, y, z = q

        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
            [2*(x*y + z*w),         1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w),         1 - 2*(x*x + y*y)],
        ], dtype=np.float32)
        return R

    def _make_world_to_cam(self, tx, ty, tz, qx, qy, qz, qw):
        """
        Build a 4x4 world_to_cam matrix from a pose [tx,ty,tz,qx,qy,qz,qw].

        Assumption:
        - (tx,ty,tz) is the camera position in world coordinates.
        - (qx,qy,qz,qw) is the camera orientation in world coordinates.
        - So T_world_cam = [R, t; 0,1], and world_to_cam = T_world_cam^{-1}.
        """
        R_wc = self._quat_to_rot(qw, qx, qy, qz)  # 3x3
        t_wc = np.array([tx, ty, tz], dtype=np.float32)

        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_wc.T
        T[:3, 3] = -R_wc.T @ t_wc
        return T

    def _update_gaussians_from_trainer(self):
        """
        Pull current Gaussian parameters from GSTrainer (GPU -> CPU) and
        convert to the list-of-dicts format used for saving + RViz.
        """
        params = self.trainer.get_gaussians_numpy()
        pos = params["positions"]  # (N,3)
        scales = params["scales"]  # (N,3)
        colors = params["colors"]  # (N,3)
        opacity = params["opacity"].reshape(-1)  # (N,)

        N = pos.shape[0]
        self.gaussians = []
        for i in range(N):
            self.gaussians.append({
                "x": float(pos[i, 0]),
                "y": float(pos[i, 1]),
                "z": float(pos[i, 2]),
                "sx": float(scales[i, 0]),
                "sy": float(scales[i, 1]),
                "sz": float(scales[i, 2]),
                "r": float(colors[i, 0]),
                "g": float(colors[i, 1]),
                "b": float(colors[i, 2]),
                "opacity": float(opacity[i]),
            })

    def load_from_file(self, filename):
        """
        Keep your existing text loader so you can visualize previously
        saved gaussians without running GS again.
        """
        if not os.path.isfile(filename):
            return False

        data = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) not in (10, 11):
                    continue
                vals = list(map(float, parts))
                cvi = vals[10] if len(vals) == 11 else 0.0
                g = {
                    "x": vals[0],
                    "y": vals[1],
                    "z": vals[2],
                    "sx": vals[3],
                    "sy": vals[4],
                    "sz": vals[5],
                    "r": vals[6],
                    "g": vals[7],
                    "b": vals[8],
                    "opacity": vals[9],
                    "cvi": cvi,
                }
                data.append(g)

        if not data:
            return False

        self.gaussians = data
        rospy.loginfo("GaussianModel: loaded %d gaussians from %s",
                      len(self.gaussians), filename)
        return True

    def save_gs_to_txt(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as f:
            f.write("# x y z sx sy sz r g b opacity cvi\n")
            for g in self.gaussians:
                f.write(
                    f"{g['x']} {g['y']} {g['z']} "
                    f"{g['sx']} {g['sy']} {g['sz']} "
                    f"{g['r']} {g['g']} {g['b']} {g['opacity']} {g.get('cvi', 0.0)}\n"
                )

    def save_gs_to_splat(self, gaussians, filename):
        """
        Save gaussians to antimatter15 .splat format.

        gaussians: list of dicts with keys:
            x,y,z,sx,sy,sz,r,g,b,opacity  (r,g,b,opacity in [0,1])
        filename: path to output .splat file
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            for g in gaussians:
                # position
                px = float(g["x"])
                py = float(g["y"])
                pz = float(g["z"])

                # scale (radii)
                sx = float(g["sx"])
                sy = float(g["sy"])
                sz = float(g["sz"])

                # color in [0,1] -> uint8 0..255
                def to_u8(v):
                    v = max(0.0, min(1.0, float(v)))
                    return int(round(v * 255.0))

                r = to_u8(g["r"])
                g_ = to_u8(g["g"])
                b = to_u8(g["b"])
                a = to_u8(g["opacity"])

                # rotation quaternion: for now, identity (0,0,0,1)
                # pack as unorm8x4 in [0,1], i.e. q in [-1,1] -> (q*0.5+0.5)*255
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

                def quat_comp_to_u8(q):
                    # map [-1,1] -> [0,1] -> [0,255]
                    v = q * 0.5 + 0.5
                    v = max(0.0, min(1.0, float(v)))
                    return int(round(v * 255.0))

                qx_u8 = quat_comp_to_u8(qx)
                qy_u8 = quat_comp_to_u8(qy)
                qz_u8 = quat_comp_to_u8(qz)
                qw_u8 = quat_comp_to_u8(qw)

                # little-endian: 6 floats + 4 u8 (color) + 4 u8 (rotation) = 32 bytes
                packed = struct.pack(
                    "<6f4B4B",
                    px, py, pz,
                    sx, sy, sz,
                    r, g_, b, a,
                    qx_u8, qy_u8, qz_u8, qw_u8
                )
                f.write(packed)

    def train_initial(self, image_list, pose_dict_raw, steps):
        """
        Initial training over a set of images.

        pose_dict_raw:
            dict keyed by image filename:
                pose_dict_raw[img_name] = [tx, ty, tz, qx, qy, qz, qw]
        """
        rospy.loginfo("GaussianModel: initial training with %d images, %d steps.",
                      len(image_list), steps)

        # Build pose_dict for GSTrainer: key = image_name, value = {"world_to_cam": 4x4 np}
        pose_dict_gs = {}

        for img_name in image_list:
            if img_name not in pose_dict_raw:
                rospy.logwarn("GaussianModel: no pose for image %s during initial training.", img_name)
                continue
            tx, ty, tz, qx, qy, qz, qw = pose_dict_raw[img_name]
            T_wc = self._make_world_to_cam(tx, ty, tz, qx, qy, qz, qw)
            pose_dict_gs[img_name] = {"world_to_cam": T_wc}

        if not pose_dict_gs:
            rospy.logwarn("GaussianModel: no valid poses for initial training, skipping.")
            return

        self.trainer.train_initial(image_list, pose_dict_gs, steps)
        self._update_gaussians_from_trainer()

    def train_incremental(self, new_images, pose_dict_raw, steps):
        """
        Incremental refinement on a set of new images.

        new_images: list of image filenames
        pose_dict_raw:
            dict keyed by image filename:
                pose_dict_raw[img_name] = [tx, ty, tz, qx, qy, qz, qw]
        """
        rospy.loginfo("GaussianModel: incremental training with %d new images, %d steps.",
                      len(new_images), steps)

        if not new_images:
            return

        pose_dict_gs = {}
        used_images = []

        for img_name in new_images:
            vals = pose_dict_raw.get(img_name, None)
            if vals is None:
                rospy.logwarn("GaussianModel: no pose for new image %s, skipping.", img_name)
                continue
            tx, ty, tz, qx, qy, qz, qw = vals
            T_wc = self._make_world_to_cam(tx, ty, tz, qx, qy, qz, qw)
            pose_dict_gs[img_name] = {"world_to_cam": T_wc}
            used_images.append(img_name)

        if not pose_dict_gs:
            rospy.logwarn("GaussianModel: no valid poses for incremental training, skipping.")
            return

        # reuse same training routine; it will only backprop through visible Gaussians
        self.trainer.train_initial(used_images, pose_dict_gs, steps)
        self._update_gaussians_from_trainer()


class GaussianSplattingNode(object):
    def __init__(self):
        rospy.init_node("gaussian_splatting_node")

        # --------------------
        # Parameters (from controller.yaml loaded under this node's ns)
        # --------------------
        # Same structure as your ControllerNode, so you reuse controller.yaml
        self.img_base_dir = rospy.get_param("~file_paths/img_dir",
                                            "/home/thanos/Documents/IROS_2026/")
        self.poses_base_dir = rospy.get_param("~file_paths/poses_dir",
                                              "/home/thanos/Documents/IROS_2026/")
        self.gaussians_base_dir = rospy.get_param(
            "~file_paths/gaussians_path",
            "/home/thanos/Documents/IROS_2026/GaussianSpace"
        )

        # Building geometry (reused)
        self.build_center_x = rospy.get_param("~building/center_x", 0.0)
        self.build_center_y = rospy.get_param("~building/center_y", 0.0)
        self.build_center_z = rospy.get_param("~building/center_z", 0.0)
        self.build_width    = rospy.get_param("~building/width", 20.0)
        self.build_length   = rospy.get_param("~building/length", 30.0)
        self.build_height   = rospy.get_param("~building/height", 10.0)

        # Camera intrinsics (from controller.yaml -> calibration)
        self.fx = rospy.get_param("~calibration/fx", 381.361145)
        self.fy = rospy.get_param("~calibration/fy", 381.361145)
        self.cx = rospy.get_param("~calibration/cx", 320.0)
        self.cy = rospy.get_param("~calibration/cy", 240.0)
        self.img_width = rospy.get_param("~calibration/width", 640)
        self.img_height = rospy.get_param("~calibration/height", 480)

        # Gaussian splatting specific params
        self.update_interval = rospy.get_param("~gaussian_splatting/update_interval", 1.0)
        self.initial_steps   = rospy.get_param("~gaussian_splatting/initial_training_steps", 30000)
        self.incremental_steps = rospy.get_param("~gaussian_splatting/incremental_steps", 500)
        self.num_init_gaussians = rospy.get_param("~gaussian_splatting/num_init_gaussians", 10000)
        # how many new images before saving txt + splat
        self.save_every_n_images = rospy.get_param(
            "~gaussian_splatting/save_every_n_images", 10
        )

        # Derived paths to match your controller node
        self.image_dir = os.path.join(self.img_base_dir, "images")
        self.pose_file = os.path.join(self.poses_base_dir, "poses", "poses.txt")
        self.gaussians_file = os.path.join(self.gaussians_base_dir, "gaussians_latest.txt")
        self.renders_dir = os.path.join(self.gaussians_base_dir, "renders")

        rospy.loginfo("GaussianSplattingNode: images_dir=%s", self.image_dir)
        rospy.loginfo("GaussianSplattingNode: pose_file=%s", self.pose_file)
        rospy.loginfo("GaussianSplattingNode: gaussians_file=%s", self.gaussians_file)
        rospy.loginfo("GaussianSplattingNode: renders_dir=%s", self.renders_dir)

        center = [self.build_center_x, self.build_center_y, self.build_center_z]
        extent = [self.build_width, self.build_length, self.build_height]

        self.model = GaussianModel(
            center=center,
            extent=extent,
            num_init_gaussians=self.num_init_gaussians,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            width=self.img_width,
            height=self.img_height,
            image_dir=self.image_dir,
            renders_dir=self.renders_dir,
            device="cuda",  # or "cpu" if you want to test on CPU
        )

        # Track which images we've already integrated
        self.known_images = set()
        # count new images since last txt + splat save
        self.images_since_last_save = 0

        # Publisher for gaussians
        self.gaussians_pub = rospy.Publisher("gaussians", PointCloud2, queue_size=1)

        # Initialize / load model
        self._initialize_or_load_model()

        # Timer for incremental updates
        self.timer = rospy.Timer(
            rospy.Duration(self.update_interval),
            self.timer_callback
        )

    # ------------- File helpers -------------

    def _list_images(self):
        if not os.path.isdir(self.image_dir):
            return []
        files = os.listdir(self.image_dir)
        imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        imgs.sort()
        return imgs

    def _load_poses(self):
        """
        Pose loader for poses.txt.

        Expected line format:
            image_id qw qx qy qz tx ty tz image_name

        Returns:
            dict keyed by image_name:
                pose_dict[image_name] = [tx, ty, tz, qx, qy, qz, qw]
        """
        pose_dict = {}
        if not os.path.isfile(self.pose_file):
            return pose_dict

        with open(self.pose_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 9:
                raise IndexError(
                    "It should be in the form image_id qw qx qy qz tx ty tz image_name , i.e. 9 elements"
                )
            image_name = parts[-1]
            # parts[1:-1] = [qw, qx, qy, qz, tx, ty, tz]
            vals_raw = list(map(float, parts[1:-1]))
            qw, qx, qy, qz, tx, ty, tz = vals_raw
            # reorder to [tx, ty, tz, qx, qy, qz, qw]
            vals = [tx, ty, tz, qx, qy, qz, qw]
            pose_dict[image_name] = vals

        return pose_dict

    # ------------- Initialization -------------

    def _initialize_or_load_model(self):
        # Existing model?
        if self.model.load_from_file(self.gaussians_file):
            # If loaded, assume initial training already done.
            rospy.loginfo("GaussianSplattingNode: loaded existing Gaussians.")
        else:
            # No model yet: run initial GS training.
            rospy.loginfo("GaussianSplattingNode: running initial Gaussian Splatting.")

            images = self._list_images()
            self.known_images = set(images)

            poses = self._load_poses()
            self.model.train_initial(images, poses, self.initial_steps)

            self.model.save_gs_to_txt(self.gaussians_file)
            splat_path = os.path.join(self.gaussians_base_dir, "gaussians_latest.splat")
            self.model.save_gs_to_splat(self.model.gaussians, splat_path)

        # Publish once after init
        self.publish_gaussians()

    # ------------- Timer callback -------------

    def timer_callback(self, event):
        # Check for new images in the directory
        images = self._list_images()
        candidate_new_imgs = [img for img in images if img not in self.known_images]

        if not candidate_new_imgs:
            return

        poses = self._load_poses()
        if not poses:
            # no poses yet; wait
            return

        # keep only those for which we already have poses
        new_imgs = [img for img in candidate_new_imgs if img in poses]

        if not new_imgs:
            return

        for img in new_imgs:
            self.known_images.add(img)

        # Incrementally train on these new views
        self.model.train_incremental(new_imgs, poses, self.incremental_steps)

        # update counter for periodic saving
        self.images_since_last_save += len(new_imgs)

        if self.images_since_last_save >= self.save_every_n_images:
            self.model.save_gs_to_txt(self.gaussians_file)
            splat_path = os.path.join(self.gaussians_base_dir, "gaussians_latest.splat")
            self.model.save_gs_to_splat(self.model.gaussians, splat_path)
            self.images_since_last_save = 0

        # Publish updated gaussians
        self.publish_gaussians()

    # ------------- Publishing -------------

    def publish_gaussians(self):
        if not self.model.gaussians:
            return

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "world"  # adapt to your world frame

        # x,y,z + r,g,b,opacity as float32
        fields = [
            PointField("x", 0,  PointField.FLOAT32, 1),
            PointField("y", 4,  PointField.FLOAT32, 1),
            PointField("z", 8,  PointField.FLOAT32, 1),
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1),
            PointField("opacity", 24, PointField.FLOAT32, 1),
        ]

        points = []
        for g in self.model.gaussians:
            points.append([
                g["x"], g["y"], g["z"],
                g["r"], g["g"], g["b"], g["opacity"]
            ])

        cloud = pc2.create_cloud(header, fields, points)
        self.gaussians_pub.publish(cloud)


if __name__ == "__main__":
    node = GaussianSplattingNode()
    rospy.spin()
