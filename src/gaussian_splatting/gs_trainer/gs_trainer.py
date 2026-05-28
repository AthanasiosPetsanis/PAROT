import os
from typing import Dict, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- Gaussian representation -----------------


class GaussianField(nn.Module):
    """
    Simple 3D Gaussian field.

    Each Gaussian has:
      - position: (x, y, z)
      - anisotropic scale: (sx, sy, sz)
      - orientation: quaternion (w, x, y, z)
      - color: (r, g, b) in [0, 1]
      - opacity: alpha in (0, 1)
    """

    def __init__(
        self,
        num_gaussians: int,
        roi_center: torch.Tensor,  # (3,)
        roi_extent: torch.Tensor,  # (3,) full size of box
        device: str = "cuda",
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.register_buffer("roi_center", roi_center.to(device))
        self.register_buffer("roi_extent", roi_extent.to(device))

        # Positions: uniform in axis-aligned RoI box
        positions = self._init_positions_uniform()
        self.positions = nn.Parameter(positions)  # (N, 3)

        # Scales (log-space)
        log_scales = torch.full((num_gaussians, 3), -2.0, device=device)
        self.log_scales = nn.Parameter(log_scales)  # (N, 3)

        # Orientations as quats (w, x, y, z), start identity
        quats = torch.zeros((num_gaussians, 4), device=device)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)  # (N, 4)

        # Colors (RGB), start gray
        colors = torch.full((num_gaussians, 3), 0.5, device=device)
        self.colors = nn.Parameter(colors)  # (N, 3)

        # Opacity logits
        opacity_logits = torch.full((num_gaussians, 1), 0.0, device=device)
        self.opacity_logits = nn.Parameter(opacity_logits)  # (N, 1)

    def _init_positions_uniform(self) -> torch.Tensor:
        """
        Sample positions uniformly inside the RoI box defined by
        roi_center and roi_extent.
        """
        # roi_min = center - extent/2, roi_max = center + extent/2
        roi_min = self.roi_center - 0.5 * self.roi_extent
        roi_max = self.roi_center + 0.5 * self.roi_extent
        u = torch.rand((self.num_gaussians, 3), device=self.roi_center.device)
        return roi_min + u * (roi_max - roi_min)

    def forward(self) -> Dict[str, torch.Tensor]:
        """
        Return current Gaussian parameters in a normalized form.
        """
        quats = self.quats / (self.quats.norm(dim=-1, keepdim=True) + 1e-8)
        scales = torch.exp(self.log_scales)
        opacity = torch.sigmoid(self.opacity_logits)

        return {
            "positions": self.positions,  # (N, 3)
            "scales": scales,             # (N, 3)
            "quats": quats,               # (N, 4)
            "colors": self.colors,        # (N, 3)
            "opacity": opacity,           # (N, 1)
        }


# ----------------- Camera and projection -----------------


class Camera:
    """
    Pinhole camera with intrinsics (fx, fy, cx, cy) and image size.
    """

    def __init__(self, fx, fy, cx, cy, width, height, device="cuda"):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height
        self.device = device

    def project(self, points_world: torch.Tensor, world_to_cam: torch.Tensor):
        """
        points_world: (N, 3)
        world_to_cam: (4, 4) transform (homogeneous) from world to camera.
        Returns:
          uv: (N, 2) pixel coords
          depth: (N,) positive depth in camera z
        """
        N = points_world.shape[0]
        ones = torch.ones((N, 1), device=points_world.device)
        pts_h = torch.cat([points_world, ones], dim=-1)  # (N, 4)

        # (4,4) @ (N,4)^T -> (4,N) -> (N,4)
        pts_cam_h = (world_to_cam @ pts_h.T).T
        x = pts_cam_h[:, 0]
        y = pts_cam_h[:, 1]
        z = pts_cam_h[:, 2]

        # Avoid divide-by-zero
        z_safe = torch.clamp(z, min=1e-6)

        u = self.fx * (x / z_safe) + self.cx
        v = self.fy * (y / z_safe) + self.cy

        uv = torch.stack([u, v], dim=-1)  # (N, 2)
        return uv, z


# ----------------- Very simple renderer -----------------

def render_gaussians_point_splat(
    field: GaussianField,
    camera: Camera,
    world_to_cam: torch.Tensor,
    image_height: int,
    image_width: int,
):
    def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized quaternions (w,x,y,z) to rotation matrices.
        quat: (N,4)
        returns: (N,3,3)
        """
        w, x, y, z = quat.unbind(-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rot = torch.stack(
            [
                torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1),
                torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1),
                torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1),
            ],
            dim=-2,
        )
        return rot

    """
    Upgraded differentiable renderer:

    - Projects Gaussian centers to pixels.
    - Computes a screen-space radius in pixels from 3D scale and depth.
    - Splat each Gaussian into a small (2k+1)x(2k+1) neighborhood
      with a Gaussian kernel (soft blob).
    """

    device = field.positions.device
    params = field()
    positions = params["positions"]      # (N,3)
    colors = params["colors"]            # (N,3)
    opacity = params["opacity"].squeeze(-1)  # (N,)
    scales = params["scales"]            # (N,3)
    quats = params["quats"]              # (N,4)

    # Camera transform
    R_wc = world_to_cam[:3, :3].to(device)
    t_wc = world_to_cam[:3, 3].to(device)

    # World -> camera
    pts_cam = positions @ R_wc.T + t_wc  # (N,3)
    x_c, y_c, z_c = pts_cam.unbind(-1)
    z_safe = torch.clamp(z_c, min=1e-6)

    # Project to image plane
    u = camera.fx * (x_c / z_safe) + camera.cx
    v = camera.fy * (y_c / z_safe) + camera.cy
    uv = torch.stack([u, v], dim=-1)
    depth = z_c

    u = uv[:, 0]
    v = uv[:, 1]

    iu = torch.round(u).long()
    iv = torch.round(v).long()

    # Valid pixels (front-facing and inside image)
    valid = (
        (depth > 0.0)
        & (iu >= 0)
        & (iu < image_width)
        & (iv >= 0)
        & (iv < image_height)
    )

    if valid.sum() == 0:
        # Tie to parameters so graph exists
        dummy = field.positions.sum() * 0.0
        img = torch.zeros(3, image_height, image_width, device=device) + dummy
        return img

    iu = iu[valid]            # (M,)
    iv = iv[valid]            # (M,)
    colors = colors[valid]    # (M,3)
    alpha = opacity[valid]    # (M,)
    scales = scales[valid]    # (M,3)
    quats = quats[valid]      # (M,4)
    pts_cam = pts_cam[valid]  # (M,3)

    # Ellipsoid projection to screen space
    rot_g = quat_to_rotmat(quats)                    # (M,3,3)
    scale_mat = torch.diag_embed(scales ** 2)        # (M,3,3)
    sigma_world = rot_g @ scale_mat @ rot_g.transpose(-1, -2)  # (M,3,3)

    R_wc_exp = R_wc.unsqueeze(0).expand_as(sigma_world)        # (M,3,3)
    sigma_cam = torch.matmul(R_wc_exp, torch.matmul(sigma_world, R_wc.T))  # (M,3,3)

    x_c, y_c, z_c = pts_cam.unbind(-1)
    z_safe = torch.clamp(z_c, min=1e-6)
    J_row1 = torch.stack(
        [camera.fx / z_safe, torch.zeros_like(z_safe), -camera.fx * x_c / (z_safe * z_safe)],
        dim=-1,
    )
    J_row2 = torch.stack(
        [torch.zeros_like(z_safe), camera.fy / z_safe, -camera.fy * y_c / (z_safe * z_safe)],
        dim=-1,
    )
    J = torch.stack([J_row1, J_row2], dim=1)  # (M,2,3)

    sigma_img = torch.bmm(J, torch.bmm(sigma_cam, J.transpose(1, 2)))  # (M,2,2)
    sigma_img = sigma_img + 1e-4 * torch.eye(2, device=device).unsqueeze(0)
    sigma_inv = torch.linalg.inv(sigma_img)
    sigma_inv = 0.5 * (sigma_inv + sigma_inv.transpose(1, 2))  # enforce symmetry

    # Kernel half-size (in pixels)
    k = 3  # => kernel size = 7x7
    ks = 2 * k + 1

    # Precompute offset grid
    offsets_y, offsets_x = torch.meshgrid(
        torch.arange(-k, k + 1, device=device),
        torch.arange(-k, k + 1, device=device),
        indexing="ij",
    )
    offsets_x = offsets_x.reshape(-1)  # (K2,)
    offsets_y = offsets_y.reshape(-1)  # (K2,)
    K2 = offsets_x.shape[0]

    # Expand per-Gaussian center and radius over the kernel
    M = iu.shape[0]
    iu_exp = iu[:, None] + offsets_x[None, :]   # (M,K2)
    iv_exp = iv[:, None] + offsets_y[None, :]   # (M,K2)

    a = sigma_inv[:, 0, 0][:, None]
    b = sigma_inv[:, 0, 1][:, None]
    c = sigma_inv[:, 1, 1][:, None]
    dx = offsets_x[None, :].float()
    dy = offsets_y[None, :].float()
    exponent = a * (dx * dx) + 2.0 * b * (dx * dy) + c * (dy * dy)  # (M,K2)
    w = torch.exp(-0.5 * exponent)
    w_alpha = w * alpha[:, None]                         # (M,K2)

    # Keep only pixels inside image
    inside = (
        (iu_exp >= 0) & (iu_exp < image_width) &
        (iv_exp >= 0) & (iv_exp < image_height)
    )  # (M,K2)

    if inside.sum() == 0:
        dummy = field.positions.sum() * 0.0
        img = torch.zeros(3, image_height, image_width, device=device) + dummy
        return img

    # Flatten valid contributions
    iu_flat = iu_exp[inside]          # (L,)
    iv_flat = iv_exp[inside]          # (L,)
    w_flat = w_alpha[inside]          # (L,)

    # For colors, we need per-contribution weights for each channel.
    # Expand colors to match kernel then mask.
    colors_exp = colors[:, None, :] * w[:, :, None]  # (M,K2,3)
    colors_flat = colors_exp[inside]                 # (L,3)

    # Prepare accumulation buffers
    img_acc = torch.zeros(3, image_height, image_width, device=device)
    alpha_acc = torch.zeros(image_height, image_width, device=device)

    # 1D indices
    linear_idx = iv_flat * image_width + iu_flat
    num_pixels = image_height * image_width

    alpha_flat = alpha_acc.view(-1)
    alpha_flat.scatter_add_(0, linear_idx, w_flat)

    for c in range(3):
        img_flat_c = img_acc[c].view(-1)
        img_flat_c.scatter_add_(0, linear_idx, colors_flat[:, c])

    eps = 1e-6
    alpha_acc = alpha_acc.clamp(min=eps)
    img = img_acc / alpha_acc.unsqueeze(0)
    img = img.clamp(0.0, 1.0)
    return img


# ----------------- Trainer -----------------

class GSTrainer:
    """
    Wraps a GaussianField and a simple renderer to provide a
    train_initial(...) function.

    Expected pose_dict format:

        pose_dict[image_name] = {
            "world_to_cam": np.ndarray shape (4,4), float32
        }
    """

    def __init__(
        self,
        roi_center: np.ndarray,
        roi_extent: np.ndarray,
        num_gaussians: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: int,
        height: int,
        image_dir: str,
        device: str = "cuda",
        render_dir: str = None,
        save_render_interval: int = 100,
    ):
        self.device = device
        self.image_dir = image_dir

        # where to save rendered + original images (can be None)
        self.render_dir = render_dir
        self.save_render_interval = save_render_interval
        if self.render_dir is not None:
            os.makedirs(self.render_dir, exist_ok=True)

        roi_center_t = torch.tensor(roi_center, dtype=torch.float32, device=device)
        roi_extent_t = torch.tensor(roi_extent, dtype=torch.float32, device=device)

        self.field = GaussianField(
            num_gaussians=num_gaussians,
            roi_center=roi_center_t,
            roi_extent=roi_extent_t,
            device=device,
        )

        self.camera = Camera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            device=device,
        )

        # Simple optimizer (single LR for now; you can later split into param groups)
        self.optimizer = torch.optim.Adam(
            [
                self.field.positions,
                self.field.log_scales,
                self.field.quats,
                self.field.colors,
                self.field.opacity_logits,
            ],
            lr=1e-3,
        )

        self.height = height
        self.width = width

    def _load_image(self, filename: str) -> torch.Tensor:
        """
        Load an RGB image from disk and convert to float32 tensor [0,1],
        shape (3, H, W).
        """
        path = os.path.join(self.image_dir, filename)
        img = Image.open(path).convert("RGB")
        img = img.resize((self.width, self.height), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
        arr = np.transpose(arr, (2, 0, 1))                # (3,H,W)
        return torch.from_numpy(arr).to(self.device)

    def _pose_to_tensor(self, pose_entry) -> torch.Tensor:
        """
        Convert pose_dict entry to a torch (4,4) world_to_cam matrix.

        Here we assume pose_entry["world_to_cam"] is already a (4,4) np array.
        """
        mat = pose_entry["world_to_cam"]
        assert mat.shape == (4, 4)
        return torch.from_numpy(mat).float().to(self.device)

    def train_initial(
        self,
        image_list: List[str],
        pose_dict: Dict,
        steps: int,
        batch_size: int = 1,
        save_img: bool = True,
    ):
        """
        Basic training loop:

        - image_list: list of image file names (e.g. ["image_000001.png", ...])
        - pose_dict: dict mapping image name to:
              {"world_to_cam": np.ndarray (4,4)}
        - steps: number of optimization iterations
        - batch_size: how many views per step (keep small for now)
        """
        if len(image_list) == 0:
            print("GSTrainer.train_initial: no images, skipping.")
            return

        num_views = len(image_list)

        for step in range(steps):
            # Pick a random mini-batch of views
            idxs = np.random.choice(num_views, size=batch_size, replace=False)
            loss_acc = 0.0

            self.optimizer.zero_grad()

            last_pred_img = None
            last_gt_img = None
            last_img_name = None

            for idx in idxs:
                img_name = image_list[idx]

                # Load image
                gt_img = self._load_image(img_name)  # (3,H,W)

                # Get pose
                pose_entry = pose_dict[img_name]
                world_to_cam = self._pose_to_tensor(pose_entry)

                # Render
                pred_img = render_gaussians_point_splat(
                    self.field,
                    self.camera,
                    world_to_cam,
                    self.height,
                    self.width,
                )

                # Simple L2 loss
                loss = F.mse_loss(pred_img, gt_img)
                loss_acc = loss_acc + loss

                # keep last view in this batch for optional saving
                last_pred_img = pred_img
                last_gt_img = gt_img
                last_img_name = img_name

            # Backprop through all images in the batch
            loss_acc.backward()
            self.optimizer.step()

            # Optional snapshot of rendered vs original
            if (
                save_img
                and self.render_dir is not None
                and last_pred_img is not None
                and (step + 1) % self.save_render_interval == 0
            ):
                os.makedirs(self.render_dir, exist_ok=True)
                pred_np = (
                    last_pred_img.detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 255
                ).astype(np.uint8)
                gt_np = (
                    last_gt_img.detach()
                    .cpu()
                    .numpy()
                    .transpose(1, 2, 0)
                    * 255
                ).astype(np.uint8)

                # reuse original filename to keep correspondence clear
                rendered_name = f"rendered_step{step+1}_{last_img_name}"
                original_name = f"original_step{step+1}_{last_img_name}"

                out_path_rendered = os.path.join(self.render_dir, rendered_name)
                out_path_original = os.path.join(self.render_dir, original_name)

                Image.fromarray(pred_np).save(out_path_rendered)
                Image.fromarray(gt_np).save(out_path_original)

            if (step + 1) % 100 == 0:
                print(f"[GSTrainer] step {step+1}/{steps}, loss = {loss_acc.item():.6f}")

    def get_gaussians_numpy(self):
        """
        Convenience: pull current Gaussians to CPU as numpy arrays for saving
        into your existing text format from GaussianModel.
        """
        params = self.field()
        out = {
            k: v.detach().cpu().numpy() for k, v in params.items()
        }
        return out
