# examples/02_mip360_test.py
# MipNeRF-360 triangle-soup radiance field optimisation with DiffSoup.
#
# Usage:
#   python examples/02_mip360_test.py --scene_root ./datasets/360_v2/kitchen
#
# Dependencies (beyond diffsoup):
#   pip install open3d imageio torchvision tqdm pytorch-msssim matplotlib scipy
#
# Note: LPIPS is intentionally excluded from this example.  For fair
# comparison with other methods, ensure you use a consistent LPIPS model
# and weights (e.g. VGG vs AlexNet, v0.0 vs v0.1) across all baselines.

from __future__ import annotations

import argparse
import os
import random
from typing import List, Optional

import imageio.v2 as iio
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from torch.optim import Adam
from tqdm.auto import tqdm

import diffsoup as ds
from utils import (
    SSIM_BACKEND,
    ssim,
    load_mipnerf360_scene,
    mvp_from_K_Tcw,
    read_points3D,
    project_vertices,
    psnr_fn,
    exp_decay_mult,
    count_visible_triangles,
    build_keep_map,
    split_edges_from_training_views,
)

# ── Reproducibility ──────────────────────────────────────────────────

SEED = 0
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ── Main ─────────────────────────────────────────────────────────────

def main(
    scene_root: str = "./datasets/360_v2/kitchen",
    batch_size: int = 4,
    steps: int = 10_000,
    schedule_steps: int = 10_000,
    n_points: Optional[int] = 15_000,
    downscale: int = 4,
    flip_z: bool = True,
    out_dir: Optional[str] = None,
    overwrite: bool = False,
    prior_root: Optional[str] = None,
    normal_prior_weight: float = 0.0,
    depth_prior_weight: float = 0.0,
    prior_start: int = 5_501,
    prior_ramp_steps: int = 500,
    prior_samples_per_view: int = 16_384,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ssim] backend={SSIM_BACKEND}")
    scene_name = os.path.basename(os.path.normpath(scene_root))
    if out_dir is None:
        out_dir = os.path.join("./results/02_mip360", scene_name)
    if (
        os.path.isdir(out_dir)
        and os.listdir(out_dir)
        and not overwrite
    ):
        raise FileExistsError(
            f"output directory is not empty: {out_dir}; "
            "choose a new --out_dir or pass --overwrite"
        )
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────

    train_data = load_mipnerf360_scene(
        scene_root, split="train", downscale=downscale, device=device,
    )
    K, H, W = train_data["K"], train_data["H"], train_data["W"]
    frames = train_data["frames"]
    N_train = len(frames)
    print(f"[views] train={N_train}  folder={train_data['folder']} size={H}x{W}")
    if schedule_steps <= 0:
        raise ValueError("schedule_steps must be positive")
    if (
        not np.isfinite(normal_prior_weight)
        or not np.isfinite(depth_prior_weight)
        or normal_prior_weight < 0
        or depth_prior_weight < 0
    ):
        raise ValueError("prior weights must be nonnegative")
    if prior_start < 1:
        raise ValueError("prior_start must be at least 1")
    if prior_ramp_steps < 0:
        raise ValueError("prior_ramp_steps must be nonnegative")
    if prior_samples_per_view <= 0:
        raise ValueError("prior_samples_per_view must be positive")
    use_geometry_prior = normal_prior_weight > 0 or depth_prior_weight > 0
    if use_geometry_prior and prior_root is None:
        # Scene-level depth/normal PNGs are the default prior interface.
        prior_root = scene_root

    # ── Geometry: triangle soup from COLMAP points ───────────────────

    xyz_np = read_points3D(scene_root)
    print(f"[points3D] loaded original {xyz_np.shape[0]:,} points")
    N_total = xyz_np.shape[0]
    target_faces = 15_000 if n_points is None else int(n_points)
    if target_faces <= 0 or target_faces > N_total:
        raise ValueError(
            f"n_points must be in [1, {N_total}], got {target_faces}"
        )
    random_count = target_faces // 3
    fps_count = target_faces - random_count

    sel = np.random.choice(xyz_np.shape[0], random_count, replace=False)
    xyz_sel = xyz_np[sel]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_np)
    pcd_down = pcd.farthest_point_down_sample(fps_count)
    xyz_np = np.asanyarray(pcd_down.points)

    xyz_np = np.concatenate([xyz_np, xyz_sel], axis=0)

    spacing = 0.25 * ds.nn_spacing(xyz_np, reduction="mean")

    print(f"[points3D] using {len(xyz_np):,} / {N_total:,} points (subsampled)")
    xyz = torch.from_numpy(xyz_np).to(device=device, dtype=torch.float32)
    V_single, F = ds.triangle_soup_from_points(xyz, scale=spacing)

    # ── Feature buffers ──────────────────────────────────────────────
    #   Rmin, Rmax: multi-resolution level range.  Both colour and opacity
    #   features share the same range and are lifted together at step 5 000.

    Rmin, Rmax = 0, 0
    feat_dim = 7

    feat_src = ds.build_multires_triangle_color(
        F.shape[0], Rmin, Rmax, feat_dim,
    ).to(device="cuda")
    feat_src.requires_grad = True

    alpha_src = ds.build_multires_triangle_color(
        F.shape[0], Rmin, Rmax, feat_dim=1,
    ).to(device="cuda")
    alpha_src.requires_grad = True

    # ── Precompute GT and MVPs ───────────────────────────────────────

    gt_rgb = torch.stack([fr["image"].clamp(0, 1) for fr in frames], dim=0)

    z_near_train, z_near_test, z_far = 0.01, 0.5, 100.0

    MVPs = torch.stack([
        mvp_from_K_Tcw(K, fr["Tcw"], (H, W), z_near=z_near_train, z_far=z_far, flip_z=flip_z)
        for fr in frames
    ], dim=0)
    MVPs_inv = torch.inverse(MVPs).contiguous()
    Tcws = None
    prior_store = None
    if use_geometry_prior:
        assert prior_root is not None
        Tcws = torch.stack([fr["Tcw"] for fr in frames], dim=0)
        view_names = [os.path.basename(fr["img_path"]) for fr in frames]
        prior_store = ds.GeometryPriorStore(
            prior_root,
            view_names,
            (H, W),
            downscale=downscale,
        )
        print(
            f"[prior] normal_weight={normal_prior_weight:g} "
            f"depth_weight={depth_prior_weight:g} start={prior_start} "
            f"ramp={prior_ramp_steps} samples/view={prior_samples_per_view:,}"
        )

    # ``gt_rgb`` owns the stacked images; retaining the per-frame tensors
    # keeps a second full copy of the training set resident on the GPU.
    del frames, train_data
    torch.cuda.empty_cache()

    # ── Colour MLP ───────────────────────────────────────────────────

    color_mlp = ds.ColorMLP(
        input_dim=feat_dim + 9, hidden_dim=16, n_layers=2, output_dim=3,
    ).to(device="cuda")

    V_single = V_single.requires_grad_(True)

    # ── Initial renders ──────────────────────────────────────────────

    eval_ids = torch.linspace(0, max(N_train - 1, 0), steps=min(3, N_train)).round().long().unique()
    eval_MVP = MVPs[eval_ids]
    eval_MVP_inv = MVPs_inv[eval_ids]

    with torch.no_grad():
        pred_init = []
        for view_idx in range(eval_MVP.shape[0]):
            V_clip = project_vertices(V_single, eval_MVP[view_idx : view_idx + 1])
            rast_out = ds.rasterize_multires_triangle_alpha(
                (H, W), V_clip, F,
                level=Rmax,
                alpha_src=ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid(),
            )
            feat = ds.multires_triangle_color(
                rast_out, level=Rmax,
                feat=ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid(),
            ).view(-1, H, W, feat_dim)
            view_feat = ds.encode_view_dir_sh2(rast_out, eval_MVP_inv[view_idx : view_idx + 1])
            color = color_mlp.forward(
                feat, mask=rast_out[..., -1] > 0, extra_features=view_feat,
            ).view(1, H, W, 3)
            pred_init.append(color.squeeze(0))
        pred_init = torch.stack(pred_init, dim=0)
    for j in range(pred_init.shape[0]):
        iio.imwrite(
            os.path.join(out_dir, f"initial_pred_{j}.png"),
            (pred_init[j].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
        )
    del pred_init, color, feat, view_feat, rast_out, V_clip
    print(f"[save] initial renders → {out_dir}/initial_pred_*.png")

    # ── Optimisers ───────────────────────────────────────────────────

    optimizer_soup = Adam([
        {"params": [feat_src], "lr": 5e-2},
        {"params": [alpha_src], "lr": 5e-2},
    ])
    optimizer_vert = ds.optimize.VectorAdam(params=[V_single], lr=1e-2)
    optimizer_shader = Adam([{"params": color_mlp.parameters(), "lr": 1e-2}])

    base_soup_lrs = [pg["lr"] for pg in optimizer_soup.param_groups]
    base_vert_lrs = [pg["lr"] for pg in optimizer_vert.param_groups]
    base_shader_lrs = [pg["lr"] for pg in optimizer_shader.param_groups]

    # ── Training loop ────────────────────────────────────────────────

    losses: List[float] = []
    loss_terms = {
        "photometric": [],
        "normal_prior": [],
        "depth_prior": [],
        "prior_ramp": [],
        "weighted_normal_prior": [],
        "weighted_depth_prior": [],
        "normal_hit_fraction": [],
        "depth_hit_fraction": [],
        "vertex_grad_norm": [],
        "faces": [],
    }
    # Keep one CPU schedule for prior sampling and mirror it to the GPU once
    # per epoch for camera and image indexing.
    view_order_rng = torch.Generator(device="cpu")
    view_order_rng.manual_seed(SEED)
    perm_cpu = torch.randperm(N_train, generator=view_order_rng)
    perm = perm_cpu.to(device=device)
    ptr = 0
    geometry_prior_rng = (
        np.random.default_rng(SEED + 10_003) if use_geometry_prior else None
    )

    pbar = tqdm(range(1, steps + 1), desc="optimising", leave=True)
    for i_iter in pbar:
        end = min(ptr + batch_size, N_train)
        batch_idx_cpu = perm_cpu[ptr:end]
        batch_idx = perm[ptr:end]
        ptr = end
        if ptr == N_train:
            perm_cpu = torch.randperm(N_train, generator=view_order_rng)
            perm = perm_cpu.to(device=device)
            ptr = 0

        # LR schedule
        mult = exp_decay_mult(
            min(i_iter, schedule_steps), schedule_steps, final_mult=0.01,
        )
        for opt, base_lrs in (
            (optimizer_soup, base_soup_lrs),
            (optimizer_vert, base_vert_lrs),
            (optimizer_shader, base_shader_lrs),
        ):
            for pg, base_lr in zip(opt.param_groups, base_lrs):
                pg["lr"] = base_lr * mult

        batch_MVP = MVPs[batch_idx]
        batch_MVP_inv = MVPs_inv[batch_idx]
        batch_gt_rgb = gt_rgb[batch_idx]

        V_clip = project_vertices(V_single, batch_MVP)
        alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
        feat_acc = ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid()

        rast_out, fragments = ds.rasterize_multires_triangle_alpha(
            (H, W), V_clip, F, level=Rmax, alpha_src=alpha_acc,
            return_fragments=True,
        )
        feat = ds.multires_triangle_color(
            rast_out, level=Rmax, feat=feat_acc,
        ).view(-1, H, W, feat_dim)
        view_feat = ds.encode_view_dir_sh2(rast_out, batch_MVP_inv)
        color = color_mlp.forward(
            feat, mask=rast_out[..., -1] > 0, extra_features=view_feat,
        ).view(-1, H, W, 3)

        # Opacity auxiliary loss (zero-valued; hooks gradient into alpha_src)
        aux_loss = ds.opacity_aux_loss(
            color.detach(), batch_gt_rgb, rast_out, V_clip, F,
            level=Rmax, alpha_src=alpha_acc, fragments=fragments,
        )
        color = ds.edge_grad(color, rast_out, V_clip, F)
        l1_loss = (batch_gt_rgb - color).abs().mean()
        ssim_loss = 0.5 * (1 - ssim(
            color.permute(0, 3, 1, 2), batch_gt_rgb.permute(0, 3, 1, 2),
        ))
        photometric_loss = aux_loss + 0.8 * l1_loss + 0.2 * ssim_loss

        if not use_geometry_prior or i_iter < prior_start:
            prior_ramp = 0.0
        elif prior_ramp_steps == 0:
            prior_ramp = 1.0
        else:
            prior_ramp = min(
                1.0, (i_iter - prior_start + 1) / prior_ramp_steps,
            )

        normal_prior_loss = None
        depth_prior_loss = None
        prior_samples = None
        expected_surface = None
        normal_hit_fraction = 0.0
        depth_hit_fraction = 0.0
        if use_geometry_prior and prior_ramp > 0:
            assert prior_store is not None
            assert geometry_prior_rng is not None
            assert Tcws is not None
            prior_samples = prior_store.sample_joint_uniform(
                batch_idx_cpu,
                prior_samples_per_view,
                geometry_prior_rng,
                device=device,
                dtype=V_single.dtype,
            )
            batch_Tcw = Tcws[batch_idx]
            expected_surface = ds.vertex_expected_surface_samples(
                V_single,
                F,
                fragments.frag_pix,
                fragments.frag_attrs,
                fragments.frag_alpha,
                K,
                batch_Tcw,
                prior_samples.pixels_b_y_x,
                (H, W),
            )
            del batch_Tcw
            if normal_prior_weight > 0:
                normal_prior_loss = ds.normal_prior_loss(
                    expected_surface,
                    prior_samples.normal_camera,
                    prior_samples.normal_valid,
                )
                normal_count = int(prior_samples.normal_valid.sum().item())
                if normal_count > 0:
                    normal_hit_fraction = float(
                        (
                            expected_surface.valid
                            & prior_samples.normal_valid
                        ).sum().detach().item() / normal_count
                    )
            if depth_prior_weight > 0:
                depth_prior_loss = ds.inverse_depth_prior_loss(
                    expected_surface,
                    prior_samples.inverse_camera_z,
                    prior_samples.depth_valid,
                )
                depth_count = int(prior_samples.depth_valid.sum().item())
                if depth_count > 0:
                    depth_hit_fraction = float(
                        (
                            expected_surface.valid
                            & prior_samples.depth_valid
                        ).sum().detach().item() / depth_count
                    )
        del fragments

        loss = photometric_loss
        if normal_prior_loss is not None:
            loss = loss + prior_ramp * normal_prior_weight * normal_prior_loss
        if depth_prior_loss is not None:
            loss = loss + prior_ramp * depth_prior_weight * depth_prior_loss

        loss.backward()
        vertex_grad_norm = (
            float(V_single.grad.detach().norm().item())
            if V_single.grad is not None else 0.0
        )
        optimizer_soup.step()
        optimizer_vert.step()
        optimizer_shader.step()

        # Gradients are not used after the update.  Releasing them here keeps
        # the large post-lift source gradients out of the next forward pass
        # and, most importantly, out of lift/resampling allocations.
        optimizer_soup.zero_grad(set_to_none=True)
        optimizer_vert.zero_grad(set_to_none=True)
        optimizer_shader.zero_grad(set_to_none=True)

        l = float(loss.detach().item())
        losses.append(l)
        photo_value = float(photometric_loss.detach().item())
        normal_value = (
            float(normal_prior_loss.detach().item())
            if normal_prior_loss is not None else 0.0
        )
        depth_value = (
            float(depth_prior_loss.detach().item())
            if depth_prior_loss is not None else 0.0
        )
        loss_terms["photometric"].append(photo_value)
        loss_terms["normal_prior"].append(normal_value)
        loss_terms["depth_prior"].append(depth_value)
        loss_terms["prior_ramp"].append(prior_ramp)
        loss_terms["weighted_normal_prior"].append(
            prior_ramp * normal_prior_weight * normal_value
        )
        loss_terms["weighted_depth_prior"].append(
            prior_ramp * depth_prior_weight * depth_value
        )
        loss_terms["normal_hit_fraction"].append(normal_hit_fraction)
        loss_terms["depth_hit_fraction"].append(depth_hit_fraction)
        loss_terms["vertex_grad_norm"].append(vertex_grad_norm)
        loss_terms["faces"].append(int(F.shape[0]))
        pbar.set_postfix(
            loss=f"{l:.6f}",
            Ln=f"{normal_value:.4f}",
            Ld=f"{depth_value:.4f}",
            hit=f"{max(normal_hit_fraction, depth_hit_fraction):.3f}",
        )

        # Python function scope otherwise retains these outputs until the next
        # assignment, overlapping them with topology-changing allocations.
        del batch_MVP, batch_MVP_inv, batch_gt_rgb
        del V_clip, alpha_acc, feat_acc, rast_out, feat, view_feat, color
        del aux_loss, l1_loss, ssim_loss, photometric_loss, loss
        if normal_prior_loss is not None:
            del normal_prior_loss
        if depth_prior_loss is not None:
            del depth_prior_loss
        if prior_samples is not None:
            del prior_samples, expected_surface

        # ── Lift multi-resolution levels at step 5 000 ───────────────
        if i_iter == 5_000 and i_iter < steps:
            old_lr_feat = optimizer_soup.param_groups[0]["lr"]
            old_lr_alpha = optimizer_soup.param_groups[1]["lr"]
            old_lr_vert = optimizer_vert.param_groups[0]["lr"]
            del optimizer_soup, optimizer_vert

            with torch.no_grad():
                feat_src_lifted = ds.accumulate_to_level(0, 0, feat_src, target_level=2)
                Rmin, Rmax = 2, 5
                new_feat_src = ds.build_multires_triangle_color(
                    F.shape[0], Rmin, Rmax, feat_dim=feat_dim,
                ).to(device="cuda")
                new_feat_src[..., : feat_src_lifted.shape[1], :] = feat_src_lifted
                feat_src = new_feat_src

                alpha_src = ds.build_multires_triangle_color(
                    F.shape[0], Rmin, Rmax, feat_dim=1,
                ).to(device="cuda")

            feat_src.requires_grad = True
            alpha_src.requires_grad = True
            del feat_src_lifted, new_feat_src

            optimizer_soup = Adam([
                {"params": [feat_src], "lr": old_lr_feat},
                {"params": [alpha_src], "lr": old_lr_alpha},
            ])
            optimizer_vert = ds.optimize.VectorAdam(params=[V_single], lr=old_lr_vert)

        # ── Resample soup ────────────────────────────────────────────
        if i_iter % 100 == 0 and i_iter < 9_550:
            old_lr_feat = optimizer_soup.param_groups[0]["lr"]
            old_lr_alpha = optimizer_soup.param_groups[1]["lr"]
            old_lr_vert = optimizer_vert.param_groups[0]["lr"]
            del optimizer_soup, optimizer_vert

            with torch.no_grad():
                source_feat = feat_src
                source_alpha = alpha_src
                parent_map = torch.arange(F.shape[0], device=F.device)
                alpha_acc = None
                topology_changed = False

                if F.shape[0] > target_faces:
                    alpha_acc = ds.accumulate_to_level(
                        Rmin, Rmax, source_alpha,
                    ).sigmoid()
                    tri_counts = count_visible_triangles(
                        (H // 2, W // 2), MVPs, V_single, F,
                        level=Rmax, alpha_src=alpha_acc, batch_size=1,
                    )
                    keep_map = build_keep_map(
                        tri_counts, remove=F.shape[0] - target_faces,
                    )
                    F = F[keep_map]
                    V_single, F = ds.remove_unreferenced_vertices_from_soup(V_single, F)
                    parent_map = parent_map[keep_map]
                    alpha_acc = alpha_acc[keep_map]
                    topology_changed = True

                if i_iter < 9_500:
                    if alpha_acc is None:
                        alpha_acc = ds.accumulate_to_level(
                            Rmin, Rmax, source_alpha,
                        ).sigmoid()
                    V_single, F, face_map = split_edges_from_training_views(
                        (H // 2, W // 2), MVPs, V_single, F,
                        Rmax, alpha_acc,
                        tau_ratio=1 / 5, num_views_cap=20,
                    )
                    parent_map = parent_map[face_map]
                    topology_changed = True

                if topology_changed:
                    feat_src = ds.expand_by_index(source_feat, parent_map)
                    alpha_src = ds.expand_by_index(source_alpha, parent_map)

            del source_feat, source_alpha, parent_map, alpha_acc, topology_changed

            print(f"  [resample] verts={V_single.shape[0]:,}  faces={F.shape[0]:,}")

            V_single.requires_grad = True
            feat_src.requires_grad = True
            alpha_src.requires_grad = True

            optimizer_soup = Adam([
                {"params": [feat_src], "lr": old_lr_feat},
                {"params": [alpha_src], "lr": old_lr_alpha},
            ])
            optimizer_vert = ds.optimize.VectorAdam(params=[V_single], lr=old_lr_vert)

            # Resampling briefly allocates buffers close to the VRAM limit at
            # Rmax=5. Release those cached blocks before the next iteration so
            # WDDM does not page subsequent training allocations.
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()

    # ── Final renders ────────────────────────────────────────────────

    with torch.no_grad():
        for j in range(eval_MVP.shape[0]):
            V_clip = project_vertices(V_single, eval_MVP[j : j + 1])
            rast_out = ds.rasterize_multires_triangle_alpha(
                (H, W), V_clip, F,
                level=Rmax,
                alpha_src=ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid(),
                stochastic=False,
            )
            feat = ds.multires_triangle_color(
                rast_out, level=Rmax,
                feat=ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid(),
            ).view(-1, H, W, feat_dim)
            view_feat = ds.encode_view_dir_sh2(rast_out, eval_MVP_inv[j : j + 1])
            color = color_mlp.forward(
                feat, mask=rast_out[..., -1] > 0, extra_features=view_feat,
            ).view(1, H, W, 3)
            iio.imwrite(
                os.path.join(out_dir, f"final_pred_{j}.png"),
                (color.squeeze(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8),
            )
    print(f"[save] final renders → {out_dir}/final_pred_*.png")

    # ── Loss curve ───────────────────────────────────────────────────

    plt.figure()
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.xlabel("step"); plt.ylabel("loss"); plt.title("Training Loss")
    plt.grid(True, alpha=0.2)
    loss_png = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_png, bbox_inches="tight"); plt.close()
    print(f"[save] loss curve → {loss_png}")
    if use_geometry_prior:
        plt.figure()
        plt.plot(
            np.arange(1, len(losses) + 1),
            loss_terms["normal_prior"],
            label="L_n",
        )
        plt.plot(
            np.arange(1, len(losses) + 1),
            loss_terms["depth_prior"],
            label="L_d",
        )
        plt.xlabel("step"); plt.ylabel("raw prior loss")
        plt.title("Vertex-only Geometry Priors")
        plt.grid(True, alpha=0.2); plt.legend()
        geometry_loss_png = os.path.join(out_dir, "geometry_loss_curve.png")
        plt.savefig(geometry_loss_png, bbox_inches="tight"); plt.close()
        print(f"[save] geometry loss curve → {geometry_loss_png}")

    # ── Checkpoint ───────────────────────────────────────────────────

    with torch.no_grad():
        alpha_acc = ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid()
        feat_acc = ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid()

    ckpt_path = os.path.join(out_dir, "final_params.pt")
    torch.save({
        "feat_acc": feat_acc.detach().cpu(),
        "alpha_acc": alpha_acc.detach().cpu(),
        "V": V_single.detach().cpu(),
        "F": F.detach().cpu(),
        "color_mlp": color_mlp.state_dict(),
        "Rmin": Rmin,
        "Rmax": Rmax,
        "feat_dim": feat_dim,
        "K": K.detach().cpu(), "H": H, "W": W,
        "flip_z": flip_z,
        "steps": steps,
        "schedule_steps": schedule_steps,
        "losses": losses,
        "loss_terms": loss_terms,
        "geometry_prior": {
            "root": prior_root,
            "downscale": downscale,
            "normal_weight": normal_prior_weight,
            "depth_weight": depth_prior_weight,
            "start": prior_start,
            "ramp_steps": prior_ramp_steps,
            "samples_per_view": prior_samples_per_view,
        },
        "seed": SEED,
    }, ckpt_path)
    print(f"[save] checkpoint → {ckpt_path}")

    # ── Test evaluation ──────────────────────────────────────────────

    test_data = load_mipnerf360_scene(
        scene_root, split="test", downscale=downscale, device=device,
    )
    test_frames = test_data["frames"]
    print(f"[views] test={len(test_frames)}")

    out_dir_test = os.path.join(out_dir, "test_views")
    os.makedirs(out_dir_test, exist_ok=True)

    psnrs, ssims = [], []

    with torch.no_grad():
        for i, fr in enumerate(test_frames):
            MVP = mvp_from_K_Tcw(
                K, fr["Tcw"], (H, W), z_near=z_near_test, z_far=z_far, flip_z=flip_z,
            ).unsqueeze(0)
            MVP_inv = torch.inverse(MVP).contiguous()
            V_clip = project_vertices(V_single, MVP)

            rast_out = ds.rasterize_multires_triangle_alpha(
                (H, W), V_clip, F,
                level=Rmax,
                alpha_src=ds.accumulate_to_level(Rmin, Rmax, alpha_src).sigmoid(),
                stochastic=False,
            )
            feat = ds.multires_triangle_color(
                rast_out, level=Rmax,
                feat=ds.accumulate_to_level(Rmin, Rmax, feat_src).sigmoid(),
            ).view(-1, H, W, feat_dim)
            view_feat = ds.encode_view_dir_sh2(rast_out, MVP_inv)
            color = color_mlp.forward(
                feat, mask=rast_out[..., -1] > 0, extra_features=view_feat,
            ).view(1, H, W, 3)

            pred_lin = color.squeeze(0).clamp(0, 1)
            gt_lin = fr["image"].clamp(0, 1)

            psnrs.append(float(psnr_fn(pred_lin, gt_lin).item()))
            pred_nchw = pred_lin.permute(2, 0, 1).unsqueeze(0)
            gt_nchw = gt_lin.permute(2, 0, 1).unsqueeze(0)
            ssims.append(float(ssim(pred_nchw, gt_nchw).item()))

            stem = os.path.splitext(os.path.basename(fr["img_path"]))[0]
            iio.imwrite(
                os.path.join(out_dir_test, f"{i:04d}_{stem}_pred.png"),
                (pred_lin.cpu().numpy() * 255).astype(np.uint8),
            )
            iio.imwrite(
                os.path.join(out_dir_test, f"{i:04d}_{stem}_gt.png"),
                (gt_lin.cpu().numpy() * 255).astype(np.uint8),
            )

    print(f"[save] test renders → {out_dir_test}/")

    if psnrs:
        print(f"[metrics] PSNR  {np.mean(psnrs):.3f} dB")
        print(f"[metrics] SSIM  {np.mean(ssims):.4f}")

        with open(os.path.join(out_dir_test, "metrics.txt"), "w") as f:
            for i, (p, s) in enumerate(zip(psnrs, ssims)):
                f.write(f"{i:04d} PSNR={p:.3f} SSIM={s:.4f}\n")
            f.write(f"\nmean PSNR={np.mean(psnrs):.3f} SSIM={np.mean(ssims):.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffSoup MipNeRF-360 example")
    parser.add_argument("--scene_root", type=str, default="./datasets/360_v2/kitchen")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument(
        "--schedule_steps", type=int, default=10_000,
        help="Reference horizon for LR decay; keep 10000 for comparable short smokes",
    )
    parser.add_argument("--n_points", type=int, default=15_000)
    parser.add_argument("--downscale", type=int, default=4, choices=[0, 1, 2, 4, 8],
                        help="Image downscale factor: 0/1=images/, 2=images_2/, 4=images_4/, 8=images_8/")
    parser.add_argument(
        "--flip_z", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: ./results/02_mip360/<scene>)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Allow writing into a nonempty output directory",
    )
    parser.add_argument(
        "--prior_root", type=str, default=None,
        help="Prior scene root; defaults to --scene_root when a prior loss is enabled.",
    )
    parser.add_argument("--normal_prior_weight", type=float, default=0.0)
    parser.add_argument("--depth_prior_weight", type=float, default=0.0)
    parser.add_argument("--prior_start", type=int, default=5_501)
    parser.add_argument("--prior_ramp_steps", type=int, default=500)
    parser.add_argument("--prior_samples_per_view", type=int, default=16_384)
    args = parser.parse_args()

    main(
        scene_root=args.scene_root,
        batch_size=args.batch_size,
        steps=args.steps,
        schedule_steps=args.schedule_steps,
        n_points=args.n_points,
        downscale=args.downscale,
        flip_z=args.flip_z,
        out_dir=args.out_dir,
        overwrite=args.overwrite,
        prior_root=args.prior_root,
        normal_prior_weight=args.normal_prior_weight,
        depth_prior_weight=args.depth_prior_weight,
        prior_start=args.prior_start,
        prior_ramp_steps=args.prior_ramp_steps,
        prior_samples_per_view=args.prior_samples_per_view,
    )
