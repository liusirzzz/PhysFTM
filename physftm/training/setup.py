"""Setup builders for the PhysFTM training entrypoint."""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time as _time
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from physftm.rkhs.basis import RKHSFunctionalTucker2D, RKHSFunctionalTucker3D
from physftm.rkhs.losses import build_sparse_observation_index
from physftm.distributed import is_main_process
from physftm.checkpointing.spatial import build_spatial_method_tag
from physftm.data.normalization import (
    apply_saved_normalization,
    load_or_compute_channelwise_zscore_stats,
)
from physftm.data.splits import resolve_split_paths
from physftm.pde.training import _normalize_pde_point_sample_mode, pde_channel_order
from physftm.training.runtime import set_requires_grad
from physftm.training.configs import load_rkhs_runtime_config
from physftm.training.wrapper import RKHSFTMWrapper
from physftm.data.axes import ordered_spatial_values, spatial_axis_order_for_dataset
from physftm.data.loading import align_3d_mask_to_data, load_large_data


def set_seed(seed=123):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


class TeeStdout:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            (getattr(stream, "isatty", lambda: False)() for stream in self.streams)
        )


@dataclass
class ExperimentContext:
    experiment_root_dir: str
    latest_save_dir: str
    best_save_dir: str
    log_file_handle: object
    original_stdout: object
    original_stderr: object


@dataclass
class DataContext:
    train_data_path: str
    train_metadata_path: str
    val_data_path: str
    val_metadata_path: str
    manifest: object
    dataset_pde_context: object
    ind_input: tuple
    inducing_points: tuple
    t_ind_uni: torch.Tensor
    train_loader: DataLoader
    val_loader: DataLoader
    train_sampler: object
    val_sampler: object
    norm_stats: dict
    num_channels: int
    shared_basis_multichannel: bool
    obs_index: object
    train_num_samples: int
    train_num_frames: int
    spatial_dim: int
    spatial_axis_order: tuple[str, ...]
    t_coords_physical: object


@dataclass
class ModelContext:
    model: torch.nn.Module
    basis_function: torch.nn.Module
    optimizer: object
    core_adam_lr: float
    chol_t: object
    rank: tuple
    use_rkhs_basis: bool
    use_basis_solve: bool
    use_core_adam: bool
    use_core_cg: bool
    k_inv_t: object
    k_inv_u: object
    k_inv_v: object
    k_inv_w: object


@dataclass
class PdeContext:
    lambda_pde_point: float
    use_pde_loss: bool
    use_pde_monitor: bool
    use_pde_module: bool
    pde_kwargs: dict
    pde_channel_order: object
    pde_point_sample_mode: str


def load_rkhs_yaml_config(config_path):
    return load_rkhs_runtime_config(config_path)


def parse_rkhs_config_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PhysFTM from a YAML config file."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a grouped YAML config file."
    )
    args = parser.parse_args(argv)
    return load_rkhs_yaml_config(args.config)


def _select_uniform_inducing_points(coord, requested_count, *, min_count, axis_name):
    coord_t = torch.as_tensor(coord, dtype=torch.float32).reshape(-1)
    available = int(coord_t.numel())
    if available <= 0:
        raise ValueError(
            f"Cannot select inducing points from empty {axis_name}-axis coordinates."
        )
    if requested_count is None:
        count = available
    else:
        count = int(requested_count)
        if count <= 0:
            raise ValueError(
                f"rkhs.num_inducing_{axis_name} must be positive or null, got {requested_count}"
            )
        if count > available:
            raise ValueError(
                f"rkhs.num_inducing_{axis_name}={count} exceeds the {available} available {axis_name}-axis coordinates."
            )
    if count < int(min_count):
        raise ValueError(
            f"rkhs.num_inducing_{axis_name} resolves to {count}, but the corresponding Tucker rank is {int(min_count)}."
        )
    if count == available:
        return coord_t
    indices = (
        torch.linspace(
            0, available - 1, steps=count, dtype=torch.float64, device=coord_t.device
        )
        .round()
        .long()
    )
    if int(torch.unique(indices).numel()) != count:
        raise RuntimeError(
            f"Uniform {axis_name}-axis inducing-point selection produced duplicate indices."
        )
    return coord_t.index_select(0, indices)


def _validate_lengthscale(value, *, axis_name):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"rkhs.lengthscale_{axis_name} must be a positive finite value or null, got {value}"
        )
    return value


def _validate_rkhs_kernel_config(config):
    return {
        "u": _validate_lengthscale(config.lengthscale_u, axis_name="u"),
        "v": _validate_lengthscale(config.lengthscale_v, axis_name="v"),
        "w": _validate_lengthscale(config.lengthscale_w, axis_name="w"),
    }


def setup_experiment_logging(config, ddp_state, device):
    experiment_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_method_tag = build_spatial_method_tag(config)
    experiment_tag = f"{config.data_name}_{experiment_method_tag}_{config.R[0]}x{config.R[1]}x{config.R[2]}_{experiment_timestamp}"
    experiment_root_dir = os.path.join(".", "artifacts", experiment_tag)
    latest_save_dir = os.path.join(experiment_root_dir, "latest")
    best_save_dir = os.path.join(experiment_root_dir, "best")
    log_file_handle = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if is_main_process(ddp_state):
        os.makedirs(experiment_root_dir, exist_ok=True)
        source_config_path = getattr(config, "source_config_path", None)
        copied_config_path = os.path.join(experiment_root_dir, "config.yaml")
        if source_config_path and os.path.exists(source_config_path):
            shutil.copy2(source_config_path, copied_config_path)
        resolved_config_path = os.path.join(experiment_root_dir, "resolved_config.json")
        if hasattr(config, "to_nested_dict"):
            with open(resolved_config_path, "w", encoding="utf-8") as f:
                json.dump(config.to_nested_dict(), f, indent=2)
        log_path = os.path.join(experiment_root_dir, "train.log")
        log_file_handle = open(log_path, "a", encoding="utf-8", buffering=1)
        tee_stdout = TeeStdout(original_stdout, log_file_handle)
        sys.stdout = tee_stdout
        sys.stderr = tee_stdout
        print("device:", device)
        print("artifact root:", experiment_root_dir)
        print("log path:", log_path)
        if source_config_path:
            print("source config:", source_config_path)
            print("copied config:", copied_config_path)
        print("resolved config:", resolved_config_path)
    return ExperimentContext(
        experiment_root_dir=experiment_root_dir,
        latest_save_dir=latest_save_dir,
        best_save_dir=best_save_dir,
        log_file_handle=log_file_handle,
        original_stdout=original_stdout,
        original_stderr=original_stderr,
    )


def close_experiment_logging(experiment_ctx):
    if experiment_ctx is not None and experiment_ctx.log_file_handle is not None:
        sys.stdout = experiment_ctx.original_stdout
        sys.stderr = experiment_ctx.original_stderr
        experiment_ctx.log_file_handle.close()


def build_data_context(config, ddp_state, device):
    _validate_rkhs_kernel_config(config)
    train_data_path, train_metadata_path, manifest = resolve_split_paths(
        data_manifest_path=config.data_manifest_path or None,
        split="train",
        data_path=config.data_path,
        metadata_path=config.metadata_path,
    )
    val_data_path, val_metadata_path, _ = resolve_split_paths(
        data_manifest_path=config.data_manifest_path or None,
        split="val",
        data_path=config.data_path,
        metadata_path=config.metadata_path,
    )
    dataset_pde_context = None
    if is_main_process(ddp_state):
        print(
            f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始加载训练数据 (load_large_data)...",
            flush=True,
        )
    t0 = _time.time()
    ind_uni, data_extract, mask_tr = load_large_data(
        train_data_path, train_metadata_path
    )
    metadata_payload = np.load(train_metadata_path, allow_pickle=True).item()["data"]
    t_coords_physical = np.asarray(
        metadata_payload.get("t_coords_physical", metadata_payload["t_ind_uni"]),
        dtype=np.float32,
    ).reshape(-1)
    is_3d_data = int(np.asarray(ind_uni[2]).size) > 1
    if is_3d_data and np.asarray(data_extract).ndim == 5:
        data_extract = np.asarray(data_extract)[..., None]
    if is_main_process(ddp_state):
        t1 = _time.time()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 训练数据加载完成！耗时: {t1 - t0:.2f}s, 形状: {data_extract.shape}",
            flush=True,
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 开始加载验证数据...", flush=True
        )
    _, val_data_extract, val_mask_tr = load_large_data(val_data_path, val_metadata_path)
    if is_3d_data and np.asarray(val_data_extract).ndim == 5:
        val_data_extract = np.asarray(val_data_extract)[..., None]
    if is_main_process(ddp_state):
        t2 = _time.time()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 验证数据加载完成！耗时: {t2 - t1:.2f}s, 形状: {val_data_extract.shape}",
            flush=True,
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 开始执行 normalize_data...",
            flush=True,
        )
    data_extract, norm_stats = load_or_compute_channelwise_zscore_stats(
        data_extract,
        train_data_path=train_data_path,
        train_metadata_path=train_metadata_path,
        data_manifest_path=config.data_manifest_path or None,
        dataset_type=config.dataset_type,
        data_name=config.data_name,
    )
    train_min_after_norm = float(np.min(data_extract))
    train_max_after_norm = float(np.max(data_extract))
    if is_main_process(ddp_state):
        t3 = _time.time()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] normalize_data 完成！耗时: {t3 - t2:.2f}s | dtype={data_extract.dtype} | train_range=[{train_min_after_norm:.6g}, {train_max_after_norm:.6g}] | norm_cache={('loaded' if bool(norm_stats.get('norm_cache_loaded', False)) else 'computed')} path={norm_stats.get('norm_cache_path', '-')}",
            flush=True,
        )
    val_data_extract = apply_saved_normalization(val_data_extract, norm_stats)
    val_min_after_norm = float(np.min(val_data_extract))
    val_max_after_norm = float(np.max(val_data_extract))
    if is_main_process(ddp_state):
        t4 = _time.time()
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 验证集归一化完成！耗时: {t4 - t3:.2f}s | val_range=[{val_min_after_norm:.6g}, {val_max_after_norm:.6g}] | train_frames={(data_extract.shape[1] if data_extract.ndim >= 2 else 'NA')}",
            flush=True,
        )
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] 开始进行 PyTorch Tensor 转换以及准备 DataLoader...",
            flush=True,
        )
    u_ind_uni = torch.as_tensor(ind_uni[0], dtype=torch.float32, device=device)
    v_ind_uni = torch.as_tensor(ind_uni[1], dtype=torch.float32, device=device)
    w_ind_uni = torch.as_tensor(ind_uni[2], dtype=torch.float32, device=device)
    t_ind_uni = torch.as_tensor(ind_uni[3], dtype=torch.float32, device=device)
    print(f"config.R: {config.R}")
    data = torch.as_tensor(data_extract, dtype=torch.float32)
    val_data = torch.as_tensor(val_data_extract, dtype=torch.float32)
    mask_tr = torch.as_tensor(mask_tr, dtype=torch.float32)
    val_mask_tr = torch.as_tensor(val_mask_tr, dtype=torch.float32)
    if data.dim() == 4:
        data = data.unsqueeze(-1)
    if val_data.dim() == 4:
        val_data = val_data.unsqueeze(-1)
    spatial_dim = 3 if int(w_ind_uni.numel()) > 1 else 2
    if spatial_dim == 3:
        if data.dim() == 5:
            data = data.unsqueeze(-1)
        if val_data.dim() == 5:
            val_data = val_data.unsqueeze(-1)
        if data.dim() != 6 or val_data.dim() != 6:
            raise ValueError(
                f"3D data must be (N,T,D,H,W[,C]), got {tuple(data.shape)}"
            )
        mask_tr = align_3d_mask_to_data(mask_tr, data)
        val_mask_tr = align_3d_mask_to_data(val_mask_tr, val_data)
    num_channels = int(data.shape[-1])
    shared_basis_multichannel = num_channels > 1
    if shared_basis_multichannel:
        expected_dim = 6 if spatial_dim == 3 else 5
        if data.dim() != expected_dim or val_data.dim() != expected_dim:
            raise ValueError(
                "shared_basis_multichannel data has an invalid spatial/channel layout."
            )
        if num_channels <= 1:
            raise ValueError("shared_basis_multichannel requires data.shape[-1] > 1.")
        if spatial_dim == 2 and (not False) and (int(config.R[2]) != 1):
            raise ValueError(
                "shared_basis_multichannel per-time 2D path requires R[2] == 1."
            )
    spatial_axis_order = spatial_axis_order_for_dataset(
        config.dataset_type, spatial_dim
    )
    spatial_coords = ordered_spatial_values(
        spatial_axis_order,
        u=u_ind_uni,
        v=v_ind_uni,
        w=w_ind_uni if spatial_dim == 3 else None,
    )
    ind_input = spatial_coords
    if spatial_dim == 3:
        coord_by_axis = {"u": u_ind_uni, "v": v_ind_uni, "w": w_ind_uni}
        inducing_count_by_axis = {
            "u": config.num_inducing_u,
            "v": config.num_inducing_v,
            "w": getattr(config, "num_inducing_w", None),
        }
        inducing_points = tuple(
            (
                _select_uniform_inducing_points(
                    coord_by_axis[axis],
                    inducing_count_by_axis[axis],
                    min_count=int(config.R[factor_idx]),
                    axis_name=axis,
                )
                for factor_idx, axis in enumerate(spatial_axis_order)
            )
        )
    else:
        inducing_points = (
            _select_uniform_inducing_points(
                u_ind_uni,
                config.num_inducing_u,
                min_count=int(config.R[0]),
                axis_name="u",
            ),
            _select_uniform_inducing_points(
                v_ind_uni,
                config.num_inducing_v,
                min_count=int(config.R[1]),
                axis_name="v",
            ),
        )
    if is_main_process(ddp_state):
        full_counts = tuple(
            (int(torch.as_tensor(coord).numel()) for coord in ind_input)
        )
        inducing_counts = tuple(
            (int(torch.as_tensor(coord).numel()) for coord in inducing_points)
        )
        print(
            f"shared_basis_multichannel: {shared_basis_multichannel} | num_channels: {num_channels} | core_parameterization={'per_time'} spatial_dim={spatial_dim} | spatial_axis_order={spatial_axis_order} | coordinate_counts={full_counts} | inducing_counts={inducing_counts}"
        )
    batch_ind = torch.arange(data.size(0), dtype=torch.long)
    val_batch_ind = torch.arange(val_data.size(0), dtype=torch.long)
    train_dataset = TensorDataset(data, mask_tr, batch_ind)
    val_dataset = TensorDataset(val_data, val_mask_tr, val_batch_ind)
    seed = int(getattr(config, "seed", 123))
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True, seed=seed)
        if ddp_state["distributed"]
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False, seed=seed)
        if ddp_state["distributed"]
        else None
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=False,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=False,
    )
    with torch.no_grad():
        if mask_tr.dim() == 5:
            representative_mask = mask_tr[0]
        else:
            representative_mask = mask_tr
        obs_index = build_sparse_observation_index(representative_mask)
    return DataContext(
        train_data_path=train_data_path,
        train_metadata_path=train_metadata_path,
        val_data_path=val_data_path,
        val_metadata_path=val_metadata_path,
        manifest=manifest,
        dataset_pde_context=dataset_pde_context,
        ind_input=ind_input,
        inducing_points=inducing_points,
        t_ind_uni=t_ind_uni,
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        norm_stats=norm_stats,
        num_channels=num_channels,
        shared_basis_multichannel=shared_basis_multichannel,
        obs_index=obs_index,
        train_num_samples=int(data.size(0)),
        train_num_frames=int(data.size(1)),
        spatial_dim=spatial_dim,
        spatial_axis_order=spatial_axis_order,
        t_coords_physical=t_coords_physical,
    )


def _build_temporal_cholesky(t_ind_uni, device):
    with torch.no_grad():
        t = torch.as_tensor(t_ind_uni, dtype=torch.float32, device=device).flatten()
        span = float(t.max() - t.min() + 1e-06)
        lengthscale_t = max(span / max(t.numel() - 1, 1), 0.01)
        sqdist_t = (t[:, None] - t[None, :]).pow(2)
        Kt = torch.exp(-0.5 * sqdist_t / lengthscale_t**2)
        Kt = Kt + 1e-06 * torch.eye(t.numel(), device=device, dtype=Kt.dtype)
        return torch.linalg.cholesky(Kt)


def build_model_context(config, data_ctx, ddp_state, device):
    chol_t = None
    chol_t = _build_temporal_cholesky(data_ctx.t_ind_uni, device)
    rank = tuple(config.R)
    use_rkhs_basis = True
    lengthscale_by_axis = {
        "u": config.lengthscale_u,
        "v": config.lengthscale_v,
        "w": config.lengthscale_w,
    }
    basis_function = (
        RKHSFunctionalTucker3D(
            ranks=rank[:3],
            inducing_points=data_ctx.inducing_points,
            lengthscales=tuple(
                (lengthscale_by_axis[axis] for axis in data_ctx.spatial_axis_order)
            ),
            kernel_type="matern32",
            coef_scale=config.coef_scale,
        )
        if data_ctx.spatial_dim == 3
        else RKHSFunctionalTucker2D(
            ranks=rank[:2],
            inducing_points=data_ctx.inducing_points,
            lengthscales=(config.lengthscale_u, config.lengthscale_v),
            kernel_type="matern32",
            coef_scale=config.coef_scale,
        )
    ).to(device)
    if is_main_process(ddp_state):
        effective_lengthscales = tuple(
            (
                float(mode.kernel.log_lengthscale.detach().exp().cpu())
                for mode in (
                    (
                        basis_function.mode_u,
                        basis_function.mode_v,
                        basis_function.mode_w,
                    )
                    if data_ctx.spatial_dim == 3
                    else (basis_function.mode_u, basis_function.mode_v)
                )
            )
        )
        print(
            f"basis: {'RKHS'} | kernel={'matern32'} lengthscales={effective_lengthscales} query_chunk={int(config.pde_query_chunk_size)}"
        )
    use_basis_solve = str(config.basis_update_mode).lower() == "solve"
    if use_basis_solve:
        if data_ctx.spatial_dim == 3:
            raise ValueError(
                "basis_update_mode=solve is not implemented for 3D RKHS; use basis_update_mode=adam."
            )
        if not data_ctx.shared_basis_multichannel and data_ctx.num_channels != 1:
            raise ValueError(
                "basis_update_mode=solve is only implemented for 2D scalar fields or shared-basis multichannel 2D data."
            )
    if is_main_process(ddp_state):
        print(
            f"spatial basis update: mode={config.basis_update_mode} value_ridge={float(config.basis_solve_value_ridge):.2e} target_clip={float(config.basis_solve_target_clip):.2f}"
        )
    core_update_mode = str(config.core_update_mode).lower()
    use_core_adam = core_update_mode == "adam"
    use_core_cg = core_update_mode == "cg"
    core_table_device = torch.device("cpu") if use_core_adam else device
    if data_ctx.spatial_dim == 3 and data_ctx.shared_basis_multichannel:
        tucker_core_init = (
            torch.ones(
                data_ctx.train_num_samples,
                data_ctx.train_num_frames,
                data_ctx.num_channels,
                rank[0],
                rank[1],
                rank[2],
                device=core_table_device,
            )
            / 2
        )
    elif data_ctx.spatial_dim == 3:
        tucker_core_init = (
            torch.ones(
                data_ctx.train_num_samples,
                data_ctx.train_num_frames,
                rank[0],
                rank[1],
                rank[2],
                device=core_table_device,
            )
            / 2
        )
    elif data_ctx.shared_basis_multichannel:
        tucker_core_init = (
            torch.ones(
                data_ctx.train_num_samples,
                data_ctx.train_num_frames,
                data_ctx.num_channels,
                rank[0],
                rank[1],
                device=core_table_device,
            )
            / 2
        )
    else:
        tucker_core_init = (
            torch.ones(
                data_ctx.train_num_samples,
                data_ctx.train_num_frames,
                rank[0],
                rank[1],
                device=core_table_device,
            )
            / 2
        )
    model = RKHSFTMWrapper(
        basis_function,
        tucker_core_init,
        shared_basis_multichannel=data_ctx.shared_basis_multichannel,
        core_table_on_cpu=use_core_adam,
        core_parameterization="per_time",
    ).to(device)
    model.tucker_core.requires_grad_(False)
    if use_basis_solve:
        set_requires_grad(model.basis_function.mode_u, False)
        set_requires_grad(model.basis_function.mode_v, False)
        if data_ctx.spatial_dim == 3:
            set_requires_grad(model.basis_function.mode_w, False)
    if is_main_process(ddp_state):
        print(
            f"core update: mode={config.core_update_mode} core_lr={(float(config.core_learning_rate) if float(config.core_learning_rate) > 0.0 else float(config.learning_rate)):.2e} table_device={('cpu' if use_core_adam else str(device))} lambda_core_ridge={float(config.lambda_core_ridge):.2e} beta_core_tv={float(config.beta_core_tv or 0.0):.2e} als_core_cg_iter={int(config.als_core_cg_iter)} als_core_cg_tol={float(config.als_core_cg_tol):.2e} als_core_steps={int(config.als_core_steps)}"
        )
        if use_core_adam:
            print(
                "[warn] core_update_mode=adam skips CG core solve; lambda_core_ridge, beta_core_tv, and als_core_* only apply to core_update_mode=cg."
            )
    if ddp_state["distributed"]:
        model = DDP(
            model,
            device_ids=[ddp_state["local_rank"]],
            output_device=ddp_state["local_rank"],
        )
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = (
        optim.AdamW(optimizer_params, lr=config.learning_rate)
        if len(optimizer_params) > 0
        else None
    )
    core_adam_lr = (
        float(config.core_learning_rate)
        if float(config.core_learning_rate) > 0.0
        else float(config.learning_rate)
    )
    k_inv_t = None
    k_inv_u = None
    k_inv_v = None
    k_inv_w = None
    if use_rkhs_basis and (not use_basis_solve) and (optimizer is not None):
        with torch.no_grad():
            k_inv_u = torch.linalg.inv(basis_function.mode_u.kernel.gram()).to(device)
            k_inv_v = torch.linalg.inv(basis_function.mode_v.kernel.gram()).to(device)
            if data_ctx.spatial_dim == 3:
                k_inv_w = torch.linalg.inv(basis_function.mode_w.kernel.gram()).to(
                    device
                )
    return ModelContext(
        model=model,
        basis_function=basis_function,
        optimizer=optimizer,
        core_adam_lr=core_adam_lr,
        chol_t=chol_t,
        rank=rank,
        use_rkhs_basis=use_rkhs_basis,
        use_basis_solve=use_basis_solve,
        use_core_adam=use_core_adam,
        use_core_cg=use_core_cg,
        k_inv_t=k_inv_t,
        k_inv_u=k_inv_u,
        k_inv_v=k_inv_v,
        k_inv_w=k_inv_w,
    )


def build_pde_context(config, data_ctx, ddp_state):
    lambda_pde_point = float(config.lambda_pde_point)
    use_pde_loss = lambda_pde_point > 0.0
    use_pde_monitor = bool(config.pde_monitor_always) and bool(config.pde_monitor_point)
    use_pde_module = bool(use_pde_loss or use_pde_monitor)
    pde_kwargs = {}
    channel_order = pde_channel_order(
        config.dataset_type, num_channels=data_ctx.num_channels
    )
    point_sample_mode = _normalize_pde_point_sample_mode(config.pde_point_sample_mode)
    if use_pde_module:
        if config.dataset_type == "ac":
            pde_dt = 0.01 if config.pde_dt is None else float(config.pde_dt)
            pde_query_dt = (
                0.001 if config.pde_query_dt is None else float(config.pde_query_dt)
            )
            pde_kwargs = dict(
                Lx=1.0 if config.pde_Lx == 4.0 else config.pde_Lx,
                Ly=config.pde_Ly,
                dt=pde_dt,
                query_dt=pde_query_dt,
                ac_lambda=config.pde_ac_lambda,
                epsilon=config.pde_ac_epsilon,
                fd_dx=(
                    None
                    if float(config.pde_ac_fd_dx) <= 0.0
                    else float(config.pde_ac_fd_dx)
                ),
                fd_dy=(
                    None
                    if float(config.pde_ac_fd_dy) <= 0.0
                    else float(config.pde_ac_fd_dy)
                ),
                fd_space_order=int(config.pde_ac_fd_space_order),
                fd_time_order=int(config.pde_ac_fd_time_order),
            )
        elif config.dataset_type == "ns":
            pde_dt = 0.1 if config.pde_dt is None else float(config.pde_dt)
            pde_query_dt = (
                0.01 if config.pde_query_dt is None else float(config.pde_query_dt)
            )
            pde_kwargs = dict(
                Lx=1.0 if config.pde_Lx == 4.0 else config.pde_Lx,
                Ly=1.0 if config.pde_Ly == 1.0 else config.pde_Ly,
                dt=pde_dt,
                query_dt=pde_query_dt,
                nu=float(config.pde_ns_nu),
                forcing_scale=float(config.pde_ns_forcing_scale),
                fd_dx=(
                    None
                    if float(config.pde_ns_fd_dx) <= 0.0
                    else float(config.pde_ns_fd_dx)
                ),
                fd_dy=(
                    None
                    if float(config.pde_ns_fd_dy) <= 0.0
                    else float(config.pde_ns_fd_dy)
                ),
                fd_space_order=int(config.pde_ns_fd_space_order),
                fd_time_order=int(config.pde_ns_fd_time_order),
            )
        elif config.dataset_type == "heat3d":
            if data_ctx.spatial_dim != 3 or data_ctx.shared_basis_multichannel:
                raise ValueError("Heat3D PDE requires scalar 3D per-time data.")
            pde_kwargs = dict(
                Lx=float(config.pde_Lx),
                Ly=float(config.pde_Ly),
                Lz=float(config.pde_Lz),
                dt=None if config.pde_dt is None else float(config.pde_dt),
                query_dt=(
                    None if config.pde_query_dt is None else float(config.pde_query_dt)
                ),
                alpha=float(config.pde_heat3d_alpha),
                fd_dx=(
                    None
                    if float(config.pde_heat3d_fd_dx) <= 0.0
                    else float(config.pde_heat3d_fd_dx)
                ),
                fd_dy=(
                    None
                    if float(config.pde_heat3d_fd_dy) <= 0.0
                    else float(config.pde_heat3d_fd_dy)
                ),
                fd_dz=(
                    None
                    if float(config.pde_heat3d_fd_dz) <= 0.0
                    else float(config.pde_heat3d_fd_dz)
                ),
                fd_space_order=int(config.pde_heat3d_fd_space_order),
                fd_time_order=int(getattr(config, "pde_heat3d_fd_time_order", 2)),
                time_coords_physical=data_ctx.t_coords_physical,
            )
        elif config.dataset_type == "ocean":
            pde_kwargs = {}
        pde_kwargs["point_sample_mode"] = point_sample_mode
        pde_kwargs["query_chunk_size"] = int(config.pde_query_chunk_size)
        if is_main_process(ddp_state):
            print(
                f"PDE loss setup: dataset_type={config.dataset_type}, lambdas={{'pde_point': {lambda_pde_point}}}, start_step={int(config.pde_start_step)}, monitor_always={bool(config.pde_monitor_always)}, monitor_point={bool(config.pde_monitor_point)}, point_sample_ratio={float(config.pde_point_sample_ratio)}, point_max_points={int(config.pde_point_max_points)}, kwargs={pde_kwargs}"
            )
    return PdeContext(
        lambda_pde_point=lambda_pde_point,
        use_pde_loss=use_pde_loss,
        use_pde_monitor=use_pde_monitor,
        use_pde_module=use_pde_module,
        pde_kwargs=pde_kwargs,
        pde_channel_order=channel_order,
        pde_point_sample_mode=point_sample_mode,
    )
