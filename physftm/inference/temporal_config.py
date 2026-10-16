"""Structured configuration for per-time temporal RKHS reconstruction."""

from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from omegaconf import OmegaConf
from physftm.training.configs import (
    RkhsPdeAcConfig,
    RkhsPdeNsConfig,
    RuntimeConfig,
    load_structured_config,
)


@dataclass
class TemporalRunConfig:
    results_dir: str | None = None
    results_prefix: str = "temporal_rkhs"
    device: str | None = None
    do_inference: bool = True


@dataclass
class TemporalArtifactConfig:
    ftm_manifest_path: str | None = None


@dataclass
class TemporalDataConfig:
    data_manifest_path: str | None = None
    evaluation_ground_truth_data_path: str | None = None
    batch_size: int = 4


@dataclass
class TemporalCoreConfig:
    lambda_core_ridge: float = 0.01
    beta_core_tv: float = 0.01
    cg_max_iter: int = 30
    cg_tol: float = 0.0001
    prior_core_mode: str = "mean"


@dataclass
class TemporalQueryConfig:
    times: list[float] = field(default_factory=list)
    start: float | None = None
    end: float | None = None
    num: int = 0
    match_tol: float = 1e-07


@dataclass
class TemporalMaskConfig:
    template: str = "random_drop_half"
    template_seed: int = 123
    mode: str = "identity"
    observed_ratio: float = 0.0
    seed: int = 123


@dataclass
class TemporalOdernnConfig:
    hidden_dim: int = 256
    ode_hidden_dim: int = 256
    train_epochs: int = 20
    train_batch_size: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 1e-06
    train_time_mask_mechanisms: list[str] = field(
        default_factory=lambda: ["random_drop_half", "blockmask"]
    )
    val_time_mask_mechanisms: list[str] = field(default_factory=list)
    train_keep_ratio: float = 0.5
    train_seed: int = 123
    val_seed: int = 123
    observed_ratio_list: list[float] = field(default_factory=lambda: [0.5])
    core_loss_weight: float = 1.0


@dataclass
class TemporalHeadConfig:
    method: str = "odernn"
    model_space: str = "core"
    override_anchor_times: bool = True
    load_if_exists: bool = True
    save_model: bool = True
    model_path: str | None = None
    allow_checkpoint_data_manifest_mismatch: bool = False
    query: TemporalQueryConfig = field(default_factory=TemporalQueryConfig)
    inference_mask: TemporalMaskConfig = field(default_factory=TemporalMaskConfig)
    odernn: TemporalOdernnConfig = field(default_factory=TemporalOdernnConfig)


@dataclass
class TemporalPdeDomainConfig:
    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0
    dt: float | None = None


@dataclass
class TemporalPdeHeat3dConfig:
    alpha: float = 0.01
    fd_dx: float = 0.0
    fd_dy: float = 0.0
    fd_dz: float = 0.0
    fd_space_order: int = 2


@dataclass
class TemporalPdeConfig:
    dataset_type: str = "auto"
    lambda_pde_point: float = 0.0
    point_sample_ratio: float = 0.25
    point_max_points: int = 0
    query_chunk_size: int = 512
    time_query_substeps: int = 10
    start_step: int = 0
    monitor_always: bool = False
    monitor_point: bool = False
    domain: TemporalPdeDomainConfig = field(default_factory=TemporalPdeDomainConfig)
    ac: RkhsPdeAcConfig = field(default_factory=RkhsPdeAcConfig)
    ns: RkhsPdeNsConfig = field(default_factory=RkhsPdeNsConfig)
    heat3d: TemporalPdeHeat3dConfig = field(default_factory=TemporalPdeHeat3dConfig)


@dataclass
class TemporalEvalConfig:
    n_plot_samples: int = 3
    plot_time_indices: list[int] = field(default_factory=lambda: [0, 49, 50, 51, 99])
    save_dtype: str = "float32"
    skip_save_predictions: bool = False
    skip_ssim: bool = False


@dataclass
class TemporalPlotsConfig:
    reconstruction_on_log: bool = True
    pde_residual_on_log: bool = True
    pde_residual_sample_index: int = 0
    pde_residual_sample_indices: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass
class TemporalInferenceConfig:
    run: TemporalRunConfig = field(default_factory=TemporalRunConfig)
    artifact: TemporalArtifactConfig = field(default_factory=TemporalArtifactConfig)
    data: TemporalDataConfig = field(default_factory=TemporalDataConfig)
    core: TemporalCoreConfig = field(default_factory=TemporalCoreConfig)
    temporal: TemporalHeadConfig = field(default_factory=TemporalHeadConfig)
    pde: TemporalPdeConfig = field(default_factory=TemporalPdeConfig)
    eval: TemporalEvalConfig = field(default_factory=TemporalEvalConfig)
    plots: TemporalPlotsConfig = field(default_factory=TemporalPlotsConfig)


TEMPORAL_FLAT_PATHS = {
    "results_dir": ("run", "results_dir"),
    "results_prefix": ("run", "results_prefix"),
    "device": ("run", "device"),
    "do_inference": ("run", "do_inference"),
    "ftm_manifest_path": ("artifact", "ftm_manifest_path"),
    "data_manifest_path": ("data", "data_manifest_path"),
    "evaluation_ground_truth_data_path": ("data", "evaluation_ground_truth_data_path"),
    "batch_size": ("data", "batch_size"),
    "lambda_core_ridge": ("core", "lambda_core_ridge"),
    "beta_core_tv": ("core", "beta_core_tv"),
    "cg_max_iter": ("core", "cg_max_iter"),
    "cg_tol": ("core", "cg_tol"),
    "prior_core_mode": ("core", "prior_core_mode"),
    "temporal_method": ("temporal", "method"),
    "temporal_model_space": ("temporal", "model_space"),
    "override_anchor_times": ("temporal", "override_anchor_times"),
    "temporal_load_if_exists": ("temporal", "load_if_exists"),
    "temporal_save_model": ("temporal", "save_model"),
    "temporal_model_path": ("temporal", "model_path"),
    "allow_checkpoint_data_manifest_mismatch": (
        "temporal",
        "allow_checkpoint_data_manifest_mismatch",
    ),
    "query_times": ("temporal", "query", "times"),
    "query_time_start": ("temporal", "query", "start"),
    "query_time_end": ("temporal", "query", "end"),
    "query_num": ("temporal", "query", "num"),
    "query_time_match_tol": ("temporal", "query", "match_tol"),
    "inference_time_template": ("temporal", "inference_mask", "template"),
    "inference_time_seed": ("temporal", "inference_mask", "template_seed"),
    "inference_time_mask_mode": ("temporal", "inference_mask", "mode"),
    "inference_obs_ratio": ("temporal", "inference_mask", "observed_ratio"),
    "inference_mask_seed": ("temporal", "inference_mask", "seed"),
    "temporal_hidden_dim": ("temporal", "odernn", "hidden_dim"),
    "temporal_ode_hidden_dim": ("temporal", "odernn", "ode_hidden_dim"),
    "temporal_train_epochs": ("temporal", "odernn", "train_epochs"),
    "temporal_train_batch_size": ("temporal", "odernn", "train_batch_size"),
    "temporal_train_lr": ("temporal", "odernn", "learning_rate"),
    "temporal_train_weight_decay": ("temporal", "odernn", "weight_decay"),
    "temporal_train_time_mask_mechanisms": (
        "temporal",
        "odernn",
        "train_time_mask_mechanisms",
    ),
    "temporal_val_time_mask_mechanisms": (
        "temporal",
        "odernn",
        "val_time_mask_mechanisms",
    ),
    "temporal_train_keep_ratio": ("temporal", "odernn", "train_keep_ratio"),
    "temporal_train_seed": ("temporal", "odernn", "train_seed"),
    "temporal_val_seed": ("temporal", "odernn", "val_seed"),
    "temporal_obs_ratio_list": ("temporal", "odernn", "observed_ratio_list"),
    "temporal_core_loss_weight": ("temporal", "odernn", "core_loss_weight"),
    "pde_dataset_type": ("pde", "dataset_type"),
    "lambda_pde_point": ("pde", "lambda_pde_point"),
    "pde_point_sample_ratio": ("pde", "point_sample_ratio"),
    "pde_point_max_points": ("pde", "point_max_points"),
    "pde_query_chunk_size": ("pde", "query_chunk_size"),
    "pde_time_query_substeps": ("pde", "time_query_substeps"),
    "pde_start_step": ("pde", "start_step"),
    "pde_monitor_always": ("pde", "monitor_always"),
    "pde_monitor_point": ("pde", "monitor_point"),
    "pde_Lx": ("pde", "domain", "Lx"),
    "pde_Ly": ("pde", "domain", "Ly"),
    "pde_Lz": ("pde", "domain", "Lz"),
    "pde_dt": ("pde", "domain", "dt"),
    "pde_ac_fd_dx": ("pde", "ac", "fd_dx"),
    "pde_ac_fd_dy": ("pde", "ac", "fd_dy"),
    "pde_ac_fd_space_order": ("pde", "ac", "fd_space_order"),
    "pde_ac_fd_time_order": ("pde", "ac", "fd_time_order"),
    "pde_ac_lambda": ("pde", "ac", "lambda_"),
    "pde_ac_epsilon": ("pde", "ac", "epsilon"),
    "pde_ns_fd_dx": ("pde", "ns", "fd_dx"),
    "pde_ns_fd_dy": ("pde", "ns", "fd_dy"),
    "pde_ns_fd_space_order": ("pde", "ns", "fd_space_order"),
    "pde_ns_fd_time_order": ("pde", "ns", "fd_time_order"),
    "pde_ns_nu": ("pde", "ns", "nu"),
    "pde_ns_forcing_scale": ("pde", "ns", "forcing_scale"),
    "pde_heat3d_alpha": ("pde", "heat3d", "alpha"),
    "pde_heat3d_fd_dx": ("pde", "heat3d", "fd_dx"),
    "pde_heat3d_fd_dy": ("pde", "heat3d", "fd_dy"),
    "pde_heat3d_fd_dz": ("pde", "heat3d", "fd_dz"),
    "pde_heat3d_fd_space_order": ("pde", "heat3d", "fd_space_order"),
    "n_plot_samples": ("eval", "n_plot_samples"),
    "plot_time_indices": ("eval", "plot_time_indices"),
    "save_dtype": ("eval", "save_dtype"),
    "skip_save_predictions": ("eval", "skip_save_predictions"),
    "skip_ssim": ("eval", "skip_ssim"),
    "plot_reconstruction_on_log": ("plots", "reconstruction_on_log"),
    "plot_pde_residual_on_log": ("plots", "pde_residual_on_log"),
    "plot_pde_residual_sample_index": ("plots", "pde_residual_sample_index"),
    "plot_pde_residual_sample_indices": ("plots", "pde_residual_sample_indices"),
}


def _validate_temporal_config(config: RuntimeConfig) -> None:
    if not str(config.ftm_manifest_path or "").strip():
        raise ValueError("artifact.ftm_manifest_path is required")
    choices = {
        "temporal_method": {"odernn"},
        "temporal_model_space": {"core"},
        "prior_core_mode": {"mean"},
        "pde_dataset_type": {"none", "auto", "ac", "ns", "heat3d", "ocean"},
        "inference_time_template": {"all", "random_drop_half"},
        "inference_time_mask_mode": {"identity", "random_drop_half", "blockmask"},
        "save_dtype": {"float16", "float32"},
    }
    for name, allowed in choices.items():
        value = str(getattr(config, name)).lower()
        if value not in allowed:
            raise ValueError(
                f"Unsupported {name}={value!r}; expected one of {sorted(allowed)}"
            )
    mask_modes = {"identity", "random_drop_half", "blockmask"}
    for name in (
        "temporal_train_time_mask_mechanisms",
        "temporal_val_time_mask_mechanisms",
    ):
        unknown = {str(x).lower() for x in getattr(config, name)} - mask_modes
        if unknown:
            raise ValueError(f"Unsupported {name} values: {sorted(unknown)}")
    if int(config.batch_size) <= 0 or int(config.temporal_train_batch_size) <= 0:
        raise ValueError(
            "data.batch_size and temporal.odernn.train_batch_size must be positive"
        )
    if int(config.temporal_hidden_dim) <= 0 or int(config.temporal_ode_hidden_dim) <= 0:
        raise ValueError("temporal.odernn hidden dimensions must be positive")
    if int(config.temporal_train_epochs) < 0:
        raise ValueError("temporal.odernn.train_epochs must be non-negative")
    if float(config.lambda_core_ridge) < 0.0 or float(config.beta_core_tv) < 0.0:
        raise ValueError("core ridge and TV weights must be non-negative")
    if int(config.cg_max_iter) <= 0 or float(config.cg_tol) <= 0.0:
        raise ValueError("core.cg_max_iter and core.cg_tol must be positive")
    if not 0.0 <= float(config.inference_obs_ratio) <= 1.0:
        raise ValueError("temporal.inference_mask.observed_ratio must be in [0,1]")
    for name in ("temporal_train_keep_ratio",):
        if not 0.0 < float(getattr(config, name)) <= 1.0:
            raise ValueError(f"{name} must be in (0,1]")
    for value in config.temporal_obs_ratio_list:
        if not 0.0 < float(value) <= 1.0:
            raise ValueError(
                "temporal.odernn.observed_ratio_list values must be in (0,1]"
            )
    if int(config.query_num) < 0:
        raise ValueError("temporal.query.num must be non-negative")
    if float(config.lambda_pde_point) != 0.0:
        raise ValueError(
            "Temporal training uses core loss only; pde.lambda_pde_point must be zero"
        )
    if not 0.0 < float(config.pde_point_sample_ratio) <= 1.0:
        raise ValueError("pde.point_sample_ratio must be in (0,1]")
    if int(config.pde_point_max_points) < 0 or int(config.pde_query_chunk_size) <= 0:
        raise ValueError(
            "pde.point_max_points must be non-negative and query_chunk_size must be positive"
        )
    if int(config.pde_time_query_substeps) <= 0:
        raise ValueError("pde.time_query_substeps must be positive")
    if int(config.pde_ac_fd_space_order) not in (2, 4) or int(
        config.pde_ac_fd_time_order
    ) not in (2, 4):
        raise ValueError("AC finite-difference orders must be 2 or 4")
    if int(config.pde_ns_fd_space_order) not in (2, 4) or int(
        config.pde_ns_fd_time_order
    ) not in (2, 4):
        raise ValueError("NS finite-difference orders must be 2 or 4")
    if int(config.pde_heat3d_fd_space_order) != 2:
        raise ValueError("Heat3D finite-difference space order must be 2")


def load_temporal_inference_config(config_path: str) -> RuntimeConfig:
    config = load_structured_config(
        config_path, TemporalInferenceConfig, TEMPORAL_FLAT_PATHS
    )
    _validate_temporal_config(config)
    return config


def save_resolved_temporal_inference_config(
    config: RuntimeConfig, output_path: str
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(config.to_nested_dict()), f=str(output))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-time RKHS reconstruction with the Hermite-conditioned ODE-RNN temporal head."
    )
    parser.add_argument(
        "--config", required=True, help="Temporal inference YAML configuration."
    )
    parser.add_argument("--ftm_manifest_path", default=None)
    parser.add_argument("--data_manifest_path", default=None)
    parser.add_argument(
        "--evaluation_ground_truth_data_path",
        default=None,
        help="Optional clean test data used only for evaluation. When omitted, use splits.test.clean_data_path when present, otherwise the observation data.",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--results_prefix", default=None)
    parser.add_argument("--temporal_method", choices=["odernn"], default=None)
    parser.add_argument("--temporal_model_path", default=None)
    parser.add_argument(
        "--allow_checkpoint_data_manifest_mismatch",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow a dimension-compatible temporal checkpoint to evaluate a different mask protocol and time sampling grid.",
    )
    parser.add_argument("--lambda_pde_point", type=float, default=None)
    parser.add_argument("--pde_point_sample_ratio", type=float, default=None)
    parser.add_argument("--pde_point_max_points", type=int, default=None)
    parser.add_argument("--pde_query_chunk_size", type=int, default=None)
    parser.add_argument(
        "--pde_dataset_type",
        choices=["none", "auto", "ac", "ns", "heat3d", "ocean"],
        default=None,
    )
    parser.add_argument(
        "--inference_time_template", choices=["all", "random_drop_half"], default=None
    )
    parser.add_argument(
        "--do_inference", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--n_plot_samples", type=int, default=None)
    parser.add_argument(
        "--skip_save_predictions", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--skip_ssim", action=argparse.BooleanOptionalAction, default=None
    )
    cli = parser.parse_args(argv)
    config = load_temporal_inference_config(cli.config)
    for name in (
        "ftm_manifest_path",
        "data_manifest_path",
        "evaluation_ground_truth_data_path",
        "batch_size",
        "device",
        "results_dir",
        "results_prefix",
        "temporal_method",
        "temporal_model_path",
        "allow_checkpoint_data_manifest_mismatch",
        "lambda_pde_point",
        "pde_point_sample_ratio",
        "pde_point_max_points",
        "pde_query_chunk_size",
        "pde_dataset_type",
        "inference_time_template",
        "do_inference",
        "n_plot_samples",
        "skip_save_predictions",
        "skip_ssim",
    ):
        value = getattr(cli, name)
        if value is not None:
            setattr(config, name, value)
    _validate_temporal_config(config)
    return config
