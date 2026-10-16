import argparse
from dataclasses import dataclass, field
from physftm.training.configs import (
    DatasetType,
    RkhsPdeAcConfig,
    RkhsPdeDomainConfig,
    RkhsPdeNsConfig,
    RkhsPdeHeat3DConfig,
    RuntimeConfig,
    load_structured_config,
)


@dataclass
class InferenceRunConfig:
    results_dir: str | None = None
    results_prefix: str = "basis_split_reconstruction"
    device: str | None = None


@dataclass
class InferenceArtifactConfig:
    ftm_manifest_path: str | None = None


@dataclass
class InferenceDataConfig:
    data_manifest_path: str | None = None
    split: str = "test"
    batch_size: int = 8
    auto_crop_time_to_core: str = "none"


@dataclass
class InferenceCoreConfig:
    lambda_core_ridge: float = 0.01
    beta_core_tv: float = 0.01
    cg_max_iter: int = 30
    cg_tol: float = 0.0001
    prior_core_mode: str = "mean"


@dataclass
class InferencePdeConfig:
    dataset_type: DatasetType | str = "auto"
    point_sample_ratio: float = 0.25
    point_max_points: int = 0
    query_chunk_size: int = 8192
    domain: RkhsPdeDomainConfig = field(default_factory=RkhsPdeDomainConfig)
    ac: RkhsPdeAcConfig = field(default_factory=RkhsPdeAcConfig)
    ns: RkhsPdeNsConfig = field(default_factory=RkhsPdeNsConfig)
    heat3d: RkhsPdeHeat3DConfig = field(default_factory=RkhsPdeHeat3DConfig)


@dataclass
class InferenceEvalConfig:
    skip_ssim: bool = False
    n_plot_samples: int = 3
    plot_time_indices: str = "0,100,200"
    save_dtype: str = "float32"


@dataclass
class RkhsInferenceConfig:
    run: InferenceRunConfig = field(default_factory=InferenceRunConfig)
    artifact: InferenceArtifactConfig = field(default_factory=InferenceArtifactConfig)
    data: InferenceDataConfig = field(default_factory=InferenceDataConfig)
    core: InferenceCoreConfig = field(default_factory=InferenceCoreConfig)
    pde: InferencePdeConfig = field(default_factory=InferencePdeConfig)
    eval: InferenceEvalConfig = field(default_factory=InferenceEvalConfig)


INFERENCE_FLAT_PATHS = {
    "results_dir": ("run", "results_dir"),
    "results_prefix": ("run", "results_prefix"),
    "device": ("run", "device"),
    "ftm_manifest_path": ("artifact", "ftm_manifest_path"),
    "data_manifest_path": ("data", "data_manifest_path"),
    "split": ("data", "split"),
    "batch_size": ("data", "batch_size"),
    "auto_crop_time_to_core": ("data", "auto_crop_time_to_core"),
    "lambda_core_ridge": ("core", "lambda_core_ridge"),
    "beta_core_tv": ("core", "beta_core_tv"),
    "cg_max_iter": ("core", "cg_max_iter"),
    "cg_tol": ("core", "cg_tol"),
    "prior_core_mode": ("core", "prior_core_mode"),
    "pde_dataset_type": ("pde", "dataset_type"),
    "pde_point_sample_ratio": ("pde", "point_sample_ratio"),
    "pde_point_max_points": ("pde", "point_max_points"),
    "pde_query_chunk_size": ("pde", "query_chunk_size"),
    "pde_Lx": ("pde", "domain", "Lx"),
    "pde_Ly": ("pde", "domain", "Ly"),
    "pde_Lz": ("pde", "domain", "Lz"),
    "pde_dt": ("pde", "domain", "dt"),
    "pde_query_dt": ("pde", "domain", "query_dt"),
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
    "pde_heat3d_fd_time_order": ("pde", "heat3d", "fd_time_order"),
    "skip_ssim": ("eval", "skip_ssim"),
    "n_plot_samples": ("eval", "n_plot_samples"),
    "plot_time_indices": ("eval", "plot_time_indices"),
    "save_dtype": ("eval", "save_dtype"),
}


def load_rkhs_inference_config(config_path: str) -> RuntimeConfig:
    return load_structured_config(
        config_path, RkhsInferenceConfig, INFERENCE_FLAT_PATHS
    )


def _apply_cli_overrides(config, cli_args):
    for key in INFERENCE_FLAT_PATHS:
        if hasattr(cli_args, key):
            value = getattr(cli_args, key)
            if value is not None:
                setattr(config, key, value)
    return config


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Use a trained RKHS basis artifact to reconstruct an AC2D/NS2D/Heat3D/Ocean split."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="YAML inference config path."
    )
    parser.add_argument(
        "--ftm_manifest_path",
        type=str,
        default=None,
        help="Override artifact manifest path from config.",
    )
    parser.add_argument("--basis_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--core_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--data_manifest_path",
        type=str,
        default=None,
        help="Override temporal-consistent data manifest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Override split to reconstruct.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override inference batch size."
    )
    parser.add_argument(
        "--auto_crop_time_to_core",
        type=str,
        default=None,
        choices=["none", "front"],
        help="Override time-axis crop policy.",
    )
    parser.add_argument(
        "--lambda_core_ridge",
        type=float,
        default=None,
        help="Override core-solve ridge weight.",
    )
    parser.add_argument(
        "--beta_core_tv",
        type=float,
        default=None,
        help="Override core-solve temporal TV weight.",
    )
    parser.add_argument(
        "--cg_max_iter", type=int, default=None, help="Override maximum CG iterations."
    )
    parser.add_argument(
        "--cg_tol", type=float, default=None, help="Override CG tolerance."
    )
    parser.add_argument(
        "--prior_core_mode",
        type=str,
        default=None,
        choices=["mean"],
        help="Override core prior mode.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None, help="Override output directory."
    )
    parser.add_argument(
        "--results_prefix",
        type=str,
        default=None,
        help="Override timestamped results folder prefix.",
    )
    parser.add_argument(
        "--pde_dataset_type",
        type=str,
        default=None,
        choices=["none", "auto", "ac", "ns", "heat3d", "ocean"],
        help="Override PDE evaluation type.",
    )
    parser.add_argument(
        "--pde_point_sample_ratio",
        type=float,
        default=None,
        help="Override PDE residual spatial sample ratio.",
    )
    parser.add_argument(
        "--pde_point_max_points",
        type=int,
        default=None,
        help="Override PDE residual max sampled points.",
    )
    parser.add_argument(
        "--pde_query_chunk_size",
        type=int,
        default=None,
        help="Override PDE query chunk size.",
    )
    parser.add_argument(
        "--skip_ssim", action="store_true", default=None, help="Skip SSIM."
    )
    parser.add_argument(
        "--n_plot_samples",
        type=int,
        default=None,
        help="Override number of samples to plot.",
    )
    parser.add_argument(
        "--plot_time_indices",
        type=str,
        default=None,
        help="Override comma-separated time indices to visualize.",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        default=None,
        choices=["float16", "float32"],
        help="Override prediction array storage dtype.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Override torch device."
    )
    hidden_float_keys = [
        "pde_ac_fd_dx",
        "pde_ac_fd_dy",
        "pde_ns_fd_dx",
        "pde_ns_fd_dy",
        "pde_Lx",
        "pde_Ly",
        "pde_Lz",
        "pde_dt",
        "pde_query_dt",
        "pde_ns_nu",
        "pde_ns_forcing_scale",
        "pde_ac_lambda",
        "pde_ac_epsilon",
        "pde_heat3d_alpha",
        "pde_heat3d_fd_dx",
        "pde_heat3d_fd_dy",
        "pde_heat3d_fd_dz",
    ]
    for key in hidden_float_keys:
        parser.add_argument(
            f"--{key}", type=float, default=None, help=argparse.SUPPRESS
        )
    hidden_int_keys = [
        "pde_ac_fd_space_order",
        "pde_ac_fd_time_order",
        "pde_ns_fd_space_order",
        "pde_ns_fd_time_order",
        "pde_heat3d_fd_space_order",
    ]
    for key in hidden_int_keys:
        parser.add_argument(
            f"--{key}", type=int, default=None, choices=[2, 4], help=argparse.SUPPRESS
        )
    parser.add_argument(
        "--axis_energy_tau", type=int, default=None, help=argparse.SUPPRESS
    )
    cli_args = parser.parse_args(argv)
    config = load_rkhs_inference_config(cli_args.config)
    return _apply_cli_overrides(config, cli_args)
