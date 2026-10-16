from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence, get_args, get_origin, get_type_hints

try:
    from omegaconf import OmegaConf
except ImportError as exc:
    raise ImportError(
        "OmegaConf is required for structured training config loading. Install project dependencies from requirements.txt."
    ) from exc
DEFAULT_DATA_MANIFEST = ""


class _ChoiceStrEnum(str, Enum):

    def __str__(self):
        return str(self.value)


class CoreParameterization(_ChoiceStrEnum):
    PER_TIME = "per_time"


class NormMode(_ChoiceStrEnum):
    CHANNELWISE_ZSCORE = "channelwise_zscore"


class CoreUpdateMode(_ChoiceStrEnum):
    ADAM = "adam"
    CG = "cg"


class BasisUpdateMode(_ChoiceStrEnum):
    ADAM = "adam"
    SOLVE = "solve"


class DatasetType(_ChoiceStrEnum):
    AC = "ac"
    NS = "ns"
    HEAT3D = "heat3d"
    OCEAN = "ocean"


class PointSampleMode(_ChoiceStrEnum):
    CONTINUOUS = "continuous"
    OBSERVED_GRID = "observed_grid"
    UNOBSERVED_GRID = "unobserved_grid"


class KernelType(_ChoiceStrEnum):
    MATERN32 = "matern32"


def _int_list(default: Sequence[int]) -> list[int]:
    return [int(v) for v in default]


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _to_primitive(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, SimpleNamespace):
        return {str(k): _to_primitive(v) for k, v in value.__dict__.items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _to_primitive(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_to_primitive(v) for v in value]
    if isinstance(value, list):
        return [_to_primitive(v) for v in value]
    return value


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(k): _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _get_path(root: Any, path: Sequence[str]) -> Any:
    node = root
    for key in path:
        node = getattr(node, key)
    return node


def _set_path(root: Any, path: Sequence[str], value: Any) -> None:
    node = root
    for key in path[:-1]:
        node = getattr(node, key)
    setattr(node, path[-1], value)


class RuntimeConfig:

    def __init__(self, structured_obj: Any, flat_paths: Mapping[str, Sequence[str]]):
        tree = _to_namespace(_to_primitive(structured_obj))
        object.__setattr__(self, "_tree", tree)
        object.__setattr__(
            self, "_flat_paths", {str(k): tuple(v) for k, v in flat_paths.items()}
        )
        for name, value in tree.__dict__.items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        flat_paths = object.__getattribute__(self, "_flat_paths")
        if name in flat_paths:
            return _get_path(object.__getattribute__(self, "_tree"), flat_paths[name])
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        flat_paths = object.__getattribute__(self, "_flat_paths")
        if name in flat_paths:
            _set_path(object.__getattribute__(self, "_tree"), flat_paths[name], value)
            return
        object.__setattr__(self, name, value)

    def to_nested_dict(self) -> dict[str, Any]:
        return _to_primitive(object.__getattribute__(self, "_tree"))

    def to_flat_dict(self) -> dict[str, Any]:
        tree = object.__getattribute__(self, "_tree")
        flat_paths = object.__getattribute__(self, "_flat_paths")
        return {
            flat: _to_primitive(_get_path(tree, path))
            for flat, path in flat_paths.items()
        }

    def to_namespace(self) -> SimpleNamespace:
        return object.__getattribute__(self, "_tree")


def _field_kind(field_type: Any) -> Any:
    origin = get_origin(field_type)
    if origin is not None:
        return origin
    return field_type


def _validate_loaded_node(value: Any, schema_type: Any, dotted: str) -> None:
    if value is None:
        return
    kind = _field_kind(schema_type)
    if is_dataclass(kind):
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Config section {dotted} must be a mapping, got {type(value).__name__}."
            )
        schema_fields = get_type_hints(kind)
        for key, child in value.items():
            if key not in schema_fields:
                raise KeyError(f"Unknown config key {dotted}.{key}.")
            _validate_loaded_node(child, schema_fields[key], f"{dotted}.{key}")
        return
    if kind is bool and (not isinstance(value, bool)):
        raise TypeError(f"Config key {dotted} expects a YAML boolean, got {value!r}.")
    if kind is int and isinstance(value, bool):
        raise TypeError(
            f"Config key {dotted} expects an integer, got boolean {value!r}."
        )
    if kind is float and isinstance(value, bool):
        raise TypeError(f"Config key {dotted} expects a number, got boolean {value!r}.")
    if kind is list:
        if not isinstance(value, list):
            raise TypeError(f"Config key {dotted} expects a YAML list, got {value!r}.")
        item_type = get_args(schema_type)[0] if get_args(schema_type) else Any
        item_kind = _field_kind(item_type)
        for idx, item in enumerate(value):
            if item_kind is bool and (not isinstance(item, bool)):
                raise TypeError(
                    f"Config key {dotted}[{idx}] expects a YAML boolean, got {item!r}."
                )
            if item_kind is int and isinstance(item, bool):
                raise TypeError(
                    f"Config key {dotted}[{idx}] expects an integer, got boolean {item!r}."
                )
            if item_kind is float and isinstance(item, bool):
                raise TypeError(
                    f"Config key {dotted}[{idx}] expects a number, got boolean {item!r}."
                )


def _normalize_loaded_node(value: Any, schema_type: Any) -> Any:
    if value is None:
        return None
    kind = _field_kind(schema_type)
    if is_dataclass(kind):
        schema_fields = get_type_hints(kind)
        return {
            str(key): _normalize_loaded_node(child, schema_fields[str(key)])
            for key, child in value.items()
        }
    if isinstance(kind, type) and issubclass(kind, Enum) and isinstance(value, str):
        for member in kind:
            if value == member.value or value == member.name:
                return member.name
        return value
    if kind is list and isinstance(value, list):
        item_type = get_args(schema_type)[0] if get_args(schema_type) else Any
        return [_normalize_loaded_node(item, item_type) for item in value]
    return value


def load_structured_config(
    config_path: str, schema_cls: type[Any], flat_paths: Mapping[str, Sequence[str]]
) -> RuntimeConfig:
    schema = OmegaConf.structured(schema_cls)
    loaded = OmegaConf.load(config_path)
    if loaded is None:
        loaded = OmegaConf.create({})
    loaded_container = OmegaConf.to_container(loaded, resolve=False)
    if not isinstance(loaded_container, Mapping):
        raise TypeError(
            f"Config root must be a mapping, got {type(loaded_container).__name__}."
        )
    _validate_loaded_node(loaded_container, schema_cls, "<root>")
    loaded = OmegaConf.create(_normalize_loaded_node(loaded_container, schema_cls))
    merged = OmegaConf.merge(schema, loaded)
    structured_obj = OmegaConf.to_object(merged)
    runtime_config = RuntimeConfig(structured_obj, flat_paths)
    runtime_config.source_config_path = str(Path(config_path).resolve())
    return runtime_config


@dataclass
class RkhsRunConfig:
    data_name: str = "newAC"
    seed: int = 123


@dataclass
class RkhsDataConfig:
    data_manifest_path: str = DEFAULT_DATA_MANIFEST
    data_path: str | None = None
    metadata_path: str | None = None
    dataset_type: DatasetType = DatasetType.AC
    batch_size: int = 1
    num_workers: int = 0


@dataclass
class RkhsModelConfig:
    R: list[int] = field(default_factory=lambda: _int_list([48, 48, 1]))
    core_parameterization: CoreParameterization = CoreParameterization.PER_TIME
    norm_mode: NormMode = NormMode.CHANNELWISE_ZSCORE


@dataclass
class RkhsOptimizationConfig:
    learning_rate: float = 0.0002
    max_epochs: int = 500
    print_every: int = 10
    eval_every: int = 10
    profile_every: int = 0


@dataclass
class RkhsCoreConfig:
    core_update_mode: CoreUpdateMode = CoreUpdateMode.CG
    core_learning_rate: float = 0.0
    lambda_core_ridge: float = 0.01
    beta_core_tv: float = 0.01
    als_core_cg_iter: int = 5
    als_core_cg_tol: float = 0.001
    als_core_steps: int = 1


@dataclass
class RkhsBasisConfig:
    als_basis_steps: int = 1
    basis_update_mode: BasisUpdateMode = BasisUpdateMode.ADAM
    basis_solve_value_ridge: float = 1e-06
    basis_solve_target_clip: float = 0.0


@dataclass
class RkhsKernelConfig:
    lambda_rkhs: float = 0.0001
    lambda_time: float = 1e-05
    coef_scale: float = 0.01
    kernel_type: KernelType = KernelType.MATERN32
    num_inducing_u: int | None = None
    num_inducing_v: int | None = None
    num_inducing_w: int | None = None
    lengthscale_u: float | None = None
    lengthscale_v: float | None = None
    lengthscale_w: float | None = None


@dataclass
class RkhsPdeDomainConfig:
    Lx: float = 4.0
    Ly: float = 1.0
    Lz: float = 1.0
    dt: float | None = None
    query_dt: float | None = None


@dataclass
class RkhsPdeAcConfig:
    fd_dx: float = 0.0004
    fd_dy: float = 0.0004
    fd_space_order: int = 2
    fd_time_order: int = 2
    lambda_: float = 1.0
    epsilon: float = 0.01


@dataclass
class RkhsPdeNsConfig:
    fd_dx: float = 0.0007
    fd_dy: float = 0.0007
    fd_space_order: int = 2
    fd_time_order: int = 2
    nu: float = 0.0001
    forcing_scale: float = 0.1


@dataclass
class RkhsPdeHeat3DConfig:
    alpha: float = 0.01
    fd_dx: float = 0.0
    fd_dy: float = 0.0
    fd_dz: float = 0.0
    fd_space_order: int = 2
    fd_time_order: int = 2


@dataclass
class RkhsPdeConfig:
    lambda_pde_point: float = 0.0
    point_sample_ratio: float = 0.05
    point_max_points: int = 0
    query_chunk_size: int = 8192
    point_sample_mode: PointSampleMode = PointSampleMode.CONTINUOUS
    start_step: int = 0
    monitor_always: bool = False
    monitor_point: bool = False
    domain: RkhsPdeDomainConfig = field(default_factory=RkhsPdeDomainConfig)
    ac: RkhsPdeAcConfig = field(default_factory=RkhsPdeAcConfig)
    ns: RkhsPdeNsConfig = field(default_factory=RkhsPdeNsConfig)
    heat3d: RkhsPdeHeat3DConfig = field(default_factory=RkhsPdeHeat3DConfig)


@dataclass
class RkhsPlotsConfig:
    reconstruction_on_log: bool = True
    pde_residual_on_log: bool = True
    pde_residual_sample_index: int = 0
    pde_residual_sample_indices: list[int] = field(
        default_factory=lambda: _int_list([0, 1, 2])
    )


@dataclass
class RkhsTrainConfig:
    run: RkhsRunConfig = field(default_factory=RkhsRunConfig)
    data: RkhsDataConfig = field(default_factory=RkhsDataConfig)
    model: RkhsModelConfig = field(default_factory=RkhsModelConfig)
    optimization: RkhsOptimizationConfig = field(default_factory=RkhsOptimizationConfig)
    core: RkhsCoreConfig = field(default_factory=RkhsCoreConfig)
    basis: RkhsBasisConfig = field(default_factory=RkhsBasisConfig)
    rkhs: RkhsKernelConfig = field(default_factory=RkhsKernelConfig)
    pde: RkhsPdeConfig = field(default_factory=RkhsPdeConfig)
    plots: RkhsPlotsConfig = field(default_factory=RkhsPlotsConfig)


RKHS_FLAT_PATHS = {
    "data_name": ("run", "data_name"),
    "seed": ("run", "seed"),
    "data_manifest_path": ("data", "data_manifest_path"),
    "data_path": ("data", "data_path"),
    "metadata_path": ("data", "metadata_path"),
    "dataset_type": ("data", "dataset_type"),
    "batch_size": ("data", "batch_size"),
    "num_workers": ("data", "num_workers"),
    "R": ("model", "R"),
    "core_parameterization": ("model", "core_parameterization"),
    "norm_mode": ("model", "norm_mode"),
    "learning_rate": ("optimization", "learning_rate"),
    "max_epochs": ("optimization", "max_epochs"),
    "print_every": ("optimization", "print_every"),
    "eval_every": ("optimization", "eval_every"),
    "profile_every": ("optimization", "profile_every"),
    "core_update_mode": ("core", "core_update_mode"),
    "core_learning_rate": ("core", "core_learning_rate"),
    "lambda_core_ridge": ("core", "lambda_core_ridge"),
    "beta_core_tv": ("core", "beta_core_tv"),
    "als_core_cg_iter": ("core", "als_core_cg_iter"),
    "als_core_cg_tol": ("core", "als_core_cg_tol"),
    "als_core_steps": ("core", "als_core_steps"),
    "als_basis_steps": ("basis", "als_basis_steps"),
    "basis_update_mode": ("basis", "basis_update_mode"),
    "basis_solve_value_ridge": ("basis", "basis_solve_value_ridge"),
    "basis_solve_target_clip": ("basis", "basis_solve_target_clip"),
    "lambda_rkhs": ("rkhs", "lambda_rkhs"),
    "lambda_time": ("rkhs", "lambda_time"),
    "coef_scale": ("rkhs", "coef_scale"),
    "kernel_type": ("rkhs", "kernel_type"),
    "num_inducing_u": ("rkhs", "num_inducing_u"),
    "num_inducing_v": ("rkhs", "num_inducing_v"),
    "num_inducing_w": ("rkhs", "num_inducing_w"),
    "lengthscale_u": ("rkhs", "lengthscale_u"),
    "lengthscale_v": ("rkhs", "lengthscale_v"),
    "lengthscale_w": ("rkhs", "lengthscale_w"),
    "lambda_pde_point": ("pde", "lambda_pde_point"),
    "pde_point_sample_ratio": ("pde", "point_sample_ratio"),
    "pde_point_max_points": ("pde", "point_max_points"),
    "pde_query_chunk_size": ("pde", "query_chunk_size"),
    "pde_point_sample_mode": ("pde", "point_sample_mode"),
    "pde_start_step": ("pde", "start_step"),
    "pde_monitor_always": ("pde", "monitor_always"),
    "pde_monitor_point": ("pde", "monitor_point"),
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
    "plot_reconstruction_on_log": ("plots", "reconstruction_on_log"),
    "plot_pde_residual_on_log": ("plots", "pde_residual_on_log"),
    "plot_pde_residual_sample_index": ("plots", "pde_residual_sample_index"),
    "plot_pde_residual_sample_indices": ("plots", "pde_residual_sample_indices"),
}


def load_rkhs_runtime_config(config_path: str) -> RuntimeConfig:
    return load_structured_config(config_path, RkhsTrainConfig, RKHS_FLAT_PATHS)
