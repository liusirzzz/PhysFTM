"""Training loop components for PhysFTM.

The entrypoint builds contexts and callbacks; this module owns only the train
step mechanics so diagnostics and artifact writing can be moved independently.
"""

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from physftm.rkhs.core_solvers import (
    ridge_tv_solve_core_2d_from_sparse_mask,
    ridge_tv_solve_core_3d_from_sparse_mask,
)
from physftm.rkhs.projection import (
    ridge_projected_solve_rkhs_mode_coef_2d_from_sparse_mask,
)
from physftm.rkhs.regularizers import temporal_gp_penalty
from physftm.distributed import is_main_process, reduce_mean_scalar
from physftm.decoding.tucker import decode_spatial_from_precomputed_basis
from physftm.pde.training import (
    _restore_active_pde_query_chunk_size,
    _set_active_pde_query_chunk_size,
    compute_pde_loss_for_training,
)
from physftm.solvers.shared_basis import (
    solve_shared_basis_multichannel_core_2d_from_sparse_mask,
    solve_shared_basis_multichannel_core_3d_from_sparse_mask,
)
from physftm.training.evaluation import evaluate
from physftm.training.runtime import (
    current_model,
    format_loss_component_log,
    sync_if_cuda,
)
from physftm.training.runtime import current_optimizer_lr
from physftm.training.wrapper import (
    reconstruction_loss_from_output,
    broadcast_mask_channels,
)


@dataclass
class BatchLogPayload:
    step: int
    step_tag: str
    batch_idx: torch.Tensor
    data_batch: torch.Tensor
    mask_batch: torch.Tensor


@dataclass
class EvalPayload:
    step: int
    rmse: float
    mae: float
    rel_l2: float
    best_rmse: float
    best_mae: float
    best_rel_l2: float
    best_step: int
    is_new_best: bool


@dataclass
class LoopResult:
    best_rmse: float
    best_mae: float
    best_rel_l2: float
    best_step: int


@dataclass
class LossResult:
    loss: torch.Tensor
    components: dict
    dense_output: torch.Tensor


class _BatchCoreAdamW:
    """AdamW state for batch-local cores backed by a CPU core table."""

    def __init__(
        self, core_table, *, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01
    ):
        if int(core_table.ndim) < 1:
            raise ValueError(
                f"Core table must have a sample axis, got shape={tuple(core_table.shape)}"
            )
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.exp_avg = torch.zeros_like(core_table, device="cpu")
        self.exp_avg_sq = torch.zeros_like(core_table, device="cpu")
        self.steps = torch.zeros(
            int(core_table.shape[0]), dtype=torch.int64, device="cpu"
        )

    @torch.no_grad()
    def step(self, batch_idx, core_batch):
        if core_batch.grad is None:
            return
        idx = batch_idx.detach().cpu().long().reshape(-1)
        if int(core_batch.shape[0]) != int(idx.numel()):
            raise ValueError(
                f"Core batch size {int(core_batch.shape[0])} does not match index count {int(idx.numel())}"
            )
        device = core_batch.device
        dtype = core_batch.dtype
        grad = core_batch.grad.detach()
        exp_avg = self.exp_avg.index_select(0, idx).to(
            device=device, dtype=dtype, non_blocking=True
        )
        exp_avg_sq = self.exp_avg_sq.index_select(0, idx).to(
            device=device, dtype=dtype, non_blocking=True
        )
        step_cpu = self.steps.index_select(0, idx) + 1
        exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
        step = step_cpu.to(dtype=torch.float64)
        view_shape = (int(step.numel()),) + (1,) * (int(core_batch.ndim) - 1)
        bias_correction1 = (
            (1.0 - torch.pow(self.beta1, step))
            .to(device=device, dtype=dtype)
            .reshape(view_shape)
        )
        bias_correction2_sqrt = (
            torch.sqrt(1.0 - torch.pow(self.beta2, step))
            .to(device=device, dtype=dtype)
            .reshape(view_shape)
        )
        denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
        step_size = self.lr / bias_correction1
        core_batch.mul_(1.0 - self.lr * self.weight_decay)
        core_batch.add_(-step_size * exp_avg / denom)
        self.exp_avg.index_copy_(0, idx, exp_avg.cpu())
        self.exp_avg_sq.index_copy_(0, idx, exp_avg_sq.cpu())
        self.steps.index_copy_(0, idx, step_cpu)


class LossComputer:
    """Shared loss accounting for initial evaluation and basis/core updates."""

    def __init__(
        self,
        *,
        config,
        device,
        ind_input,
        t_ind_uni,
        norm_stats,
        train_reg_cfg,
        pde_kwargs,
        pde_channel_order,
        dataset_pde_context,
        shared_basis_multichannel,
        lambda_pde_point,
        chol_t,
    ):
        self.config = config
        self.device = device
        self.ind_input = ind_input
        self.t_ind_uni = t_ind_uni
        self.norm_stats = norm_stats
        self.train_reg_cfg = train_reg_cfg
        self.pde_kwargs = pde_kwargs
        self.pde_channel_order = pde_channel_order
        self.dataset_pde_context = dataset_pde_context
        self.shared_basis_multichannel = bool(shared_basis_multichannel)
        self.lambda_pde_point = float(lambda_pde_point)
        self.chol_t = chol_t

    def compute(
        self,
        *,
        model_ref,
        basis_module,
        core_batch,
        data_batch,
        mask_batch,
        batch_idx,
        pde_active,
        pde_monitor,
        pde_compute_point,
        pde_grad_enabled=False,
    ):
        basis_module.mode = "training"
        if getattr(basis_module, "spatial_dim", 2) == 3:
            basis_u, basis_v, basis_w = basis_module(input_ind_train=self.ind_input)
            basis_extra = (basis_w,)
        else:
            basis_u, basis_v = basis_module(input_ind_train=self.ind_input)
            basis_extra = None
        total_out = decode_spatial_from_precomputed_basis(
            basis_u,
            basis_v,
            basis_extra,
            core_batch,
            shared_basis_multichannel=self.shared_basis_multichannel,
            core_parameterization="per_time",
        )
        rec = reconstruction_loss_from_output(
            total_out, data_batch, mask_batch=mask_batch, reduction="rmse"
        )
        smooth = (
            temporal_gp_penalty(core_batch, self.t_ind_uni, chol=self.chol_t)
            if not False and self.train_reg_cfg["lambda_time"] > 0
            else 0.0
        )
        rkhs_penalty = basis_module.rkhs_penalty_2d()
        loss = (
            rec
            + self.train_reg_cfg["lambda_rkhs"] * rkhs_penalty
            + self.train_reg_cfg["lambda_time"] * smooth
        )
        pde_log = {}
        if pde_monitor:
            pde_grad_context = (
                torch.enable_grad() if pde_grad_enabled else torch.no_grad()
            )
            with pde_grad_context:
                prev_chunk = _set_active_pde_query_chunk_size(
                    basis_module, int(self.config.pde_query_chunk_size)
                )
                try:
                    pde_losses = compute_pde_loss_for_training(
                        model_ref=model_ref,
                        core_batch=core_batch,
                        dense_output=total_out,
                        norm_stats=self.norm_stats,
                        dataset_type=self.config.dataset_type,
                        pde_kwargs=self.pde_kwargs,
                        channel_order=self.pde_channel_order,
                        point_sample_ratio=self.config.pde_point_sample_ratio,
                        point_max_points=self.config.pde_point_max_points,
                        coord_uniques=self.ind_input,
                        shared_basis_multichannel=self.shared_basis_multichannel,
                        lambda_pde_point=self.lambda_pde_point,
                        compute_pde_point=pde_compute_point,
                        batch_idx=batch_idx,
                        pde_context=self.dataset_pde_context,
                        mask_batch=mask_batch,
                        backward_pde=bool(pde_active and pde_grad_enabled),
                    )
                finally:
                    _restore_active_pde_query_chunk_size(basis_module, prev_chunk)
            pde_grad_accumulated = bool(pde_losses.pop("_pde_grad_accumulated", False))
            for pde_name, pde_val in pde_losses.items():
                lam = self.lambda_pde_point if pde_name == "pde_point" else 0.0
                pde_log[pde_name] = float(pde_val.detach())
                if pde_active and lam > 0 and (not pde_grad_accumulated):
                    loss = loss + lam * pde_val
        components = {
            "rec": float(rec.detach()),
            "rkhs": float(
                rkhs_penalty.detach() if torch.is_tensor(rkhs_penalty) else rkhs_penalty
            ),
            "time": float(smooth.detach() if torch.is_tensor(smooth) else smooth),
        }
        components.update(pde_log)
        return LossResult(loss=loss, components=components, dense_output=total_out)


class CoreUpdater:
    """Solve the batch core with fixed basis factors, then write it to the table."""

    def __init__(
        self,
        *,
        config,
        train_reg_cfg,
        ind_input,
        shared_basis_multichannel,
        use_core_cg,
        use_basis_solve,
    ):
        self.config = config
        self.train_reg_cfg = train_reg_cfg
        self.ind_input = ind_input
        self.shared_basis_multichannel = bool(shared_basis_multichannel)
        self.use_core_cg = bool(use_core_cg)
        self.use_basis_solve = bool(use_basis_solve)

    def precompute_basis(self, basis_module):
        if not (self.use_core_cg or self.use_basis_solve):
            return None
        basis_module.mode = "training"
        return tuple((b.detach() for b in basis_module(input_ind_train=self.ind_input)))

    def update(
        self,
        *,
        model_ref,
        basis_module,
        data_batch,
        mask_batch,
        batch_idx,
        precomputed_basis,
        device,
    ):
        if not self.use_core_cg:
            return
        for _ in range(max(1, self.config.als_core_steps)):
            with torch.no_grad():
                core_prev = model_ref.get_core_batch(
                    batch_idx, detach=True, device=device
                )
                if (
                    getattr(basis_module, "spatial_dim", 2) == 3
                    and self.shared_basis_multichannel
                ):
                    core_solved = (
                        solve_shared_basis_multichannel_core_3d_from_sparse_mask(
                            basis=basis_module,
                            y=data_batch,
                            mask=mask_batch,
                            coord_uniques=self.ind_input,
                            rank=tuple(self.config.R),
                            lambda_ridge=self.train_reg_cfg["lambda_core_ridge"],
                            beta_tv=self.train_reg_cfg["beta_core_tv"],
                            max_cg_iter=self.config.als_core_cg_iter,
                            cg_tol=self.config.als_core_cg_tol,
                            initial_core=core_prev,
                            precomputed_basis=precomputed_basis,
                        )
                    )
                elif getattr(basis_module, "spatial_dim", 2) == 3:
                    core_solved = ridge_tv_solve_core_3d_from_sparse_mask(
                        basis=basis_module,
                        y=data_batch,
                        mask=mask_batch,
                        coord_uniques=self.ind_input,
                        rank=tuple(self.config.R),
                        lambda_ridge=self.train_reg_cfg["lambda_core_ridge"],
                        beta_tv=self.train_reg_cfg["beta_core_tv"],
                        max_cg_iter=self.config.als_core_cg_iter,
                        cg_tol=self.config.als_core_cg_tol,
                        initial_core=core_prev,
                        precomputed_basis=precomputed_basis,
                    )
                elif self.shared_basis_multichannel:
                    core_solved = (
                        solve_shared_basis_multichannel_core_2d_from_sparse_mask(
                            basis=basis_module,
                            y=data_batch,
                            mask=mask_batch,
                            coord_uniques=self.ind_input,
                            rank=tuple(self.config.R),
                            lambda_ridge=self.train_reg_cfg["lambda_core_ridge"],
                            beta_tv=self.train_reg_cfg["beta_core_tv"],
                            max_cg_iter=self.config.als_core_cg_iter,
                            cg_tol=self.config.als_core_cg_tol,
                            initial_core=core_prev,
                            precomputed_basis=precomputed_basis,
                        )
                    )
                else:
                    core_solved = ridge_tv_solve_core_2d_from_sparse_mask(
                        basis=basis_module,
                        y=data_batch,
                        mask=mask_batch,
                        coord_uniques=self.ind_input,
                        rank=tuple(self.config.R),
                        lambda_ridge=self.train_reg_cfg["lambda_core_ridge"],
                        beta_tv=self.train_reg_cfg["beta_core_tv"],
                        max_cg_iter=self.config.als_core_cg_iter,
                        cg_tol=self.config.als_core_cg_tol,
                        initial_core=core_prev,
                        precomputed_basis=precomputed_basis,
                    )
                model_ref.put_core_batch(batch_idx, core_solved)


class BasisUpdater:
    """Update RKHS basis coefficients and optional batch-local Adam cores."""

    def __init__(
        self,
        *,
        config,
        device,
        ind_input,
        train_reg_cfg,
        loss_computer,
        optimizer,
        core_adam_lr,
        use_core_adam,
        use_basis_solve,
        use_rkhs_basis,
        k_inv_t,
        k_inv_u,
        k_inv_v,
        k_inv_w=None,
    ):
        self.config = config
        self.device = device
        self.ind_input = ind_input
        self.train_reg_cfg = train_reg_cfg
        self.loss_computer = loss_computer
        self.optimizer = optimizer
        self.core_adam_lr = float(core_adam_lr)
        self.use_core_adam = bool(use_core_adam)
        self.use_basis_solve = bool(use_basis_solve)
        self.use_rkhs_basis = bool(use_rkhs_basis)
        self.k_inv_t = k_inv_t
        self.k_inv_u = k_inv_u
        self.k_inv_v = k_inv_v
        self.k_inv_w = k_inv_w
        self.core_adam = None

    def _solve_basis(
        self, *, basis_module, core_batch, data_batch, mask_batch, precomputed_basis
    ):
        mask_batch = broadcast_mask_channels(mask_batch, data_batch)
        coef_u_solved = ridge_projected_solve_rkhs_mode_coef_2d_from_sparse_mask(
            model=basis_module,
            core_batch=core_batch,
            y=data_batch,
            mask=mask_batch,
            coord_uniques=self.ind_input,
            mode="u",
            lambda_rkhs=self.train_reg_cfg["lambda_rkhs"],
            value_ridge=self.config.basis_solve_value_ridge,
            target_clip=self.config.basis_solve_target_clip,
            precomputed_basis=precomputed_basis,
        )
        basis_module.mode_u.coef.data.copy_(coef_u_solved)
        basis_module.mode = "training"
        basis_u_tmp, basis_v_tmp = basis_module(input_ind_train=self.ind_input)
        coef_v_solved = ridge_projected_solve_rkhs_mode_coef_2d_from_sparse_mask(
            model=basis_module,
            core_batch=core_batch,
            y=data_batch,
            mask=mask_batch,
            coord_uniques=self.ind_input,
            mode="v",
            lambda_rkhs=self.train_reg_cfg["lambda_rkhs"],
            value_ridge=self.config.basis_solve_value_ridge,
            target_clip=self.config.basis_solve_target_clip,
            precomputed_basis=(basis_u_tmp.detach(), basis_v_tmp.detach()),
        )
        basis_module.mode_v.coef.data.copy_(coef_v_solved)

    def _precondition_basis_gradients(self, basis_module):
        if not (self.use_rkhs_basis and (not self.use_basis_solve)):
            return
        with torch.no_grad():
            coef_u = basis_module.mode_u.coef
            coef_v = basis_module.mode_v.coef
            if coef_u.grad is not None:
                coef_u.grad.data = self.k_inv_u @ coef_u.grad.data
            if coef_v.grad is not None:
                coef_v.grad.data = self.k_inv_v @ coef_v.grad.data
            if hasattr(basis_module, "mode_w") and self.k_inv_w is not None:
                coef_w = basis_module.mode_w.coef
                if coef_w.grad is not None:
                    coef_w.grad.data = self.k_inv_w @ coef_w.grad.data

    def update(
        self,
        *,
        model_ref,
        basis_module,
        data_batch,
        mask_batch,
        batch_idx,
        precomputed_basis,
        pde_active,
        pde_monitor,
        pde_compute_point,
    ):
        losses = []
        components = []
        for _ in range(max(1, self.config.als_basis_steps)):
            if self.use_core_adam:
                if self.core_adam is None:
                    self.core_adam = _BatchCoreAdamW(
                        model_ref.tucker_core, lr=self.core_adam_lr
                    )
                core_batch = nn.Parameter(
                    model_ref.get_core_batch(batch_idx, detach=True, device=self.device)
                )
            else:
                core_batch = model_ref.get_core_batch(
                    batch_idx, detach=True, device=self.device
                )
            if self.use_basis_solve:
                with torch.no_grad():
                    self._solve_basis(
                        basis_module=basis_module,
                        core_batch=core_batch,
                        data_batch=data_batch,
                        mask_batch=mask_batch,
                        precomputed_basis=precomputed_basis,
                    )
            if self.optimizer is not None or self.core_adam is not None:
                if self.optimizer is not None:
                    self.optimizer.zero_grad(set_to_none=True)
                if self.core_adam is not None:
                    core_batch.grad = None
                result = self.loss_computer.compute(
                    model_ref=model_ref,
                    basis_module=basis_module,
                    core_batch=core_batch,
                    data_batch=data_batch,
                    mask_batch=mask_batch,
                    batch_idx=batch_idx,
                    pde_active=pde_active,
                    pde_monitor=pde_monitor,
                    pde_compute_point=pde_compute_point,
                    pde_grad_enabled=pde_active,
                )
                components.append(result.components)
                if result.loss.requires_grad:
                    result.loss.backward()
                    self._precondition_basis_gradients(basis_module)
                    if self.optimizer is not None:
                        self.optimizer.step()
                    if self.core_adam is not None:
                        self.core_adam.step(batch_idx, core_batch)
                        model_ref.put_core_batch(batch_idx, core_batch)
                losses.append(float(result.loss.detach().item()))
            else:
                with torch.no_grad():
                    result = self.loss_computer.compute(
                        model_ref=model_ref,
                        basis_module=basis_module,
                        core_batch=core_batch,
                        data_batch=data_batch,
                        mask_batch=mask_batch,
                        batch_idx=batch_idx,
                        pde_active=pde_active,
                        pde_monitor=pde_monitor,
                        pde_compute_point=pde_compute_point,
                        pde_grad_enabled=False,
                    )
                    components.append(result.components)
                    losses.append(float(result.loss.detach().item()))
        return (losses, components)


class TrainingLoop:
    """Epoch orchestration for ALS training.

    Snapshot and checkpoint work stays behind callbacks so Stage 3 can move
    diagnostics/artifact code without changing updater internals.
    """

    def __init__(
        self,
        *,
        config,
        ddp_state,
        device,
        data_ctx,
        model_ctx,
        pde_ctx,
        snapshot_callback=None,
        checkpoint_callback=None,
    ):
        self.config = config
        self.ddp_state = ddp_state
        self.device = device
        self.data_ctx = data_ctx
        self.model_ctx = model_ctx
        self.pde_ctx = pde_ctx
        self.snapshot_callback = snapshot_callback
        self.checkpoint_callback = checkpoint_callback
        self.train_reg_cfg = {
            "lambda_core_ridge": float(config.lambda_core_ridge),
            "beta_core_tv": float(config.beta_core_tv or 0.0),
            "lambda_rkhs": float(config.lambda_rkhs),
            "lambda_time": float(config.lambda_time),
        }
        self.loss_computer = LossComputer(
            config=config,
            device=device,
            ind_input=data_ctx.ind_input,
            t_ind_uni=data_ctx.t_ind_uni,
            norm_stats=data_ctx.norm_stats,
            train_reg_cfg=self.train_reg_cfg,
            pde_kwargs=pde_ctx.pde_kwargs,
            pde_channel_order=pde_ctx.pde_channel_order,
            dataset_pde_context=data_ctx.dataset_pde_context,
            shared_basis_multichannel=data_ctx.shared_basis_multichannel,
            lambda_pde_point=pde_ctx.lambda_pde_point,
            chol_t=model_ctx.chol_t,
        )
        self.core_updater = CoreUpdater(
            config=config,
            train_reg_cfg=self.train_reg_cfg,
            ind_input=data_ctx.ind_input,
            shared_basis_multichannel=data_ctx.shared_basis_multichannel,
            use_core_cg=model_ctx.use_core_cg,
            use_basis_solve=model_ctx.use_basis_solve,
        )
        self.basis_updater = BasisUpdater(
            config=config,
            device=device,
            ind_input=data_ctx.ind_input,
            train_reg_cfg=self.train_reg_cfg,
            loss_computer=self.loss_computer,
            optimizer=model_ctx.optimizer,
            core_adam_lr=model_ctx.core_adam_lr,
            use_core_adam=model_ctx.use_core_adam,
            use_basis_solve=model_ctx.use_basis_solve,
            use_rkhs_basis=model_ctx.use_rkhs_basis,
            k_inv_t=model_ctx.k_inv_t,
            k_inv_u=model_ctx.k_inv_u,
            k_inv_v=model_ctx.k_inv_v,
            k_inv_w=model_ctx.k_inv_w,
        )

    def _pde_flags(self, step, *, log_step=False):
        pde_active = bool(self.pde_ctx.use_pde_loss) and int(step) >= int(
            self.config.pde_start_step
        )
        monitor_only = (
            bool(self.config.pde_monitor_always)
            and bool(self.config.pde_monitor_point)
            and bool(log_step)
        )
        pde_monitor = bool(self.pde_ctx.use_pde_module) and (pde_active or monitor_only)
        pde_compute_point = bool(pde_monitor) and (
            self.pde_ctx.lambda_pde_point > 0.0 or monitor_only
        )
        return (pde_active, pde_monitor, pde_compute_point)

    def _lambda_map(self, pde_active):
        return {
            "rec": 1.0,
            "rkhs": self.train_reg_cfg["lambda_rkhs"],
            "time": self.train_reg_cfg["lambda_time"],
            "pde_point": self.pde_ctx.lambda_pde_point if pde_active else 0.0,
        }

    def _snapshot(self, payload):
        if self.snapshot_callback is not None:
            self.snapshot_callback(payload)

    def _run_initial_pass(self):
        model = self.model_ctx.model
        model.train()
        pde_active, pde_monitor, pde_compute_point = self._pde_flags(0, log_step=True)
        init_losses = []
        init_components = []
        init_time_total = 0.0
        init_batch_counter = 0
        last_payload = None
        with torch.no_grad():
            for data_batch, mask_batch, batch_idx in self.data_ctx.train_loader:
                init_batch_counter += 1
                t_batch0 = datetime.now().timestamp()
                data_batch = data_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)
                batch_idx = batch_idx.to(self.device, non_blocking=True)
                model_ref = current_model(model, self.ddp_state)
                basis_module = model_ref.basis_function
                core_batch = model_ref.get_core_batch(
                    batch_idx, detach=True, device=self.device
                )
                result = self.loss_computer.compute(
                    model_ref=model_ref,
                    basis_module=basis_module,
                    core_batch=core_batch,
                    data_batch=data_batch,
                    mask_batch=mask_batch,
                    batch_idx=batch_idx,
                    pde_active=pde_active,
                    pde_monitor=pde_monitor,
                    pde_compute_point=pde_compute_point,
                    pde_grad_enabled=False,
                )
                init_components.append(result.components)
                init_losses.append(float(result.loss.detach().item()))
                init_time_total += datetime.now().timestamp() - t_batch0
                last_payload = BatchLogPayload(
                    step=0,
                    step_tag="step00000_preupdate",
                    batch_idx=batch_idx,
                    data_batch=data_batch,
                    mask_batch=mask_batch,
                )
        init_mean_loss = (
            reduce_mean_scalar(float(np.mean(init_losses)), self.ddp_state, self.device)
            if init_losses
            else 0.0
        )
        if is_main_process(self.ddp_state):
            comp_str = ""
            if init_components:
                comp_str = format_loss_component_log(
                    init_components, self._lambda_map(pde_active), use_latest_pde=True
                )
            print(
                f"epoch: 0  loss: {init_mean_loss:.6f}  [spatial_only core_update={self.config.core_update_mode} pde_active={pde_active} pde_monitor={pde_monitor} pde_start={int(self.config.pde_start_step)} pre_update=True | {comp_str}]"
            )
            if init_batch_counter > 0:
                print(
                    f"[timing step=0 summary] per-batch avg: total={init_time_total / init_batch_counter:.3f}s core=0.000s basis=0.000s other={init_time_total / init_batch_counter:.3f}s"
                )
                if last_payload is not None:
                    self._snapshot(last_payload)

    def _initial_eval(self):
        init_eval_every = max(int(self.config.eval_every), 1)
        if 0 % init_eval_every == 0 or int(self.config.max_epochs) == 0:
            init_rmse, init_mae, init_rel_l2 = evaluate(
                self.model_ctx.model,
                self.data_ctx.train_loader,
                self.data_ctx.ind_input,
                self.ddp_state,
                self.device,
                self.data_ctx.norm_stats,
            )
            if is_main_process(self.ddp_state):
                print(
                    f"full field eval (physical, step=0 pre-update) lr={current_optimizer_lr(self.model_ctx.optimizer, self.config.learning_rate):.3e} rmse: {init_rmse} mae: {init_mae} rel_l2: {init_rel_l2}"
                )

    def run(self):
        self._run_initial_pass()
        self._initial_eval()
        best_rmse = float("inf")
        best_mae = float("inf")
        best_rel_l2 = float("inf")
        best_step = -1
        iterator = range(1, int(self.config.max_epochs) + 1)
        if is_main_process(self.ddp_state):
            iterator = tqdm(iterator, desc="Epoch")
        print_every = max(int(self.config.print_every), 1)
        for step in iterator:
            if self.data_ctx.train_sampler is not None:
                self.data_ctx.train_sampler.set_epoch(step)
            self.model_ctx.model.train()
            should_print_step = step % print_every == 0
            pde_active, pde_monitor, pde_compute_point = self._pde_flags(
                step, log_step=should_print_step
            )
            losses = []
            loss_components_accum = []
            step_time_total = 0.0
            step_time_core = 0.0
            step_time_basis = 0.0
            step_time_other = 0.0
            batch_counter = 0
            last_payload = None
            for data_batch, mask_batch, batch_idx in self.data_ctx.train_loader:
                batch_counter += 1
                t_batch0 = datetime.now().timestamp()
                data_batch = data_batch.to(self.device, non_blocking=True)
                mask_batch = mask_batch.to(self.device, non_blocking=True)
                batch_idx = batch_idx.to(self.device, non_blocking=True)
                model_ref = current_model(self.model_ctx.model, self.ddp_state)
                basis_module = model_ref.basis_function
                t_core0 = datetime.now().timestamp()
                if (
                    is_main_process(self.ddp_state)
                    and self.config.profile_every
                    and (batch_counter % self.config.profile_every == 0)
                ):
                    sync_if_cuda(self.device)
                with torch.no_grad():
                    precomputed_basis = self.core_updater.precompute_basis(basis_module)
                self.core_updater.update(
                    model_ref=model_ref,
                    basis_module=basis_module,
                    data_batch=data_batch,
                    mask_batch=mask_batch,
                    batch_idx=batch_idx,
                    precomputed_basis=precomputed_basis,
                    device=self.device,
                )
                if (
                    is_main_process(self.ddp_state)
                    and self.config.profile_every
                    and (batch_counter % self.config.profile_every == 0)
                ):
                    sync_if_cuda(self.device)
                t_core1 = datetime.now().timestamp()
                t_basis0 = datetime.now().timestamp()
                if (
                    is_main_process(self.ddp_state)
                    and self.config.profile_every
                    and (batch_counter % self.config.profile_every == 0)
                ):
                    sync_if_cuda(self.device)
                batch_losses, batch_components = self.basis_updater.update(
                    model_ref=model_ref,
                    basis_module=basis_module,
                    data_batch=data_batch,
                    mask_batch=mask_batch,
                    batch_idx=batch_idx,
                    precomputed_basis=precomputed_basis,
                    pde_active=pde_active,
                    pde_monitor=pde_monitor,
                    pde_compute_point=pde_compute_point,
                )
                losses.extend(batch_losses)
                loss_components_accum.extend(batch_components)
                if (
                    is_main_process(self.ddp_state)
                    and self.config.profile_every
                    and (batch_counter % self.config.profile_every == 0)
                ):
                    sync_if_cuda(self.device)
                t_basis1 = datetime.now().timestamp()
                dt_total = datetime.now().timestamp() - t_batch0
                dt_core = t_core1 - t_core0
                dt_basis = t_basis1 - t_basis0
                step_time_total += dt_total
                step_time_core += dt_core
                step_time_basis += dt_basis
                step_time_other += max(0.0, dt_total - dt_core - dt_basis)
                if (
                    is_main_process(self.ddp_state)
                    and self.config.profile_every
                    and (batch_counter % self.config.profile_every == 0)
                ):
                    print(
                        f"[timing step={step} batch={batch_counter}] total={dt_total:.3f}s core={dt_core:.3f}s basis={dt_basis:.3f}s other={max(0.0, dt_total - dt_core - dt_basis):.3f}s (als_basis_steps={max(1, self.config.als_basis_steps)} als_core_steps={max(1, self.config.als_core_steps)})"
                    )
                last_payload = BatchLogPayload(
                    step=step,
                    step_tag=f"step{int(step):05d}",
                    batch_idx=batch_idx,
                    data_batch=data_batch,
                    mask_batch=mask_batch,
                )
            if should_print_step:
                mean_loss = reduce_mean_scalar(
                    float(np.mean(losses)), self.ddp_state, self.device
                )
                if is_main_process(self.ddp_state):
                    comp_str = ""
                    if loss_components_accum:
                        comp_str = format_loss_component_log(
                            loss_components_accum,
                            self._lambda_map(pde_active),
                            use_latest_pde=True,
                        )
                    print(
                        f"epoch: {step}  loss: {mean_loss:.6f}  [spatial_only core_update={self.config.core_update_mode} pde_active={pde_active} pde_monitor={pde_monitor} pde_start={int(self.config.pde_start_step)} lr={current_optimizer_lr(self.model_ctx.optimizer, self.config.learning_rate):.3e} | {comp_str}]"
                    )
                    if batch_counter > 0:
                        print(
                            f"[timing step={step} summary] per-batch avg: total={step_time_total / batch_counter:.3f}s core={step_time_core / batch_counter:.3f}s basis={step_time_basis / batch_counter:.3f}s other={step_time_other / batch_counter:.3f}s"
                        )
                        if last_payload is not None:
                            self._snapshot(last_payload)
            if step % self.config.eval_every == 0 or step == self.config.max_epochs:
                rmse, mae, rel_l2 = evaluate(
                    self.model_ctx.model,
                    self.data_ctx.train_loader,
                    self.data_ctx.ind_input,
                    self.ddp_state,
                    self.device,
                    self.data_ctx.norm_stats,
                )
                is_new_best = rmse < best_rmse
                if is_new_best:
                    best_rmse = rmse
                    best_mae = mae
                    best_rel_l2 = rel_l2
                    best_step = step
                if is_main_process(self.ddp_state):
                    print(
                        f"full field eval (physical) lr={current_optimizer_lr(self.model_ctx.optimizer, self.config.learning_rate):.3e} rmse: {rmse} mae: {mae} rel_l2: {rel_l2}"
                    )
                if self.checkpoint_callback is not None:
                    self.checkpoint_callback(
                        EvalPayload(
                            step=step,
                            rmse=rmse,
                            mae=mae,
                            rel_l2=rel_l2,
                            best_rmse=best_rmse,
                            best_mae=best_mae,
                            best_rel_l2=best_rel_l2,
                            best_step=best_step,
                            is_new_best=is_new_best,
                        )
                    )
        return LoopResult(
            best_rmse=best_rmse,
            best_mae=best_mae,
            best_rel_l2=best_rel_l2,
            best_step=best_step,
        )
