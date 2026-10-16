"""Callbacks that connect RKHS training events to diagnostics and artifacts."""

from physftm.distributed import is_main_process
from physftm.checkpointing.spatial import (
    gather_core_table_for_saving,
    save_checkpoint_bundle,
)
from physftm.diagnostics.snapshots import (
    parse_pde_snapshot_sample_indices,
    save_pde_residual_snapshot,
    save_reconstruction_snapshot,
)
from physftm.pde.training import (
    _restore_active_pde_query_chunk_size,
    _set_active_pde_query_chunk_size,
)
from physftm.training.runtime import current_model


def build_rkhs_snapshot_callback(
    *, config, ddp_state, data_ctx, model_ctx, pde_ctx, experiment_root_dir
):
    """Create the log-time snapshot callback used by TrainingLoop."""

    def _snapshot_callback(payload):
        if not is_main_process(ddp_state):
            return
        model_ref = current_model(model_ctx.model, ddp_state)
        basis_module = model_ref.basis_function
        if bool(config.plot_pde_residual_on_log) and bool(pde_ctx.use_pde_module):
            prev_chunk = _set_active_pde_query_chunk_size(
                basis_module, int(config.pde_query_chunk_size)
            )
            snapshot_sample_ratio = float(config.pde_point_sample_ratio)
            snapshot_max_points = int(config.pde_point_max_points)
            print(
                f"PDE residual snapshot sampling: ratio={snapshot_sample_ratio} max_points={snapshot_max_points} source={'pde_point_*'}"
            )
            try:
                save_pde_residual_snapshot(
                    model=model_ctx.model,
                    ddp_state=ddp_state,
                    batch_idx=payload.batch_idx,
                    data_batch=payload.data_batch,
                    mask_batch=payload.mask_batch,
                    ind_input=data_ctx.ind_input,
                    norm_stats=data_ctx.norm_stats,
                    pde_kwargs={
                        **pde_ctx.pde_kwargs,
                        "dataset_type": config.dataset_type,
                        "channel_order": pde_ctx.pde_channel_order,
                    },
                    point_sample_ratio=snapshot_sample_ratio,
                    point_max_points=snapshot_max_points,
                    shared_basis_multichannel=data_ctx.shared_basis_multichannel,
                    experiment_root_dir=experiment_root_dir,
                    step_tag=payload.step_tag,
                    sample_in_batch=parse_pde_snapshot_sample_indices(config),
                    pde_context=data_ctx.dataset_pde_context,
                )
            except Exception as exc:
                print(
                    f"[warn] failed to save PDE residual snapshot at step {payload.step}: {exc}"
                )
            finally:
                _restore_active_pde_query_chunk_size(basis_module, prev_chunk)
        elif bool(config.plot_reconstruction_on_log) and (
            not bool(pde_ctx.use_pde_module) or str(config.dataset_type).lower() == "ns"
        ):
            try:
                save_reconstruction_snapshot(
                    model=model_ctx.model,
                    ddp_state=ddp_state,
                    batch_idx=payload.batch_idx,
                    data_batch=payload.data_batch,
                    mask_batch=payload.mask_batch,
                    ind_input=data_ctx.ind_input,
                    norm_stats=data_ctx.norm_stats,
                    shared_basis_multichannel=data_ctx.shared_basis_multichannel,
                    experiment_root_dir=experiment_root_dir,
                    step_tag=payload.step_tag,
                    sample_in_batch=parse_pde_snapshot_sample_indices(config),
                )
            except Exception as exc:
                print(
                    f"[warn] failed to save reconstruction snapshot at step {payload.step}: {exc}"
                )

    return _snapshot_callback


def build_rkhs_checkpoint_callback(
    *,
    config,
    ddp_state,
    device,
    data_ctx,
    model_ctx,
    experiment_root_dir,
    latest_save_dir,
    best_save_dir,
    train_start_time,
):
    """Create the eval-time checkpoint callback used by TrainingLoop."""

    def _checkpoint_callback(payload):
        if ddp_state["distributed"]:
            core_to_save = gather_core_table_for_saving(
                model_ctx.model, data_ctx.train_sampler, ddp_state, device
            )
        else:
            core_to_save = model_ctx.model.tucker_core.detach().cpu()
        if not is_main_process(ddp_state):
            return
        basis_to_save = (
            model_ctx.model.module.basis_function
            if ddp_state["distributed"]
            else model_ctx.model.basis_function
        )
        latest_paths = save_checkpoint_bundle(
            save_dir=latest_save_dir,
            experiment_root_dir=experiment_root_dir,
            save_kind="latest",
            core_to_save=core_to_save,
            basis_to_save=basis_to_save,
            norm_stats=data_ctx.norm_stats,
            ind_input=data_ctx.ind_input,
            t_coords=data_ctx.t_coords_physical,
            config=config,
            rank=model_ctx.rank,
            shared_basis_multichannel=data_ctx.shared_basis_multichannel,
            num_channels=data_ctx.num_channels,
            best_rmse=payload.best_rmse,
            best_mae=payload.best_mae,
            best_rel_l2=payload.best_rel_l2,
            eval_rmse=payload.rmse,
            eval_mae=payload.mae,
            eval_rel_l2=payload.rel_l2,
            best_step=payload.best_step,
            step=payload.step,
            train_start_time=train_start_time,
        )
        print("saved latest:", *latest_paths)
        if payload.is_new_best:
            best_paths = save_checkpoint_bundle(
                save_dir=best_save_dir,
                experiment_root_dir=experiment_root_dir,
                save_kind="best",
                core_to_save=core_to_save,
                basis_to_save=basis_to_save,
                norm_stats=data_ctx.norm_stats,
                ind_input=data_ctx.ind_input,
                t_coords=data_ctx.t_coords_physical,
                config=config,
                rank=model_ctx.rank,
                shared_basis_multichannel=data_ctx.shared_basis_multichannel,
                num_channels=data_ctx.num_channels,
                best_rmse=payload.best_rmse,
                best_mae=payload.best_mae,
                best_rel_l2=payload.best_rel_l2,
                eval_rmse=payload.rmse,
                eval_mae=payload.mae,
                eval_rel_l2=payload.rel_l2,
                best_step=payload.best_step,
                step=payload.step,
                train_start_time=train_start_time,
            )
            print("saved best:", *best_paths)

    return _checkpoint_callback
