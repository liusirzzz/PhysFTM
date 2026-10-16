import time as _time
import torch
from physftm.distributed import setup_distributed, cleanup_distributed, is_main_process
from physftm.artifacts.callbacks import (
    build_rkhs_checkpoint_callback,
    build_rkhs_snapshot_callback,
)
from physftm.training.setup import (
    build_data_context,
    build_model_context,
    build_pde_context,
    close_experiment_logging,
    parse_rkhs_config_args,
    set_seed,
    setup_experiment_logging,
)
from physftm.training.loop import TrainingLoop


def main():
    ddp_state = setup_distributed(
        default_device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    device = ddp_state["device"]
    config = parse_rkhs_config_args()
    seed = set_seed(config.seed)
    experiment_ctx = setup_experiment_logging(config, ddp_state, device)
    experiment_root_dir = experiment_ctx.experiment_root_dir
    latest_save_dir = experiment_ctx.latest_save_dir
    best_save_dir = experiment_ctx.best_save_dir
    if is_main_process(ddp_state):
        print("seed:", seed)
    data_ctx = build_data_context(config, ddp_state, device)
    model_ctx = build_model_context(config, data_ctx, ddp_state, device)
    pde_ctx = build_pde_context(config, data_ctx, ddp_state)
    train_start_time = _time.time()
    snapshot_callback = build_rkhs_snapshot_callback(
        config=config,
        ddp_state=ddp_state,
        data_ctx=data_ctx,
        model_ctx=model_ctx,
        pde_ctx=pde_ctx,
        experiment_root_dir=experiment_root_dir,
    )
    checkpoint_callback = build_rkhs_checkpoint_callback(
        config=config,
        ddp_state=ddp_state,
        device=device,
        data_ctx=data_ctx,
        model_ctx=model_ctx,
        experiment_root_dir=experiment_root_dir,
        latest_save_dir=latest_save_dir,
        best_save_dir=best_save_dir,
        train_start_time=train_start_time,
    )
    loop = TrainingLoop(
        config=config,
        ddp_state=ddp_state,
        device=device,
        data_ctx=data_ctx,
        model_ctx=model_ctx,
        pde_ctx=pde_ctx,
        snapshot_callback=snapshot_callback,
        checkpoint_callback=checkpoint_callback,
    )
    loop.run()
    cleanup_distributed(ddp_state)
    close_experiment_logging(experiment_ctx)


if __name__ == "__main__":
    main()
