import torch
import numpy as np
import h5py


def load_large_data(data_path, metadata_path):
    data_path = str(data_path)
    if data_path.lower().endswith((".h5", ".hdf5")):
        with h5py.File(data_path, "r") as f:
            data_extract = f["data"][:]
    elif data_path.lower().endswith(".npy"):
        payload = np.load(data_path, allow_pickle=True).item()
        if isinstance(payload, dict):
            if "data" not in payload:
                raise KeyError(
                    f"Numpy payload at {data_path} is missing required key 'data'"
                )
            data_extract = np.asarray(payload["data"])
        else:
            raise ValueError(
                f"Unsupported numpy payload type at {data_path}: {type(payload)}"
            )
    else:
        raise ValueError(
            f"Unsupported data file format for {data_path}; expected .h5/.hdf5 or .npy"
        )
    d = np.load(metadata_path, allow_pickle=True).item()
    d = d["data"]
    if "mask_tr" in d:
        mask = d["mask_tr"]
    elif "mask_val" in d:
        mask = d["mask_val"]
    elif "mask_te" in d:
        mask = d["mask_te"]
    else:
        raise KeyError(
            f"No mask found in metadata {metadata_path}; expected one of mask_tr/mask_val/mask_te"
        )
    depth_key = (
        "w_ind_uni" if "w_ind_uni" in d else "d_ind_uni" if "d_ind_uni" in d else None
    )
    if depth_key is not None:
        ind_uni = (d["u_ind_uni"], d["v_ind_uni"], d[depth_key], d["t_ind_uni"])
        if mask.ndim not in (4, 5, 6):
            raise ValueError(
                f"Metadata 3D mask must be (N,T,H,W[,C]) or (N,T,D,H,W[,C]); got {mask.shape}"
            )
    else:
        w_ind_uni = np.zeros((1,), dtype=np.float32)
        ind_uni = (d["u_ind_uni"], d["v_ind_uni"], w_ind_uni, d["t_ind_uni"])
        if mask.ndim == 4:
            mask = mask[:, :, :, :, None]
        elif mask.ndim == 5:
            pass
        else:
            raise ValueError(
                f"Metadata mask must be per-sequence (N,T,H,W[,1]) after fix; got {mask.shape} in {metadata_path}"
            )
    return (ind_uni, data_extract, mask)


def align_3d_mask_to_data(mask: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
    """Align voxel-wise or depth-shared masks to (N,T,D,H,W,C)."""
    if data.dim() != 6:
        raise ValueError(f"3D data must be (N,T,D,H,W,C), got {tuple(data.shape)}")
    n, t, depth, height, width, channels = (int(x) for x in data.shape)
    if int(mask.shape[0]) != n or int(mask.shape[1]) != t:
        raise ValueError(
            f"3D mask sample/time shape {tuple(mask.shape[:2])} does not match data {(n, t)}"
        )
    if mask.dim() == 4 and tuple(mask.shape[2:]) == (height, width):
        mask = mask.unsqueeze(2).unsqueeze(-1).expand(n, t, depth, height, width, 1)
    elif mask.dim() == 5 and tuple(mask.shape[2:]) == (depth, height, width):
        mask = mask.unsqueeze(-1)
    elif (
        mask.dim() == 5
        and tuple(mask.shape[2:4]) == (height, width)
        and (int(mask.shape[-1]) in (1, channels))
    ):
        mask = mask.unsqueeze(2).expand(n, t, depth, height, width, int(mask.shape[-1]))
    elif mask.dim() == 6:
        if tuple(mask.shape[2:5]) != (depth, height, width):
            raise ValueError(
                f"3D mask spatial shape {tuple(mask.shape[2:5])} does not match data {(depth, height, width)}"
            )
    else:
        raise ValueError(
            f"3D mask must be voxel-wise (N,T,D,H,W[,C]) or depth-shared (N,T,H,W[,C]); got {tuple(mask.shape)}"
        )
    if int(mask.shape[-1]) not in (1, channels):
        raise ValueError(
            f"3D mask channels={int(mask.shape[-1])} are incompatible with data channels={channels}"
        )
    return mask


def get_mask_batch(mask, batch_ind, data_batch=None, device=None):
    """Return a per-batch mask tensor aligned with `data_batch`.

        Accepts:
            - mask: torch.Tensor or np.ndarray
                * preferred: (N,H,W,C) spatial mask shared across time
                * also supports legacy: (N,T,H,W,C)
      - batch_ind: torch.LongTensor of shape (B,) giving sequence indices.

    Returns:
      - mask_batch: torch.Tensor of shape (B,H,W,C) or (B,T,H,W,C)
    """
    if not torch.is_tensor(mask):
        mask_t = torch.as_tensor(mask)
    else:
        mask_t = mask
    if mask_t.dim() not in (4, 5, 6):
        raise ValueError(
            f"Mask must be per-sequence 2D/3D (N,H,W[,C])/(N,D,H,W[,C]) or time-aware; got dim={mask_t.dim()} shape={tuple(mask_t.shape)}"
        )
    index_t = (
        batch_ind.to(device=mask_t.device, dtype=torch.long)
        if torch.is_tensor(batch_ind)
        else torch.as_tensor(batch_ind, device=mask_t.device, dtype=torch.long)
    )
    mask_batch = mask_t.index_select(0, index_t)
    if device is not None:
        mask_batch = mask_batch.to(device)
    if data_batch is not None:
        if mask_batch.dim() == 4 and data_batch.dim() == 5:
            mask_batch = mask_batch.unsqueeze(1)
        if (
            mask_batch.dim() == 5
            and data_batch.dim() == 5
            and (mask_batch.shape[1] == 1)
            and (data_batch.shape[1] != 1)
        ):
            mask_batch = mask_batch.expand(
                mask_batch.shape[0], data_batch.shape[1], *mask_batch.shape[2:]
            )
        if mask_batch.shape[-1] == 1 and data_batch.shape[-1] != 1:
            mask_batch = mask_batch.expand(*mask_batch.shape[:-1], data_batch.shape[-1])
    return mask_batch
