# PhysFTM

## Repository Structure

| Path | Purpose |
|---|---|
| `train_spatial.py` / `reconstruct_spatial.py` | Spatial training / reconstruction |
| `train_temporal.py` / `reconstruct_temporal.py` | Temporal training / missing-frame reconstruction |
| `configs/<dataset>/` | Four YAML configurations per dataset for the main workflow, 16 in total |
| `physftm/` | RKHS factors, core solvers, temporal models, PDE constraints, and result export |
| `scripts/generate/` | Synthetic data generation and ocean data download |
| `scripts/preprocess/` | Data splitting, normalization metadata, and sensor masks |
| `scripts/protocols/` | Random-drop-half and temporal-interval-missing evaluation protocols |

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

For NS2D:

```bash
python scripts/generate/ns2d.py
python scripts/preprocess/ns2d.py
```

For AC2D and Heat3D, use the corresponding `ac2d.py` and `heat3d.py` scripts in the same directories. The generators use the dataset sizes reported in the paper. Check the parameters at the top of each script before running.

For Ocean3D, download the source data first:

```bash
python scripts/generate/ocean3d_download.py
```

The preprocessing input and output directories can be configured through command-line arguments.

Generate temporal evaluation masks, for example, for NS2D:

```bash
python scripts/protocols/random_drop_half.py --dataset NS2D --ns2d-dir data/processed/ns2d
python scripts/protocols/interval_missing.py --dataset NS2D --ns2d-dir data/processed/ns2d
```

## Main Workflow

The following example uses NS2D. Replace the placeholder paths with the absolute paths to your files.

```bash
export PHYSFTM_NS_BASE_MANIFEST=/absolute/path/base_manifest.json
python train_spatial.py --config configs/ns2d/spatial_train.yaml

# Spatial training saves ftm_*.json and its dependencies under artifacts/<experiment>/best/.
export PHYSFTM_NS_FTM_MANIFEST=/absolute/path/artifacts/experiment/best/ftm_model.json
python reconstruct_spatial.py --config configs/ns2d/spatial_infer.yaml

export PHYSFTM_NS_TEMPORAL_MODEL=/absolute/path/models/ns_temporal.pth
python train_temporal.py --config configs/ns2d/temporal_train.yaml

export PHYSFTM_NS_EVAL_MANIFEST=/absolute/path/temporal_protocol_manifest.json
python reconstruct_temporal.py --config configs/ns2d/temporal_infer.yaml
```