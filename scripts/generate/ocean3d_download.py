#!/usr/bin/env python3
"""Download the annual native-grid inputs used by PhysFTM Ocean3D.

Requires Python >=3.10. Authenticate with `copernicusmarine login` first.
Default: 1993--2017, daily uo/vo, released 64x64 crop, first 32 depth levels.
No interpolation, resampling, normalization, or missing-value replacement.
"""

import argparse
import calendar
from datetime import date, timedelta
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import tempfile


DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
DATASET_VERSION = "202311"
YEARS = range(1993, 2018)
LAT_RANGE = (-2.6666667461395264, 2.5833332538604736)
LON_RANGE = (167.3333282470703, 172.5833282470703)
DEPTH_VALUES = [
    0.49402499198913574, 1.5413750410079956, 2.6456689834594727,
    3.8194949626922607, 5.078224182128906, 6.440614223480225,
    7.92956018447876, 9.572997093200684, 11.404999732971191,
    13.467140197753906, 15.810070037841797, 18.495559692382812,
    21.598819732666016, 25.211410522460938, 29.444730758666992,
    34.43415069580078, 40.344051361083984, 47.37369155883789,
    55.76428985595703, 65.80726623535156, 77.85385131835938,
    92.3260726928711, 109.72930145263672, 130.66600036621094,
    155.85069274902344, 186.12559509277344, 222.47520446777344,
    266.0403137207031, 318.1274108886719, 380.2130126953125,
    453.9377136230469, 541.0889282226562,
]


def filename(year):
    # Keep the existing preprocessor's discovery pattern and annual boundaries.
    return f"uv_fulldepth_{year}-01-01_to_{year}-12-31.nc"


def subset_request(year, output_dir):
    # Tiny padding avoids excluding boundary points after decimal rounding.
    # It is much smaller than a grid cell or a depth-level separation.
    eps = 2e-5
    return dict(
        dataset_id=DATASET_ID,
        dataset_version=DATASET_VERSION,
        variables=["uo", "vo"],
        minimum_latitude=LAT_RANGE[0] - eps,
        maximum_latitude=LAT_RANGE[1] + eps,
        minimum_longitude=LON_RANGE[0] - eps,
        maximum_longitude=LON_RANGE[1] + eps,
        minimum_depth=0.0,
        maximum_depth=DEPTH_VALUES[-1] + eps,
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T23:59:59",
        coordinates_selection_method="inside",
        output_directory=str(output_dir),
        output_filename=filename(year),
        file_format="netcdf",
        netcdf_compression_level=1,
        overwrite=True,
    )


def validate_file(path, year):
    """Check the coordinate/time/variable contract consumed by preprocessing."""
    import netCDF4 as nc
    import numpy as np

    with nc.Dataset(path) as ds:
        coords = {}
        for name, count, bounds in (
            ("latitude", 64, LAT_RANGE), ("longitude", 64, LON_RANGE)
        ):
            var = ds.variables[name]
            values = np.asarray(var[:], dtype=np.float64)
            if var.dimensions != (name,) or values.shape != (count,):
                raise ValueError(f"{path.name}: {name} must contain {count} points")
            if not np.all(np.diff(values) > 0) or not np.allclose(
                values[[0, -1]], bounds, rtol=0, atol=2e-5
            ) or not np.allclose(np.diff(values), 1 / 12, rtol=0, atol=2e-5):
                raise ValueError(f"{path.name}: unexpected native {name} grid")
            coords[name] = values.astype(np.float32).tolist()
        depth = np.asarray(ds.variables["depth"][:], dtype=np.float32)
        if depth.shape != (32,) or not np.allclose(
            depth, DEPTH_VALUES, rtol=1e-6, atol=1e-6
        ):
            raise ValueError(f"{path.name}: first 32 source depths do not match")
        coords["depth"] = depth.tolist()
        t = ds.variables["time"]
        timestamps = nc.num2date(
            t[:], units=t.units, calendar=getattr(t, "calendar", "standard")
        )
        expected_days = 366 if calendar.isleap(year) else 365
        expected_dates = [
            (date(year, 1, 1) + timedelta(days=i)).isoformat()
            for i in range(expected_days)
        ]
        if [v.strftime("%Y-%m-%d") for v in timestamps] != expected_dates or any(
            (b - a).total_seconds() != 86400
            for a, b in zip(timestamps[:-1], timestamps[1:])
        ):
            raise ValueError(f"{path.name}: expected a complete consecutive daily year")
        for name in ("uo", "vo"):
            var = ds.variables[name]
            if var.dimensions != ("time", "depth", "latitude", "longitude") or (
                var.shape != (expected_days, 32, 64, 64)
            ):
                raise ValueError(f"{path.name}: incompatible {name} dimensions")
            # Read boundary slabs to catch unreadable or nonfinite unmasked data.
            # Retain source masks; the supplied preprocessor fills masked cells.
            for index in (0, expected_days - 1):
                values = np.ma.asarray(var[index]).compressed()
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{path.name}: nonfinite unmasked {name} values")
        if getattr(ds, "physftm_dataset_version", None) != DATASET_VERSION or (
            getattr(ds, "physftm_dataset_id", None) != DATASET_ID
        ):
            raise ValueError(
                f"{path.name}: missing/mismatched downloader provenance; "
                "use a clean output directory or --overwrite"
            )
    return {"file": path.name, "year": year, "days": expected_days, **coords}


def sample_plan():
    records = []
    for year in YEARS:
        for block in range(4):
            start = date(year, 1, 1) + timedelta(days=90 * block)
            records.append(dict(
                sample_index=len(records), source_file=filename(year),
                time_start_index=90 * block, time_stop_index_exclusive=90 * (block + 1),
                start_date=start.isoformat(), end_date=(start + timedelta(days=89)).isoformat(),
            ))
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ocean/daily_my"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print 25 requests offline; no login/download")
    mode.add_argument("--check-only", action="store_true", help="Validate all 25 existing annual files offline")
    mode.add_argument("--overwrite", action="store_true", help="Replace existing annual files after validating downloads")
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if args.dry_run:
        print(json.dumps([subset_request(y, output_dir) for y in YEARS], indent=2))
        return 0

    expected_files = {filename(y) for y in YEARS}
    extras = {p.name for p in output_dir.glob("uv_fulldepth_*.nc")} - expected_files
    if extras:
        raise ValueError("Extra input files may alter preprocessing order: " + ", ".join(sorted(extras)))

    if not args.check_only:
        import copernicusmarine as cm
        # Fail before downloading if an incompatible Toolbox API is installed.
        inspect.signature(cm.subset).bind(**subset_request(1993, output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
    annual = []
    for year in YEARS:
        target = output_dir / filename(year)
        if args.check_only or (target.exists() and not args.overwrite):
            info = validate_file(target, year)
            print(f"Validated existing: {target.name}", flush=True)
        else:
            print(f"Downloading: {target.name}", flush=True)
            # A failed/interrupted transfer never becomes a discoverable annual file.
            with tempfile.TemporaryDirectory(prefix=f".download-{year}-", dir=output_dir) as stage:
                cm.subset(**subset_request(year, Path(stage)))
                staged = Path(stage) / filename(year)
                import netCDF4 as nc
                with nc.Dataset(staged, "a") as ds:
                    ds.physftm_dataset_id = DATASET_ID
                    ds.physftm_dataset_version = DATASET_VERSION
                    ds.physftm_source_subset = "Native 64x64 crop; source depth indices 0:32"
                info = validate_file(staged, year)
                if annual and any(info[k] != annual[0][k] for k in ("latitude", "longitude", "depth")):
                    raise ValueError(f"{target.name}: coordinates differ between annual files")
                staged.replace(target)
        if annual and any(info[k] != annual[0][k] for k in ("latitude", "longitude", "depth")):
            raise ValueError(f"{target.name}: coordinates differ between annual files")
        annual.append(info)

    records = sample_plan()
    splits = {}
    for name, begin, end in (("train", 0, 80), ("val", 80, 90), ("test", 90, 100)):
        splits[name] = dict(sample_indices=list(range(begin, end)),
                            start_date=records[begin]["start_date"],
                            end_date=records[end - 1]["end_date"])
    versions = {}
    for package in ("copernicusmarine", "numpy", "netCDF4"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    manifest = dict(dataset_id=DATASET_ID, dataset_version=DATASET_VERSION,
                    annual_files=annual, sample_records=records, splits=splits,
                    versions=versions, source_records=sum(r["days"] for r in annual),
                    retained_records=9000, sample_shape=[90, 32, 64, 64, 2],
                    sequence_rule="Four consecutive nonoverlapping 90-day blocks per annual file; discard trailing days")
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Ready: 25 annual files, 100 sequences, 9000 retained daily fields.\n{manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Download/validation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
