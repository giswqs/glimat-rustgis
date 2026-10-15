"""Runs the Rasters to NetCDF tool as RustGIS does."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parent.parent / "main.py"


def run(params, cwd):
    """Runs ``main.py`` with a raster-to-nc request and returns its result message.

    Args:
        params: The tool's parameters.
        cwd: The folder to run in.

    Returns:
        The last message: ``{"ok": ...}`` or ``{"error": ...}``.
    """
    request = {
        "abi": 1,
        "kind": "tool",
        "id": "raster-to-nc",
        "params": params,
        "layers": {},
        "lang": "en",
    }
    proc = subprocess.run(
        [sys.executable, str(MAIN)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()][-1]


def test_rasters_become_one_model_file(tmp_path):
    import rasterio
    import xarray as xr
    from rasterio.transform import from_origin

    rasters = tmp_path / "thk"
    rasters.mkdir()
    for i, year in enumerate([1000, 1100, 1200]):
        data = np.full((4, 5), 10.0 * (i + 1), dtype="float32")
        with rasterio.open(
            rasters / f"thk_{year}.tif",
            "w",
            driver="GTiff",
            height=4,
            width=5,
            count=1,
            dtype="float32",
            crs="EPSG:3413",
            transform=from_origin(0, 2000, 500, 500),
        ) as dst:
            dst.write(data, 1)
    out = tmp_path / "out" / "model.nc"
    reply = run({"thick_folder": str(rasters), "output_nc": str(out)}, tmp_path)
    assert "ok" in reply, reply
    assert str(out) in reply["ok"]["messages"][0]
    ds = xr.open_dataset(out, decode_times=False)
    assert ds.sizes["time"] == 3 and ds.sizes["x"] == 5 and ds.sizes["y"] == 4
    thk = next(v for v in ds.data_vars if "th" in v.lower())
    assert np.allclose(ds[thk].isel(time=2).values, 30.0)


def test_missing_folders_are_errors(tmp_path):
    assert (
        "is required" in run({"output_nc": str(tmp_path / "m.nc")}, tmp_path)["error"]
    )
    assert (
        "no folder at"
        in run(
            {
                "thick_folder": str(tmp_path / "nope"),
                "output_nc": str(tmp_path / "m.nc"),
            },
            tmp_path,
        )["error"]
    )
