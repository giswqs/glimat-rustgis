"""Runs the Point Comparison (STPC) tool as RustGIS does, on a synthetic model."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MAIN = Path(__file__).resolve().parent.parent / "main.py"
CRS = "EPSG:3413"


def run(request, cwd):
    """Runs ``main.py`` with ``request``; returns its last message.

    Args:
        request: The request, as RustGIS sends it.
        cwd: The folder to run in.

    Returns:
        The result message: ``{"ok": ...}`` or ``{"error": ...}``.
    """
    proc = subprocess.run(
        [sys.executable, str(MAIN)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()][-1]


@pytest.fixture
def model(tmp_path):
    """A model whose ice is 100 m thick everywhere at time step 1, 200 m at 2, and so on.

    Args:
        tmp_path: pytest's temporary folder.

    Returns:
        The NetCDF file's path.
    """
    import xarray as xr

    x = np.arange(0, 20_000, 500.0) + 250
    y = np.arange(0, 20_000, 500.0) + 250
    thk = np.stack(
        [np.full((len(y), len(x)), 100.0 * t, dtype="float32") for t in range(6)]
    )
    ds = xr.Dataset(
        {"thk": (("time", "y", "x"), thk)},
        coords={"time": np.arange(6.0), "y": y, "x": x},
        attrs={"proj": CRS},
    )
    path = tmp_path / "model.nc"
    ds.to_netcdf(path)
    return path


def points_layer():
    """Measured thickness of 300 m at four points, as RustGIS sends a layer (WGS 84 GeoJSON).

    Returns:
        A GeoJSON FeatureCollection.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"Thick": [300.0] * 4, "site": ["a", "a", "b", "b"]},
        geometry=[
            Point(3_000, 3_000),
            Point(7_000, 12_000),
            Point(15_000, 4_000),
            Point(10_000, 18_000),
        ],
        crs=CRS,
    )
    return json.loads(gdf.to_crs("EPSG:4326").to_json())


def request(model_path, params=None):
    """An STPC request for ``model_path`` and the measured points.

    Args:
        model_path: The model output's path.
        params: Parameters to add or override.

    Returns:
        The request dict.
    """
    p = {
        "model": str(model_path),
        "points": "Measured",
        "value_field": "Thick",
        "variable": "thk",
        "save_plots": False,
    }
    p.update(params or {})
    return {
        "abi": 1,
        "kind": "tool",
        "id": "stpc",
        "params": p,
        "layers": {"Measured": points_layer()},
        "lang": "en",
    }


def test_stpc_finds_the_matching_time_step(model, tmp_path):
    reply = run(request(model), tmp_path)
    assert "ok" in reply, reply
    result = reply["ok"]
    metrics = [f["properties"] for f in result["tables"][0]["geojson"]["features"]]
    # Time step 0 has no ice (thickness below 1 m): every point counts as 0 m, still a row.
    assert [m["time_idx"] for m in metrics] == [0, 1, 2, 3, 4, 5]
    best = min(metrics, key=lambda m: m["rmse_m"])
    assert best["time_idx"] == 3 and best["rmse_m"] == pytest.approx(0.0, abs=1e-6)
    by_t = {m["time_idx"]: m for m in metrics}
    assert by_t[1]["rmse_m"] == pytest.approx(200.0) and by_t[5][
        "mean_diff_m"
    ] == pytest.approx(200.0)
    layer = result["layers"][0]
    assert "time step 3" in layer["name"]
    values = [f["properties"]["modeled_value_m"] for f in layer["geojson"]["features"]]
    assert values == pytest.approx([300.0] * 4)
    assert any("RMSE 0.00 m" in m for m in result["messages"])
    csv = (
        model.parent
        / "model_glimat_stpc"
        / "csv"
        / "model_glimat_stpc_point_comparison.csv"
    )
    assert csv.is_file()


def test_stpc_groups_by_id(model, tmp_path):
    reply = run(request(model, {"id_field": "site"}), tmp_path)
    result = reply["ok"]
    ids = {f["properties"]["id"] for f in result["tables"][0]["geojson"]["features"]}
    assert ids == {"a", "b"}
    assert len(result["layers"]) == 2


def test_stpc_reports_bad_input_as_errors(model, tmp_path):
    reply = run(request(model, {"value_field": "nope"}), tmp_path)
    assert "no field 'nope'" in reply["error"]
    reply = run(request(model, {"id_field": "nope"}), tmp_path)
    assert "no field 'nope'" in reply["error"]
