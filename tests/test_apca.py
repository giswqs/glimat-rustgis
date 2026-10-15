"""The Ice Margin Offset (APCA) tool, run as RustGIS runs it: one request on stdin."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MAIN = Path(__file__).resolve().parent.parent / "main.py"
CRS = "EPSG:3413"


def run(request, cwd):
    """Runs ``main.py`` with ``request`` and returns its last message.

    Args:
        request: The request, as RustGIS sends it.
        cwd: The folder to run in.

    Returns:
        The last message: ``{"ok": ...}`` or ``{"error": ...}``.
    """
    proc = subprocess.run(
        [sys.executable, str(MAIN)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )
    messages = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return messages[-1]


@pytest.fixture
def model(tmp_path):
    """A model whose ice is a square growing by 2 km a side per time step, over 5 steps.

    Args:
        tmp_path: pytest's temporary folder.

    Returns:
        The NetCDF file's path.
    """
    import xarray as xr

    x = np.arange(0, 20_000, 500.0) + 250
    y = np.arange(0, 20_000, 500.0) + 250
    thk = np.zeros((5, len(y), len(x)), dtype="float32")
    for t in range(5):
        half = 2_000 + 1_000 * t
        inside = (np.abs(y[:, None] - 10_000) < half) & (
            np.abs(x[None, :] - 10_000) < half
        )
        thk[t][inside] = 100.0
    ds = xr.Dataset(
        {"thk": (("time", "y", "x"), thk)},
        coords={"time": np.arange(5.0), "y": y, "x": x},
        attrs={"proj": CRS},
    )
    path = tmp_path / "model.nc"
    ds.to_netcdf(path)
    return path


def margin_layer(half):
    """A mapped ice margin: the outline of the square of half-side ``half`` metres, in WGS 84.

    Args:
        half: Half the square's side, in metres.

    Returns:
        A GeoJSON FeatureCollection, as RustGIS sends a layer.
    """
    import geopandas as gpd
    from shapely.geometry import box

    ring = box(10_000 - half, 10_000 - half, 10_000 + half, 10_000 + half).exterior
    gdf = gpd.GeoDataFrame({"name": ["margin"]}, geometry=[ring], crs=CRS)
    return json.loads(gdf.to_crs("EPSG:4326").to_json())


def request(model_path, params=None):
    """An APCA request for ``model_path`` and the margin of time step 3.

    Args:
        model_path: The model output's path.
        params: Parameters to add or override.

    Returns:
        The request dict.
    """
    p = {"model": str(model_path), "margin": "Margin", "save_plots": False}
    p.update(params or {})
    return {
        "abi": 1,
        "kind": "tool",
        "id": "apca",
        "params": p,
        "layers": {"Margin": margin_layer(5_000)},
        "lang": "en",
    }


def test_apca_finds_the_matching_time_step(model, tmp_path):
    reply = run(request(model, {"save_shapefiles": False}), tmp_path)
    assert "ok" in reply, reply
    result = reply["ok"]
    rows = [f["properties"] for f in result["tables"][0]["geojson"]["features"]]
    assert [r["time_idx"] for r in rows] == [0, 1, 2, 3, 4]
    best = min(rows, key=lambda r: r["mean_offset_mean"])
    # The margin is time step 3's; GLIMAT's grid sits half a cell off (see README), so the
    # offset is a fraction of a cell rather than 0, and the other steps are kilometres off.
    assert best["time_idx"] == 3 and best["mean_offset_mean"] < 500
    assert all(r["mean_offset_mean"] > 500 for r in rows if r["time_idx"] != 3)
    assert "time step 3" in result["layers"][0]["name"]
    assert result["tables"][1]["name"] == "APCA best match offsets, time step 3"
    assert any(m.startswith("Best match: time step 3") for m in result["messages"])
    out = model.parent / "model_glimat_apca" / "csv"
    assert (out / "model_glimat_apca_apca_time_series_results.csv").is_file()


def test_apca_reports_bad_input_as_errors(model, tmp_path):
    reply = run(request(model, {"id_field": "nope"}), tmp_path)
    assert "no field 'nope'" in reply["error"]
    bad = request(model)
    bad["params"]["margin"] = "Missing"
    assert "no layer named 'Missing'" in run(bad, tmp_path)["error"]
