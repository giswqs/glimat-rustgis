"""Runs the AFDA tool as RustGIS does, on a model whose flow turns from north to east to south."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

MAIN = Path(__file__).resolve().parent.parent / "main.py"
CRS = "EPSG:3413"


def run(request, cwd):
    """Runs ``main.py`` with ``request`` and returns its result message.

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
    """A model full of ice flowing north, then east, then south (one direction per step).

    Args:
        tmp_path: pytest's temporary folder.

    Returns:
        The NetCDF file's path.
    """
    import xarray as xr

    x = np.arange(0, 20_000, 500.0) + 250
    y = np.arange(0, 20_000, 500.0) + 250
    shape = (3, len(y), len(x))
    u = np.zeros(shape, dtype="float32")
    v = np.zeros(shape, dtype="float32")
    v[0], u[1], v[2] = 50.0, 50.0, -50.0
    ds = xr.Dataset(
        {
            "thk": (("time", "y", "x"), np.full(shape, 100.0, dtype="float32")),
            "uvelbase": (("time", "y", "x"), u),
            "vvelbase": (("time", "y", "x"), v),
        },
        coords={"time": np.arange(3.0), "y": y, "x": x},
        attrs={"proj": CRS},
    )
    path = tmp_path / "model.nc"
    ds.to_netcdf(path)
    return path


def lineations():
    """Five lineations pointing east, as RustGIS sends a layer (WGS 84 GeoJSON).

    Returns:
        A GeoJSON FeatureCollection.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    lines = [LineString([(4_000, y), (16_000, y)]) for y in range(4_000, 17_000, 3_000)]
    gdf = gpd.GeoDataFrame({"id": range(len(lines))}, geometry=lines, crs=CRS)
    return json.loads(gdf.to_crs("EPSG:4326").to_json())


def request(model_path, params=None):
    """An AFDA request for ``model_path`` and the eastward lineations.

    Args:
        model_path: The model output's path.
        params: Parameters to add or override.

    Returns:
        The request dict.
    """
    p = {"model": str(model_path), "lineations": "Lineations"}
    p.update(params or {})
    return {
        "abi": 1,
        "kind": "tool",
        "id": "afda",
        "params": p,
        "layers": {"Lineations": lineations()},
        "lang": "en",
    }


def test_afda_finds_the_step_whose_flow_follows_the_lineations(model, tmp_path):
    reply = run(request(model), tmp_path)
    assert "ok" in reply, reply
    result = reply["ok"]
    rows = [f["properties"] for f in result["tables"][0]["geojson"]["features"]]
    assert [r["time_idx"] for r in rows] == [0, 1, 2]
    by_step = {r["time_idx"]: r for r in rows}
    # East-flowing ice matches the lineations exactly; north and south are 90° off.
    assert by_step[1]["vec_mean_deg"] == pytest.approx(0.0, abs=1e-6)
    assert by_step[0]["vec_mean_deg"] == pytest.approx(90.0, abs=1e-3)
    assert by_step[2]["vec_mean_deg"] == pytest.approx(90.0, abs=1e-3)
    assert all(r["normal_alignment_pct"] == pytest.approx(100.0) for r in rows)
    assert any(m.startswith("Best agreement: time step 1:") for m in result["messages"])
    out = (
        model.parent
        / "model_glimat_afda"
        / "csv"
        / "model_glimat_afda_afda_time_series.csv"
    )
    assert out.is_file()


def test_afda_reports_bad_input_as_errors(model, tmp_path):
    reply = run(request(model, {"ice_type": "nope"}), tmp_path)
    assert "ice type must be one of" in reply["error"]
    req = request(model)
    req["layers"] = {}
    assert "no layer named 'Lineations'" in run(req, tmp_path)["error"]
