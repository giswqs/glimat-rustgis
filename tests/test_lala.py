"""Runs the LALA tool as RustGIS does, on a synthetic model, and checks GLIMAT's numbers."""

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
    """A model with no ice at time step 0, then thick grounded ice flowing east everywhere.

    Args:
        tmp_path: pytest's temporary folder.

    Returns:
        The NetCDF file's path.
    """
    import xarray as xr

    x = np.arange(0, 20_000, 500.0) + 250
    y = np.arange(0, 20_000, 500.0) + 250
    shape = (4, len(y), len(x))
    thk = np.full(shape, 100.0)
    thk[0] = 0.0
    mask = np.full(shape, 2, dtype="int8")
    mask[0] = 0
    ds = xr.Dataset(
        {
            "thk": (("time", "y", "x"), thk),
            "mask": (("time", "y", "x"), mask),
            "uvelbase": (("time", "y", "x"), np.full(shape, 10.0)),
            "vvelbase": (("time", "y", "x"), np.zeros(shape)),
        },
        coords={"time": np.arange(4.0), "y": y, "x": x},
        attrs={"proj": CRS},
    )
    path = tmp_path / "model.nc"
    ds.to_netcdf(path)
    return path


def lineations():
    """Two lineations through the middle: one north-south, one east-west, as RustGIS sends them.

    Returns:
        A GeoJSON FeatureCollection in WGS 84.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    gdf = gpd.GeoDataFrame(
        {"name": ["north-south", "east-west"]},
        geometry=[
            LineString([(10_000, 8_000), (10_000, 12_000)]),
            LineString([(8_000, 10_000), (12_000, 10_000)]),
        ],
        crs=CRS,
    )
    return json.loads(gdf.to_crs("EPSG:4326").to_json())


def request(model_path, params=None):
    """A LALA request for ``model_path`` and the two lineations.

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
        "id": "lala",
        "params": p,
        "layers": {"Lineations": lineations()},
        "lang": "en",
    }


def test_lala_scores_lineations_as_glimat_does(model, tmp_path):
    from scipy.stats import vonmises

    reply = run(request(model), tmp_path)
    assert "ok" in reply, reply
    result = reply["ok"]
    points = [f["properties"] for f in result["layers"][0]["geojson"]["features"]]
    assert len(points) == 2
    ns, ew = points
    # Ice could have formed both in the 3 time steps with ice.
    assert ns["time_count"] == ew["time_count"] == 3
    assert ns["direction"] == pytest.approx(0.0, abs=1e-6)
    assert ew["direction"] == pytest.approx(90.0, abs=1e-6)

    # GLIMAT's LALA (v1.0.0): lam spreads the formation likelihood over the possible cells;
    # each lineation sums, over its possible times, lam times a von Mises density of the
    # model's flow angle (radians, here 0: east) minus the lineation's azimuth as it stores it
    # (degrees clockwise from north), either way round.
    cells = 40 * 40
    lam_star = 0.01 / cells
    lam = (1 - cells * lam_star) / (3 * cells)

    def score(azimuth):
        pdf = 0.5 * vonmises.pdf(0 - azimuth, 10) + 0.5 * vonmises.pdf(
            0 - azimuth + np.pi, 10
        )
        return 3 * lam * pdf

    assert ns["score"] == pytest.approx(score(0.0), rel=1e-9)
    assert ew["score"] == pytest.approx(score(90.0), rel=1e-6)
    assert ns["score"] > ew["score"]

    summary = result["tables"][0]["geojson"]["features"][0]["properties"]
    assert summary["lineations"] == 2 and summary["lineations_with_possible_times"] == 2
    assert summary["A"] == 3 * cells
    expected = (
        np.log(score(0.0)) + np.log(score(90.0)) - (lam * 3 * cells + lam_star * cells)
    )
    assert summary["final_likelihood"] == pytest.approx(expected, rel=1e-6)
    assert any(m.startswith("Final log-likelihood") for m in result["messages"])
    out = (
        model.parent
        / "model_glimat_lala"
        / "csv"
        / "model_glimat_lala_flowset_scores_final.csv"
    )
    assert out.is_file()


def test_lala_reports_bad_input_as_errors(model, tmp_path):
    reply = run(request(model, {"u_var": "nope"}), tmp_path)
    assert "no variable nope" in reply["error"]
    bad = request(model)
    bad["layers"] = {}
    assert "no layer named" in run(bad, tmp_path)["error"]
