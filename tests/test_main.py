"""Runs the plugin as RustGIS does: one request on standard input, JSON lines back."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "main.py"
CRS = "EPSG:3413"


def run(request, cwd):
    """Runs ``main.py`` with ``request`` and returns its log lines and its result message.

    Args:
        request: The request, as RustGIS sends it.
        cwd: The folder to run in.

    Returns:
        A tuple of (log lines, the last message: ``{"ok": ...}`` or ``{"error": ...}``).
    """
    messages = run_messages(request, cwd)
    logs = [m["log"] for m in messages if "log" in m]
    return logs, messages[-1]


def run_messages(request, cwd):
    """Runs ``main.py`` with ``request`` and returns every message it wrote.

    Args:
        request: The request, as RustGIS sends it.
        cwd: The folder to run in.

    Returns:
        The messages, in order; the last is ``{"ok": ...}`` or ``{"error": ...}``.
    """
    proc = subprocess.run(
        [sys.executable, str(MAIN)],
        input=json.dumps(request) + "\n",
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )
    return [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]


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


def field_layer(half):
    """A mapped ice extent: the square of half-side ``half`` metres, as RustGIS sends it.

    Args:
        half: Half the square's side, in metres.

    Returns:
        A GeoJSON FeatureCollection in WGS 84.
    """
    import geopandas as gpd
    from shapely.geometry import box

    gdf = gpd.GeoDataFrame(
        {"name": ["mapped"]},
        geometry=[box(10_000 - half, 10_000 - half, 10_000 + half, 10_000 + half)],
        crs=CRS,
    )
    return json.loads(gdf.to_crs("EPSG:4326").to_json())


def request(model_path, params=None, tool="apoa"):
    """An APOA request for ``model_path`` and the field layer of time step 3.

    Args:
        model_path: The model output's path.
        params: Parameters to add or override.
        tool: The tool id.

    Returns:
        The request dict.
    """
    p = {"model": str(model_path), "field": "Mapped", "save_plots": False}
    p.update(params or {})
    return {
        "abi": 1,
        "kind": "tool",
        "id": tool,
        "params": p,
        "layers": {"Mapped": field_layer(5_000)},
        "lang": "en",
    }


def test_apoa_finds_the_matching_time_step(model, tmp_path):
    logs, reply = run(request(model), tmp_path)
    assert "ok" in reply, reply
    result = reply["ok"]
    metrics = [f["properties"] for f in result["tables"][0]["geojson"]["features"]]
    assert [m["time_idx"] for m in metrics] == [0, 1, 2, 3, 4]
    best = max(metrics, key=lambda m: m["f1_score"])
    # GLIMAT (v1.0.0) places its grid by the outermost cell centres (x.min(), y.max()) rather
    # than the cell edges, so its model polygons sit half a cell (250 m here) off in x and y.
    # A perfect match therefore scores (9750 / 10000)² rather than 1: this checks the port
    # reproduces GLIMAT's numbers exactly, as the QGIS plugin computes them.
    assert best["time_idx"] == 3 and best["f1_score"] == pytest.approx(
        0.950625, abs=1e-6
    )
    assert result["layers"] and "time step 3" in result["layers"][0]["name"]
    assert any("Best match" in m and "F1 0.9506" in m for m in result["messages"])
    out = (
        model.parent
        / "model_glimat_apoa"
        / "csv"
        / "model_glimat_apoa_overlap_time_series.csv"
    )
    assert out.is_file()


def test_glimat_output_shows_as_progress(model, tmp_path):
    messages = run_messages(request(model), tmp_path)
    progress = [m["progress"] for m in messages if "progress" in m]
    # GLIMAT's lines, one by one, while it runs; not its rules or blank lines.
    assert any("Time steps to analyze: 5" in p for p in progress), progress
    assert all(any(c.isalnum() for c in p) and p == p.strip() for p in progress)
    # The result keeps only the summary.
    result = messages[-1]["ok"]
    assert not set(progress) & set(result["messages"])


def test_progress_reports_whole_lines(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(MAIN.parent))
    from rustgis_glimat import protocol

    sent = []
    monkeypatch.setattr(protocol, "send", sent.append)
    out = protocol.Progress()
    out.write("Time step ")
    out.write("3 of 5\n" + "=" * 20 + "\n\n  10%\r  50%\rdone\nhalf")
    assert sent == [{"progress": "Time step 3 of 5"}, {"progress": "done"}]
    assert capsys.readouterr().err.startswith("Time step 3 of 5\n")


def test_apoa_reports_bad_input_as_errors(model, tmp_path):
    _, reply = run(request(tmp_path / "missing.nc"), tmp_path)
    assert "no model output file" in reply["error"]
    _, reply = run(request(model, {"id_field": "nope"}), tmp_path)
    assert "no field 'nope'" in reply["error"]
    _, reply = run(request(model, tool="nope"), tmp_path)
    assert "GLIMAT has no tool 'nope'" in reply["error"]
