"""GLIMAT as a RustGIS process plugin.

RustGIS starts this program for each tool run and writes the request as one line of JSON on
its standard input. The program answers with lines of JSON on its standard output: ``{"log":
...}`` lines, then ``{"ok": ...}`` or ``{"error": ...}``. See RustGIS's ``docs/plugins.md``
(Process plugins).

GLIMAT's own analysis code is in ``glimat/``, unchanged. This file turns a request into the
arguments GLIMAT's QGIS dialog passes, runs the analysis, and turns its results into RustGIS
layers and tables. GLIMAT's CSV files, plots and summaries are written to an output folder, as
in QGIS.

Set ``GLIMAT_PYTHON`` to the Python interpreter that has GLIMAT's libraries (see
``requirements.txt``) when the one on ``PATH`` doesn't.
"""

import json
import math
import os
import sys
import traceback
from pathlib import Path

# The protocol has standard output to itself: anything else printed (by GLIMAT or the
# libraries it uses) goes to standard error, which RustGIS logs.
_PROTOCOL = sys.stdout
sys.stdout = sys.stderr

HERE = Path(__file__).resolve().parent

# GLIMAT's libraries, and the packages they come in.
REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "geopandas": "geopandas",
    "shapely": "shapely",
    "rasterio": "rasterio",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "matplotlib": "matplotlib",
    "pyproj": "pyproj",
    "scipy": "scipy",
}


def _json_default(value):
    """Converts what ``json`` can't encode (NumPy numbers and arrays, paths) to what it can.

    Args:
        value: The value ``json`` couldn't encode.

    Returns:
        A JSON-encodable equivalent: a plain number or list, or text.
    """
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _finite(value):
    """Replaces NaN and infinities, which JSON can't carry, with ``None``, at any depth.

    Args:
        value: A JSON-encodable value.

    Returns:
        The value with every non-finite float replaced by ``None``.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def send(message):
    """Writes one protocol message to RustGIS.

    Args:
        message: A JSON-encodable dict: ``{"log": ...}``, ``{"ok": ...}`` or ``{"error": ...}``.
    """
    try:
        text = json.dumps(message, default=_json_default, allow_nan=False)
    except ValueError:
        # A NaN somewhere (a metric of an empty time step): sent as null.
        text = json.dumps(
            _finite(json.loads(json.dumps(message, default=_json_default)))
        )
    _PROTOCOL.write(text + "\n")
    _PROTOCOL.flush()


def log(text):
    """Adds a line to the tool run's messages in RustGIS.

    Args:
        text: The line.
    """
    send({"log": str(text)})


def use_configured_python():
    """Restarts under ``GLIMAT_PYTHON`` when it names another interpreter.

    The request is still unread on standard input, so the new interpreter reads it.
    """
    want = os.environ.get("GLIMAT_PYTHON", "").strip()
    if want and os.path.realpath(want) != os.path.realpath(sys.executable):
        os.execv(want, [want, str(Path(__file__).resolve())])


def check_libraries():
    """Raises a clear error naming the libraries GLIMAT needs that this Python lacks.

    Raises:
        RuntimeError: When one or more libraries can't be imported.
    """
    missing = []
    for module, package in REQUIRED.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            f"GLIMAT needs these Python packages, which {sys.executable} doesn't have: "
            f"{', '.join(missing)}. Install them (pip install -r requirements.txt in the "
            "plugin's folder), or set GLIMAT_PYTHON to a Python that has them."
        )


def number(params, name, default=None):
    """A number parameter (forms may send numbers as text).

    Args:
        params: The tool's parameters.
        name: The parameter's name.
        default: The value when it's missing or empty.

    Returns:
        The number, or ``default``.
    """
    value = params.get(name)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return float(value)


def flag(params, name, default=False):
    """A yes/no parameter.

    Args:
        params: The tool's parameters.
        name: The parameter's name.
        default: The value when it's missing.

    Returns:
        The parameter as a bool.
    """
    value = params.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1", "on")
    return bool(value)


def fix_time_coordinates(ds):
    """Replaces a model's time coordinate with time step numbers, as GLIMAT's dialog does.

    Copied from GLIMAT's ``glacier_analysis_toolkit_dialog.py`` (v1.0.0, Yingkui Li, GPL-3.0),
    where model files with unreadable times are given one number per time step.

    Args:
        ds: The model output, an ``xarray.Dataset`` with a ``time`` dimension.

    Returns:
        The dataset with ``time`` = 0, 1, 2, ...
    """
    import numpy as np

    ds = ds.assign_coords(time=np.arange(len(ds.time)))
    ds.time.attrs["long_name"] = "time"
    ds.time.attrs["units"] = "years since simulation start"
    return ds


def open_model(path):
    """Opens a model output file, read in place, as GLIMAT's dialog does.

    Args:
        path: The NetCDF file's path.

    Returns:
        The model output as an ``xarray.Dataset``.

    Raises:
        RuntimeError: When the file is missing or has no time dimension.
    """
    import xarray as xr

    path = Path(path).expanduser()
    if not path.is_file():
        raise RuntimeError(f"no model output file at {path}")
    ds = xr.open_dataset(path, decode_times=False)
    if "time" not in ds.dims:
        raise RuntimeError(f"{path.name} has no time dimension")
    return fix_time_coordinates(ds)


def model_crs(ds):
    """Reads the model grid's coordinate system from a NetCDF's metadata, if it says.

    Looks at the CF grid mapping of the variables (``crs_wkt``, ``spatial_ref``,
    ``proj4_params``, ``proj4``, ``epsg_code``), then at global attributes PISM and others write (``proj``,
    ``proj4``, ``crs``, ``epsg``).

    Args:
        ds: The model output.

    Returns:
        A ``pyproj.CRS``, or ``None`` when the file doesn't say.
    """
    from pyproj import CRS

    candidates = []
    for var in ds.data_vars.values():
        mapping = var.attrs.get("grid_mapping")
        if mapping and mapping in ds.variables:
            attrs = ds[mapping].attrs
            candidates += [
                attrs.get(k)
                for k in (
                    "crs_wkt",
                    "spatial_ref",
                    "proj4_params",
                    "proj4text",
                    "proj4",
                    "epsg_code",
                )
            ]
    candidates += [
        ds.attrs.get(k) for k in ("proj", "proj4", "crs", "epsg", "spatial_ref")
    ]
    for c in candidates:
        if c is None or (isinstance(c, str) and not c.strip()):
            continue
        try:
            return CRS.from_user_input(
                c.replace("+init=", "") if isinstance(c, str) else c
            )
        except Exception:  # nosec B112 - try the next candidate
            continue
    return None


def layer_in(req, param, crs):
    """A layer parameter's features as a GeoDataFrame in the model's coordinate system.

    Args:
        req: The request.
        param: The layer parameter's name.
        crs: The model's coordinate system.

    Returns:
        A ``geopandas.GeoDataFrame``.

    Raises:
        RuntimeError: When the layer is missing or has no features.
    """
    import geopandas as gpd

    name = req["params"].get(param)
    fc = req.get("layers", {}).get(name)
    if not name or fc is None:
        raise RuntimeError(f"no layer named '{name}'")
    features = fc.get("features", [])
    if not features:
        raise RuntimeError(f"{name} has no features")
    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    return gdf.to_crs(crs)


def table(df):
    """A pandas DataFrame as a RustGIS table: a FeatureCollection without geometries.

    Args:
        df: The table.

    Returns:
        A GeoJSON FeatureCollection dict.
    """
    rows = json.loads(df.to_json(orient="records"))
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": None, "properties": r} for r in rows
        ],
    }


def layer_out(polygons, crs, properties):
    """Polygons in the model's coordinate system as a WGS 84 GeoJSON FeatureCollection.

    Args:
        polygons: Shapely polygons.
        crs: Their coordinate system.
        properties: Attributes for every feature.

    Returns:
        A GeoJSON FeatureCollection dict.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {k: [v] * len(polygons) for k, v in properties.items()}
        | {"area_km2": [p.area / 1e6 for p in polygons]},
        geometry=list(polygons),
        crs=crs,
    ).to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


def output_dirs(params, model_path, tool):
    """Creates GLIMAT's output folders for a run.

    Args:
        params: The tool's parameters.
        model_path: The model output's path (the default folder is beside it).
        tool: The tool's id, for the default folder's name.

    Returns:
        GLIMAT's ``output_subdirs`` dict of folder paths.
    """
    given = str(params.get("output_folder") or "").strip()
    model = Path(model_path).expanduser()
    out = (
        Path(given).expanduser()
        if given
        else model.parent / f"{model.stem}_glimat_{tool}"
    )
    subdirs = {
        "main": out,
        "plots": out / "plots",
        "time_step_plots": out / "plots" / "time_steps",
        "best_match": out / "best_match",
        "csv": out / "csv",
        "shapefiles": out / "shapefiles",
    }
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return subdirs


def time_indices(params, n_times):
    """The time steps to analyse, from the first, last and interval parameters.

    Args:
        params: The tool's parameters.
        n_times: How many time steps the model has.

    Returns:
        A list of time step indices.
    """
    start = int(number(params, "start", 0))
    end = number(params, "end")
    end = n_times - 1 if end is None else int(end)
    step = max(1, int(number(params, "step", 1)))
    return list(range(max(0, start), min(end + 1, n_times), step))


def run_apoa(req):
    """Runs GLIMAT's polygon overlap analysis (APOA).

    Args:
        req: The request.

    Returns:
        The tool's result: the metrics per time step as a table, the best-matching model ice
        extent as a layer, and a summary in the messages.

    Raises:
        RuntimeError: When an input is missing or the analysis produced nothing.
    """
    from glimat.overlap_polygon import PolygonOverlapAnalyzer

    params = req["params"]
    model_path = params.get("model", "")
    ds = open_model(model_path)
    crs = str(params.get("crs") or "").strip() or model_crs(ds)
    if crs is None:
        raise RuntimeError(
            "the model file doesn't say its coordinate system: give it (an EPSG code, such as EPSG:3413)"
        )
    field = layer_in(req, "field", crs)
    id_field = str(params.get("id_field") or "").strip() or None
    if id_field is not None and id_field not in field.columns:
        raise RuntimeError(f"the mapped ice extent has no field '{id_field}'")

    subdirs = output_dirs(params, model_path, "apoa")
    use_window = flag(params, "use_time_window")
    analyzer = PolygonOverlapAnalyzer(
        pism_dataset=ds,
        field_gdf=field,
        output_subdirs=subdirs,
        prefix=subdirs["main"].name,
        time_indices=time_indices(params, len(ds.time)),
        min_ice_thickness=number(params, "min_ice_thickness", 1.0),
        ice_variable=str(params.get("ice_variable") or "thk"),
        use_time_window=use_window,
        window_size=int(number(params, "window_size", 5)),
        use_accumulative_ice=flag(params, "use_accumulative_ice"),
        id_field=id_field,
        min_polygon_area_km2=number(params, "min_polygon_area_km2", 0.01),
        save_plots=flag(params, "save_plots", True),
    )
    analyzer.set_console(lambda msg: print(msg, file=sys.stderr))
    results = analyzer.analyze_time_series(
        save_plots=flag(params, "save_plots", True),
        save_shapefiles=False,
        use_parallel=False,
    )
    if results is None:
        raise RuntimeError(
            "the analysis produced no results (no ice in the time steps chosen?)"
        )

    layers, messages = [], []
    for uid, best in analyzer.best_matches.items():
        t = best.get("time_idx")
        if t is None:
            continue
        polygons = (
            analyzer.get_model_polygons_window(t)
            if use_window
            else analyzer.get_model_polygons(t)
        )
        id_text = "" if uid is None else f" (ID {uid})"
        if polygons:
            props = {"time_idx": t, "f1_score": best["f1_score"]} | (
                {} if uid is None else {"id": uid}
            )
            layers.append(
                {
                    "name": f"APOA best match{id_text}, time step {t}",
                    "geojson": layer_out(polygons, crs, props),
                }
            )
        res = best["result"]
        messages.append(
            f"Best match{id_text}: time step {t}: F1 {best['f1_score']:.4f}, IoU {res['iou']:.4f}, "
            f"precision {res['precision']:.4f}, recall {res['recall']:.4f}"
        )
    messages.append(f"GLIMAT's files are in {subdirs['main']}")
    return {
        "layers": layers,
        "tables": [{"name": "APOA metrics", "geojson": table(analyzer.time_series)}],
        "messages": messages,
    }


TOOLS = {"apoa": run_apoa}


def main():
    """Answers one request from RustGIS."""
    use_configured_python()
    line = sys.stdin.readline()
    try:
        req = json.loads(line)
        tool = TOOLS.get(req.get("id")) if req.get("kind") == "tool" else None
        if tool is None:
            raise RuntimeError(
                f"GLIMAT has no {req.get('kind', 'request')} '{req.get('id')}'"
            )
        check_libraries()
        sys.path.insert(0, str(HERE))
        reply = {"ok": tool(req)}
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        reply = {"error": str(exc)}
    send(reply)


if __name__ == "__main__":
    main()
