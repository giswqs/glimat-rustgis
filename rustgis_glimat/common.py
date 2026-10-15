"""Helpers the GLIMAT tools share: parameters, the model output, layers in and out, folders."""

import json
from pathlib import Path


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


def open_model(path, decode_times=False):
    """Opens a model output file, read in place, as GLIMAT's dialog does.

    Args:
        path: The NetCDF file's path.
        decode_times: Whether xarray decodes the times: GLIMAT's dialog leaves it on for APCA,
            AFDA and LALA, and turns it off for the other tools.

    Returns:
        The model output as an ``xarray.Dataset``.

    Raises:
        RuntimeError: When the file is missing or has no time dimension.
    """
    import xarray as xr

    path = Path(path).expanduser()
    if not path.is_file():
        raise RuntimeError(f"no model output file at {path}")
    ds = xr.open_dataset(path, decode_times=decode_times)
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
    rows = json.loads(df.to_json(orient="records", double_precision=15))
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


def output_dirs(params, model_path, tool, folders=None):
    """Creates GLIMAT's output folders for a run.

    Args:
        params: The tool's parameters.
        model_path: The model output's path (the default folder is beside it).
        tool: The tool's id, for the default folder's name.
        folders: The subfolders the tool's analyzer writes to, by GLIMAT's key (``csv``:
            ``"csv"``); APOA's when ``None``.

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
    if folders is None:
        folders = {
            "plots": "plots",
            "time_step_plots": "plots/time_steps",
            "best_match": "best_match",
            "csv": "csv",
            "shapefiles": "shapefiles",
        }
    subdirs = {"main": out} | {k: out / v for k, v in folders.items()}
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


def layer_file(req, param, crs, folder):
    """A layer parameter's features written to a GeoPackage in the model's coordinate system.

    For GLIMAT's analyzers that read their field data from a file (AFDA, LALA).

    Args:
        req: The request.
        param: The layer parameter's name.
        crs: The model's coordinate system.
        folder: Where to write the file (the run's output folder).

    Returns:
        The GeoPackage's path.
    """
    gdf = layer_in(req, param, crs)
    path = Path(folder) / f"{param}_input.gpkg"
    gdf.to_file(path, driver="GPKG")
    return path


def require_crs(params, ds):
    """The model's coordinate system: the one given, else the one the NetCDF says.

    Args:
        params: The tool's parameters (``crs``).
        ds: The model output.

    Returns:
        The coordinate system (text or ``pyproj.CRS``).

    Raises:
        RuntimeError: When neither says.
    """
    crs = str(params.get("crs") or "").strip() or model_crs(ds)
    if crs is None:
        raise RuntimeError(
            "the model file doesn't say its coordinate system: give it (an EPSG code, such as EPSG:3413)"
        )
    return crs
