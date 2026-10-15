"""STPC (Surface/Thickness Point Comparison): modelled vs measured values at points, per time step."""

import json
import sys

from rustgis_glimat.common import (
    flag,
    layer_in,
    number,
    open_model,
    output_dirs,
    require_crs,
    table,
    time_indices,
)

# Columns of GLIMAT's results that hold lists or dicts per row, not values for a table.
NOT_TABLE = ("modeled_values", "observed_values", "metrics")


def points_out(point_data, crs, properties):
    """GLIMAT's per-point values (model coordinates) as a WGS 84 point FeatureCollection.

    Args:
        point_data: GLIMAT's point dicts (``x``, ``y``, ``modeled_value_m``, ...).
        crs: The model's coordinate system.
        properties: Attributes added to every feature.

    Returns:
        A GeoJSON FeatureCollection dict.
    """
    import geopandas as gpd
    import pandas as pd

    df = pd.DataFrame(point_data)
    df["difference_m"] = df["modeled_value_m"] - pd.to_numeric(
        df["observed_value_m"], errors="coerce"
    )
    for k, v in properties.items():
        df[k] = v
    gdf = gpd.GeoDataFrame(
        df.drop(columns=["x", "y"]),
        geometry=gpd.points_from_xy(df["x"], df["y"]),
        crs=crs,
    ).to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


def run(req):
    """Runs GLIMAT's point comparison analysis (STPC).

    Args:
        req: The request.

    Returns:
        The tool's result: the metrics per time step (and ID) as a table, the measured points
        with the modelled values at the best-matching time step as a layer, and a summary in the
        messages.

    Raises:
        RuntimeError: When an input is missing or the analysis produced nothing.
    """
    from glimat.point_compare import PointCompareAnalyzer

    params = req["params"]
    model_path = params.get("model", "")
    ds = open_model(model_path)
    crs = require_crs(params, ds)
    points = layer_in(req, "points", crs)
    value_field = str(params.get("value_field") or "").strip() or None
    if value_field is None or value_field not in points.columns:
        raise RuntimeError(f"the measured points have no field '{value_field}'")
    id_field = str(params.get("id_field") or "").strip() or None
    if id_field is not None and id_field not in points.columns:
        raise RuntimeError(f"the measured points have no field '{id_field}'")

    subdirs = output_dirs(
        params,
        model_path,
        "stpc",
        {
            "plots": "plots",
            "time_step_plots": "plots/time_steps",
            "best_match": "best_match",
            "csv": "csv",
        },
    )
    save_plots = flag(params, "save_plots", True)
    analyzer = PointCompareAnalyzer(
        pism_dataset=ds,
        points_gdf=points,
        output_subdirs=subdirs,
        prefix=subdirs["main"].name,
        time_indices=time_indices(params, len(ds.time)),
        var_name=str(params.get("variable") or "usurf"),
        use_time_window=flag(params, "use_time_window"),
        window_size=int(number(params, "window_size", 5)),
        value_field=value_field,
        id_field=id_field,
        min_ice_thickness=number(params, "min_ice_thickness", 1.0),
        use_accumulative=flag(params, "use_accumulative"),
        save_plots=save_plots,
    )
    analyzer.set_console(lambda msg: print(msg, file=sys.stderr))
    results = analyzer.analyze_time_series(use_parallel=True, max_workers=6)
    if results is None:
        raise RuntimeError(
            "the analysis produced no results (no modelled values at the points?)"
        )

    layers, messages = [], []
    for uid, best in analyzer.best_matches.items():
        t = best.get("time_idx")
        if t is None:
            continue
        id_text = "" if uid is None else f" (ID {uid})"
        _, _, point_data, _, _ = analyzer._extract_point_data(t, uid)
        if point_data:
            layers.append(
                {
                    "name": f"STPC best match{id_text}, time step {t}",
                    "geojson": points_out(point_data, crs, {"time_idx": t}),
                }
            )
        res = best["result"]
        messages.append(
            f"Best match{id_text}: time step {t}: RMSE {res['rmse_m']:.2f} m, MAE {res['mae_m']:.2f} m, "
            f"mean difference {res['mean_diff_m']:.2f} m, R² {res['r2_corr']:.4f}, {res['count']} points"
        )
    messages.append(f"GLIMAT's files are in {subdirs['main']}")
    metrics = analyzer.time_series.drop(
        columns=[c for c in NOT_TABLE if c in analyzer.time_series.columns]
    )
    return {
        "layers": layers,
        "tables": [{"name": "STPC metrics", "geojson": table(metrics)}],
        "messages": messages,
    }
