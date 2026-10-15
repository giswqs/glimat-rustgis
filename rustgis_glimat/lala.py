"""LALA (Likelihood of Accordant Lineations Analysis): how likely a simulation's ice extent and
flow directions are, given mapped flow lineations (Archer et al., 2023)."""

import json
import sys

from rustgis_glimat.common import (
    layer_file,
    number,
    open_model,
    output_dirs,
    require_crs,
    table,
)


def _text(params, name, default):
    """A text parameter, or ``default`` when it's empty.

    Args:
        params: The tool's parameters.
        name: The parameter's name.
        default: The value when it's missing or empty.

    Returns:
        The text.
    """
    return str(params.get(name) or "").strip() or default


def _scalar(value):
    """A plain number from what GLIMAT computed: its LALA sums xarray arrays, so its totals are
    0-dimensional ``DataArray`` objects rather than numbers.

    Args:
        value: A number, or a 0-dimensional array of one.

    Returns:
        The number as a ``float``.
    """
    import numpy as np

    return float(np.asarray(value).item())


def _points_out(scores, crs):
    """The lineations' midpoints, with their scores, as a WGS 84 GeoJSON FeatureCollection.

    Args:
        scores: GLIMAT's per-lineation table, with ``X_Coord`` and ``Y_Coord`` in the model's
            coordinate system.
        crs: The model's coordinate system.

    Returns:
        A GeoJSON FeatureCollection dict.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        scores.drop(columns=["X_Coord", "Y_Coord"]),
        geometry=gpd.points_from_xy(scores["X_Coord"], scores["Y_Coord"]),
        crs=crs,
    ).to_crs("EPSG:4326")
    return json.loads(gdf.to_json())


def run(req):
    """Runs GLIMAT's LALA, as its QGIS dialog does.

    Args:
        req: The request.

    Returns:
        The tool's result: the lineations' midpoints with their scores as a layer, a summary
        table, and the final log-likelihood in the messages.

    Raises:
        RuntimeError: When an input is missing or the analysis failed.
    """
    import pandas as pd

    from glimat.lala import LALA_Analyzer

    params = req["params"]
    model_path = params.get("model", "")
    # GLIMAT's dialog opens the model with xarray's default time decoding for LALA.
    ds = open_model(model_path, decode_times=True)
    crs = require_crs(params, ds)
    variables = {
        "u_var": _text(params, "u_var", "uvelbase"),
        "v_var": _text(params, "v_var", "vvelbase"),
        "thk_var": _text(params, "thk_var", "thk"),
        "mask_var": _text(params, "mask_var", "mask"),
    }
    missing = [v for v in variables.values() if v not in ds.variables]
    if missing:
        raise RuntimeError(f"the model has no variable {', '.join(missing)}")

    subdirs = output_dirs(
        params,
        model_path,
        "lala",
        {"plots": "plots", "csv": "csv", "shape_files": "shape_files"},
    )
    lineations = layer_file(req, "lineations", crs, subdirs["main"])
    prefix = subdirs["main"].name
    analyzer = LALA_Analyzer(
        pism_dataset=ds,
        shapefile_path=str(lineations),
        output_subdirs=subdirs,
        prefix=prefix,
        thk_threshold=number(params, "thk_threshold", 10),
        velocity_threshold=number(params, "velocity_threshold", 1.0),
        grounded_ice=number(params, "grounded_ice", 2.0),
        kappa=number(params, "kappa", 10),
        p_value=number(params, "p_value", 0.01),
        **variables,
    )
    analyzer.set_console(lambda msg: print(msg, file=sys.stderr))
    results = analyzer.runLALA()
    if results is None:
        raise RuntimeError(
            "the LALA analysis failed; RustGIS's log has GLIMAT's messages"
        )

    csv = subdirs["csv"]
    final = pd.read_csv(csv / f"{prefix}_flowset_scores_final.csv")
    summary = pd.read_csv(csv / f"{prefix}_lineation_summary.csv")
    scores = pd.DataFrame(
        {
            "lineation_id": final["Flowset"],
            "score": final["Score"],
            "has_possible_times": final["Has_Possible_Times"],
            "time_count": final["Time_Count"],
            "time_avg_nu": summary["Time_Avg_Nu"],
            "max_nu": summary["Max_Nu"],
            "max_time_index": summary["Max_Time_Index"],
            "direction": final["Direction_Field"],
            "X_Coord": final["X_Coord"],
            "Y_Coord": final["Y_Coord"],
        }
    )
    valid = int(final["Has_Possible_Times"].sum())
    likelihood = _scalar(results["final_likelihood"])
    overview = pd.DataFrame(
        [
            {
                "final_likelihood": likelihood,
                "lineations": int(results["number_lineations"]),
                "lineations_with_possible_times": valid,
                "time_steps": int(results["time_steps"]),
                "A": _scalar(results["A"]),
                "lambda": _scalar(results["lam"]),
                "lambda_star": _scalar(results["lam_star"]),
                "likelihood_location": _scalar(results["likelihood_location"]),
            }
        ]
    )
    messages = [
        f"Final log-likelihood: {likelihood:.6f}",
        f"{valid} of {results['number_lineations']} lineations could have formed "
        f"in the {results['time_steps']} time steps",
        f"GLIMAT's files are in {subdirs['main']}",
    ]
    return {
        "layers": [
            {"name": "LALA lineation scores", "geojson": _points_out(scores, crs)}
        ],
        "tables": [{"name": "LALA summary", "geojson": table(overview)}],
        "messages": messages,
    }
