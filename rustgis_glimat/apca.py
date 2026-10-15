"""APCA (Automated Proximity and Conformity Analysis): modelled vs mapped ice margins."""

import sys

from rustgis_glimat.common import (
    flag,
    layer_in,
    layer_out,
    number,
    open_model,
    output_dirs,
    require_crs,
    table,
    time_indices,
)

# The folders GLIMAT's dialog makes for APCA.
FOLDERS = {
    "plots": "plots",
    "time_step_plots": "plots/time_steps",
    "best_match": "best_match",
    "csv": "csv",
    "ice_outlines": "ice_outlines",
}


def run(req):
    """Runs GLIMAT's ice margin offset analysis (APCA), as its QGIS dialog does.

    For each time step, the modelled ice outline is compared with the mapped ice margin
    features (moraines, lines or polygons): the distance from each feature to the modelled
    margin, summarised per time step. The best match is the time step with the smallest mean
    offset.

    Args:
        req: The request.

    Returns:
        The tool's result: the offsets per time step and the best match's offsets per mapped
        feature as tables, the best-matching modelled ice outline as a layer, and a summary in
        the messages.

    Raises:
        RuntimeError: When an input is missing or the analysis produced nothing.
    """
    from glimat.apca import TimeSeriesAPCA

    params = req["params"]
    model_path = params.get("model", "")
    # The dialog opens the model with xarray's default time decoding for APCA.
    ds = open_model(model_path, decode_times=True)
    crs = require_crs(params, ds)
    margins = layer_in(req, "margin", crs)
    id_field = str(params.get("id_field") or "").strip() or None
    if id_field is not None and id_field not in margins.columns:
        raise RuntimeError(f"the mapped ice margin has no field '{id_field}'")

    subdirs = output_dirs(params, model_path, "apca", FOLDERS)
    analyzer = TimeSeriesAPCA(
        pism_dataset=ds,
        target_features=margins,
        output_subdirs=subdirs,
        prefix=subdirs["main"].name,
        time_indices=time_indices(params, len(ds.time)),
        ice_threshold=number(params, "ice_threshold", 0.5),
        min_ice_thickness=number(params, "min_ice_thickness", 1.0),
        cell_size=number(params, "cell_size", 500),
        crs=margins.crs,
        ice_variable=str(params.get("ice_variable") or "thk"),
        use_accumulative_ice=flag(params, "use_accumulative_ice", True),
    )
    analyzer.set_console(lambda msg: print(msg, file=sys.stderr))
    time_series = analyzer.analyze_time_series(
        id_field=id_field,
        save_plots=flag(params, "save_plots", True),
        save_shapefiles=flag(params, "save_shapefiles", True),
    )
    if time_series is None:
        raise RuntimeError(
            "the analysis produced no results (no ice, or no modelled margin near the mapped "
            "features, in the time steps chosen?)"
        )
    # As the dialog does, whether or not time step plots are saved.
    analyzer.generate_best_match_map()
    analyzer.plot_time_series()
    analyzer.export_results()

    t = analyzer.best_match_time
    best = analyzer.best_match_apca_results
    std = float(best["std_offset_m"].mean()) if best is not None else float("nan")
    layers = []
    outline = analyzer.best_match_outline
    if outline is not None and not outline.empty:
        layers.append(
            {
                "name": f"APCA best match, time step {t}",
                "geojson": layer_out(
                    list(outline.geometry),
                    crs,
                    {"time_idx": t, "mean_offset_m": analyzer.best_match_offset},
                ),
            }
        )
    tables = [{"name": "APCA offsets", "geojson": table(time_series)}]
    if best is not None:
        tables.append(
            {"name": f"APCA best match offsets, time step {t}", "geojson": table(best)}
        )
    messages = [
        f"Best match: time step {t}: mean offset {analyzer.best_match_offset:.2f} m "
        f"(standard deviation {std:.2f} m) over {0 if best is None else len(best)} features",
        f"Trend: {analyzer.rate_of_change:.3f} m per time step "
        f"(R² {analyzer.trend_r2:.3f}, p {analyzer.trend_pvalue:.4f})",
        f"GLIMAT's files are in {subdirs['main']}",
    ]
    return {"layers": layers, "tables": tables, "messages": messages}
