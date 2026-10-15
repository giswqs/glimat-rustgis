"""APOA (Automated Polygon Overlap Analysis): modelled vs mapped ice extent, per time step."""

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


def run(req):
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
    crs = require_crs(params, ds)
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
