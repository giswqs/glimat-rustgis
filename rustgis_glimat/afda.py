"""AFDA (Automated Flow Direction Analysis): modelled ice flow directions vs mapped lineations."""

import sys

from rustgis_glimat.common import (
    flag,
    layer_file,
    number,
    open_model,
    output_dirs,
    require_crs,
    table,
    time_indices,
)

# The ice variable's kinds GLIMAT's AFDA understands.
ICE_TYPES = ("thickness", "fraction", "mask")


def run(req):
    """Runs GLIMAT's flow direction analysis (AFDA).

    Mirrors GLIMAT's QGIS dialog: the model is opened with times decoded, the lineations are
    read from a file (here, the RustGIS layer written in the model's coordinate system), and
    ``PISMAFDA.analyze_time_series`` compares the modelled flow direction with the
    lineations' direction at each time step.

    Args:
        req: The request.

    Returns:
        The tool's result: the angular statistics per time step as a table, and the time step
        whose flow agrees best with the lineations (smallest mean angular difference) in the
        messages.

    Raises:
        RuntimeError: When an input is missing or the analysis produced nothing.
    """
    from glimat.afda import PISMAFDA

    params = req["params"]
    model_path = params.get("model", "")
    ds = open_model(model_path, decode_times=True)
    crs = require_crs(params, ds)
    ice_type = str(params.get("ice_type") or "thickness").strip()
    if ice_type not in ICE_TYPES:
        raise RuntimeError(f"ice type must be one of {', '.join(ICE_TYPES)}")

    subdirs = output_dirs(
        params,
        model_path,
        "afda",
        {"plots": "plots", "csv": "csv", "rasters": "rasters"},
    )
    lineations = layer_file(req, "lineations", crs, subdirs["main"])
    analyzer = PISMAFDA(
        pism_dataset=ds,
        shapefile_path=str(lineations),
        output_subdirs=subdirs,
        prefix=subdirs["main"].name,
        time_indices=time_indices(params, len(ds.time)),
        ice_threshold=number(params, "ice_threshold", 0.5),
        velocity_threshold=number(params, "velocity_threshold", 1.0),
        cell_size=number(params, "cell_size", 500),
        save_rasters=flag(params, "save_rasters"),
        u_var=str(params.get("u_var") or "uvelbase"),
        v_var=str(params.get("v_var") or "vvelbase"),
        ice_var=str(params.get("ice_var") or "thk"),
        ice_type=ice_type,
        use_time_window=flag(params, "use_time_window"),
        window_size=int(number(params, "window_size", 5)),
        use_accumulative_ice=flag(params, "use_accumulative_ice", True),
    )
    analyzer.set_console(lambda msg: print(msg, file=sys.stderr))
    results = analyzer.analyze_time_series()
    if results is None:
        raise RuntimeError(
            "the analysis produced no results (do the lineations overlap the model's ice?)"
        )

    ts = analyzer.time_series
    best = ts.loc[ts["vec_mean_deg"].idxmin()]
    messages = [
        f"Best agreement: time step {int(best['time_idx'])}: mean angular difference "
        f"{best['vec_mean_deg']:.2f}°, vector strength {best['vec_strength']:.4f}, "
        f"normal {best['normal_alignment_pct']:.1f}%, frozen bed {best['frozen_bed_pct']:.1f}%, "
        f"ice-free {best['ice_free_pct']:.1f}%",
        f"GLIMAT's files are in {subdirs['main']}",
    ]
    return {
        "layers": [],
        "tables": [{"name": "AFDA statistics", "geojson": table(ts)}],
        "messages": messages,
    }
