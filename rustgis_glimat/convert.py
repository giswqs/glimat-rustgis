"""GLIMAT's TIF/ASC to NetCDF converter: a folder of rasters per variable into one model file."""

import sys
from pathlib import Path

from rustgis_glimat.common import flag, number


def folder(params, name, required=False):
    """A folder parameter: an existing folder, or ``None`` when it's empty and optional.

    Args:
        params: The tool's parameters.
        name: The parameter's name.
        required: Whether it must be given.

    Returns:
        The folder's path as text, or ``None``.

    Raises:
        RuntimeError: When a folder given doesn't exist, or a required one is missing.
    """
    value = str(params.get(name) or "").strip()
    if not value:
        if required:
            raise RuntimeError(f"'{name}' is required")
        return None
    path = Path(value).expanduser()
    if not path.is_dir():
        raise RuntimeError(f"no folder at {path}")
    return str(path)


def run(req):
    """Converts folders of TIF/ASC rasters (one per time step) into one NetCDF.

    Runs GLIMAT's ``RasterToNetCDFConverter`` with the arguments its QGIS dialog passes. The
    time of each raster is read from its file name (``thk_1000.tif``: 1000).

    Args:
        req: The request.

    Returns:
        The tool's result: the NetCDF written, in the messages.

    Raises:
        RuntimeError: When an input is missing or nothing was written.
    """
    from glimat.raster_to_nc import RasterToNetCDFConverter

    params = req["params"]
    output = str(params.get("output_nc") or "").strip()
    if not output:
        raise RuntimeError("'output_nc' is required")
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    def index(name):
        value = number(params, name)
        return None if value is None else int(value)

    converter = RasterToNetCDFConverter(
        thick_folder=folder(params, "thick_folder", required=True),
        output_nc=str(output),
        surface_folder=folder(params, "surface_folder"),
        uvel_folder=folder(params, "uvel_folder"),
        vvel_folder=folder(params, "vvel_folder"),
        start_idx=index("start"),
        end_idx=index("end"),
        step=index("step"),
        use_compression=flag(params, "use_compression", True),
        fill_value=-9999.0,
    )
    converter.set_console(lambda msg: print(msg, file=sys.stderr))
    written = converter.convert()
    if not written or not Path(written).is_file():
        raise RuntimeError(
            "the conversion wrote nothing (no rasters with a time in their names?)"
        )
    return {
        "layers": [],
        "tables": [],
        "messages": [f"Model output written to {written}"],
    }
