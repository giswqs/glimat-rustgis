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

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# First, so the protocol keeps the real standard output; anything else printed (by GLIMAT or the
# libraries it uses) goes to standard error, which RustGIS logs.
from rustgis_glimat.protocol import send  # noqa: E402

sys.stdout = sys.stderr

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


# The tools: their ids in plugin.json, and the modules in rustgis_glimat/ that run them.
TOOLS = {"apoa": "apoa", "lala": "lala"}


def main():
    """Answers one request from RustGIS."""
    use_configured_python()
    line = sys.stdin.readline()
    try:
        req = json.loads(line)
        module = TOOLS.get(req.get("id")) if req.get("kind") == "tool" else None
        if module is None:
            raise RuntimeError(
                f"GLIMAT has no {req.get('kind', 'request')} '{req.get('id')}'"
            )
        check_libraries()
        tool = importlib.import_module(f"rustgis_glimat.{module}")
        reply = {"ok": tool.run(req)}
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        reply = {"error": str(exc)}
    send(reply)


if __name__ == "__main__":
    main()
