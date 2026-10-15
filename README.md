# GLIMAT for RustGIS

A port of [GLIMAT](https://github.com/yingkui2003/glimat) (Glacier Lineation and Ice Margin Analysis Toolkit), the QGIS plugin by Yingkui Li, to a [RustGIS](https://github.com/opengeos/RustGIS) plugin.

GLIMAT validates glacier and ice sheet model output (NetCDF) against geomorphological data with five tools:

- **APCA**, Automated Proximity and Conformity Analysis: modelled vs mapped ice margins.
- **AFDA**, Automated Flow Direction Analysis: modelled flow directions vs mapped lineations.
- **LALA**, Likelihood of Accordant Lineations Analysis.
- **APOA**, Automated Polygon Overlap Analysis: F1, IoU, precision and recall of ice extent.
- **STPC**, Surface/Thickness Point Comparison: RMSE, MAE and R² at measured points.

## Status

Work in progress. Done: **APOA** (Polygon Overlap). Next: STPC, APCA, AFDA and LALA, each checked against the QGIS plugin on the [GLIMAT test dataset](https://zenodo.org/records/22836601).

The plugin is a RustGIS **process plugin**: RustGIS runs `main.py` with Python, which runs GLIMAT's own code in `glimat/` (unchanged from GLIMAT v1.0.0) and hands the results back as layers and tables. That lets it read model output of any size in place, as the QGIS plugin does. It needs a RustGIS with [process plugins](https://github.com/opengeos/RustGIS/issues/241), on the desktop or the command line.

## Install

In RustGIS, open **Plugins › Plugin Manager › Settings** and add this folder as a plugin folder, or run `rustgis-cli plugins install /path/to/glimat-rustgis`.

RustGIS then makes GLIMAT a Python environment of its own, with Python 3.12 and the packages in `requirements.txt`. The desktop app does it in the background and shows the progress in the Plugin Manager; `rustgis-cli` does it before the install returns. You don't need Python installed. The first time downloads about 600 MB. This needs a RustGIS with [plugin Python environments](https://github.com/opengeos/RustGIS/pull/246).

For development, `GLIMAT_PYTHON` names another interpreter to run GLIMAT with, such as a conda environment's `python` that has the packages in `requirements.txt`.

## Tools

The tools are in the **Tools** pane under **Plugins › GLIMAT**.

### Polygon Overlap (APOA)

Compares the modelled ice extent (cells with ice thickness at or above a minimum) with a mapped ice extent polygon at each time step, and reports F1-score, IoU, precision and recall.

- **Model output (NetCDF path)**: the model's NetCDF, read in place.
- **Mapped ice extent**: a polygon layer in RustGIS. It's reprojected to the model's coordinate system, which is read from the NetCDF, or given as **Model coordinate system** (an EPSG code).
- **ID field**: compares each ID's polygons separately.
- Time steps, a moving time window and the other options work as in the QGIS plugin.

Results: an **APOA metrics** table (one row per time step and ID), the best-matching modelled ice extent as a layer, and the best scores in the run's messages. GLIMAT's CSV files, summary and plots are written to the output folder (by default `<model>_glimat_apoa` beside the model).

A note on the numbers: GLIMAT v1.0.0 positions the model grid by its outermost cell centres rather than its cell edges, so modelled polygons are offset by half a cell. The port reproduces GLIMAT's results exactly, offset included.

## Development

Install the [pre-commit](https://pre-commit.com) hooks once, and they check each commit:

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files   # check everything now
python -m pytest tests        # run the plugin as RustGIS does, on a synthetic model
```

GLIMAT's own modules in `glimat/` are left out of the hooks, so they stay identical to the upstream release.

## Credits and licence

GLIMAT is by Yingkui Li (University of Tennessee). The analysis code in `glimat/` is his, from [yingkui2003/glimat](https://github.com/yingkui2003/glimat), licensed under the GNU General Public License v3.0. This port is licensed under the same terms; see [LICENSE](LICENSE).

`requirements-upstream.txt` lists the library versions GLIMAT was tested with.
