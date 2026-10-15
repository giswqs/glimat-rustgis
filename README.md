# GLIMAT for RustGIS

A port of [GLIMAT](https://github.com/yingkui2003/glimat) (Glacier Lineation and Ice Margin Analysis Toolkit), the QGIS plugin by Yingkui Li, to a [RustGIS](https://github.com/opengeos/RustGIS) plugin.

GLIMAT validates glacier and ice sheet model output (NetCDF) against geomorphological data with five tools:

- **APCA**, Automated Proximity and Conformity Analysis: modelled vs mapped ice margins.
- **AFDA**, Automated Flow Direction Analysis: modelled flow directions vs mapped lineations.
- **LALA**, Likelihood of Accordant Lineations Analysis.
- **APOA**, Automated Polygon Overlap Analysis: F1, IoU, precision and recall of ice extent.
- **STPC**, Surface/Thickness Point Comparison: RMSE, MAE and R² at measured points.

## Status

Work in progress. The plan:

1. `glimat/`: GLIMAT's analysis modules from GLIMAT v1.0.0, unchanged.
2. A RustGIS **process plugin** (`plugin.json` and `main.py`): each GLIMAT tool as a RustGIS geoprocessing tool, with results added to the map as layers and tables. Process plugins run GLIMAT's Python code as it is, which a sandboxed plugin can't do with model output of several GB. Needs [RustGIS process plugins](https://github.com/opengeos/RustGIS/issues/241).
3. Results checked against the QGIS plugin on the [GLIMAT test dataset](https://zenodo.org/records/22836601).

## Development

Install the [pre-commit](https://pre-commit.com) hooks once, and they check each commit:

```sh
pip install pre-commit
pre-commit install
pre-commit run --all-files   # check everything now
```

GLIMAT's own modules in `glimat/` are left out of the hooks, so they stay identical to the upstream release.

## Credits and licence

GLIMAT is by Yingkui Li (University of Tennessee). The analysis code in `glimat/` is his, from [yingkui2003/glimat](https://github.com/yingkui2003/glimat), licensed under the GNU General Public License v3.0. This port is licensed under the same terms; see [LICENSE](LICENSE).

`requirements-upstream.txt` lists the library versions GLIMAT was tested with.
