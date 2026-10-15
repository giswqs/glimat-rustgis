"""
/*********************************************************************************
* Surface/Thickness Point Comparison (STPC) for PISM NetCDF Data                 *   
* Compares modeled ice surface elevation/thickness with field observation points *
*                                                                                *
* This analysis:                                                                 *
* - Extracts modeled values at field observation point locations                 *
* - Supports moving time window with accumulative (max) or mean value            *
* - Calculates statistics (mean, std, min, max, RMSE)                            *
* - Best match based on minimum RMSE                                             *
* - Supports parallel processing with ThreadPoolExecutor                         * 
*                                                                                *
* This file is part of GLIMAT Plugin.                                            *
*                                                                                *
* Author: Yingkui Li                                                             *
* Department of Geography & Sustainability                                       *
* University of Tennessee, Knoxville, TN 37996                                   * 
* Email: yli32@utk.edu                                                           *
* Start date: 06/07/2026                                                         *
* Finalized date: 09/17/2026                                                     *
* Copyright(C): 2026 Yingkui Li                                                  * 
*                                                                                *
* Version: 1.0.0                                                                 *
*********************************************************************************/
"""
import gc
import multiprocessing
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib as mpl

mpl.use('Agg')  # Non-interactive backend
# Force TrueType fonts instead of Type 3 paths in PDF/PS
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings('ignore')

# PISM variable names we may encounter for ice thickness / surface elevation
THICKNESS_VARS = ('thk', 'land_ice_thickness', 'ice_thickness')
SURFACE_VARS = ('usurf', 'surface_altitude', 'topg')

# Plot colors used when drawing one line per point-group ID
PLOT_COLORS = ('blue', 'green', 'red', 'orange', 'purple', 'brown')

def _empty_metrics():
    return {
        'count': 0,
        'mean_diff': np.nan,
        'std_diff': np.nan,
        'min_diff': np.nan,
        'max_diff': np.nan,
        'rmse': np.nan,
        'mae': np.nan,
        'r2': np.nan,
        'r2_corr': np.nan,
        'slope': np.nan,
        'intercept': np.nan,
        'modeled_mean': np.nan,
        'observed_mean': np.nan,
    }

def _empty_result(time_idx, uid):
    """Placeholder result row used when a (time, id) task yields no data."""
    m = _empty_metrics()
    return {
        'time_idx': time_idx,
        'time_label': f"t{time_idx:04d}",
        'id': uid,
        'count': 0,
        'mean_diff_m': m['mean_diff'],
        'std_diff_m': m['std_diff'],
        'min_diff_m': m['min_diff'],
        'max_diff_m': m['max_diff'],
        'rmse_m': m['rmse'],
        'mae_m': m['mae'],
        'r2': m['r2'],
        'r2_corr': m['r2_corr'],
        'slope': m['slope'],
        'intercept': m['intercept'],
        'modeled_values': [],
        'observed_values': [],
        'metrics': m,
        'modeled_mean_m': m['modeled_mean'],
        'observed_mean_m': m['observed_mean'],
    }

class PointCompareAnalyzer:
    """
    Point-based elevation/thickness comparison for glacier model validation.
    """

    def __init__(self, pism_dataset, points_gdf, output_subdirs, prefix='',
                 time_indices=None, var_name='usurf', value_field=None,
                 use_time_window=False, window_size=5, use_accumulative=False,
                 id_field=None, progress_callback=None, cancel_checker=None,
                 save_plots=True, thickness_var='thk', min_ice_thickness=1.0):
        """
        Initialize point-based comparison analyzer.

        Parameters
        ----------
        pism_dataset : xarray.Dataset
            PISM NetCDF dataset.
        points_gdf : geopandas.GeoDataFrame
            Points with field observations.
        output_subdirs : dict
            Output directory structure; must contain keys 'csv', 'plots',
            'best_match', 'time_step_plots'.
        prefix : str
            Prefix for output files.
        time_indices : list, optional
            Time indices to analyze.
        var_name : str
            Variable to extract ('usurf' for surface elevation, 'thk' for thickness).
        value_field : str
            Column in `points_gdf` holding the observed values.
        use_time_window : bool
            Whether to aggregate over a moving time window.
        window_size : int
            Window size (forced to an odd number).
        use_accumulative : bool
            If True, use maximum over the window; else the mean.
        id_field : str, optional
            Column used to group points (e.g., by field campaign).
        progress_callback : callable, optional
            Progress callback(percent, message).
        cancel_checker : callable, optional
            Returns True if the user has requested cancellation.
        save_plots : bool
            Whether to save per-timestep scatter plots.
        thickness_var : str
            Preferred name of the thickness variable in the dataset.
        min_ice_thickness : float
            Thickness threshold (m) below which a point is considered ice-free.
        """
        self.pism_dataset = pism_dataset
        self.points_gdf = points_gdf
        self.output_subdirs = output_subdirs
        self.prefix = prefix
        self.var_name = var_name
        self.value_field = value_field
        self.use_time_window = use_time_window
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.use_accumulative = use_accumulative
        self.id_field = id_field
        self.progress_callback = progress_callback
        self.cancel_checker = cancel_checker
        self.save_plots = save_plots
        self._is_cancelled = False
        self.thickness_var = thickness_var
        self.min_ice_thickness = min_ice_thickness
        self.console = None

        # Results
        self.results = []
        self.time_series = None
        self.best_matches = {}
        self.best_match_time = None
        self.best_match_rmse = None

        # Validate observation field
        if self.value_field is None or self.value_field not in points_gdf.columns:
            available_numeric = [
                c for c in points_gdf.columns
                if c != 'geometry' and pd.api.types.is_numeric_dtype(points_gdf[c])
            ]
            raise ValueError(
                f"Value field '{self.value_field}' not found. "
                f"Available numeric fields: {available_numeric}"
            )

        self._log(f"\nUsing observation field: {self.value_field}")

        # Group points by ID if specified
        if self.id_field is not None and self.id_field in points_gdf.columns:
            self._log(f"\nGrouping points by ID field: {self.id_field}")
            self.unique_ids = list(points_gdf[self.id_field].unique())
            self._log(f"  Found {len(self.unique_ids)} unique IDs: {self.unique_ids[:5]}...")
        else:
            self.unique_ids = [None]
            if self.id_field is not None:
                self._log(f"Warning: ID field '{self.id_field}' not found. Analyzing all together.")

        # Determine time indices
        if 'time' in pism_dataset.dims:
            self.n_times = len(pism_dataset.time)
            if time_indices is None:
                if self.use_time_window and self.window_size > 1:
                    half_win = self.window_size // 2
                    self.time_indices = list(range(half_win, self.n_times - half_win))
                else:
                    self.time_indices = list(range(self.n_times))
            else:
                self.time_indices = [i for i in time_indices if i < self.n_times]
        else:
            self.n_times = 1
            self.time_indices = [0]

        self._log(f"\nPoint Compare Analyzer initialized")
        self._log(f"  Variable: {self.var_name}")
        self._log(f"  Observation field: {self.value_field}")
        self._log(f"  Time steps to analyze: {len(self.time_indices)}")
        self._log(f"  Number of point groups: {len(self.unique_ids)}")
        self._log(f"  Number of points: {len(points_gdf)}")
        if self.use_time_window:
            self._log(f"  Moving time window: size={self.window_size}")
            self._log(f"  Window mode: "
                      f"{'Accumulative (maximum)' if self.use_accumulative else 'Mean'}")

    # logging

    def set_console(self, console_callback):
        """Set console callback for debug messages."""
        self.console = console_callback

    def _log(self, message):
        """Log message to console if available, otherwise print."""
        if self.console:
            self.console(message)
        else:
            print(message)

    # cancellation

    def cancel(self):
        """Request cancellation."""
        self._is_cancelled = True

    def is_cancelled(self):
        """Return True if cancellation has been requested."""
        if self.cancel_checker:
            return self.cancel_checker()
        return self._is_cancelled

    # extraction

    def _build_point_cache(self):
        """Precompute per-point grid indices and full time series for fast lookups."""
        if getattr(self, '_cache_built', False):
            return

        ds = self.pism_dataset
        has_time = 'time' in ds.dims

        # Grid coords are time-invariant
        ds0 = ds.isel(time=0) if has_time else ds
        x_coords = ds0.x.values
        y_coords = ds0.y.values

        n_points = len(self.points_gdf)
        row_idx = np.full(n_points, -1, dtype=int)
        col_idx = np.full(n_points, -1, dtype=int)
        for k, (_, row) in enumerate(self.points_gdf.iterrows()):
            geom = row.geometry
            if geom.geom_type != 'Point':
                continue
            col_idx[k] = int(np.argmin(np.abs(x_coords - geom.x)))
            row_idx[k] = int(np.argmin(np.abs(y_coords - geom.y)))

        self._point_row_idx = row_idx
        self._point_col_idx = col_idx
        valid = row_idx >= 0

        # Resolve variable names (same fallbacks as get_variable_value)
        if self.var_name in ds:
            var_used = self.var_name
        else:
            var_used = next(
                (a for a in SURFACE_VARS + THICKNESS_VARS if a in ds), None)
            if var_used is None:
                raise ValueError(
                    f"None of '{self.var_name}' or fallbacks found in dataset")
        self._var_used = var_used

        thick_used = next(
            (t for t in (self.thickness_var,) + THICKNESS_VARS if t in ds), None)
        self._thick_used = thick_used

        def _load_time_first(name):
            arr = ds[name].values
            if not has_time:
                return arr[np.newaxis, ...]
            tdim = ds[name].dims.index('time')
            if tdim != 0:
                arr = np.moveaxis(arr, tdim, 0)
            return arr

        var_arr = _load_time_first(var_used)
        n_times = var_arr.shape[0]

        var_series = np.full((n_times, n_points), np.nan, dtype=float)
        if valid.any():
            var_series[:, valid] = var_arr[:, row_idx[valid], col_idx[valid]]

        if thick_used is not None:
            thick_arr = _load_time_first(thick_used)
            thick_series = np.full((n_times, n_points), np.nan, dtype=float)
            if valid.any():
                thick_series[:, valid] = thick_arr[:, row_idx[valid], col_idx[valid]]
            self._thick_series = thick_series
            var_series = np.where(thick_series >= self.min_ice_thickness,
                                var_series, np.nan)
        else:
            self._thick_series = None

        # NoData masking (same rule as get_variable_value)
        var_series = np.where(np.isnan(var_series) | (var_series < -1e9),
                            np.nan, var_series)

        # Moving-window reduction, vectorized across all points at once.
        # This exactly mirrors the original integer index range
        # [max(0, t-half), min(n, t+half+1)) used by extract_point_value_window.
        if self.use_time_window and self.window_size > 1:
            half = self.window_size // 2
            rolled = np.full_like(var_series, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for t in range(n_times):
                    lo = max(0, t - half)
                    hi = min(n_times, t + half + 1)
                    window = var_series[lo:hi]
                    rolled[t] = (np.nanmax(window, axis=0)
                                if self.use_accumulative
                                else np.nanmean(window, axis=0))
            var_series = rolled

        self._var_series = var_series
        self._cache_built = True

    def get_variable_value(self, ds_slice, x, y, var_name): ##return zero if the point is ice-free, out of bounds, or NoData
        """Extract a variable value at (x, y) with thickness masking.

        Returns zero if the point is ice-free, out of bounds, or NoData.
        """
        try:
            if 'x' not in ds_slice.coords or 'y' not in ds_slice.coords:
                return 0

            x_coords = ds_slice.x.values
            y_coords = ds_slice.y.values
            x_idx = int(np.argmin(np.abs(x_coords - x)))
            y_idx = int(np.argmin(np.abs(y_coords - y)))

            # Thickness masking
            if self.thickness_var in ds_slice:
                thickness = ds_slice[self.thickness_var].values[y_idx, x_idx]
                if thickness < self.min_ice_thickness:
                    return 0

            # Target variable (with fallbacks)
            if var_name in ds_slice:
                value = ds_slice[var_name].values[y_idx, x_idx]
            else:
                for alt in SURFACE_VARS + THICKNESS_VARS:
                    if alt in ds_slice:
                        value = ds_slice[alt].values[y_idx, x_idx]
                        break
                else:
                    return 0

            if np.isnan(value) or value < -1e9:
                return 0
            return float(value)
        except Exception:
            return 0

    def get_variable_value_window(self, dataset, time_indices, x, y, var_name,
                                  use_accumulative=False):
        """Aggregate the variable over a moving window using max or mean."""
        values = []
        for t_idx in time_indices:
            ds_slice = dataset.isel(time=t_idx)
            val = self.get_variable_value(ds_slice, x, y, var_name)
            if val is not None:
                values.append(val)

        if not values:
            return None
        return float(np.max(values) if use_accumulative else np.mean(values))

    def _extract_point_data(self, time_idx, uid):
        self._build_point_cache()

        if uid is not None:
            sel = (self.points_gdf[self.id_field] == uid).to_numpy()
        else:
            sel = np.ones(len(self.points_gdf), dtype=bool)
        positions = np.where(sel)[0]
        if positions.size == 0:
            return [], [], [], 0, 0

        vals = self._var_series[time_idx, positions]
        thick = (self._thick_series[time_idx, positions]
                if self._thick_series is not None else None)
        sub = self.points_gdf.iloc[positions]

        modeled_values, observed_values, point_data = [], [], []
        n_no_ice = 0
        n_no_data = 0

        for k, (idx, row) in enumerate(sub.iterrows()):
            geom = row.geometry
            if geom.geom_type != 'Point':
                continue

            obs_value = row[self.value_field]
            thickness = None if thick is None else float(thick[k])

            # Treat ice-free / NoData as modeled = 0.0
            if np.isnan(vals[k]):
                modeled_val = 0.0
                if thickness is not None and thickness < self.min_ice_thickness:
                    n_no_ice += 1
                else:
                    n_no_data += 1
            else:
                modeled_val = float(vals[k])

            point_info = {
                'point_index': idx,
                'x': geom.x,
                'y': geom.y,
                'modeled_value_m': modeled_val,
                'observed_value_m': obs_value,
                'has_ice': thickness is not None and thickness >= self.min_ice_thickness,
                'thickness_m': thickness if thickness is not None else np.nan,
            }
            if uid is not None:
                point_info['id'] = uid
            elif self.id_field is not None and self.id_field in row:
                point_info['id'] = row[self.id_field]
            point_data.append(point_info)

            # Observed value must still be present
            if obs_value is None or (isinstance(obs_value, float)
                                    and np.isnan(obs_value)):
                continue

            modeled_values.append(modeled_val)
            observed_values.append(float(obs_value))

        return modeled_values, observed_values, point_data, n_no_ice, n_no_data

    def extract_point_value(self, time_idx, point_x, point_y):
        """Extract modeled value at a point for a single time step."""
        if 'time' in self.pism_dataset.dims:
            ds_slice = self.pism_dataset.isel(time=time_idx)
        else:
            ds_slice = self.pism_dataset
        return self.get_variable_value(ds_slice, point_x, point_y, self.var_name)

    def extract_point_value_window(self, time_idx, point_x, point_y):
        """Extract modeled value using the moving time window."""
        half_window = self.window_size // 2
        start_idx = max(0, time_idx - half_window)
        end_idx = min(self.n_times, time_idx + half_window + 1)
        time_indices = list(range(start_idx, end_idx))
        return self.get_variable_value_window(
            self.pism_dataset, time_indices, point_x, point_y,
            self.var_name, self.use_accumulative,
        )

    # statistics

    def calculate_metrics(self, modeled_values, observed_values):
        """Compute comparison statistics between modeled and observed values.

        Returns two R-squared values:

        - 'r2'      : Nash–Sutcliffe efficiency, 1 - SS_res/SS_tot where the
                    "prediction" is the modeled value itself and the residual
                    is (modeled - observed).  This is the R² of the 1:1 line.
                    Range: (-inf, 1].  Negative when the model is worse than
                    predicting the observed mean.  Sensitive to bias, slope,
                    and intercept errors.

        - 'r2_corr' : Squared Pearson correlation between observed and modeled.
                    Range: [0, 1].  Matches Excel's RSQ and the R² shown on
                    an Excel trendline.  Ignores bias and slope/intercept.
        """
        valid_pairs = [
            (m, o) for m, o in zip(modeled_values, observed_values)
            if m is not None and o is not None
            and not (isinstance(m, float) and np.isnan(m))
            and not (isinstance(o, float) and np.isnan(o))
        ]
        if not valid_pairs:
            return _empty_metrics()

        modeled = np.asarray([m for m, _ in valid_pairs], dtype=float)
        observed = np.asarray([o for _, o in valid_pairs], dtype=float)
        diff = modeled - observed
        squared_diff = diff ** 2

        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff))
        min_diff = float(np.min(diff))
        max_diff = float(np.max(diff))
        rmse = float(np.sqrt(np.mean(squared_diff)))
        mae = float(np.mean(np.abs(diff)))
        modeled_mean = float(np.mean(modeled))
        observed_mean = float(np.mean(observed))

        # ---- R² #1: Nash–Sutcliffe efficiency (1 - SS_res / SS_tot) ----
        ss_res = float(np.sum(squared_diff))
        ss_tot = float(np.sum((observed - observed_mean) ** 2))
        if ss_tot > 1e-10:
            r2_nse = 1.0 - ss_res / ss_tot
        elif np.allclose(modeled, observed_mean, rtol=1e-5):
            r2_nse = 1.0  # Constant observations, model matches them
        else:
            r2_nse = 0.0

        # ---- R² #2: Pearson correlation squared (matches Excel RSQ) ----
        if (modeled.size >= 2
                and np.std(modeled) > 0
                and np.std(observed) > 0):
            r = float(np.corrcoef(observed, modeled)[0, 1])
            r2_corr = r ** 2
            slope, intercept = np.polyfit(observed, modeled, 1)
            slope = float(slope)
            intercept = float(intercept)
        else:
            r2_corr = 0
            slope = 0
            intercept = 0

        return {
            'count': len(valid_pairs),
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'min_diff': min_diff,
            'max_diff': max_diff,
            'rmse': rmse,
            'mae': mae,
            'r2': float(r2_nse),        # Nash–Sutcliffe (1:1 line)
            'r2_corr': float(r2_corr),  # Pearson (Excel-style)
            'slope': slope,             # modeled = slope * observed + intercept
            'intercept': intercept,
            'modeled_mean': modeled_mean,
            'observed_mean': observed_mean,
        }

    # per-task

    def _find_thickness_var(self, ds_slice):
        """Return the first available thickness variable name in `ds_slice`."""
        for name in (self.thickness_var,) + THICKNESS_VARS:
            if name in ds_slice:
                return name
        return None

    def _save_thickness_export(self, time_idx, uid, point_data):
        """Write the per-point thickness export CSV for one (time, id)."""
        if not point_data:
            return None
        df_points = pd.DataFrame(point_data)
        thickness_dir = self.output_subdirs['csv'] / 'thickness_export'
        thickness_dir.mkdir(parents=True, exist_ok=True)
        id_suffix = f"_id{uid}" if uid is not None else ""
        csv_file = thickness_dir / f"{self.prefix}_thickness_t{time_idx:04d}{id_suffix}.csv"
        df_points.to_csv(csv_file, index=False)
        self._log(f"      Exported point thickness data: {csv_file}")
        return csv_file

    def _analyze_one(self, time_idx, uid):
        """Analyze one (time_idx, uid) task, returning a result dict or None."""
        if self.is_cancelled():
            return None

        modeled, observed, point_data, skipped_no_ice, skipped_no_data = \
            self._extract_point_data(time_idx, uid)

        if point_data:
            self._save_thickness_export(time_idx, uid, point_data)

        total_points = len(point_data)
        if not modeled:
            self._log(f"      No valid points: {total_points} total, "
                      f"{skipped_no_ice} outside ice mask, "
                      f"{skipped_no_data} no data")
            return None

        self._log(f"      Valid points: {len(modeled)}/{total_points} "
                  f"(excluded: {skipped_no_ice} outside ice, "
                  f"{skipped_no_data} no data)")

        metrics = self.calculate_metrics(modeled, observed)
        if metrics['count'] == 0:
            return None

        return {
            'time_idx': time_idx,
            'time_label': f"t{time_idx:04d}",
            'id': uid,
            'count': metrics['count'],
            'mean_diff_m': metrics['mean_diff'],
            'std_diff_m': metrics['std_diff'],
            'min_diff_m': metrics['min_diff'],
            'max_diff_m': metrics['max_diff'],
            'rmse_m': metrics['rmse'],
            'mae_m': metrics['mae'],
            'r2': metrics['r2'],
            'r2_corr': metrics['r2_corr'],
            'slope': metrics['slope'],
            'intercept': metrics['intercept'],
            'modeled_values': modeled,
            'observed_values': observed,
            'metrics': metrics,
            'modeled_mean_m': metrics['modeled_mean'],
            'observed_mean_m': metrics['observed_mean'],
        }

    # Processes

    def analyze_time_series_sequential(self):
        """Analyze all (time, id) tasks sequentially."""
        self._log("\n" + "=" * 80)
        self._log("POINT-BASED COMPARISON (SEQUENTIAL)")
        self._log("=" * 80)
        self._log(f"Number of time steps: {len(self.time_indices)}")
        self._log(f"Number of point groups: {len(self.unique_ids)}")
        self._log(f"Variable: {self.var_name}")
        self._log(f"Saving time step plots: {self.save_plots}")
        if self.use_time_window:
            self._log(f"Moving time window: size={self.window_size}")
            self._log(f"Window mode: "
                      f"{'Accumulative (maximum)' if self.use_accumulative else 'Mean'}")
        self._log("=" * 80)

        all_results = []
        total_tasks = len(self.time_indices) * len(self.unique_ids)
        completed = 0
        failed_tasks = []
        best_match_per_id = {
            uid: {'rmse': float('inf'), 'time_idx': None, 'result': None}
            for uid in self.unique_ids
        }

        for time_idx in self.time_indices:
            if self.is_cancelled():
                self._log(f"\nAnalysis cancelled at time step {time_idx}")
                break

            self._log(f"\nProcessing time index: {time_idx} "
                      f"({completed + 1}/{total_tasks} tasks)")

            for uid in self.unique_ids:
                if self.is_cancelled():
                    break

                completed += 1
                if self.progress_callback:
                    percent = 30 + int(completed / total_tasks * 60)
                    id_text = f"ID {uid}" if uid is not None else "All IDs"
                    self.progress_callback(
                        percent,
                        f"Processing time {time_idx}, {id_text} "
                        f"({completed}/{total_tasks})",
                    )

                try:
                    result = self._analyze_one(time_idx, uid)
                except Exception as e:
                    self._log(f"  ✗ ERROR Time {time_idx}, ID {uid}: "
                              f"{type(e).__name__}: {str(e)[:100]}")
                    failed_tasks.append((time_idx, uid))
                    all_results.append(_empty_result(time_idx, uid))
                    continue

                if result and result['count'] > 0:
                    all_results.append(result)
                    self._log(f"  ✓ Time {time_idx}, ID {uid}: "
                              f"RMSE={result['rmse_m']:.2f}m, "
                              f"Mean Diff={result['mean_diff_m']:.2f}m, "
                              f"R²={result['r2_corr']:.3f}")
                    if result['rmse_m'] < best_match_per_id[uid]['rmse']:
                        best_match_per_id[uid]['rmse'] = result['rmse_m']
                        best_match_per_id[uid]['time_idx'] = time_idx
                        best_match_per_id[uid]['result'] = result
                        self._log(f"    *** New best match for ID {uid} at "
                                  f"time {time_idx}! RMSE={result['rmse_m']:.2f}m ***")
                else:
                    all_results.append(_empty_result(time_idx, uid))
                    failed_tasks.append((time_idx, uid))
                    self._log(f"  ✗ Time {time_idx}, ID {uid}: No valid results (count=0)")

            self._log(f"  Progress: {completed}/{total_tasks} tasks completed")
            if time_idx % 10 == 0:
                gc.collect()
                self._log(f"  Memory cleanup performed at time step {time_idx}")

        return self._finalize(
            all_results, best_match_per_id, failed_tasks, total_tasks
        )

    def analyze_time_series_parallel(self, max_workers=None):
        """Analyze all (time, id) tasks in parallel using ThreadPoolExecutor."""
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        total_tasks = len(self.time_indices) * len(self.unique_ids)

        self._log("\n" + "=" * 80)
        self._log("POINT-BASED COMPARISON (PARALLEL - THREADS)")
        self._log("=" * 80)
        self._log(f"Number of time steps: {len(self.time_indices)}")
        self._log(f"Number of point groups: {len(self.unique_ids)}")
        self._log(f"Total tasks: {total_tasks}")
        self._log(f"Parallel workers: {max_workers}")
        self._log(f"Variable: {self.var_name}")
        self._log(f"Saving time step plots: {self.save_plots}")
        if self.use_time_window:
            self._log(f"Moving time window: size={self.window_size}")
            self._log(f"Window mode: "
                      f"{'Accumulative (maximum)' if self.use_accumulative else 'Mean'}")
        self._log("=" * 80)

        all_results = []
        failed_tasks = []
        completed = 0
        lock = threading.Lock()
        best_match_per_id = {
            uid: {'rmse': float('inf'), 'time_idx': None, 'result': None}
            for uid in self.unique_ids
        }

        tasks = [
            (time_idx, uid)
            for time_idx in self.time_indices
            for uid in self.unique_ids
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self._analyze_one, t, u): (t, u)
                              for t, u in tasks}

            for future in as_completed(future_to_task):
                time_idx, uid = future_to_task[future]
                try:
                    result = future.result(timeout=300)
                except Exception as e:
                    self._log(f"  ✗ ERROR processing time {time_idx}, ID {uid}: "
                              f"{type(e).__name__}: {str(e)[:100]}")
                    with lock:
                        failed_tasks.append((time_idx, uid))
                        all_results.append(_empty_result(time_idx, uid))
                    continue

                if result and result['count'] > 0:
                    with lock:
                        all_results.append(result)
                        completed += 1
                        if self.progress_callback:
                            percent = 30 + int(completed / total_tasks * 40)
                            self.progress_callback(
                                percent,
                                f"Processed {completed}/{total_tasks}",
                            )
                        self._log(f"  Completed time {time_idx}, ID {uid}: "
                                  f"RMSE={result['rmse_m']:.2f}m, "
                                  f"R²={result['r2_corr']:.3f}")
                        if result['rmse_m'] < best_match_per_id[uid]['rmse']:
                            best_match_per_id[uid]['rmse'] = result['rmse_m']
                            best_match_per_id[uid]['time_idx'] = time_idx
                            best_match_per_id[uid]['result'] = result
                            self._log(f"    *** New best match for ID {uid} at "
                                      f"time {time_idx}! "
                                      f"RMSE={result['rmse_m']:.2f}m ***")
                else:
                    with lock:
                        all_results.append(_empty_result(time_idx, uid))
                        failed_tasks.append((time_idx, uid))
                    self._log(f"  ✗ Time {time_idx}, ID {uid}: No valid results")

        return self._finalize(
            all_results, best_match_per_id, failed_tasks, total_tasks
        )

    def _finalize(self, all_results, best_match_per_id, failed_tasks, total_tasks):
        """Common tail: summary logging, save results, generate plots."""
        self._log("\n" + "=" * 80)
        self._log("ANALYSIS COMPLETE - SUMMARY")
        self._log("=" * 80)
        successful = [r for r in all_results if r.get('count', 0) > 0]
        self._log(f"Total tasks: {total_tasks}")
        self._log(f"Successful results: {len(successful)}")
        self._log(f"Failed/Empty tasks: {len(failed_tasks)}")

        if failed_tasks:
            self._log(f"\nFailed/Empty tasks ({len(failed_tasks)}):")
            for t, uid in failed_tasks[:10]:
                self._log(f"  - Time {t}, ID {uid}")
            if len(failed_tasks) > 10:
                self._log(f"  ... and {len(failed_tasks) - 10} more")

        # Sort by time index for stable plotting / CSV output
        all_results.sort(key=lambda x: x['time_idx'])

        self.results = all_results
        self.best_matches = best_match_per_id
        self.time_series = pd.DataFrame(all_results)

        if not successful:
            self._log("\nNo successful results generated!")
            return None

        # Overall best (min RMSE across all IDs)
        overall_best = min(
            best_match_per_id.values(),
            key=lambda x: x['rmse'] if x['rmse'] is not None else float('inf'),
        )
        self.best_match_time = overall_best['time_idx']
        self.best_match_rmse = overall_best['rmse']

        self._save_results()

        if self.save_plots:
            self.generate_all_time_step_plots()

        self._create_plots()
        #self.generate_best_match_plot()

        self._log("\n" + "=" * 80)
        self._log("BEST MATCHES SUMMARY")
        self._log("=" * 80)
        for uid, best in best_match_per_id.items():
            if best['time_idx'] is not None:
                id_text = f"ID {uid}" if uid is not None else "All IDs"
                self._log(f"  {id_text}: Time {best['time_idx']} "
                          f"with RMSE = {best['rmse']:.2f}m")

        return self.time_series

    def analyze_time_series(self, use_parallel=False, max_workers=None):
        """Run the analysis (parallel or sequential) and report runtime."""
        start_time = time.time()
        if use_parallel and len(self.time_indices) > 1 and len(self.unique_ids) > 0:
            results = self.analyze_time_series_parallel(max_workers)
        else:
            results = self.analyze_time_series_sequential()

        elapsed = time.time() - start_time
        if elapsed < 60:
            self._log(f"\nTotal runtime: {elapsed:.2f} seconds")
        elif elapsed < 3600:
            self._log(f"\nTotal runtime: {int(elapsed // 60)} minute(s) and "
                      f"{elapsed % 60:.2f} seconds")
        else:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            self._log(f"\nTotal runtime: {hours} hour(s), {minutes} minute(s), "
                      f"and {elapsed % 60:.2f} seconds")
        return results

    # output

    def _save_results(self):
        """Write time-series CSV and best-match summary text."""
        csv_file = self.output_subdirs['csv'] / f"{self.prefix}_point_comparison.csv"
        self.time_series.to_csv(csv_file, index=False)
        self._log(f"\nTime series saved to: {csv_file}")

        summary_file = self.output_subdirs['csv'] / f"{self.prefix}_best_match_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("POINT-BASED COMPARISON - BEST MATCH SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Variable: {self.var_name}\n")
            f.write(f"Observation field: {self.value_field}\n")
            if self.use_time_window:
                f.write(f"Moving time window: size={self.window_size}\n")
                f.write(f"Window mode: "
                        f"{'Accumulative (maximum)' if self.use_accumulative else 'Mean'}\n")
            f.write("\n")

            for uid, best in self.best_matches.items():
                if best['time_idx'] is None:
                    continue
                id_text = f"ID {uid}" if uid is not None else "All IDs"
                r = best['result']
                f.write(f"\n{id_text}:\n")
                f.write(f"  Best match time step: {best['time_idx']}\n")
                f.write(f"  RMSE: {best['rmse']:.2f} m\n")
                f.write(f"  Mean difference: {r['mean_diff_m']:.2f} m\n")
                f.write(f"  Std difference: {r['std_diff_m']:.2f} m\n")
                f.write(f"  Min difference: {r['min_diff_m']:.2f} m\n")
                f.write(f"  Max difference: {r['max_diff_m']:.2f} m\n")
                f.write(f"  R²: {r['r2_corr']:.4f}\n")
                f.write(f"  Number of points: {r['count']}\n")

        self._log(f"Summary saved to: {summary_file}")

    def _create_plots(self):
        """Create the 2x2 summary plot (RMSE, MAE, Pearson R², best-match scatter)."""
        if self.time_series is None or self.time_series.empty:
            self._log("No data to create plots.")
            return
        if 'rmse_m' not in self.time_series.columns:
            self._log("Warning: 'rmse_m' column not found in results")
            return

        self._log(f"\nCreating plots with {len(self.time_series)} rows")

        # Unique IDs (drop None entries so legend labels are cleaner)
        unique_ids = [uid for uid in self.time_series['id'].unique() if uid is not None]
        if not unique_ids:
            unique_ids = [None]

        all_time_indices = getattr(self, 'time_indices',
                                self.time_series['time_idx'].values)
        if len(all_time_indices):
            x_min, x_max = min(all_time_indices), max(all_time_indices)
        else:
            x_min, x_max = 0, len(self.time_series) - 1

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        subplot_specs = (
            ('rmse_m', 'o-', 'RMSE (m)', 'Root Mean Square Error Over Time'),
            ('mae_m', 's-', 'Mean Absolute Difference (m)',
            'Mean Absolute Difference |Modeled - Observed| Over Time'),
            ('r2_corr', 'd-', 'Pearson R²',
            'Pearson R² Over Time (matches Excel RSQ)'),
        )

        all_r2_values = []
        for ax, (col, marker, ylabel, title) in zip(axes.flat, subplot_specs):
            has_data = False
            for i, uid in enumerate(unique_ids):
                df_id = self.time_series if uid is None else \
                    self.time_series[self.time_series['id'] == uid]
                df_valid = df_id[df_id[col].notna()]
                if df_valid.empty:
                    continue
                has_data = True
                color = PLOT_COLORS[i % len(PLOT_COLORS)]
                label = f'ID {uid}' if uid is not None else 'All IDs'
                ax.plot(df_valid['time_idx'], df_valid[col],
                        marker, linewidth=2, markersize=6, color=color, label=label)
                if col == 'r2_corr':
                    all_r2_values.extend(df_valid['r2_corr'].values)

            ax.set_xlim(x_min - 1, x_max + 1)
            ax.set_xlabel('Time Step')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            if has_data:
                ax.legend()
            ax.grid(True, alpha=0.3)

        # Pearson R² is always in [0, 1]
        ax3 = axes[1, 0]
        if all_r2_values:
            ax3.set_ylim(0.0, max(1.0, max(all_r2_values) + 0.05))
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=1.0, alpha=0.5)

        # ---- 4th panel: best-match scatter ----------------------------------
        ax4 = axes[1, 1]
        if len(unique_ids) == 1:
            # Single ID (or None): draw in the panel
            self._draw_best_match_scatter(ax4, unique_ids[0])
        else:
            # Multiple IDs: hide the panel, save one scatter PDF per ID
            ax4.axis('off')
            for uid in unique_ids:
                fig_bm, ax_bm = plt.subplots(figsize=(8, 8))
                self._draw_best_match_scatter(ax_bm, uid)
                plt.tight_layout()

                id_suffix = f"_id{uid}" if uid is not None else ""
                bm_path = (self.output_subdirs['best_match']
                        / f"{self.prefix}_best_match{id_suffix}.pdf")
                bm_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(bm_path, dpi=150, bbox_inches='tight')
                plt.close(fig_bm)
                gc.collect()
                self._log(f"Best match scatter saved to: {bm_path}")

        window_mode = (
            f"Window: {self.window_size} steps "
            f"({'Accumulative' if self.use_accumulative else 'Mean'})"
            if self.use_time_window else "No window"
        )
        plt.suptitle(
            f'Point-Based Comparison - {self.var_name.upper()} ({window_mode})',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()

        plot_path = self.output_subdirs['plots'] / f"{self.prefix}_point_comparison.pdf"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        self._log(f"Summary plot saved to: {plot_path}")

    def generate_all_time_step_plots(self):
        """Generate one scatter plot per (time, id) task."""
        if not self.results:
            self._log("No results available. Run analyze_time_series*() first.")
            return

        self._log("\n" + "=" * 80)
        self._log("GENERATING TIME STEP PLOTS")
        self._log("=" * 80)

        total = len(self.results)
        for i, result in enumerate(self.results):
            if self.is_cancelled():
                self._log("Plot generation cancelled")
                return

            time_idx = result['time_idx']
            uid = result['id']

            if result.get('count', 0) > 0:
                plot_file = self._save_time_step_plot(
                    time_idx, uid,
                    result.get('modeled_values', []),
                    result.get('observed_values', []),
                    result.get('metrics', {}),
                )
                self._log(f"  Generated plot {i + 1}/{total}: {plot_file}")
            else:
                self._log(f"  Skipping plot {i + 1}/{total}: "
                          f"No valid data for time {time_idx}, ID {uid}")

            if self.progress_callback:
                percent = 70 + int((i + 1) / total * 30)
                self.progress_callback(percent, f"Generating plot {i + 1}/{total}")

        self._log("=" * 80)
        self._log("All plots generated successfully!")
        self._log("=" * 80)

    def _save_time_step_plot(self, time_idx, uid, modeled_values,
                             observed_values, metrics):
        """Save a scatter plot for one (time, id) task."""
        if not modeled_values or not observed_values:
            return None

        fig, ax = plt.subplots(figsize=(8, 6))
        modeled = np.asarray(modeled_values)
        observed = np.asarray(observed_values)

        ax.scatter(observed, modeled, c='blue', alpha=0.6, s=50, edgecolor='black')

        min_val = min(observed.min(), modeled.min())
        max_val = max(observed.max(), modeled.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
                label='1:1 line')

        z = np.polyfit(observed, modeled, 1)
        p = np.poly1d(z)
        ax.plot(observed, p(observed), 'g-', linewidth=2,
                label=f'Best fit: y={z[0]:.2f}x+{z[1]:.2f}')

        text_str = (f'RMSE: {metrics.get("rmse", np.nan):.2f}m\n'
                    f'R²: {metrics.get("r2_corr", np.nan):.3f}\n'
                    f'n={metrics.get("count", 0)}')
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

        ax.set_xlabel('Observed (m)')
        ax.set_ylabel('Modeled (m)')
        title = f'Time Step {time_idx}'
        if uid is not None:
            title += f', ID {uid}'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        id_suffix = f"_id{uid}" if uid is not None else ""
        plot_path = (self.output_subdirs['time_step_plots']
                     / f"{self.prefix}_point_t{time_idx:04d}{id_suffix}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        return plot_path

    def generate_best_match_plot(self):
        """Scatter of modeled vs observed at each ID's best-match time."""
        n_ids = len(self.unique_ids)
        if n_ids == 1:
            self._generate_single_best_match_plot(self.unique_ids[0])
            return

        n_cols = min(3, n_ids)
        n_rows = (n_ids + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, uid in enumerate(self.unique_ids):
            self._draw_best_match_scatter(axes[i], uid)
        for j in range(len(self.unique_ids), len(axes)):
            axes[j].axis('off')

        plt.suptitle('Best Match Scatter Plots by Field ID',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = self.output_subdirs['best_match'] / \
            f"{self.prefix}_best_match_scatter.pdf"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        self._log(f"Best match scatter plots saved to: {plot_path}")

    def _generate_single_best_match_plot(self, uid):
        """Generate a single best-match scatter plot."""
        fig, ax = plt.subplots(figsize=(8, 8))
        self._draw_best_match_scatter(ax, uid)
        plt.tight_layout()

        plot_path = self.output_subdirs['best_match'] / \
            f"{self.prefix}_best_match_scatter.pdf"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        self._log(f"Best match scatter plot saved to: {plot_path}")

    def _draw_best_match_scatter(self, ax, uid):
        best = self.best_matches.get(uid)
        if not best or best['time_idx'] is None:
            ax.text(0.5, 0.5, f'No valid match for ID {uid}',
                    ha='center', va='center')
            ax.set_title(f'ID {uid}')
            return

        self._build_point_cache()
        t = best['time_idx']

        if uid is None:
            sel = np.ones(len(self.points_gdf), dtype=bool)
        else:
            sel = (self.points_gdf[self.id_field] == uid).to_numpy()
        positions = np.where(sel)[0]

        vals = self._var_series[t, positions]
        sub = self.points_gdf.iloc[positions]

        modeled_values, observed_values = [], []
        for k, (_, row) in enumerate(sub.iterrows()):
            if row.geometry.geom_type != 'Point':
                continue
            mv = vals[k]
            if np.isnan(mv):
                continue
            obs_value = row[self.value_field]
            if obs_value is None or (isinstance(obs_value, float)
                                    and np.isnan(obs_value)):
                continue
            modeled_values.append(float(mv))
            observed_values.append(float(obs_value))

        if not modeled_values:
            ax.text(0.5, 0.5, f'No data points for ID {uid}',
                    ha='center', va='center')
            ax.set_title(f'ID {uid}')
            return

        modeled = np.asarray(modeled_values)
        observed = np.asarray(observed_values)

        ax.scatter(observed, modeled, c='blue', alpha=0.6, s=50, edgecolor='black')

        min_val = min(observed.min(), modeled.min())
        max_val = max(observed.max(), modeled.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--',
                linewidth=2, label='1:1 line')

        z = np.polyfit(observed, modeled, 1)
        p = np.poly1d(z)
        ax.plot(observed, p(observed), 'g-', linewidth=2,
                label=f'y={z[0]:.2f}x+{z[1]:.2f}')

        r = best['result']
        text_str = f'Time step: {best["time_idx"]}\nRMSE: {r["rmse_m"]:.2f}m\nR²: {r["r2_corr"]:.3f}\nn={r["count"]}'
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

        ax.set_xlabel('Observed (m)')
        ax.set_ylabel('Modeled (m)')
        window_text = (f"Window: {self.window_size} steps"
                       if self.use_time_window else "No window")
        ax.set_title(f'ID {uid} - Time {best["time_idx"]} ({window_text})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
