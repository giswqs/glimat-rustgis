"""
/*********************************************************************************
* Automated Proximity and Conformity Analysis (APCA )for ice margin comparison   * 
* =============================================================================  *
* Time Series APCA Analysis for NetCDF Data                                      *
* Enhanced with moving time window support for ice mask extraction               * 
*                                                                                * 
* Features:                                                                      *
* - Handles corrupted time coordinates in NetCDF files                           *
* - Checks and matches projections between NetCDF and shapefile                  *
* - Calculates mean and standard deviation offsets between modeled and           *
*   field-mapped ice margin (line feature)                                       *
* - Identifies best match time (smallest offset)                                 *
* - Generates ice cover map for the best match time with field ice margin overlay*
* - Saves plots for each time step to output folder                              *
* - Exports all results to organized output directory                            *
* - MOVING TIME WINDOW: Smooth ice masks by using accumulative or consistent ice *
*   presence                                                                     *
*                                                                                * 
* Developed based on the original APCA code by Li et al. (2008)                  *
* This file is part of GLIMAT Plugin.                                            *
*                                                                                *
* Author: Yingkui Li                                                             *
* Department of Geography & Sustainability                                       *
* University of Tennessee, Knoxville, TN 37996                                   * 
* Email: yli32@utk.edu                                                           *
* Start date: 06/01/2026                                                         *
* Finalized date: 09/17/2026                                                     *
* Copyright(C): 2026 Yingkui Li                                                  * 
*                                                                                *
* Version: 1.0.0                                                                 *
*********************************************************************************/
"""

import gc
import multiprocessing
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rasterio.features import geometry_mask, rasterize, shapes
from rasterio.transform import from_origin
from scipy import stats
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape
from shapely.ops import unary_union

mpl.use('Agg')  # Non-interactive backend
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Helper functions (module-level; use print for logging)
# --------------------------------------------------------------------------- #
def get_ice_variable(ds, ice_variable='thk'):
    """
    Find the appropriate ice variable in the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset or time slice.
    ice_variable : str
        Name of the ice-thickness variable (user-specified).
    """
    if ice_variable in ds.data_vars:
        print(f"Using ice variable: {ice_variable}")
        return ds[ice_variable]

    ice_vars = ('thk', 'land_ice_thickness', 'ice_thickness', 'usurf', 'topg')
    for var in ice_vars:
        if var in ds.data_vars:
            print(f"Using ice variable: {var}")
            return ds[var]

    print(f"Available variables: {list(ds.data_vars.keys())}")
    for var in ds.data_vars:
        if var not in ('time_bounds', 'mapping', 'pism_config',
                       'run_stats', 'timestamp'):
            print(f"Using variable: {var}")
            return ds[var]

    raise ValueError(
        f"No ice thickness variable found. Tried: {ice_variable} and common names"
    )


def get_ice_mask_window(dataset, time_indices, min_ice_thickness=1.0,
                        ice_variable='thk', use_accumulative=True):
    """
    Build an ice mask over a time window (cumulative or per-time).

    Parameters
    ----------
    dataset : xarray.Dataset
        PISM NetCDF dataset.
    time_indices : list of int
        Time indices to include in the window.
    min_ice_thickness : float
        Minimum ice thickness to consider as ice (meters).
    ice_variable : str
        Name of the ice-thickness variable.
    use_accumulative : bool
        If True, ice is present if ANY time step in the window has ice.
        If False, ice is present if ALL time steps in the window have ice.

    Returns
    -------
    ice_mask : numpy.ndarray of bool
        Boolean mask where ice is present.
    ice_thickness_mean : numpy.ndarray
        Mean ice thickness over the window.
    """
    ice_mask_total = None
    ice_thickness_sum = None
    count = 0

    for t_idx in time_indices:
        ds_slice = dataset.isel(time=t_idx)

        ice_thickness = get_ice_variable(ds_slice, ice_variable)
        if hasattr(ice_thickness, 'values'):
            ice_thickness = ice_thickness.values

        ice_present_step = ice_thickness >= min_ice_thickness

        if ice_mask_total is None:
            ice_mask_total = ice_present_step.copy()
            ice_thickness_sum = ice_thickness.copy()
        else:
            if use_accumulative:
                ice_mask_total = ice_mask_total | ice_present_step
            else:
                ice_mask_total = ice_mask_total & ice_present_step
            ice_thickness_sum += ice_thickness
        count += 1

    ice_thickness_mean = (ice_thickness_sum / count
                          if count > 0 else np.zeros_like(ice_mask_total))
    return ice_mask_total, ice_thickness_mean


def extract_ice_outline_window(dataset, time_idx, min_ice_thickness=1.0,
                               cell_size=500, crs=None, ice_variable='thk',
                               use_time_window=False, window_size=5,
                               use_accumulative_ice=True):
    """
    Extract an ice outline using either a single time step or a moving window.

    Returns
    -------
    outline_gdf : geopandas.GeoDataFrame or None
    ice_area_km2 : float
    ice_thickness : numpy.ndarray or None
    extent : list or None
    """
    if use_time_window and window_size > 1:
        half_window = window_size // 2
        start_idx = max(0, time_idx - half_window)
        end_idx = min(len(dataset.time), time_idx + half_window + 1)
        time_indices = list(range(start_idx, end_idx))

        print(f"    Time window: indices {start_idx} to {end_idx - 1} "
              f"({len(time_indices)} steps)")

        ice_mask, ice_thickness = get_ice_mask_window(
            dataset, time_indices, min_ice_thickness,
            ice_variable, use_accumulative_ice,
        )
    else:
        ds_slice = dataset.isel(time=time_idx)
        ice_thickness = get_ice_variable(ds_slice, ice_variable)
        if hasattr(ice_thickness, 'values'):
            ice_thickness = ice_thickness.values
        ice_mask = ice_thickness >= min_ice_thickness

    if not np.any(ice_mask):
        return None, 0, None, None

    # Coordinates / transform
    try:
        if 'x' in dataset.dims and 'y' in dataset.dims:
            x = dataset.x.values
            y = dataset.y.values
            transform = from_origin(
                west=float(x.min()),
                north=float(y.max()),
                xsize=float(abs(x[1] - x[0])) if len(x) > 1 else cell_size,
                ysize=float(abs(y[1] - y[0])) if len(y) > 1 else cell_size,
            )
            extent = [x.min(), x.max(), y.min(), y.max()]
        else:
            x = np.arange(ice_mask.shape[1])
            y = np.arange(ice_mask.shape[0])
            transform = from_origin(0, len(y), cell_size, cell_size)
            extent = [0, ice_mask.shape[1], 0, ice_mask.shape[0]]
    except Exception as e:
        print(f"  Error getting coordinates: {e}")
        return None, 0, None, None

    # Convert to polygons (flip Y for GIS)
    ice_int = np.flipud(ice_mask.astype(np.uint8))

    try:
        results = shapes(ice_int, mask=ice_int == 1, transform=transform)
        polygons = [shape(geom) for geom, value in results if value == 1]

        if polygons:
            min_area = cell_size * cell_size * 4
            polygons = [p for p in polygons if p.area >= min_area]

            if polygons:
                ice_polygon = unary_union(polygons)
                ice_area_km2 = ice_polygon.area / 1e6
                outline_gdf = gpd.GeoDataFrame(
                    {'time_idx': [time_idx], 'area_km2': [ice_area_km2]},
                    geometry=[ice_polygon],
                    crs=crs,
                )
                return outline_gdf, ice_area_km2, ice_thickness, extent
    except Exception as e:
        print(f"  Error converting to polygons: {e}")

    return None, 0, None, None

# --------------------------------------------------------------------------- #
# Main analyzer
# --------------------------------------------------------------------------- #
class TimeSeriesAPCA:
    """Time-series analysis of APCA metrics from PISM NetCDF data."""

    def __init__(self, pism_dataset, target_features, output_subdirs,
                 prefix='', time_indices=None, ice_threshold=0.5,
                 min_ice_thickness=1.0, cell_size=500, crs=None,
                 ice_variable='thk', use_time_window=False, window_size=5,
                 use_accumulative_ice=True,
                 progress_callback=None, cancel_checker=None,
                 use_dataset_extent=True, margin_factor=0.1):
        self.pism_dataset = pism_dataset
        self.ice_threshold = ice_threshold
        self.min_ice_thickness = min_ice_thickness
        self.cell_size = cell_size
        self.prefix = prefix
        self.buffer_distance = cell_size / 2
        self.crs = crs
        self.output_subdirs = output_subdirs
        self.ice_variable = ice_variable
        self.console = None

        # Moving window parameters
        self.use_time_window = use_time_window
        self.window_size = (window_size if window_size % 2 == 1
                            else window_size + 1)
        self.use_accumulative_ice = use_accumulative_ice

        self.progress_callback = progress_callback
        self.cancel_checker = cancel_checker
        self._is_cancelled = False

        self.use_dataset_extent = use_dataset_extent
        self.margin_factor = margin_factor

        # Cached dataset extent (only computed once)
        self.dataset_extent = None
        self.global_extent = None
        if use_dataset_extent:
            self.dataset_extent = self._get_dataset_extent()
            if self.dataset_extent is not None:
                width = self.dataset_extent[1] - self.dataset_extent[0]
                height = self.dataset_extent[3] - self.dataset_extent[2]
                mx = width * margin_factor
                my = height * margin_factor
                self.global_extent = (
                    self.dataset_extent[0] - mx,
                    self.dataset_extent[1] + mx,
                    self.dataset_extent[2] - my,
                    self.dataset_extent[3] + my,
                )
                self._log(f"Using dataset extent with "
                          f"{margin_factor * 100}% margin")
                self._log(f"  Final plot extent: "
                          f"X[{self.global_extent[0]:.0f}, "
                          f"{self.global_extent[1]:.0f}], "
                          f"Y[{self.global_extent[2]:.0f}, "
                          f"{self.global_extent[3]:.0f}]")

        # Load target features
        if isinstance(target_features, str):
            self._log(f"Loading shapefile: {target_features}")
            self.target_gdf = gpd.read_file(target_features)
        else:
            self.target_gdf = target_features.copy()

        if self.target_gdf.crs is None and crs is not None:
            self.target_gdf.set_crs(crs, inplace=True)

        self.target_type = self._determine_feature_type()

        # Time indices
        if 'time' in pism_dataset.dims:
            self.n_times = len(pism_dataset.time)
            if time_indices is None:
                if self.use_time_window and self.window_size > 1:
                    half_win = self.window_size // 2
                    self.time_indices = list(
                        range(half_win, self.n_times - half_win)
                    )
                else:
                    self.time_indices = list(range(self.n_times))
            else:
                self.time_indices = [i for i in time_indices
                                     if i < self.n_times]
            self.time_values = self.time_indices
            self._log(f"\nTotal time steps: {self.n_times}")
            self._log(f"Analyzing {len(self.time_indices)} time steps")
            if self.time_indices:
                self._log(f"Time range: {self.time_indices[0]} "
                          f"to {self.time_indices[-1]}")
            if self.use_time_window:
                self._log(f"Moving time window: size={self.window_size}, "
                          f"accumulative={self.use_accumulative_ice}")
        else:
            self.n_times = 1
            self.time_indices = ([0] if time_indices is None
                                 else [time_indices[0]] if time_indices
                                 else [0])
            self.time_values = [0]

        # Results placeholders
        self.results = []
        self.time_series = None
        self.best_match_time = None
        self.best_match_offset = None
        self.best_match_outline = None
        self.best_match_apca_results = None
        self.best_match_ice_data = None

        # Trend statistics
        self.trend_slope = 0
        self.trend_pvalue = 1
        self.trend_r2 = 0
        self.rate_of_change = 0

    # ------------------------------------------------------------------ #
    # Logging / cancellation
    # ------------------------------------------------------------------ #
    def set_console(self, console_callback):
        """Set a console callback for debug messages."""
        self.console = console_callback

    def _log(self, message):
        if self.console:
            self.console(message)
        print(message)

    def cancel(self):
        """Cancel the analysis."""
        self._is_cancelled = True
        self._log("\n  Cancellation requested for APCA analysis...")

    def is_cancelled(self):
        """Check whether the analysis has been cancelled."""
        if self.cancel_checker:
            return self.cancel_checker()
        return self._is_cancelled

    # ------------------------------------------------------------------ #
    # Dataset helpers
    # ------------------------------------------------------------------ #
    def _get_dataset_extent(self):
        """Return (xmin, xmax, ymin, ymax) for the full model domain."""
        try:
            if 'x' in self.pism_dataset.dims and 'y' in self.pism_dataset.dims:
                x_coords = self.pism_dataset.x.values
                y_coords = self.pism_dataset.y.values
                extent = (
                    float(x_coords.min()),
                    float(x_coords.max()),
                    float(y_coords.min()),
                    float(y_coords.max()),
                )
                self._log(f"  Dataset extent: "
                          f"X[{extent[0]:.0f}, {extent[1]:.0f}], "
                          f"Y[{extent[2]:.0f}, {extent[3]:.0f}]")
                self._log(f"  Grid dimensions: "
                          f"{len(x_coords)} x {len(y_coords)} cells")
                return extent
            self._log("  Warning: No x/y coordinates found in dataset")
            return None
        except Exception as e:
            self._log(f"  Error getting dataset extent: {e}")
            return None

    def _determine_feature_type(self):
        """Return 'line' or 'polygon' depending on the target features."""
        first_geom = self.target_gdf.geometry.iloc[0]
        geom_type = first_geom.geom_type
        self._log(f"Target feature type: {geom_type}")

        if geom_type in ('LineString', 'MultiLineString'):
            return 'line'
        if geom_type in ('Polygon', 'MultiPolygon'):
            return 'polygon'
        self._log(f"  Warning: Unsupported geometry type {geom_type}, "
                  f"treating as line")
        return 'line'

    def _extract_ice_outline(self, time_idx):
        """Extract ice outline using the configured (window or single) mode."""
        return extract_ice_outline_window(
            self.pism_dataset, time_idx, self.min_ice_thickness,
            self.cell_size, self.crs, self.ice_variable,
            self.use_time_window, self.window_size, self.use_accumulative_ice,
        )

    # ------------------------------------------------------------------ #
    # APCA calculation for a single time step
    # ------------------------------------------------------------------ #
    def _calculate_apca_for_time(self, modeled_outline_gdf, time_idx):
        """Calculate APCA metrics for a single time step."""
        modeled_polygon = modeled_outline_gdf.geometry.iloc[0]
        if modeled_polygon.geom_type == 'Polygon':
            modeled_line = modeled_polygon.boundary
        else:
            lines = [poly.boundary for poly in modeled_polygon.geoms]
            modeled_line = unary_union(lines)

        # Prepare target features
        if self.target_type == 'line':
            target_features = self.target_gdf.copy()
            target_features.geometry = target_features.geometry.buffer(
                self.buffer_distance
            )
        else:
            target_features = self.target_gdf.copy()

        target_bounds = target_features.total_bounds
        model_bounds = modeled_line.bounds

        bounds = (
            min(target_bounds[0], model_bounds[0]),
            min(target_bounds[1], model_bounds[1]),
            max(target_bounds[2], model_bounds[2]),
            max(target_bounds[3], model_bounds[3]),
        )
        padding = max((bounds[2] - bounds[0]) * 0.1, self.cell_size * 5)
        bounds = (
            bounds[0] - padding,
            bounds[1] - padding,
            bounds[2] + padding,
            bounds[3] + padding,
        )

        width = max(1, int((bounds[2] - bounds[0]) / self.cell_size) + 1)
        height = max(1, int((bounds[3] - bounds[1]) / self.cell_size) + 1)
        transform = from_origin(bounds[0], bounds[3],
                                self.cell_size, self.cell_size)

        if modeled_line.geom_type == 'MultiLineString':
            geometries = [(geom, 1) for geom in modeled_line.geoms]
        else:
            geometries = [(modeled_line, 1)]

        try:
            line_raster = rasterize(
                geometries,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            )
            distance_raster = distance_transform_edt(1 - line_raster) * self.cell_size
        except Exception as e:
            self._log(f"  Error creating distance raster: {e}")
            return None

        results_list = []
        for idx, row in target_features.iterrows():
            geom = row.geometry
            if geom.is_empty:
                continue

            minx, miny, maxx, maxy = geom.bounds
            col_min = max(0, int((minx - bounds[0]) / self.cell_size))
            col_max = min(distance_raster.shape[1] - 1,
                          int((maxx - bounds[0]) / self.cell_size))
            row_min = max(0, int((bounds[3] - maxy) / self.cell_size))
            row_max = min(distance_raster.shape[0] - 1,
                          int((bounds[3] - miny) / self.cell_size))

            if col_min >= col_max or row_min >= row_max:
                continue

            try:
                mask = geometry_mask(
                    [geom],
                    transform=transform,
                    invert=True,
                    out_shape=(height, width),
                )
                mask_subset = mask[row_min:row_max + 1, col_min:col_max + 1]
                raster_subset = distance_raster[row_min:row_max + 1,
                                                col_min:col_max + 1]
                distances = raster_subset[mask_subset]
            except Exception:
                continue

            if len(distances) == 0:
                continue

            result = {
                'feature_id': row.get('id', idx),
                'mean_offset_m': float(np.mean(distances)),
                'std_offset_m': float(np.std(distances)),
                'median_offset_m': float(np.median(distances)),
                'num_samples': len(distances),
            }
            for col in self.target_gdf.columns:
                if col not in result and col != 'geometry':
                    try:
                        result[col] = row[col]
                    except Exception:
                        pass
            results_list.append(result)

        return pd.DataFrame(results_list) if results_list else None

    # ------------------------------------------------------------------ #
    # Sequential driver
    # ------------------------------------------------------------------ #
    def analyze_time_series(self, id_field=None, save_plots=True,
                            save_shapefiles=True):
        """Analyze APCA metrics through time and find the best match."""
        self._log("\n" + "=" * 80)
        self._log("TIME SERIES APCA ANALYSIS")
        self._log("=" * 80)
        self._log(f"Number of time steps: {len(self.time_indices)}")
        self._log(f"Target feature type: {self.target_type}")
        self._log(f"Cell size: {self.cell_size} m")
        if self.use_time_window:
            self._log(f"Moving time window: size={self.window_size}, "
                      f"accumulative={self.use_accumulative_ice}")
        if self.use_dataset_extent and self.dataset_extent is not None:
            self._log("Plot extent: Using FIXED dataset extent")
            self._log(f"  Extent: X[{self.global_extent[0]:.0f}, "
                      f"{self.global_extent[1]:.0f}], "
                      f"Y[{self.global_extent[2]:.0f}, "
                      f"{self.global_extent[3]:.0f}]")
        else:
            self._log("Plot extent: Auto-extent per time step")
        self._log("=" * 80)

        all_results = []
        best_offset = float('inf')
        best_idx = None
        best_outline = None
        best_apca = None
        best_ice_data = None

        total_steps = len(self.time_indices)

        for i, time_idx in enumerate(self.time_indices):
            if self.is_cancelled():
                self._log("\n  Analysis cancelled by user.")
                if self.progress_callback:
                    self.progress_callback(100, "Analysis cancelled")
                return None

            if self.progress_callback:
                progress = 30 + int((i + 1) / total_steps * 60)
                self.progress_callback(
                    progress, f"Processing APCA time step {time_idx}"
                )

            time_label = f"t{time_idx:04d}"
            self._log(f"\nProcessing time step {i + 1}/{total_steps}: "
                      f"{time_label}")

            outline_gdf, ice_area, ice_thickness, extent = \
                self._extract_ice_outline(time_idx)

            if outline_gdf is None:
                self._log(f"  Warning: No ice outline found for "
                          f"time {time_label}")
                continue

            if save_shapefiles:
                outline_shp = (self.output_subdirs['ice_outlines']
                               / f"{self.prefix}_ice_outline_{time_label}.shp")
                outline_gdf.to_file(outline_shp)

            apca_results = self._calculate_apca_for_time(outline_gdf, time_idx)

            if apca_results is None or apca_results.empty:
                self._log(f"  Ice area: {ice_area:.2f} km²")
                self._log("  No APCA results (no overlap with target features)")
                continue

            apca_results['time_idx'] = time_idx
            apca_results['ice_area_km2'] = ice_area
            all_results.append(apca_results)

            mean_offset = apca_results['mean_offset_m'].mean()
            std_offset = apca_results['std_offset_m'].mean()
            self._log(f"  Ice area: {ice_area:.2f} km²")
            self._log(f"  Mean offset: {mean_offset:.2f} ± {std_offset:.2f} m")
            self._log(f"  Number of samples: {len(apca_results)}")

            if save_plots:
                if self.use_dataset_extent and self.dataset_extent is not None:
                    plot_file = self._save_time_step_plot_with_fixed_extent(
                        time_idx, outline_gdf, ice_thickness, extent, mean_offset
                    )
                else:
                    plot_file = self._save_time_step_plot(
                        time_idx, outline_gdf, ice_thickness, extent, mean_offset
                    )
                self._log(f"  Saved plot: {plot_file}")

            apca_csv = (self.output_subdirs['csv']
                        / f"{self.prefix}_apca_results_{time_label}.csv")
            apca_results.to_csv(apca_csv, index=False)

            if mean_offset < best_offset:
                best_offset = mean_offset
                best_idx = time_idx
                best_outline = outline_gdf
                best_apca = apca_results
                best_ice_data = {'thickness': ice_thickness, 'extent': extent}
                self._log(f"  *** New best match! "
                          f"Offset: {best_offset:.2f} m ***")

        self.best_match_time = best_idx
        self.best_match_offset = best_offset
        self.best_match_outline = best_outline
        self.best_match_apca_results = best_apca
        self.best_match_ice_data = best_ice_data

        if not all_results:
            self._log("\nNo results generated!")
            return None

        self.results = all_results
        self._aggregate_time_series(id_field)

        if self.best_match_outline is not None:
            best_shp = (self.output_subdirs['best_match']
                        / f"{self.prefix}_best_match_ice_outline_"
                          f"t{best_idx:04d}.shp")
            self.best_match_outline.to_file(best_shp)
            self._log(f"\nBest match outline saved to: {best_shp}")

        self._log("\n" + "=" * 80)
        self._log("BEST MATCH IDENTIFIED")
        self._log("=" * 80)
        self._log(f"Best match time step: {self.best_match_time}")
        self._log(f"Mean offset: {self.best_match_offset:.2f} m")
        if self.best_match_apca_results is not None:
            self._log(f"Standard deviation: "
                      f"{self.best_match_apca_results['std_offset_m'].mean():.2f} m")
            self._log(f"Number of samples: "
                      f"{len(self.best_match_apca_results)}")

        return self.time_series

    # ------------------------------------------------------------------ #
    # Parallel driver
    # ------------------------------------------------------------------ #
    def analyze_time_series_parallel(self, id_field=None, save_shapefiles=True,
                                     save_plots=True, max_workers=None):
        """
        Analyze APCA metrics using parallel processing (calculations only).
        Plots are generated sequentially afterward for thread safety.
        """
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        self._log("\n" + "=" * 80)
        self._log("TIME SERIES APCA ANALYSIS (PARALLEL - CALCULATIONS ONLY)")
        self._log("=" * 80)
        self._log(f"Number of time steps: {len(self.time_indices)}")
        self._log(f"Parallel workers: {max_workers}")
        self._log(f"Target feature type: {self.target_type}")
        self._log(f"Cell size: {self.cell_size} m")
        if self.use_time_window:
            self._log(f"Moving time window: size={self.window_size}, "
                      f"accumulative={self.use_accumulative_ice}")
        if self.use_dataset_extent and self.dataset_extent is not None:
            self._log("Plot extent: Using FIXED dataset extent")

        # Progress budget: calculations take the first half if plots follow.
        calc_progress = 30 if save_plots else 60
        if save_plots:
            self._log("NOTE: Plots will be generated SEQUENTIALLY "
                      "after calculations")
        self._log("=" * 80)

        all_results = []
        best_offset = float('inf')
        best_idx = None
        best_outline = None
        best_apca = None
        best_ice_data = None

        total_steps = len(self.time_indices)
        completed = 0
        lock = threading.Lock()

        def process_step(t_idx):
            return self.analyze_single_time_step(
                t_idx, save_plots=False, save_shapefiles=save_shapefiles
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(process_step, idx): idx
                             for idx in self.time_indices}

            for future in as_completed(future_to_idx):
                time_idx = future_to_idx[future]
                try:
                    result = future.result(timeout=300)
                except Exception as e:
                    self._log(f"  Error processing time step {time_idx}: {e}")
                    continue

                if not result:
                    continue

                with lock:
                    all_results.append(result)
                    completed += 1

                    if self.progress_callback:
                        percent = 30 + int(completed / total_steps
                                           * calc_progress)
                        self.progress_callback(
                            percent,
                            f"Processed {completed}/{total_steps} time steps"
                        )

                    self._log(f"  Completed time step {time_idx}")

                    if result.get('mean_offset') is not None:
                        mean_offset = result['mean_offset']
                        self._log(f"    Offset: {mean_offset:.2f} m")
                        if mean_offset < best_offset:
                            best_offset = mean_offset
                            best_idx = result['time_idx']
                            best_outline = result['outline_gdf']
                            best_apca = result['apca_results']
                            best_ice_data = {
                                'thickness': result['ice_thickness'],
                                'extent': result['extent'],
                            }
                            self._log(f"    *** New best match! "
                                      f"Offset: {best_offset:.2f} m ***")
                    else:
                        self._log(f"    Ice area: "
                                  f"{result['ice_area']:.2f} km² "
                                  f"(no APCA results)")

        if not all_results:
            self._log("\nNo results generated!")
            return None

        self.results = [r['apca_results'] for r in all_results
                        if r.get('apca_results') is not None]

        self.best_match_time = best_idx
        self.best_match_offset = best_offset
        self.best_match_outline = best_outline
        self.best_match_apca_results = best_apca
        self.best_match_ice_data = best_ice_data

        if self.results:
            self._aggregate_time_series(id_field)

        if self.best_match_outline is not None:
            best_shp = (self.output_subdirs['best_match']
                        / f"{self.prefix}_best_match_ice_outline_"
                          f"t{best_idx:04d}.shp")
            self.best_match_outline.to_file(best_shp)
            self._log(f"\nBest match outline saved to: {best_shp}")

        self._log("\n" + "=" * 80)
        self._log("BEST MATCH IDENTIFIED")
        self._log("=" * 80)
        self._log(f"Best match time step: {self.best_match_time}")
        self._log(f"Mean offset: {self.best_match_offset:.2f} m")
        if self.best_match_apca_results is not None:
            self._log(f"Standard deviation: "
                      f"{self.best_match_apca_results['std_offset_m'].mean():.2f} m")
            self._log(f"Number of samples: "
                      f"{len(self.best_match_apca_results)}")

        # Sequential plotting (safe for matplotlib)
        if save_plots and self.results:
            self.generate_all_plots()

        return self.time_series

    def analyze_single_time_step(self, time_idx, save_plots=False,
                                 save_shapefiles=True):
        """
        Analyze a single time step (thread-safe; no plotting).

        Called from worker threads. `save_plots` is accepted for signature
        compatibility but ignored here; plots are produced sequentially by
        `generate_all_plots`.
        """
        if self.is_cancelled():
            return None

        time_label = f"t{time_idx:04d}"

        outline_gdf, ice_area, ice_thickness, extent = \
            self._extract_ice_outline(time_idx)

        if outline_gdf is None:
            return None

        if save_shapefiles:
            try:
                outline_shp = (self.output_subdirs['ice_outlines']
                               / f"{self.prefix}_ice_outline_{time_label}.shp")
                outline_gdf.to_file(outline_shp)
            except Exception:
                pass

        apca_results = self._calculate_apca_for_time(outline_gdf, time_idx)

        if apca_results is None or apca_results.empty:
            return {
                'time_idx': time_idx,
                'time_label': time_label,
                'ice_area': ice_area,
                'outline_gdf': outline_gdf,
                'ice_thickness': ice_thickness,
                'extent': extent,
                'apca_results': None,
                'mean_offset': None,
                'std_offset': None,
            }

        apca_results['time_idx'] = time_idx
        apca_results['ice_area_km2'] = ice_area
        mean_offset = apca_results['mean_offset_m'].mean()
        std_offset = apca_results['std_offset_m'].mean()

        try:
            apca_csv = (self.output_subdirs['csv']
                        / f"{self.prefix}_apca_results_{time_label}.csv")
            apca_results.to_csv(apca_csv, index=False)
        except Exception:
            pass

        return {
            'time_idx': time_idx,
            'time_label': time_label,
            'apca_results': apca_results,
            'ice_area': ice_area,
            'mean_offset': mean_offset,
            'std_offset': std_offset,
            'outline_gdf': outline_gdf,
            'ice_thickness': ice_thickness,
            'extent': extent,
        }

    # ------------------------------------------------------------------ #
    # Time series aggregation
    # ------------------------------------------------------------------ #
    def _aggregate_time_series(self, id_field=None):
        """Aggregate per-time-step results into a single time-series DataFrame."""
        time_series_list = []

        for df in self.results:
            if id_field and id_field in df.columns:
                grouped = df.groupby(['time_idx', 'ice_area_km2']).agg({
                    'mean_offset_m': ['mean', 'std', 'median'],
                    'std_offset_m': ['mean', 'std'],
                    'num_samples': 'sum',
                }).reset_index()
                grouped.columns = [
                    'time_idx', 'ice_area_km2',
                    'mean_offset_mean', 'mean_offset_std', 'mean_offset_median',
                    'std_offset_mean', 'std_offset_std', 'total_samples',
                ]
            else:
                summary = {
                    'time_idx': df['time_idx'].iloc[0],
                    'ice_area_km2': df['ice_area_km2'].iloc[0],
                    'mean_offset_mean': df['mean_offset_m'].mean(),
                    'mean_offset_std': df['mean_offset_m'].std(),
                    'mean_offset_median': df['mean_offset_m'].median(),
                    'std_offset_mean': df['std_offset_m'].mean(),
                    'std_offset_std': df['std_offset_m'].std(),
                    'total_samples': df['num_samples'].sum(),
                    'num_features': len(df),
                }
                grouped = pd.DataFrame([summary])

            time_series_list.append(grouped)

        self.time_series = pd.concat(time_series_list, ignore_index=True)
        self.time_series = self.time_series.sort_values('time_idx')

        if len(self.time_series) > 1 and 'mean_offset_mean' in self.time_series.columns:
            x = np.arange(len(self.time_series))
            slope, _, r_value, p_value, _ = stats.linregress(
                x, self.time_series['mean_offset_mean']
            )
            self.trend_slope = slope
            self.trend_pvalue = p_value
            self.trend_r2 = r_value ** 2

            time_span = (self.time_series['time_idx'].max()
                         - self.time_series['time_idx'].min())
            self.rate_of_change = (slope / time_span * len(self.time_series)
                                   if time_span > 0 else 0)

    # ------------------------------------------------------------------ #
    # Plotting helpers
    # ------------------------------------------------------------------ #
    def _save_time_step_plot(self, time_idx, outline_gdf, ice_thickness,
                             extent, mean_offset):
        """Save a per-time-step plot with auto extent."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if ice_thickness is not None:
            thickness_masked = np.ma.masked_where(
                ice_thickness < self.min_ice_thickness, ice_thickness
            )
            im = ax.imshow(thickness_masked, extent=extent, origin='lower',
                           cmap='Blues', alpha=0.8, aspect='auto')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Ice Thickness (m)', fontsize=10)

        if outline_gdf is not None:
            outline_gdf.boundary.plot(
                ax=ax, edgecolor='blue', linewidth=2,
                label=f'Modeled Ice (Area: '
                      f'{outline_gdf.area.iloc[0] / 1e6:.1f} km²)',
            )

        self.target_gdf.plot(ax=ax, edgecolor='red', facecolor='none',
                             linewidth=1.5, linestyle='--',
                             label='Field-Mapped Outline')

        ax.set_xlabel('Easting (m)', fontsize=10)
        ax.set_ylabel('Northing (m)', fontsize=10)
        window_text = (f"Window={self.window_size}"
                       if self.use_time_window else "Single step")
        ax.set_title(f'Time Step {time_idx} | Mean Offset: '
                     f'{mean_offset:.2f}m | {window_text}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plot_filename = (self.output_subdirs['time_step_plots']
                         / f"{self.prefix}_time_step_{time_idx:04d}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

        return plot_filename

    def _save_time_step_plot_with_fixed_extent(self, time_idx, outline_gdf,
                                                ice_thickness, extent,
                                                mean_offset):
        """Save a per-time-step plot using the precomputed dataset extent."""
        fig, ax = plt.subplots(figsize=(10, 8))

        plot_extent = self.global_extent if self.global_extent is not None else extent

        if ice_thickness is not None:
            thickness_masked = np.ma.masked_where(
                ice_thickness < self.min_ice_thickness, ice_thickness
            )
            im = ax.imshow(thickness_masked, extent=extent, origin='lower',
                           cmap='Blues', alpha=0.8, aspect='auto')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Ice Thickness (m)', fontsize=10)

        if outline_gdf is not None and not outline_gdf.empty:
            outline_gdf.boundary.plot(
                ax=ax, edgecolor='blue', linewidth=2,
                label=f'Modeled Ice (Area: '
                      f'{outline_gdf.area.iloc[0] / 1e6:.1f} km²)',
            )

        if self.target_gdf is not None:
            self.target_gdf.plot(ax=ax, edgecolor='red', facecolor='none',
                                 linewidth=1.5, linestyle='--',
                                 label='Field-Mapped Outline')

        ax.set_xlim(plot_extent[0], plot_extent[1])
        ax.set_ylim(plot_extent[2], plot_extent[3])

        ax.set_xlabel('Easting (m)', fontsize=10)
        ax.set_ylabel('Northing (m)', fontsize=10)
        window_text = (f"Window={self.window_size}"
                       if self.use_time_window else "Single step")
        ax.set_title(f'Time Step {time_idx} | Mean Offset: '
                     f'{mean_offset:.2f}m | {window_text}', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

        plot_filename = (self.output_subdirs['time_step_plots']
                         / f"{self.prefix}_time_step_{time_idx:04d}.png")
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

        return plot_filename

    def generate_all_plots(self):
        """
        Generate per-time-step plots sequentially (safe for matplotlib).

        Note: only `apca_results` DataFrames are retained in `self.results`
        after a parallel run, so `ice_thickness`/`extent` are set to None here;
        plots rely on the saved outline shapefiles and the field target layer.
        """
        if not getattr(self, 'results', None):
            self._log("No results available. Run analyze_time_series() first.")
            return

        self._log("\n" + "=" * 80)
        self._log("GENERATING TIME STEP PLOTS (SEQUENTIAL)")
        self._log("=" * 80)

        total = len(self.results)
        for i, result in enumerate(self.results):
            if self.is_cancelled():
                self._log("Plot generation cancelled")
                return

            time_idx = (result['time_idx'].iloc[0]
                        if hasattr(result['time_idx'], 'iloc')
                        else result['time_idx'])
            time_label = f"t{time_idx:04d}"

            outline_shp = (self.output_subdirs['ice_outlines']
                           / f"{self.prefix}_ice_outline_{time_label}.shp")
            if outline_shp.exists():
                outline_gdf = gpd.read_file(outline_shp)
            else:
                outline_gdf = self._get_outline_for_time(time_idx)
                if outline_gdf is None:
                    self._log(f"  Warning: No outline found for "
                              f"time step {time_idx}")
                    continue

            ice_thickness = None
            extent = None
            mean_offset = (result['mean_offset_m'].mean()
                           if 'mean_offset_m' in result else 0)

            if self.use_dataset_extent and self.dataset_extent is not None:
                plot_file = self._save_time_step_plot_with_fixed_extent(
                    time_idx, outline_gdf, ice_thickness, extent, mean_offset
                )
            else:
                plot_file = self._save_time_step_plot(
                    time_idx, outline_gdf, ice_thickness, extent, mean_offset
                )

            if self.progress_callback:
                percent = 60 + int((i + 1) / total * 40)
                self.progress_callback(percent,
                                       f"Generating plot {i + 1}/{total}")

            self._log(f"  Generated plot {i + 1}/{total}: {plot_file.name}")

        self._log("=" * 80)
        self._log("All plots generated successfully!")
        self._log("=" * 80)

    def _get_outline_for_time(self, time_idx):
        """Retrieve an outline for a specific time step from saved files."""
        time_label = f"t{time_idx:04d}"
        outline_shp = (self.output_subdirs['ice_outlines']
                       / f"{self.prefix}_ice_outline_{time_label}.shp")
        return gpd.read_file(outline_shp) if outline_shp.exists() else None

    # ------------------------------------------------------------------ #
    # Best-match map
    # ------------------------------------------------------------------ #
    def generate_best_match_map(self):
        """Generate a detailed ice-cover map for the best-match time."""
        if self.best_match_outline is None:
            self._log("No best match found. Run analyze_time_series() first.")
            return

        self._log("\n" + "=" * 80)
        self._log("GENERATING BEST MATCH ICE COVER MAP")
        self._log("=" * 80)

        fig = plt.figure(figsize=(16, 12))
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)

        if self.best_match_ice_data and \
                self.best_match_ice_data['thickness'] is not None:
            extent = self.best_match_ice_data['extent']
            thickness = self.best_match_ice_data['thickness']
            thickness_masked = np.ma.masked_where(
                thickness < self.min_ice_thickness, thickness
            )
            im = ax1.imshow(thickness_masked, extent=extent, origin='lower',
                            cmap='Blues', alpha=0.8, aspect='auto')
            cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Ice Thickness (m)', fontsize=12)

        if self.best_match_outline is not None:
            self.best_match_outline.boundary.plot(
                ax=ax1, edgecolor='blue', linewidth=3,
                label='Modeled Ice Margin',
            )

        self.target_gdf.plot(ax=ax1, edgecolor='red', facecolor='none',
                             linewidth=2, linestyle='--',
                             label='Field-Mapped Outline')

        ax1.set_xlabel('Easting (m)', fontsize=12)
        ax1.set_ylabel('Northing (m)', fontsize=12)
        ax1.set_title(f'Ice Cover at Best Match Time '
                      f'(Step {self.best_match_time})',
                      fontsize=14, fontweight='bold')

        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=3,
                   label='Modeled Ice Margin'),
            Line2D([0], [0], color='red', linewidth=2, linestyle='--',
                   label='Field-Mapped Outline'),
            Patch(facecolor='lightblue', alpha=0.5, label='Ice Thickness'),
        ]
        ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot2grid((3, 3), (0, 2))
        if self.time_series is not None:
            ax2.plot(self.time_series['time_idx'],
                     self.time_series['ice_area_km2'],
                     's-', linewidth=2, markersize=6, color='green',
                     label='Ice Area')
            ax2.axvline(x=self.best_match_time, color='red', linestyle='--',
                        linewidth=2, label='Best Match')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Ice Area (km²)')
            ax2.set_title('Ice Area Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot2grid((3, 3), (1, 2))
        if self.time_series is not None and \
                'mean_offset_mean' in self.time_series.columns:
            ax3.plot(self.time_series['time_idx'],
                     self.time_series['mean_offset_mean'],
                     'o-', linewidth=2, markersize=6, color='darkblue',
                     label='Mean Offset')
            ax3.axvline(x=self.best_match_time, color='red', linestyle='--',
                        linewidth=2, label='Best Match')
            ax3.fill_between(
                self.time_series['time_idx'],
                self.time_series['mean_offset_mean']
                - self.time_series['mean_offset_std'],
                self.time_series['mean_offset_mean']
                + self.time_series['mean_offset_std'],
                alpha=0.3, color='orange',
            )
            ax3.set_xlabel('Time Step')
            ax3.set_ylabel('Mean Offset (m)')
            ax3.set_title('Mean Offset Change Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot2grid((3, 3), (2, 2))
        if self.time_series is not None and \
                'std_offset_mean' in self.time_series.columns:
            ax4.plot(self.time_series['time_idx'],
                     self.time_series['std_offset_mean'],
                     'o-', linewidth=2, markersize=6, color='darkblue',
                     label='Standard Deviation')
            ax4.axvline(x=self.best_match_time, color='red', linestyle='--',
                        linewidth=2, label='Best Match')
            ax4.set_xlabel('Time Step')
            ax4.set_ylabel('Standard Deviation (m)')
            ax4.set_title('Standard Deviation Change Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        output_image = (self.output_subdirs['best_match']
                        / f"{self.prefix}_best_match_ice_map.pdf")
        plt.savefig(output_image, dpi=300, bbox_inches='tight')
        self._log(f"Best match ice map saved to: {output_image}")
        plt.close(fig)
        gc.collect()

        return output_image

    # ------------------------------------------------------------------ #
    # Summary plots
    # ------------------------------------------------------------------ #
    def plot_time_series(self):
        """Create a 2x2 summary plot of the APCA time series."""
        if self.time_series is None:
            self._log("No time series data to plot. "
                      "Run analyze_time_series() first.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax1 = axes[0, 0]
        ax1.fill_between(self.time_series['time_idx'], 0,
                         self.time_series['ice_area_km2'],
                         alpha=0.3, color='lightblue')
        ax1.plot(self.time_series['time_idx'],
                 self.time_series['ice_area_km2'],
                 'o-', linewidth=2, markersize=6, color='steelblue')
        if self.best_match_time is not None:
            ax1.axvline(x=self.best_match_time, color='red', linestyle='--',
                        linewidth=2, alpha=0.7, label='Best Match')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Ice Area (km²)')
        ax1.set_title('Modeled Ice Area Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        if 'mean_offset_mean' in self.time_series.columns:
            ax2.errorbar(self.time_series['time_idx'],
                         self.time_series['mean_offset_mean'],
                         yerr=self.time_series['mean_offset_std'],
                         fmt='o-', capsize=5, capthick=2, elinewidth=2,
                         markersize=6, linewidth=2, color='darkblue',
                         ecolor='lightblue')

            if self.best_match_time is not None:
                ax2.axvline(x=self.best_match_time, color='red',
                            linestyle='--', linewidth=2, alpha=0.7,
                            label=f'Best Match: '
                                  f'{self.best_match_offset:.1f}m')
                ax2.plot(self.best_match_time, self.best_match_offset, 'r*',
                         markersize=15, label='Minimum Offset')

            if len(self.time_series) > 1:
                x = np.arange(len(self.time_series))
                z = np.polyfit(x, self.time_series['mean_offset_mean'], 1)
                p = np.poly1d(z)
                ax2.plot(self.time_series['time_idx'], p(x), 'g--',
                         linewidth=2, alpha=0.7,
                         label=f'Trend: {self.rate_of_change:.2f} m/step')
                ax2.legend()

        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Mean Offset (m)')
        ax2.set_title('Mean Distance to Ice Margin')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if 'std_offset_mean' in self.time_series.columns:
            ax3.fill_between(
                self.time_series['time_idx'],
                self.time_series['std_offset_mean']
                - self.time_series['std_offset_std'],
                self.time_series['std_offset_mean']
                + self.time_series['std_offset_std'],
                alpha=0.3, color='orange',
            )
            ax3.plot(self.time_series['time_idx'],
                     self.time_series['std_offset_mean'],
                     's-', linewidth=2, markersize=6, color='darkorange')
            if self.best_match_time is not None:
                ax3.axvline(x=self.best_match_time, color='red',
                            linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Standard Deviation (m)')
        ax3.set_title('Offset Variance')
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        if 'mean_offset_mean' in self.time_series.columns:
            scatter = ax4.scatter(
                self.time_series['ice_area_km2'],
                self.time_series['mean_offset_mean'],
                c=self.time_series['time_idx'], cmap='viridis',
                s=80, alpha=0.7, edgecolor='black', linewidth=1,
            )
            if self.best_match_time is not None:
                best_row = self.time_series[
                    self.time_series['time_idx'] == self.best_match_time
                ]
                if len(best_row) > 0:
                    ax4.scatter(best_row['ice_area_km2'],
                                best_row['mean_offset_mean'],
                                c='red', s=200, marker='*',
                                label='Best Match', zorder=5)
            ax4.set_xlabel('Ice Area (km²)')
            ax4.set_ylabel('Mean Offset (m)')
            ax4.set_title('Offset vs Ice Area')
            plt.colorbar(scatter, ax=ax4, label='Time Step')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            if len(self.time_series) > 2:
                corr = (self.time_series['ice_area_km2']
                        .corr(self.time_series['mean_offset_mean']))
                ax4.annotate(
                    f'Correlation: {corr:.3f}', xy=(0.05, 0.95),
                    xycoords='axes fraction', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.8),
                )

        window_text = (f"Time Window: {self.window_size} steps"
                       if self.use_time_window else "Single step")
        plt.suptitle(
            f'Time Series APCA Analysis\n'
            f'Best Match at Time Step {self.best_match_time} '
            f'(Offset: {self.best_match_offset:.2f}m) | {window_text}',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()

        output_image = (self.output_subdirs['plots']
                        / f"{self.prefix}_apca_time_series_results.png")
        plt.savefig(output_image, dpi=150, bbox_inches='tight')
        self._log(f"\nTime series plot saved to: {output_image}")
        plt.close(fig)
        gc.collect()

    # ------------------------------------------------------------------ #
    # Summary / export
    # ------------------------------------------------------------------ #
    def print_summary(self):
        """Print a comprehensive summary of the time-series analysis."""
        if self.time_series is None:
            self._log("No results to summarize. "
                      "Run analyze_time_series() first.")
            return

        self._log("\n" + "=" * 80)
        self._log("TIME SERIES APCA ANALYSIS SUMMARY")
        self._log("=" * 80)
        self._log(f"Analysis period: time steps "
                  f"{self.time_series['time_idx'].min()} to "
                  f"{self.time_series['time_idx'].max()}")
        self._log(f"Number of time steps: {len(self.time_series)}")
        self._log(f"Target feature type: {self.target_type}")
        self._log(f"Cell size: {self.cell_size} m")
        if self.use_time_window:
            self._log(f"Moving time window: size={self.window_size}, "
                      f"accumulative={self.use_accumulative_ice}")

        self._log("\n" + "-" * 40)
        self._log("BEST MATCH RESULTS")
        self._log("-" * 40)
        self._log(f"Best match time step: {self.best_match_time}")
        self._log(f"Mean offset: {self.best_match_offset:.2f} m")
        if self.best_match_apca_results is not None:
            self._log(f"Standard deviation: "
                      f"{self.best_match_apca_results['std_offset_m'].mean():.2f} m")
            self._log(f"Median offset: "
                      f"{self.best_match_apca_results['mean_offset_m'].median():.2f} m")
            self._log(f"Offset range: "
                      f"{self.best_match_apca_results['mean_offset_m'].min():.2f} - "
                      f"{self.best_match_apca_results['mean_offset_m'].max():.2f} m")

        if 'mean_offset_mean' in self.time_series.columns:
            self._log("\n" + "-" * 40)
            self._log("OFFSET STATISTICS")
            self._log("-" * 40)
            mo = self.time_series['mean_offset_mean']
            self._log(f"  Initial mean offset: {mo.iloc[0]:.2f} m")
            self._log(f"  Final mean offset: {mo.iloc[-1]:.2f} m")
            self._log(f"  Change: {mo.iloc[-1] - mo.iloc[0]:.2f} m")
            self._log(f"  Mean offset range: {mo.min():.2f} - {mo.max():.2f} m")

        self._log("\n" + "-" * 40)
        self._log("ICE AREA STATISTICS")
        self._log("-" * 40)
        ia = self.time_series['ice_area_km2']
        self._log(f"  Initial ice area: {ia.iloc[0]:.2f} km²")
        self._log(f"  Final ice area: {ia.iloc[-1]:.2f} km²")
        self._log(f"  Area change: {ia.iloc[-1] - ia.iloc[0]:.2f} km²")

        if 'mean_offset_mean' in self.time_series.columns:
            self._log("\n" + "-" * 40)
            self._log("TREND ANALYSIS")
            self._log("-" * 40)
            self._log(f"  Rate of change: {self.rate_of_change:.3f} m per time step")
            self._log(f"  Trend significance: p = {self.trend_pvalue:.4f}")
            self._log(f"  R² value: {self.trend_r2:.3f}")
            if self.trend_pvalue < 0.05:
                if self.rate_of_change > 0:
                    self._log("  → Model is increasingly overestimating "
                              "ice extent (positive trend)")
                elif self.rate_of_change < 0:
                    self._log("  → Model is increasingly underestimating "
                              "ice extent (negative trend)")
                else:
                    self._log("  → No significant trend detected")
            else:
                self._log("  → No statistically significant trend")

        self._log("=" * 80)

    def export_results(self):
        """Export all time-series results to CSV and a text summary."""
        if self.time_series is None:
            self._log("No results to export. "
                      "Run analyze_time_series() first.")
            return

        output_csv = (self.output_subdirs['csv']
                      / f"{self.prefix}_apca_time_series_results.csv")
        self.time_series.to_csv(output_csv, index=False)
        self._log(f"Time series results saved to: {output_csv}")

        if self.best_match_apca_results is not None:
            best_csv = (self.output_subdirs['best_match']
                        / f"{self.prefix}_best_match_apca_results.csv")
            self.best_match_apca_results.to_csv(best_csv, index=False)
            self._log(f"Best match APCA results saved to: {best_csv}")

        summary_file = (self.output_subdirs['csv']
                        / f"{self.prefix}_analysis_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("APCA ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Analysis date: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cell size: {self.cell_size} m\n")
            f.write(f"Min ice thickness: {self.min_ice_thickness} m\n")
            if self.use_time_window:
                f.write(f"Moving time window: size={self.window_size}, "
                        f"accumulative={self.use_accumulative_ice}\n\n")
            f.write(f"Best match time step: {self.best_match_time}\n")
            f.write(f"Best match offset: {self.best_match_offset:.2f} m\n")
            if self.best_match_apca_results is not None:
                f.write(f"Std deviation: "
                        f"{self.best_match_apca_results['std_offset_m'].mean():.2f} m\n")
            f.write(f"Initial offset: "
                    f"{self.time_series['mean_offset_mean'].iloc[0]:.2f} m\n")
            f.write(f"Final offset: "
                    f"{self.time_series['mean_offset_mean'].iloc[-1]:.2f} m\n")
            f.write(f"Trend rate: {self.rate_of_change:.3f} m/step\n")
            f.write(f"Trend p-value: {self.trend_pvalue:.4f}\n")
            f.write(f"R²: {self.trend_r2:.3f}\n")

        self._log(f"Summary saved to: {summary_file}")
