"""
/*********************************************************************************
* Automated Polygon Overlap Analysis (APOA):                                     * 
* =========================================                                      *
* Calculates precision, recall, F1-score, and IoU based on ice outline overlap   *
* using vector polygons                                                          *
*                                                                                *
* This analysis:                                                                 *
* - Converts model ice masks to polygons                                         *
* - Uses shapely overlay operations (intersection, difference)                   *
* - Provides area-based metrics directly without rasterization                   *
* - Much more accurate and efficient                                             *
* - Supports parallel processing with ThreadPoolExecutor                         *
*                                                                                *
* This file is part of GLIMAT Plugin.                                            *
*                                                                                *
* Author: Yingkui Li                                                             *
* Department of Geography & Sustainability                                       *
* University of Tennessee, Knoxville, TN 37996                                   * 
* Email: yli32@utk.edu                                                           *
* Start date: 06/05/2026                                                         *
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

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import MultiPolygon, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

mpl.use('Agg')  # Non-interactive backend
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def mask_to_polygons(mask, transform, min_area_km2=0.01):
    """
    Convert a binary mask to polygon geometries.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask (True/False).
    transform : affine.Affine
        Geotransform for the raster.
    min_area_km2 : float
        Minimum area (km²) to keep a polygon (filters noise).

    Returns
    -------
    list of shapely.geometry.Polygon
    """
    mask_int = mask.astype(np.uint8)
    results = shapes(mask_int, mask=mask_int == 1, transform=transform)
    min_area_m2 = min_area_km2 * 1e6
    polygons = []
    for geom, value in results:
        if value == 1:
            poly = shape(geom)
            if poly.area >= min_area_m2:
                polygons.append(poly)
    return polygons

# --------------------------------------------------------------------------- #
# Main analyzer
# --------------------------------------------------------------------------- #
class PolygonOverlapAnalyzer:
    """Polygon-based Precision/Recall/F1-score analysis for ice-outline overlap."""

    def __init__(self, pism_dataset, field_gdf, output_subdirs, prefix='',
                 time_indices=None, min_ice_thickness=1.0,
                 ice_variable='thk', use_time_window=False, window_size=5,
                 use_accumulative_ice=False, id_field=None,
                 min_polygon_area_km2=0.01,
                 progress_callback=None, cancel_checker=None,
                 save_rasters=False, save_plots=True):
        self.pism_dataset = pism_dataset
        self.field_gdf = field_gdf
        self.output_subdirs = output_subdirs
        self.min_ice_thickness = min_ice_thickness
        self.ice_variable = ice_variable
        self.use_time_window = use_time_window
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.use_accumulative_ice = use_accumulative_ice
        self.id_field = id_field
        self.min_polygon_area_km2 = min_polygon_area_km2
        self.save_rasters = save_rasters
        self.save_plots = save_plots
        self.progress_callback = progress_callback
        self.cancel_checker = cancel_checker
        self._is_cancelled = False
        self.prefix = prefix
        self.console = None

        # ---- Grid / transform -------------------------------------------- #
        if 'x' in pism_dataset.dims and 'y' in pism_dataset.dims:
            self.x = pism_dataset.x.values
            self.y = pism_dataset.y.values
            self.netcdf_bounds = [
                float(self.x.min()), float(self.y.min()),
                float(self.x.max()), float(self.y.max()),
            ]
            self.transform = from_origin(
                west=self.x.min(),
                north=self.y.max(),
                xsize=abs(self.x[1] - self.x[0]),
                ysize=abs(self.y[1] - self.y[0]),
            )
            self.model_shape = (len(self.y), len(self.x))
        else:
            first_var = list(pism_dataset.data_vars.values())[0]
            self.model_shape = first_var.shape[-2:]
            self.netcdf_bounds = [0, 0, self.model_shape[1], self.model_shape[0]]
            self.transform = from_origin(0, self.model_shape[0], 1, 1)

        # ---- CRS & field polygons ---------------------------------------- #
        self.crs = field_gdf.crs if field_gdf.crs is not None else None

        if self.id_field is not None and self.id_field in field_gdf.columns:
            self._log(f"\nGrouping field polygons by ID field: {self.id_field}")
            self.field_polygons_by_id = {}
            for uid in field_gdf[self.id_field].unique():
                gdf_id = field_gdf[field_gdf[self.id_field] == uid]
                union_poly = unary_union(gdf_id.geometry)
                self.field_polygons_by_id[uid] = union_poly
                self._log(f"  ID {uid}: Area = {union_poly.area / 1e6:.2f} km²")
            self.unique_ids = list(self.field_polygons_by_id.keys())
        else:
            union_poly = unary_union(field_gdf.geometry)
            self.field_polygons_by_id = {None: union_poly}
            self.unique_ids = [None]
            self._log(f"\nField polygon area: {union_poly.area / 1e6:.2f} km²")

        # ---- Time indices ------------------------------------------------- #
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
            self.time_indices = [0]

        # ---- Results placeholders ---------------------------------------- #
        self.results = []
        self.time_series = None
        self.best_matches = {}
        self.best_match_time = None
        self.best_match_f1 = None

        self._log("\nPolygon Overlap Analyzer initialized")
        self._log(f"  Time steps to analyze: {len(self.time_indices)}")
        self._log(f"  Number of field polygons: {len(self.unique_ids)}")
        self._log(f"  Min polygon area: {self.min_polygon_area_km2} km²")

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

    def is_cancelled(self):
        """Check whether the analysis has been cancelled."""
        if self.cancel_checker:
            return self.cancel_checker()
        return self._is_cancelled

    # ------------------------------------------------------------------ #
    # Model-polygon extraction
    # ------------------------------------------------------------------ #
    def get_model_polygons(self, time_idx):
        """Extract model ice polygons for a single time step."""
        ds_slice = (self.pism_dataset.isel(time=time_idx)
                    if 'time' in self.pism_dataset.dims
                    else self.pism_dataset)

        ice_thickness = None
        if self.ice_variable in ds_slice:
            ice_thickness = ds_slice[self.ice_variable].values
        else:
            for var in ('thk', 'land_ice_thickness'):
                if var in ds_slice:
                    ice_thickness = ds_slice[var].values
                    break

        if ice_thickness is None:
            return []

        ice_mask = ice_thickness >= self.min_ice_thickness
        ice_mask = np.flipud(ice_mask)  # Flip for GIS compatibility
        return mask_to_polygons(ice_mask, self.transform, self.min_polygon_area_km2)

    def get_model_polygons_window(self, time_idx):
        """Extract model polygons using a moving time window."""
        half_window = self.window_size // 2
        start_idx = max(0, time_idx - half_window)
        end_idx = min(len(self.pism_dataset.time), time_idx + half_window + 1)

        all_polygons = []
        for t_idx in range(start_idx, end_idx):
            all_polygons.extend(self.get_model_polygons(t_idx))

        if not all_polygons:
            return []

        if self.use_accumulative_ice:
            union_poly = unary_union(all_polygons)
            if union_poly.is_empty:
                return []
            if isinstance(union_poly, MultiPolygon):
                return list(union_poly.geoms)
            return [union_poly]

        return all_polygons

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def calculate_overlap_metrics(self, model_polygons, field_polygon):
        """
        Calculate precision, recall, F1-score, and IoU using polygon overlay.
        Only model polygons that intersect the field polygon are considered.
        """
        if not model_polygons:
            field_area = field_polygon.area / 1e6
            return {
                'tp_area_km2': 0, 'fp_area_km2': 0, 'fn_area_km2': field_area,
                'precision': 0, 'recall': 0, 'f1_score': 0, 'iou': 0,
                'num_polygons': 0, 'num_intersecting_polygons': 0,
                'model_area_km2': 0, 'intersecting_model_area_km2': 0,
                'field_area_km2': field_area,
            }, []

        # Spatial index for fast intersection queries
        tree = STRtree(model_polygons)
        intersecting_indices = tree.query(field_polygon, predicate='intersects')
        intersecting_polygons = [model_polygons[i] for i in intersecting_indices]

        total_model_area = sum(p.area for p in model_polygons) / 1e6
        intersecting_area = (sum(p.area for p in intersecting_polygons) / 1e6
                             if intersecting_polygons else 0)
        field_area = field_polygon.area / 1e6

        if not intersecting_polygons:
            return {
                'tp_area_km2': 0, 'fp_area_km2': 0, 'fn_area_km2': field_area,
                'precision': 0, 'recall': 0, 'f1_score': 0, 'iou': 0,
                'num_polygons': len(model_polygons),
                'num_intersecting_polygons': 0,
                'model_area_km2': total_model_area,
                'intersecting_model_area_km2': 0,
                'field_area_km2': field_area,
            }, []

        model_union = unary_union(intersecting_polygons)

        tp_area = model_union.intersection(field_polygon).area / 1e6
        fp_area = model_union.difference(field_polygon).area / 1e6
        fn_area = field_polygon.difference(model_union).area / 1e6

        precision = tp_area / (tp_area + fp_area) if (tp_area + fp_area) > 0 else 0
        recall = tp_area / (tp_area + fn_area) if (tp_area + fn_area) > 0 else 0
        f1_score = (2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0 else 0)
        iou = (tp_area / (tp_area + fp_area + fn_area)
               if (tp_area + fp_area + fn_area) > 0 else 0)

        result = {
            'tp_area_km2': tp_area,
            'fp_area_km2': fp_area,
            'fn_area_km2': fn_area,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'iou': iou,
            'num_polygons': len(model_polygons),
            'num_intersecting_polygons': len(intersecting_polygons),
            'model_area_km2': total_model_area,
            'intersecting_model_area_km2': intersecting_area,
            'field_area_km2': field_area,
        }
        return result, intersecting_polygons

    # ------------------------------------------------------------------ #
    # Single (time, id) task
    # ------------------------------------------------------------------ #
    def analyze_time_step_for_id(self, time_idx, uid, save_plots, save_shapefiles):
        """Analyze a single time step for one field polygon ID."""
        if self.is_cancelled():
            return None

        field_polygon = self.field_polygons_by_id[uid]

        if self.use_time_window and self.window_size > 1:
            model_polygons = self.get_model_polygons_window(time_idx)
        else:
            model_polygons = self.get_model_polygons(time_idx)

        metrics, intersecting_polygons = self.calculate_overlap_metrics(
            model_polygons, field_polygon
        )

        result = {
            'time_idx': time_idx,
            'time_label': f"t{time_idx:04d}",
            'id': uid,
            'tp_area_km2': metrics['tp_area_km2'],
            'fp_area_km2': metrics['fp_area_km2'],
            'fn_area_km2': metrics['fn_area_km2'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'iou': metrics['iou'],
            'model_area_km2': metrics['model_area_km2'],
            'intersecting_model_area_km2': metrics['intersecting_model_area_km2'],
            'field_area_km2': metrics['field_area_km2'],
            'num_model_polygons': metrics['num_polygons'],
            'num_intersecting_polygons': metrics['num_intersecting_polygons'],
        }

        if save_plots:
            self._save_time_step_plot(time_idx, uid, field_polygon,
                                      metrics, intersecting_polygons)
        if save_shapefiles:
            self._save_model_shapefile(time_idx, uid, intersecting_polygons)

        return result

    # ------------------------------------------------------------------ #
    # Sequential process
    # ------------------------------------------------------------------ #
    def analyze_time_series_sequential(self, save_plots=True, save_shapefiles=False):
        """Analyze all time steps sequentially."""
        self._log("\n" + "=" * 80)
        self._log("POLYGON-BASED OVERLAP ANALYSIS (SEQUENTIAL)")
        self._log("=" * 80)
        self._log(f"Number of time steps: {len(self.time_indices)}")
        self._log(f"Number of field polygons: {len(self.unique_ids)}")
        self._log(f"Moving time window: {self.use_time_window}")
        if self.use_time_window:
            self._log(f"  Window size: {self.window_size}, "
                      f"Accumulative: {self.use_accumulative_ice}")
        self._log("=" * 80)

        all_results = []
        total_tasks = len(self.time_indices) * len(self.unique_ids)
        completed = 0

        best_match_per_id = {
            uid: {'f1_score': -1, 'time_idx': None, 'result': None}
            for uid in self.unique_ids
        }

        for time_idx in self.time_indices:
            if self.is_cancelled():
                break
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

                result = self.analyze_time_step_for_id(
                    time_idx, uid, save_plots, save_shapefiles
                )
                if not result:
                    continue

                all_results.append(result)
                self._log(
                    f"  Time {time_idx}, ID {uid}: F1={result['f1_score']:.4f}, "
                    f"Precision={result['precision']:.4f}, "
                    f"Recall={result['recall']:.4f}"
                )

                if result['f1_score'] > best_match_per_id[uid]['f1_score']:
                    best_match_per_id[uid].update({
                        'f1_score': result['f1_score'],
                        'time_idx': time_idx,
                        'result': result,
                    })
                    self._log(f"    *** New best match for ID {uid} at time "
                              f"{time_idx}! F1={result['f1_score']:.4f} ***")

        self.results = all_results
        self.best_matches = best_match_per_id
        self.time_series = pd.DataFrame(all_results)
        return all_results, best_match_per_id

    # ------------------------------------------------------------------ #
    # Parallel process
    # ------------------------------------------------------------------ #
    def analyze_time_series_parallel(self, save_plots=True,
                                     save_shapefiles=False, max_workers=None):
        """Analyze all time steps using ThreadPoolExecutor."""
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()

        total_tasks = len(self.time_indices) * len(self.unique_ids)

        self._log("\n" + "=" * 80)
        self._log("POLYGON-BASED OVERLAP ANALYSIS (PARALLEL - THREADS)")
        self._log("=" * 80)
        self._log(f"Number of time steps: {len(self.time_indices)}")
        self._log(f"Number of field polygons: {len(self.unique_ids)}")
        self._log(f"Total tasks: {total_tasks}")
        self._log(f"Parallel workers: {max_workers}")
        self._log(f"Moving time window: {self.use_time_window}")
        if self.use_time_window:
            self._log(f"  Window size: {self.window_size}, "
                      f"Accumulative: {self.use_accumulative_ice}")
        self._log("=" * 80)

        all_results = []
        completed = 0
        lock = threading.Lock()

        best_match_per_id = {
            uid: {'f1_score': -1, 'time_idx': None, 'result': None}
            for uid in self.unique_ids
        }

        def process_task(t_idx, uid_):
            if self.is_cancelled():
                return None
            return self.analyze_time_step_for_id(
                t_idx, uid_, save_plots, save_shapefiles
            )

        tasks = [(t, uid) for t in self.time_indices for uid in self.unique_ids]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(process_task, t, uid): (t, uid)
                for t, uid in tasks
            }
            for future in as_completed(future_to_task):
                time_idx, uid = future_to_task[future]
                try:
                    result = future.result(timeout=300)
                except Exception as exc:
                    self._log(f"  Error processing time {time_idx}, "
                              f"ID {uid}: {exc}")
                    continue

                if not result:
                    continue

                with lock:
                    all_results.append(result)
                    completed += 1

                    if self.progress_callback:
                        percent = 30 + int(completed / total_tasks * 60)
                        self.progress_callback(
                            percent, f"Processed {completed}/{total_tasks}"
                        )

                    self._log(
                        f"  Completed time {time_idx}, ID {uid}: "
                        f"F1={result['f1_score']:.4f}, IoU={result['iou']:.4f}"
                    )

                    if result['f1_score'] > best_match_per_id[uid]['f1_score']:
                        best_match_per_id[uid].update({
                            'f1_score': result['f1_score'],
                            'time_idx': time_idx,
                            'result': result,
                        })
                        self._log(f"    *** New best match for ID {uid} at "
                                  f"time {time_idx}! "
                                  f"F1={result['f1_score']:.4f} ***")

        all_results.sort(
            key=lambda x: (x['time_idx'], x['id'] if x['id'] is not None else -1)
        )

        self.results = all_results
        self.best_matches = best_match_per_id
        self.time_series = pd.DataFrame(all_results)

        overall_best = max(best_match_per_id.values(), key=lambda x: x['f1_score'])
        self.best_match_time = overall_best['time_idx']
        self.best_match_f1 = overall_best['f1_score']

        self._log("\n" + "=" * 80)
        self._log("BEST MATCHES SUMMARY")
        self._log("=" * 80)
        for uid, best in best_match_per_id.items():
            if best['time_idx'] is not None:
                id_text = f"ID {uid}" if uid is not None else "All IDs"
                self._log(f"  {id_text}: Time {best['time_idx']} "
                          f"with F1-Score = {best['f1_score']:.4f}")

        return all_results, best_match_per_id

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def analyze_time_series(self, save_plots=True, save_shapefiles=False,
                            use_parallel=False, max_workers=None):
        """
        Analyze all time steps for all field polygon IDs.

        Parameters
        ----------
        save_shapefiles : bool
            Save model polygons as shapefiles for each time step.
        use_parallel : bool
            Use ThreadPoolExecutor for parallel processing.
        max_workers : int
            Number of parallel workers.
        """
        start_time = time.time()

        if use_parallel and len(self.time_indices) > 1 and self.unique_ids:
            all_results, best_match_per_id = self.analyze_time_series_parallel(
                save_plots, save_shapefiles, max_workers
            )
        else:
            all_results, best_match_per_id = self.analyze_time_series_sequential(
                save_plots, save_shapefiles
            )

        if not all_results:
            self._log("\nNo results generated!")
            return None

        all_results.sort(
            key=lambda x: (x['time_idx'], x['id'] if x['id'] is not None else -1)
        )

        self.results = all_results
        self.best_matches = best_match_per_id

        overall_best = max(best_match_per_id.values(), key=lambda x: x['f1_score'])
        self.best_match_time = overall_best['time_idx']
        self.best_match_f1 = overall_best['f1_score']

        self.time_series = pd.DataFrame(all_results)
        self._log(f"\nDataFrame columns: {list(self.time_series.columns)}")
        self._log(f"DataFrame shape: {self.time_series.shape}")
        if not self.time_series.empty:
            self._log(f"First row: {self.time_series.iloc[0].to_dict()}")

        self._save_results()
        self._create_plots()
        self.generate_best_match_map()

        elapsed = time.time() - start_time
        if elapsed < 60:
            self._log(f"\nTotal runtime: {elapsed:.2f} seconds")
        elif elapsed < 3600:
            minutes, seconds = divmod(elapsed, 60)
            self._log(f"\nTotal runtime: {int(minutes)} minute(s) and "
                      f"{seconds:.2f} seconds")
        else:
            hours, rem = divmod(elapsed, 3600)
            minutes, seconds = divmod(rem, 60)
            self._log(f"\nTotal runtime: {int(hours)} hour(s), "
                      f"{int(minutes)} minute(s), and {seconds:.2f} seconds")

        self._log("\n" + "=" * 80)
        self._log("BEST MATCHES SUMMARY (by F1-Score)")
        self._log("=" * 80)
        for uid, best in best_match_per_id.items():
            if best['time_idx'] is not None:
                id_text = f"ID {uid}" if uid is not None else "All IDs"
                self._log(f"  {id_text}: Time {best['time_idx']} "
                          f"with F1-Score = {best['f1_score']:.4f}")

        return self.time_series

    # ------------------------------------------------------------------ #
    # Plotting / saving helpers
    # ------------------------------------------------------------------ #
    def _save_time_step_plot(self, time_idx, uid, field_polygon,
                             metrics, intersecting_polygons):
        """Save a per-time-step plot, highlighting intersecting polygons."""
        id_suffix = f"_id{uid}" if uid is not None else ""
        plot_path = self.output_subdirs['time_step_plots'] / (
            f"{self.prefix}_overlap_polygon_t{time_idx:04d}{id_suffix}.png"
        )

        if not intersecting_polygons:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            if isinstance(field_polygon, MultiPolygon):
                for poly in field_polygon.geoms:
                    x, y = poly.exterior.xy
                    ax.fill(x, y, alpha=0.5, fc='red', ec='red')
            else:
                x, y = field_polygon.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='red', ec='red')

            ax.set_title(f'Time {time_idx}, ID {uid}\n'
                         f'No intersecting model ice found')
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            gc.collect()
            self._log(f"    Saved plot (no intersection): {plot_path}")
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        intersecting_area = sum(p.area for p in intersecting_polygons) / 1e6

        # -- Panel 1: field polygon (ground truth) ------------------------- #
        ax1 = axes[0]
        if isinstance(field_polygon, MultiPolygon):
            for poly in field_polygon.geoms:
                x, y = poly.exterior.xy
                ax1.fill(x, y, alpha=0.5, fc='red', ec='red')
        else:
            x, y = field_polygon.exterior.xy
            ax1.fill(x, y, alpha=0.5, fc='red', ec='red')
        ax1.set_title(f'Field Ice (Ground Truth)\n'
                      f'Area: {metrics["field_area_km2"]:.2f} km²')
        ax1.set_aspect('equal')

        # -- Panel 2: intersecting model polygons -------------------------- #
        ax2 = axes[1]
        for poly in intersecting_polygons:
            if poly.is_empty:
                continue
            geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
            for p in geoms:
                x, y = p.exterior.xy
                ax2.fill(x, y, alpha=0.7, fc='blue', ec='blue')
        ax2.set_title(f'Model Ice\n'
                      f'{len(intersecting_polygons)} ({intersecting_area:.2f} km²)')
        ax2.set_aspect('equal')

        # -- Panel 3: overlap (TP in green) -------------------------------- #
        ax3 = axes[2]
        model_union = unary_union(intersecting_polygons)
        intersection = model_union.intersection(field_polygon)
        if not intersection.is_empty:
            geoms = (intersection.geoms if isinstance(intersection, MultiPolygon)
                     else [intersection])
            for poly in geoms:
                x, y = poly.exterior.xy
                ax3.fill(x, y, alpha=0.7, fc='green', ec='green')

        field_geoms = (field_polygon.geoms
                       if isinstance(field_polygon, MultiPolygon)
                       else [field_polygon])
        for poly in field_geoms:
            x, y = poly.exterior.xy
            ax3.plot(x, y, 'r-', linewidth=2)

        ax3.set_title(f'Overlap (TP in Green)\n'
                      f'F1={metrics["f1_score"]:.4f}, '
                      f'IoU={metrics["iou"]:.4f}')
        ax3.set_aspect('equal')

        plt.suptitle(f'Time Step {time_idx}, ID {uid}\n'
                     f'F1: {metrics["f1_score"]:.4f}, '
                     f'Precision: {metrics["precision"]:.4f}, '
                     f'Recall: {metrics["recall"]:.4f}')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        self._log(f"    Saved plot: {plot_path}")

    def _save_model_shapefile(self, time_idx, uid, polygons=None):
        """Save model ice polygons as a shapefile."""
        if not polygons:
            return
        try:
            gdf = gpd.GeoDataFrame(
                {
                    'time_idx': [time_idx] * len(polygons),
                    'area_km2': [p.area / 1e6 for p in polygons],
                },
                geometry=polygons,
                crs=self.crs,
            )
            id_suffix = f"_id{uid}" if uid is not None else ""
            shp_path = self.output_subdirs['shapefiles'] / (
                f"{self.prefix}_model_ice_t{time_idx:04d}{id_suffix}.shp"
            )
            gdf.to_file(shp_path)
            self._log(f"    Saved model ice shapefile: {shp_path}")
        except Exception as exc:
            self._log(f"    Warning: Could not save shapefile for "
                      f"time {time_idx}: {exc}")

    def _save_results(self):
        """Save the time series to CSV and the best-match summary to TXT."""
        csv_file = self.output_subdirs['csv'] / (
            f"{self.prefix}_overlap_time_series.csv"
        )
        self.time_series.to_csv(csv_file, index=False)
        self._log(f"\nTime series saved to: {csv_file}")

        summary_file = self.output_subdirs['csv'] / (
            f"{self.prefix}_best_match_summary.txt"
        )
        with open(summary_file, 'w') as fh:
            fh.write("=" * 80 + "\n")
            fh.write("POLYGON-BASED OVERLAP ANALYSIS - BEST MATCH SUMMARY\n")
            fh.write("=" * 80 + "\n\n")
            for uid, best in self.best_matches.items():
                if best['time_idx'] is None:
                    continue
                id_text = f"ID {uid}" if uid is not None else "All IDs"
                res = best['result']
                fh.write(f"\n{id_text}:\n")
                fh.write(f"  Best match time step: {best['time_idx']}\n")
                fh.write(f"  F1-Score:  {best['f1_score']:.4f}\n")
                fh.write(f"  IoU:       {res['iou']:.4f}\n")
                fh.write(f"  Precision: {res['precision']:.4f}\n")
                fh.write(f"  Recall:    {res['recall']:.4f}\n")
                fh.write(f"  TP Area:   {res['tp_area_km2']:.2f} km²\n")
                fh.write(f"  FP Area:   {res['fp_area_km2']:.2f} km²\n")
                fh.write(f"  FN Area:   {res['fn_area_km2']:.2f} km²\n")
                fh.write(f"  Total Model Ice Area: "
                         f"{res['model_area_km2']:.2f} km²\n")
                if 'intersecting_model_area_km2' in res:
                    fh.write(f"  Model Ice Area: "
                             f"{res['intersecting_model_area_km2']:.2f} km²\n")
                fh.write(f"  Field Ice Area: "
                         f"{res['field_area_km2']:.2f} km²\n")

        self._log(f"Summary saved to: {summary_file}")

    def _create_plots(self):
        """Create summary plots (one per field polygon ID)."""
        if self.time_series is None or self.time_series.empty:
            self._log("No data to create plots.")
            return
        for uid in self.unique_ids:
            self._create_plot_for_id(uid)

    def _create_plot_for_id(self, uid):
        """Create a summary plot for a specific field polygon ID."""
        if uid is not None:
            df_id = self.time_series[self.time_series['id'] == uid]
            id_label = f"ID {uid}"
            id_suffix = f"_id{uid}"
        else:
            df_id = self.time_series
            id_label = "All IDs"
            id_suffix = ""

        if df_id.empty:
            self._log(f"No data to create plot for {id_label}")
            return

        self._log(f"\nCreating plot for {id_label}")
        self._log(f"  Time indices range: "
                  f"{df_id['time_idx'].min()} to {df_id['time_idx'].max()}")
        self._log(f"  F1-score range: "
                  f"{df_id['f1_score'].min():.4f} to "
                  f"{df_id['f1_score'].max():.4f}")
        if 'iou' in df_id.columns:
            self._log(f"  IoU range: "
                      f"{df_id['iou'].min():.4f} to "
                      f"{df_id['iou'].max():.4f}")

        best_match = df_id.loc[df_id['f1_score'].idxmax()]
        best_time = best_match['time_idx']
        best_f1 = best_match['f1_score']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # -- Panel 1: F1 + IoU over time ---------------------------------- #
        ax1 = axes[0, 0]
        ax1.plot(df_id['time_idx'], df_id['f1_score'], 'o-',
                 linewidth=2, markersize=6, color='green', label='F1-Score')
        if 'iou' in df_id.columns:
            ax1.plot(df_id['time_idx'], df_id['iou'], 's-',
                     linewidth=2, markersize=5, color='blue', label='IoU')
        ax1.axvline(x=best_time, color='red', linestyle='--', linewidth=2,
                    alpha=0.7, label=f'Best Match (F1={best_f1:.4f})')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Score')
        ax1.set_title(f'F1-Score and IoU Over Time - {id_label}')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # -- Panel 2: precision + recall ---------------------------------- #
        ax2 = axes[0, 1]
        ax2.plot(df_id['time_idx'], df_id['precision'], 'b-o',
                 label='Precision', linewidth=2, markersize=6)
        ax2.plot(df_id['time_idx'], df_id['recall'], 'orange',
                 label='Recall', linewidth=2, markersize=6)
        ax2.axvline(x=best_time, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Score')
        ax2.set_title(f'Precision and Recall Over Time - {id_label}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        # -- Panel 3: areas ----------------------------------------------- #
        ax3 = axes[1, 0]
        if 'intersecting_model_area_km2' in df_id.columns:
            ax3.plot(df_id['time_idx'], df_id['intersecting_model_area_km2'],
                     'b-', linewidth=2, label='Model Ice Area (Intersecting)')
        ax3.plot(df_id['time_idx'], df_id['field_area_km2'],
                 'r-', linewidth=2, label='Field Ice Area')
        ax3.fill_between(df_id['time_idx'], 0, df_id['tp_area_km2'],
                         alpha=0.3, color='green', label='True Positive (Overlap)')
        ax3.axvline(x=best_time, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Area (km²)')
        ax3.set_title(f'Ice Area Comparison - {id_label}')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # -- Panel 4: best-match metric bar chart ------------------------- #
        ax4 = axes[1, 1]
        categories = ['F1-Score']
        values = [best_match['f1_score']]
        colors = ['green']
        if 'iou' in best_match:
            categories.append('IoU')
            values.append(best_match['iou'])
            colors.append('blue')
        categories.append('Precision')
        values.append(best_match['precision'])
        colors.append('orange')
        categories.append('Recall')
        values.append(best_match['recall'])
        colors.append('red')

        bars = ax4.bar(categories, values, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=1)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('Score', fontsize=10)
        ax4.set_title(f'Best Match Metrics (Time {best_time}) - {id_label}',
                      fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02, f'{val:.3f}',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        area_text = ("Area Statistics:\n"
                     + "=" * 18 + "\n"
                     + f"TP: {best_match['tp_area_km2']:.1f} km²\n"
                     + f"FP: {best_match['fp_area_km2']:.1f} km²\n"
                     + f"FN: {best_match['fn_area_km2']:.1f} km²\n")
        if 'intersecting_model_area_km2' in best_match:
            area_text += f"Model: {best_match['intersecting_model_area_km2']:.1f} km²\n"
        else:
            area_text += f"Model: {best_match['model_area_km2']:.1f} km²\n"
        area_text += f"Field: {best_match['field_area_km2']:.1f} km²"

        ax4.text(0.95, 0.95, area_text, transform=ax4.transAxes,
                 fontsize=8, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle="round,pad=0.4",
                           facecolor='lightgray', alpha=0.8))

        plt.suptitle(f'Polygon-Based Overlap Analysis - {id_label}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plot_path = self.output_subdirs['plots'] / (
            f"{self.prefix}_overlap_polygon_analysis{id_suffix}.pdf"
        )
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        self._log(f"Summary plot saved to: {plot_path}")

    def generate_best_match_map(self):
        """Generate a best-match map for every field polygon ID."""
        n_ids = len(self.unique_ids)
        fig, axes = plt.subplots(1, n_ids, figsize=(6 * n_ids, 5))
        if n_ids == 1:
            axes = [axes]

        for i, uid in enumerate(self.unique_ids):
            ax = axes[i]
            best = self.best_matches[uid]

            if best['time_idx'] is None:
                ax.text(0.5, 0.5, f'No valid match for ID {uid}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'ID {uid}')
                continue

            if self.use_time_window and self.window_size > 1:
                model_polygons = self.get_model_polygons_window(best['time_idx'])
            else:
                model_polygons = self.get_model_polygons(best['time_idx'])

            field_polygon = self.field_polygons_by_id[uid]
            intersecting_polygons = []
            non_intersecting_polygons = []
            for poly in model_polygons:
                (intersecting_polygons if poly.intersects(field_polygon)
                 else non_intersecting_polygons).append(poly)

            intersecting_area = (sum(p.area for p in intersecting_polygons) / 1e6
                                 if intersecting_polygons else 0)
            non_intersecting_area = (sum(p.area for p in non_intersecting_polygons) / 1e6
                                     if non_intersecting_polygons else 0)

            # Field polygon (red)
            field_geoms = (field_polygon.geoms
                           if isinstance(field_polygon, MultiPolygon)
                           else [field_polygon])
            for poly in field_geoms:
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, fc='red', ec='red')

            # Excluded model polygons (gray)
            for poly in non_intersecting_polygons:
                if poly.is_empty:
                    continue
                geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
                for p in geoms:
                    x, y = p.exterior.xy
                    ax.fill(x, y, alpha=0.3, fc='gray', ec='gray')

            # Included model polygons (blue)
            for poly in intersecting_polygons:
                if poly.is_empty:
                    continue
                geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
                for p in geoms:
                    x, y = p.exterior.xy
                    ax.fill(x, y, alpha=0.7, fc='blue', ec='blue')

            # Overlap (green)
            if intersecting_polygons:
                model_union = unary_union(intersecting_polygons)
                intersection = model_union.intersection(field_polygon)
                if intersection and not intersection.is_empty:
                    geoms = (intersection.geoms
                             if isinstance(intersection, MultiPolygon)
                             else [intersection])
                    for poly in geoms:
                        x, y = poly.exterior.xy
                        ax.fill(x, y, alpha=0.3, fc='green', ec='green')

            ax.set_aspect('equal')
            res = best['result']
            ax.set_title(
                f"ID {uid}\n"
                f"Best Match: Time {best['time_idx']}\n"
                f"F1={res['f1_score']:.3f}, IoU={res['iou']:.3f}\n"
                f"Precision={res['precision']:.3f}, Recall={res['recall']:.3f}",
                fontsize=10,
            )

            info_text = (f"Intersecting polygons: {len(intersecting_polygons)}\n"
                         f"  Area: {intersecting_area:.2f} km²\n")
            if non_intersecting_polygons:
                info_text += (f"Excluded polygons: {len(non_intersecting_polygons)}\n"
                              f"  Area: {non_intersecting_area:.2f} km²")
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor='white', alpha=0.7))

        legend_elements = [
            Patch(facecolor='red', alpha=0.5,
                  label='Field Ice (Ground Truth)'),
            Patch(facecolor='blue', alpha=0.7,
                  label='Model Ice (Intersecting - Included)'),
            Patch(facecolor='gray', alpha=0.3,
                  label='Model Ice (Non-intersecting - Excluded)'),
            Patch(facecolor='green', alpha=0.3,
                  label='Overlap (True Positive)'),
        ]
        axes[0].legend(handles=legend_elements, loc='lower right', fontsize=8)

        plt.suptitle(
            'Best Match Maps by Field ID '
            '(Only Intersecting Model Ice Considered)',
            fontsize=14, fontweight='bold',
        )
        plt.tight_layout()
        map_path = self.output_subdirs['best_match'] / (
            f"{self.prefix}_best_match_maps.pdf"
        )
        plt.savefig(map_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()
        self._log(f"Best match maps saved to: {map_path}")