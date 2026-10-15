"""
/*********************************************************************************
* AFDA (Ice Flow Direction Analysis) with Moving Time Window:                    * 
* ============================================================                   *
* Compares modeled ice flow directions with field observations                   *
*                                                                                * 
* Features:                                                                      *
* - Derive flow direction based on start and end points of the lineations        *
* - Moving time window for smoothing U and V velocities                          *
* - Accumulative ice mask for the time window                                    *
* - Ice-free and fozen bed conditions                                            *
*                                                                                * 
* Developed based on the original AML code by Li et al (2007)                    *
* Enhanced with moving time window support                                       *
*                                                                                * 
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
import warnings

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy import stats
from scipy.ndimage import zoom
from shapely.geometry import box

mpl.use('Agg')  # Non-interactive backend
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Constants (following the original AML code by Li et al. 2007)
# --------------------------------------------------------------------------- #
ICE_FREE_VALUE = 9999.0      # Ice-free areas (AML uses 9999)
FROZEN_BED_VALUE = 888.0     # Frozen-bed areas (AML uses 888)
NO_DATA_VALUE = -9999.0      # No-data value

# Angular-difference classification thresholds (degrees)
FROZEN_BED_MIN_DIFF = 360.0    # diff > 360° and <= 2000° -> frozen bed
FROZEN_BED_MAX_DIFF = 2000.0   # diff > 2000°              -> ice-free

# --------------------------------------------------------------------------- #
# Helper functions (module-level; use print for logging)
# --------------------------------------------------------------------------- #
def get_ice_mask_window(dataset, time_indices, threshold=0.5,
                        ice_var='thk', ice_type='thickness',
                        use_accumulative=True):
    """
    Build an ice mask over a time window (cumulative or per-time).

    Parameters
    ----------
    dataset : xarray.Dataset
        PISM NetCDF dataset.
    time_indices : list of int
        Time indices to include in the window.
    threshold : float
        Threshold for ice detection.
    ice_var : str
        Name of the ice variable.
    ice_type : str
        One of 'thickness', 'fraction', or 'mask'.
    use_accumulative : bool
        If True, ice is present if ANY time step in the window has ice.
        If False, ice is present if ALL time steps in the window have ice.

    Returns
    -------
    ice_present : numpy.ndarray of bool
        Boolean mask where ice is present.
    """
    ice_present_total = None

    for t_idx in time_indices:
        ds_slice = dataset.isel(time=t_idx)

        # Get ice data for this time step
        ice_data = None
        if ice_var in ds_slice:
            ice_data = ds_slice[ice_var].values
        else:
            for var in ('thk', 'land_ice_thickness', 'mask',
                        'land_ice_area_fraction'):
                if var in ds_slice:
                    ice_data = ds_slice[var].values
                    break

        if ice_data is None:
            continue

        # Determine ice presence for this time step
        if ice_type == 'fraction':
            ice_present_step = ice_data >= threshold
        elif ice_type == 'mask':
            ice_present_step = ice_data >= 1
        else:  # thickness
            ice_present_step = ice_data >= threshold

        # Combine across the time window
        if ice_present_total is None:
            ice_present_total = ice_present_step.copy()
        else:
            if use_accumulative:
                ice_present_total = ice_present_total | ice_present_step
            else:
                ice_present_total = ice_present_total & ice_present_step

    if ice_present_total is None:
        # Fall back to the dataset's own y/x shape when available
        if 'x' in dataset.dims and 'y' in dataset.dims:
            ice_present_total = np.zeros(
                (len(dataset.y), len(dataset.x)), dtype=bool
            )
        else:
            first_var = list(dataset.data_vars.values())[0]
            ice_present_total = np.zeros(first_var.shape[-2:], dtype=bool)

    return ice_present_total


def compute_velocity_window(dataset, time_indices,
                            u_var='uvelbase', v_var='vvelbase'):
    """
    Compute average velocity components over a time window.

    Returns
    -------
    u_mean, v_mean : numpy.ndarray or (None, None)
        Mean U/V velocity over the window, or (None, None) if no data found.
    """
    u_sum = None
    v_sum = None
    count = 0

    for t_idx in time_indices:
        ds_slice = dataset.isel(time=t_idx)

        # U velocity
        u = None
        if u_var in ds_slice:
            u = ds_slice[u_var].values
        else:
            for var in ('uvelbase', 'uvelsurf', 'uvel', 'ubar'):
                if var in ds_slice:
                    u = ds_slice[var].values
                    break

        # V velocity
        v = None
        if v_var in ds_slice:
            v = ds_slice[v_var].values
        else:
            for var in ('vvelbase', 'vvelsurf', 'vvel', 'vbar'):
                if var in ds_slice:
                    v = ds_slice[var].values
                    break

        if u is not None and v is not None:
            if u_sum is None:
                u_sum = u.copy()
                v_sum = v.copy()
            else:
                u_sum += u
                v_sum += v
            count += 1

    if count > 0:
        return u_sum / count, v_sum / count
    return None, None


def compute_model_flow_direction_window(dataset, center_idx, window_size,
                                        threshold=0.5,
                                        velocity_threshold=1.0,
                                        u_var='uvelbase', v_var='vvelbase',
                                        ice_var='thk', ice_type='thickness',
                                        use_accumulative_ice=True):
    """
    Compute ice flow direction using a moving time window.

    Returns
    -------
    model_raster : numpy.ndarray
        Raster with flow directions (ICE_FREE_VALUE, FROZEN_BED_VALUE, or deg).
    ice_present : numpy.ndarray of bool
        Boolean mask where ice is present.
    """
    half_window = window_size // 2
    start_idx = max(0, center_idx - half_window)
    end_idx = min(len(dataset.time), center_idx + half_window + 1)
    time_indices = list(range(start_idx, end_idx))

    print(f"    Time window: indices {start_idx} to {end_idx - 1} "
          f"({len(time_indices)} steps)")

    u_mean, v_mean = compute_velocity_window(dataset, time_indices, u_var, v_var)
    ice_present = get_ice_mask_window(dataset, time_indices, threshold,
                                      ice_var, ice_type,
                                      use_accumulative_ice)

    model_raster = np.full(ice_present.shape, ICE_FREE_VALUE, dtype=np.float32)

    if not np.any(ice_present):
        print("    No ice present in this time window")
        return model_raster, ice_present

    if u_mean is None or v_mean is None:
        print("    Warning: No velocity data found. "
              "All ice classified as frozen.")
        model_raster[ice_present] = FROZEN_BED_VALUE
        return model_raster, ice_present

    velocity_mag = np.sqrt(u_mean ** 2 + v_mean ** 2)
    active_ice = ice_present & (velocity_mag >= velocity_threshold)
    frozen_ice = ice_present & (velocity_mag < velocity_threshold)

    model_raster[frozen_ice] = FROZEN_BED_VALUE
    print(f"    Frozen ice cells: {int(np.sum(frozen_ice)):,}")

    if np.any(active_ice):
        direction_rad = np.arctan2(v_mean, u_mean)
        direction_deg = (90 - np.degrees(direction_rad)) % 360
        model_raster[active_ice] = direction_deg[active_ice]
        print(f"    Active ice cells: {int(np.sum(active_ice)):,}")
        print(f"    Max velocity: "
              f"{float(np.max(velocity_mag[active_ice])):.2f} m/a")
    else:
        print("    No active ice (all frozen)")

    return model_raster, ice_present

# --------------------------------------------------------------------------- #
# Main analyzer
# --------------------------------------------------------------------------- #
class PISMAFDA:
    """PISM Ice Flow Direction Analysis with moving time window."""

    def __init__(self, pism_dataset, shapefile_path, output_subdirs,
                 prefix="", time_indices=None, ice_threshold=0.5,
                 velocity_threshold=1.0, cell_size=500,
                 use_basal_velocity=True, save_rasters=False,
                 u_var='uvelbase', v_var='vvelbase', ice_var='thk',
                 ice_type='thickness', use_time_window=False,
                 window_size=10, use_accumulative_ice=True,
                 progress_callback=None, cancel_checker=None):
        self.pism_dataset = pism_dataset
        self.ice_threshold = ice_threshold
        self.velocity_threshold = velocity_threshold
        self.cell_size = cell_size
        self.output_subdirs = output_subdirs
        self.use_basal_velocity = use_basal_velocity
        self.save_rasters = save_rasters
        self.prefix = prefix
        self.console = None

        self.u_var = u_var
        self.v_var = v_var
        self.ice_var = ice_var
        self.ice_type = ice_type

        self.use_time_window = use_time_window
        # Ensure odd window size
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.use_accumulative_ice = use_accumulative_ice

        self.progress_callback = progress_callback
        self.cancel_checker = cancel_checker
        self._is_cancelled = False

        self.ICE_FREE = ICE_FREE_VALUE
        self.FROZEN_BED = FROZEN_BED_VALUE
        self.NO_DATA = NO_DATA_VALUE
        self.crs = None

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
            self.transform = from_origin(0, self.model_shape[0],
                                         cell_size, cell_size)

        self._log(f"\nNetCDF bounds: {self.netcdf_bounds}")
        self._log(f"Model shape: {self.model_shape}")

        # ---- Load & rasterize field observations (once) ------------------ #
        self._log("\nLoading and rasterizing field observations (one time)...")
        self.field_gdf = self.load_field_observations(shapefile_path)

        if self.field_gdf.crs is not None:
            self.crs = self.field_gdf.crs

        self.field_raster, self.field_transform = self.rasterize_flow_lines(
            self.field_gdf, self.netcdf_bounds, self.cell_size
        )

        # Save static field raster (always; independent of save_rasters)
        field_path = (self.output_subdirs['rasters']
                      / f"{self.prefix}_field_flow_static.tif")
        field_transform_gis = from_origin(
            self.netcdf_bounds[0], self.netcdf_bounds[3],
            self.cell_size, self.cell_size,
        )
        with rasterio.open(
            field_path, 'w',
            driver='GTiff',
            height=self.field_raster.shape[0],
            width=self.field_raster.shape[1],
            count=1,
            dtype=self.field_raster.dtype,
            crs=self.crs,
            transform=field_transform_gis,
            nodata=self.NO_DATA,
        ) as dst:
            dst.write(self.field_raster, 1)
        self._log(f"    Static field raster saved to: {field_path}")

        # Precompute valid field mask
        self.valid_field_mask = (
            (self.field_raster != self.NO_DATA)
            & (self.field_raster >= 0)
            & (self.field_raster <= 360)
        )
        self._log(f"    Field cells with direction: "
                  f"{int(np.sum(self.valid_field_mask))}")

        # ---- Time indices ------------------------------------------------- #
        if 'time' in pism_dataset.dims:
            if time_indices is None:
                if self.use_time_window:
                    half_win = self.window_size // 2
                    self.time_indices = list(
                        range(half_win, len(pism_dataset.time) - half_win)
                    )
                else:
                    self.time_indices = list(range(len(pism_dataset.time)))
            else:
                self.time_indices = [i for i in time_indices
                                     if i < len(pism_dataset.time)]
            self._log(f"\nTime steps available: {len(pism_dataset.time)}")
            self._log(f"Analyzing {len(self.time_indices)} time indices")
            if self.use_time_window:
                self._log(f"Using moving time window "
                          f"(size={self.window_size}, "
                          f"accumulative_ice={self.use_accumulative_ice})")
        else:
            self.time_indices = [0]

        # ---- Results placeholders ---------------------------------------- #
        self.results = []
        self.time_series = None

        self._log("\nAFDA Analysis initialized")
        self._log(f"  Time steps to analyze: {len(self.time_indices)}")
        self._log(f"  Cell size: {self.cell_size} m")
        self._log(f"  Velocity variables: U='{self.u_var}', V='{self.v_var}'")
        self._log(f"  Ice variable: '{self.ice_var}' (type: {self.ice_type})")
        self._log(f"  Velocity threshold: {self.velocity_threshold} m/a")
        window_status = (f"Enabled (size={self.window_size})"
                         if self.use_time_window else "Disabled")
        self._log(f"  Moving time window: {window_status}")
        self._log(f"  Field raster shape: {self.field_raster.shape}")
        self._log(f"  Valid field cells: {int(np.sum(self.valid_field_mask))}")

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
        self._log("\n  Cancellation requested for AFDA analysis...")

    def is_cancelled(self):
        """Check whether the analysis has been cancelled."""
        if self.cancel_checker:
            return self.cancel_checker()
        return self._is_cancelled

    # ------------------------------------------------------------------ #
    # Field observations
    # ------------------------------------------------------------------ #
    def load_field_observations(self, shapefile_path):
        """Load field observations from line features."""
        self._log(f"Loading field line features: {shapefile_path}")

        try:
            gdf = gpd.read_file(shapefile_path, engine="pyogrio")
            self._log("  Using optimized pyogrio engine")
        except Exception:
            gdf = gpd.read_file(shapefile_path)
            self._log("  Using default engine")

        self._log(f"  Original shapefile CRS: {gdf.crs}")
        self._log(f"  Original bounds: {gdf.total_bounds}")
        self._log(f"  Number of features: {len(gdf)}")

        if self.crs is not None and gdf.crs != self.crs:
            self._log("  Reprojecting shapefile to target CRS")
            gdf = gdf.to_crs(self.crs)

        clip_box = box(*self.netcdf_bounds)
        original_count = len(gdf)
        gdf = gdf.clip(clip_box)
        self._log(f"  Features before clipping: {original_count}, "
                  f"after clipping: {len(gdf)}")

        return gdf

    def rasterize_flow_lines(self, gdf, bounds, cell_size):
        """Convert flow-direction lines into a direction raster."""
        width = max(1, int((bounds[2] - bounds[0]) / cell_size) + 1)
        height = max(1, int((bounds[3] - bounds[1]) / cell_size) + 1)
        transform = from_origin(bounds[0], bounds[3], cell_size, cell_size)

        self._log(f"    Rasterizing lines: {width} x {height} cells, "
                  f"cell size {cell_size} m")

        direction_raster = np.full((height, width),
                                   NO_DATA_VALUE, dtype=np.float32)
        point_count = 0
        line_count = 0

        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom.geom_type == 'LineString':
                lines = [geom]
            elif geom.geom_type == 'MultiLineString':
                lines = list(geom.geoms)
            else:
                continue

            for line in lines:
                if line.length == 0:
                    continue

                line_count += 1
                coords = list(line.coords)
                start, end = coords[0], coords[-1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if dx == 0 and dy == 0:
                    continue

                angle_rad = np.arctan2(dx, dy)
                angle_deg = np.degrees(angle_rad) % 360

                num_points = max(2, int(line.length / (cell_size / 2)))
                distances = np.linspace(0, line.length, num_points)
                points = [line.interpolate(d) for d in distances]

                for point in points:
                    col = int((point.x - bounds[0]) / cell_size)
                    row_idx = int((bounds[3] - point.y) / cell_size)
                    if (0 <= col < direction_raster.shape[1]
                            and 0 <= row_idx < direction_raster.shape[0]):
                        direction_raster[row_idx, col] = angle_deg
                        point_count += 1

        self._log(f"    Processed {line_count} line segments")
        self._log(f"    Rasterized {point_count} points")
        self._log(f"    Cells with direction values: "
                  f"{int(np.sum(direction_raster >= 0))}")

        return direction_raster, transform

    # ------------------------------------------------------------------ #
    # Per-time-step analysis
    # ------------------------------------------------------------------ #
    def analyze_time_step(self, time_idx):
        """Analyze flow direction for a single time step."""
        time_label = f"t{time_idx:04d}"
        self._log(f"\n  Processing time step {time_label}")

        if self.use_time_window:
            model_raster, ice_present = compute_model_flow_direction_window(
                self.pism_dataset,
                time_idx,
                self.window_size,
                threshold=self.ice_threshold,
                velocity_threshold=self.velocity_threshold,
                u_var=self.u_var,
                v_var=self.v_var,
                ice_var=self.ice_var,
                ice_type=self.ice_type,
                use_accumulative_ice=self.use_accumulative_ice,
            )
        else:
            ds_slice = self.pism_dataset.isel(time=time_idx)
            model_raster, ice_present = self.compute_single_time_step(ds_slice)

        ice_free_count = int(np.sum(model_raster == self.ICE_FREE))
        frozen_count = int(np.sum(model_raster == self.FROZEN_BED))
        active_count = int(np.sum((model_raster >= 0)
                                  & (model_raster <= 360)))

        self._log(f"    Active ice cells: {active_count}")
        self._log(f"    Frozen bed cells: {frozen_count}")
        self._log(f"    Ice-free cells: {ice_free_count}")

        # Flip Y axis once so the model raster matches the field raster
        # (row 0 = north), which is the convention used by rasterize_flow_lines.
        model_raster = np.flipud(model_raster)

        # Optionally save the flipped model raster
        if self.save_rasters:
            model_path = (self.output_subdirs['rasters']
                          / f"{self.prefix}_model_flow_t{time_idx:04d}.tif")
            with rasterio.open(
                model_path, 'w',
                driver='GTiff',
                height=model_raster.shape[0],
                width=model_raster.shape[1],
                count=1,
                dtype=model_raster.dtype,
                crs=self.crs,
                transform=self.transform,
                nodata=self.NO_DATA,
            ) as dst:
                dst.write(model_raster, 1)

        # Resample model to match field raster if needed
        if model_raster.shape != self.field_raster.shape:
            scale_y = self.field_raster.shape[0] / model_raster.shape[0]
            scale_x = self.field_raster.shape[1] / model_raster.shape[1]
            model_resampled = zoom(model_raster, (scale_y, scale_x), order=0)
            model_resampled = model_resampled[:self.field_raster.shape[0],
                                              :self.field_raster.shape[1]]
        else:
            model_resampled = model_raster

        valid_mask = self.valid_field_mask & (model_resampled != self.NO_DATA)
        self._log(f"    Valid comparison cells: {int(np.sum(valid_mask))}")

        if not np.any(valid_mask):
            self._log(f"    Warning: No valid overlapping cells for "
                      f"time {time_label}")
            return None

        (mean_angle, vec_strength,
         normal_pct, frozen_pct, ice_free_pct) = \
            self._calculate_angular_statistics(
                model_resampled, self.field_raster, valid_mask
            )

        return {
            'time_idx': time_idx,
            'time_label': time_label,
            'vec_mean_deg': mean_angle,
            'vec_strength': vec_strength,
            'normal_alignment_pct': normal_pct,
            'frozen_bed_pct': frozen_pct,
            'ice_free_pct': ice_free_pct,
            'active_ice_cells': active_count,
            'frozen_bed_cells': frozen_count,
            'ice_free_cells': ice_free_count,
            'total_valid_cells': int(np.sum(valid_mask)),
        }

    def compute_single_time_step(self, ds_slice):
        """Original single-time-step computation."""
        u = None
        v = None

        if self.u_var in ds_slice:
            u = ds_slice[self.u_var].values
        if self.v_var in ds_slice:
            v = ds_slice[self.v_var].values

        if u is None:
            for var in ('uvelbase', 'uvelsurf', 'uvel'):
                if var in ds_slice:
                    u = ds_slice[var].values
                    break
        if v is None:
            for var in ('vvelbase', 'vvelsurf', 'vvel'):
                if var in ds_slice:
                    v = ds_slice[var].values
                    break

        ice_present = self.get_ice_mask_single(ds_slice)
        model_raster = np.full(ice_present.shape,
                               ICE_FREE_VALUE, dtype=np.float32)

        if np.any(ice_present) and u is not None and v is not None:
            velocity_mag = np.sqrt(u ** 2 + v ** 2)
            active_ice = ice_present & (velocity_mag >= self.velocity_threshold)
            frozen_ice = ice_present & (velocity_mag < self.velocity_threshold)

            model_raster[frozen_ice] = FROZEN_BED_VALUE

            if np.any(active_ice):
                direction_rad = np.arctan2(v, u)
                direction_deg = (90 - np.degrees(direction_rad)) % 360
                model_raster[active_ice] = direction_deg[active_ice]
                self._log(f"    Active ice cells: "
                          f"{int(np.sum(active_ice)):,}")
                self._log(f"    Frozen ice cells: "
                          f"{int(np.sum(frozen_ice)):,}")
            else:
                self._log("    No active ice (all frozen)")
        elif np.any(ice_present):
            self._log("    No velocity data. All ice classified as frozen.")
            model_raster[ice_present] = FROZEN_BED_VALUE

        return model_raster, ice_present

    def get_ice_mask_single(self, ds_slice):
        """Single-time-step ice mask."""
        ice_present = None

        if self.ice_var in ds_slice:
            ice_data = ds_slice[self.ice_var].values
            if self.ice_type == 'fraction':
                ice_present = ice_data >= self.ice_threshold
            elif self.ice_type == 'mask':
                ice_present = ice_data >= 1
            else:
                ice_present = ice_data >= self.ice_threshold

        if ice_present is None:
            for var in ('thk', 'land_ice_thickness', 'mask'):
                if var in ds_slice:
                    ice_data = ds_slice[var].values
                    ice_present = ice_data >= self.ice_threshold
                    break

        if ice_present is None:
            ice_present = np.zeros(self.model_shape, dtype=bool)

        return ice_present

    def _calculate_angular_statistics(self, model_raster, field_raster,
                                      valid_mask):
        """Compute angular statistics following the original AML logic."""
        model_vals = model_raster[valid_mask]
        field_vals = field_raster[valid_mask]

        diff = np.abs(model_vals - field_vals)

        normal_mask = diff <= FROZEN_BED_MIN_DIFF
        frozen_mask = ((diff > FROZEN_BED_MIN_DIFF)
                       & (diff <= FROZEN_BED_MAX_DIFF))
        ice_free_mask = diff > FROZEN_BED_MAX_DIFF

        normal_count = int(np.sum(normal_mask))
        frozen_count = int(np.sum(frozen_mask))
        ice_free_count = int(np.sum(ice_free_mask))
        total_valid = normal_count + frozen_count + ice_free_count

        self._log(f"    The number of normal cells: {normal_count}")
        self._log(f"    The number of frozen cells: {frozen_count}")
        self._log(f"    The number of ice-free cells: {ice_free_count}")

        if total_valid == 0:
            return 180, 100, 0, 0, 0

        adjusted_diff = np.zeros_like(diff)
        normal_angles = diff[normal_mask]
        normal_adjusted = np.where(normal_angles > 180,
                                   360 - normal_angles, normal_angles)
        adjusted_diff[normal_mask] = normal_adjusted
        adjusted_diff[frozen_mask] = 180.0
        adjusted_diff[ice_free_mask] = 180.0

        angles_rad = np.deg2rad(adjusted_diff)
        mean_angle_deg = np.rad2deg(stats.circmean(angles_rad))
        if mean_angle_deg < 0:
            mean_angle_deg += 360

        # Vector strength uses only the "normal" cells
        valid_diff = adjusted_diff[normal_mask]
        if len(valid_diff) > 0:
            vec_strength = 1.0 - stats.circvar(np.deg2rad(valid_diff))
        else:
            vec_strength = 0

        normal_pct = (normal_count / total_valid) * 100
        frozen_pct = (frozen_count / total_valid) * 100
        ice_free_pct = (ice_free_count / total_valid) * 100

        return mean_angle_deg, vec_strength, normal_pct, frozen_pct, ice_free_pct

    # ------------------------------------------------------------------ #
    # Full time-series driver
    # ------------------------------------------------------------------ #
    def analyze_time_series(self):
        """Analyze flow direction for all time steps."""
        self._log("\n" + "=" * 80)
        self._log("AFDA TIME SERIES ANALYSIS")
        self._log("=" * 80)
        self._log(f"Number of time steps: {len(self.time_indices)}")
        if self.use_time_window:
            self._log(f"Moving time window: size={self.window_size}")
            self._log(f"Accumulative ice mask: {self.use_accumulative_ice}")
        self._log("=" * 80)

        all_results = []
        total_steps = len(self.time_indices)

        for time_idx in self.time_indices:
            if self.is_cancelled():
                self._log("\n  Analysis cancelled by user.")
                if self.progress_callback:
                    self.progress_callback(100, "Analysis cancelled")
                return None

            if self.progress_callback:
                progress = 30 + int(len(all_results) / total_steps * 60)
                self.progress_callback(
                    progress, f"Processing AFDA time step: {time_idx}"
                )

            result = self.analyze_time_step(time_idx)
            if result:
                all_results.append(result)
                self._log(f"    Vector mean: {result['vec_mean_deg']:.2f}°")
                self._log(f"    Normal: {result['normal_alignment_pct']:.1f}%, "
                          f"Frozen: {result['frozen_bed_pct']:.1f}%, "
                          f"Ice-Free: {result['ice_free_pct']:.1f}%")

        if not all_results:
            self._log("\nNo results generated!")
            return None

        self.results = all_results
        self.time_series = pd.DataFrame(all_results)
        self._save_results()
        return self.time_series

    # ------------------------------------------------------------------ #
    # Saving / plotting
    # ------------------------------------------------------------------ #
    def _save_results(self):
        """Save results to CSV and a plain-text summary."""
        csv_file = (self.output_subdirs['csv']
                    / f"{self.prefix}_afda_time_series.csv")
        self.time_series.to_csv(csv_file, index=False)
        self._log(f"\nTime series saved to: {csv_file}")

        self._create_summary_plot()

        summary_file = (self.output_subdirs['csv']
                        / f"{self.prefix}_afda_summary.txt")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AFDA ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"ICE-FREE VALUE: {self.ICE_FREE}\n")
            f.write(f"FROZEN BED VALUE: {self.FROZEN_BED}\n\n")
            f.write("CLASSIFICATION:\n")
            f.write("  diff <= 360°: Normal ice movement\n")
            f.write("  360° < diff <= 2000°: Frozen bed\n")
            f.write("  diff > 2000°: Ice-free\n\n")
            if self.use_time_window:
                f.write(f"MOVING TIME WINDOW: Enabled "
                        f"(size={self.window_size})\n")
                f.write(f"ACCUMULATIVE ICE MASK: "
                        f"{self.use_accumulative_ice}\n\n")
            f.write("TIME SERIES RESULTS:\n")
            f.write("-" * 40 + "\n")
            f.write(self.time_series.to_string())

        self._log(f"Summary saved to: {summary_file}")

    def _create_summary_plot(self):
        """Create a 4-panel summary visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ts = self.time_series

        # Panel 1: vector mean
        ax1 = axes[0, 0]
        ax1.plot(ts['time_idx'], ts['vec_mean_deg'], 'o-',
                 linewidth=2, markersize=8, color='darkblue')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Vector Mean (degrees)')
        ax1.set_title('Mean Angular Difference')
        ax1.grid(True, alpha=0.3)

        # Panel 2: classification percentages
        ax2 = axes[0, 1]
        ax2.plot(ts['time_idx'], ts['normal_alignment_pct'], 'g-o',
                 label='Normal (<=360°)', linewidth=2)
        ax2.plot(ts['time_idx'], ts['frozen_bed_pct'], 'orange',
                 label='Frozen (360-2000°)', linewidth=2)
        ax2.plot(ts['time_idx'], ts['ice_free_pct'], 'r-^',
                 label='Ice-Free (>2000°)', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Flow Direction Classification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Panel 3: vector strength
        ax3 = axes[1, 0]
        ax3.plot(ts['time_idx'], ts['vec_strength'], 's-',
                 linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Vector Strength')
        ax3.set_title('Vector Strength')
        ax3.grid(True, alpha=0.3)

        # Panel 4: cell counts
        ax4 = axes[1, 1]
        ax4.plot(ts['time_idx'], ts['active_ice_cells'] / 1000, 'b-',
                 label='Active Ice', linewidth=2)
        ax4.plot(ts['time_idx'], ts['frozen_bed_cells'] / 1000, 'c--',
                 label='Frozen Bed', linewidth=2)
        ax4.plot(ts['time_idx'], ts['ice_free_cells'] / 1000, 'gray',
                 label='Ice-Free', linewidth=2)
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Cell Count (thousands)')
        ax4.set_title('Ice Conditions Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        window_text = (f"Moving Window: {self.window_size} steps"
                       if self.use_time_window else "No time window")
        plt.suptitle(f'AFDA Time Series Analysis\n{window_text}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = (self.output_subdirs['plots']
                     / f"{self.prefix}_afda_time_series.pdf")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        gc.collect()

        self._log(f"Summary plot saved to: {plot_path}")