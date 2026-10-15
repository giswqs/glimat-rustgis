"""
/*********************************************************************************
* TIFF/ASC Raster to NetCDF Converter for GLIMAT Plugin                          * 
* ===================================================                            *
* Converts individual GeoTIFF or ASCII Grid files into a single NetCDF file.     *     
*  - Original resolution and crs are preserved - NO RESAMPLING is performed.     *
*  - OPTIMIZED PARALLEL VERSION                                                  *
*  - Auto-detects time information from filenames - NO RESAMPLING                * 
*                                                                                *   
* Folder Structure:                                                              *
*  - Each folder should contain files for one variable                           *
*  - Files should contain time indices in their names (e.g., thk_0010.tif)       *
*  - All files must have the same spatial extent and resolution                  *
*                                                                                *   
* Supported Variables:                                                           *
*  - Thickness: Required for ice mask detection                                  *
*  - Surface Elevation: Optional, for point comparison                           *
*  - U/V Velocity: Optional, for flow direction analysis                         *
*                                                                                *   
* Output:                                                                        *
*  - NetCDF file with original resolution and crs ready for GLIMAT tools         *
*                                                                                *   
* This file is part of GLIMAT Plugin.                                            *
*                                                                                *
* Author: Yingkui Li                                                             *
* Department of Geography & Sustainability                                       *
* University of Tennessee, Knoxville, TN 37996                                   * 
* Email: yli32@utk.edu                                                           *
* Start date: 06/9/2026                                                          *
* Finalized date: 09/17/2026                                                     *
* Copyright(C): 2026 Yingkui Li                                                  * 
*                                                                                *
* Version: 1.0.0                                                                 *
*********************************************************************************/
"""

import glob
import multiprocessing as mp
import os
import re
import time
import warnings
from concurrent.futures import (
    ProcessPoolExecutor, ThreadPoolExecutor, as_completed,
)
from datetime import datetime

import numpy as np
import rasterio
import xarray as xr
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')

class RasterToNetCDFConverter:
    """
    Optimized parallel TIFF/ASC to NetCDF converter.
    Auto-detects time from filenames.
    """

    SUPPORTED_EXTENSIONS = ('.tif', '.tiff', '.asc')

    def __init__(self, thick_folder, output_nc,
                 surface_folder=None, uvel_folder=None, vvel_folder=None,
                 start_idx=None, end_idx=None, step=None,
                 use_compression=True, fill_value=-9999.0, crs=None,
                 progress_callback=None, cancel_checker=None,
                 max_workers=None):
        self.thick_folder = thick_folder
        self.surface_folder = surface_folder
        self.uvel_folder = uvel_folder
        self.vvel_folder = vvel_folder
        self.output_nc = output_nc
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.step = step
        self.use_compression = use_compression
        self.fill_value = fill_value
        self.crs = crs
        self.progress_callback = progress_callback
        self.cancel_checker = cancel_checker
        self._is_cancelled = False
        self.console = None
        self.last_progress = 0

        # Leave one core for the OS by default
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)

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
        self._is_cancelled = True

    def is_cancelled(self):
        if self.cancel_checker:
            return self.cancel_checker()
        return self._is_cancelled

    # ------------------------------------------------------------------ #
    # File discovery / time extraction
    # ------------------------------------------------------------------ #
    def find_files(self, folder_path):
        """Find all supported raster files in the folder."""
        if not folder_path or not os.path.exists(folder_path):
            return []

        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        return sorted(files)

    def auto_detect_time_range(self, folder_path):
        """
        Auto-detect the time range by scanning all files in the folder.

        Returns
        -------
        (start_time, end_time, step, all_times) or (None, None, None, [])
        """
        if not folder_path or not os.path.exists(folder_path):
            return None, None, None, []

        files = self.find_files(folder_path)
        all_times = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(self.extract_time_from_filename_parallel,
                                   files)
            all_times = [t for t in results if t is not None]

        if not all_times:
            return None, None, None, []

        all_times = sorted(set(all_times))

        if len(all_times) > 1:
            steps = [all_times[i + 1] - all_times[i]
                     for i in range(len(all_times) - 1)]
            most_common_step = max(set(steps), key=steps.count) if steps else 1
        else:
            most_common_step = 1

        return all_times[0], all_times[-1], most_common_step, all_times

    def extract_time_from_filename(self, filename, verbose=True):
        """Extract a time index from a filename using known patterns."""
        basename = os.path.basename(filename)

        patterns = [
            ('year_month_day',
             r'(\d{4})-(\d{2})-(\d{2})',
             lambda m: int(m.group(1)) * 10000
                       + int(m.group(2)) * 100
                       + int(m.group(3))),
            ('year_month',
             r'(\d{4})-(\d{2})',
             lambda m: int(m.group(1)) * 100 + int(m.group(2))),
            ('year', r'(\d{4})', lambda m: int(m.group(1))),
            ('pism_thk', r'thk[_\-]?(\d{4})', lambda m: int(m.group(1))),
            ('pism_surf', r'surf[_\-]?(\d{4})', lambda m: int(m.group(1))),
            ('pism_uvel', r'uvel[_\-]?(\d{4})', lambda m: int(m.group(1))),
            ('pism_vvel', r'vvel[_\-]?(\d{4})', lambda m: int(m.group(1))),
            ('pism_t', r't(\d{4})', lambda m: int(m.group(1))),
            ('prism_month_underscore',
             r'_(\d{2})_', lambda m: int(m.group(1))),
            ('prism_month_dot', r'_(\d{2})\.', lambda m: int(m.group(1))),
            ('four_digit', r'_(\d{4})\.', lambda m: int(m.group(1))),
            ('four_digit_simple', r'(\d{4})\.', lambda m: int(m.group(1))),
            ('three_digit', r'_(\d{3})\.', lambda m: int(m.group(1))),
            ('three_digit_simple', r'(\d{3})\.', lambda m: int(m.group(1))),
            ('two_digit', r'_(\d{2})\.', lambda m: int(m.group(1))),
            ('two_digit_simple', r'(\d{2})\.', lambda m: int(m.group(1))),
            ('any_number', r'(\d+)', lambda m: int(m.group(1))),
        ]

        for pattern_name, pattern, extractor in patterns:
            match = re.search(pattern, basename)
            if match:
                time_val = extractor(match)
                if verbose:
                    self._log(f"    Extracted time {time_val} from "
                              f"{basename} (pattern: {pattern_name})")
                return time_val

        if verbose:
            self._log(f"    Could not extract time from: {basename}")
        return None

    def extract_time_from_filename_parallel(self, filename):
        """Thread-safe wrapper for time extraction."""
        return self.extract_time_from_filename(filename, verbose=False)

    # ------------------------------------------------------------------ #
    # Raster reading
    # ------------------------------------------------------------------ #
    def read_raster_optimized(self, file_path, expected_shape=None):
        """Read band 1 of a raster, handling NoData and optional resampling."""
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1).astype(np.float32)

                if src.nodata is not None:
                    mask = data == src.nodata
                    if mask.any():
                        data[mask] = self.fill_value

                metadata = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'bounds': src.bounds,
                    'res': src.res,
                    'nodata': src.nodata,
                }

                if expected_shape and data.shape != expected_shape:
                    scale_y = expected_shape[0] / data.shape[0]
                    scale_x = expected_shape[1] / data.shape[1]
                    data = zoom(data, (scale_y, scale_x), order=0)

                return data, metadata, os.path.basename(file_path)

        except Exception as e:
            self._log(f"Error reading {file_path}: {e}")
            return None, None, file_path

    def validate_crs_consistency(self, folder_path):
        """Quick check that the first few rasters share the same CRS."""
        files = self.find_files(folder_path)[:5]
        if not files:
            return True

        crs_list = []
        for f in files:
            try:
                with rasterio.open(f) as src:
                    crs_list.append(src.crs)
            except Exception:
                pass

        if len(set(crs_list)) > 1:
            self._log("  WARNING: Mixed CRS detected! "
                      "Files may need reprojection.")
            return False
        return True

    # ------------------------------------------------------------------ #
    # Per-variable parallel processing
    # ------------------------------------------------------------------ #
    def process_variable_parallel(self, folder_path, var_name,
                                  start_pct, var_pct):
        """Process a variable folder with parallel file reading."""
        if not folder_path:
            return None, None, []

        self._log(f"\n  Processing {var_name} with "
                  f"{self.max_workers} workers...")

        files = self.find_files(folder_path)
        if not files:
            self._log(f"  No files found for {var_name}")
            return None, None, []

        self._log(f"  Found {len(files)} files for {var_name}")

        self.validate_crs_consistency(folder_path)

        auto_start, auto_end, auto_step, all_times = \
            self.auto_detect_time_range(folder_path)

        if all_times:
            self._log(f"  Auto-detected time range: "
                      f"{auto_start} to {auto_end}, step {auto_step}")
            preview = (f"{all_times[:20]}{'...' if len(all_times) > 20 else ''}")
            self._log(f"  Available times: {preview}")

        start = self.start_idx if self.start_idx is not None else auto_start
        end = self.end_idx if self.end_idx is not None else auto_end
        step = self.step if self.step is not None else auto_step

        if start is None or end is None:
            self._log("  Cannot determine time range. "
                      "Please specify Start and End times.")
            return None, None, []

        self._log(f"  Using time range: {start} to {end}, step {step}")

        valid_files = []
        valid_times = []
        for f in files:
            time_val = self.extract_time_from_filename(f, verbose=False)
            if time_val is not None and start <= time_val <= end:
                if (time_val - start) % step == 0:
                    valid_files.append(f)
                    valid_times.append(time_val)

        if not valid_files:
            self._log(f"  No files match time range {start}-{end} step {step}")
            return None, None, []

        self._log(f"  Selected {len(valid_files)} files for {var_name}")
        self._log(f"  Time range: {valid_times[0]} to {valid_times[-1]}")

        # Peek at the first file to get grid dimensions
        result = self.read_raster_optimized(valid_files[0])
        if result[0] is None:
            self._log(f"  ERROR: Cannot read first file: {valid_files[0]}")
            return None, None, []

        first_data, metadata, _ = result
        height, width = first_data.shape
        del first_data

        self._log(f"  Grid dimensions: {height} x {width}")
        res_x = abs(metadata['res'][0]) if metadata.get('res') else 'unknown'
        self._log(f"  Cell size: {res_x} m")

        data_array = np.zeros((len(valid_files), height, width),
                              dtype=np.float32)

        batch_size = min(self.max_workers * 2, len(valid_files), 50)
        self._log(f"  Reading {len(valid_files)} files in batches of "
                  f"{batch_size}...")

        for batch_start in range(0, len(valid_files), batch_size):
            if self.is_cancelled():
                raise InterruptedError("Conversion cancelled")

            batch_end = min(batch_start + batch_size, len(valid_files))
            batch_files = valid_files[batch_start:batch_end]

            # Map future -> (file_path, local_index) so we can write back
            # each result without a linear search.
            futures = {}
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for local_idx, f in enumerate(batch_files):
                    future = executor.submit(
                        self.read_raster_optimized, f, (height, width)
                    )
                    futures[future] = (f, local_idx)

                completed = 0
                for future in as_completed(futures):
                    if self.is_cancelled():
                        raise InterruptedError("Conversion cancelled")

                    file_path, local_idx = futures[future]
                    try:
                        data, _, _ = future.result(timeout=30)
                    except TimeoutError:
                        self._log(f"  Timeout reading file: "
                                  f"{os.path.basename(file_path)}")
                        continue
                    except InterruptedError:
                        raise
                    except Exception as e:
                        self._log(f"  Error reading "
                                  f"{os.path.basename(file_path)}: {e}")
                        continue

                    if data is not None:
                        data_array[batch_start + local_idx] = data

                    completed += 1
                    if self.progress_callback and completed % max(
                            1, len(batch_files) // 5) == 0:
                        percent = start_pct + int(
                            (batch_start + completed) / len(valid_files)
                            * var_pct
                        )
                        if percent - self.last_progress >= 5:
                            self.progress_callback(
                                percent,
                                f"Loading {var_name}: "
                                f"{batch_start + completed}/"
                                f"{len(valid_files)}",
                            )
                            self.last_progress = percent

        return data_array, metadata, valid_times

    # ------------------------------------------------------------------ #
    # Main conversion
    # ------------------------------------------------------------------ #
    def convert(self):
        """Main conversion method with optimized parallel processing."""
        start_time = time.time()
        self.last_progress = 0

        self._log("\n" + "=" * 80)
        self._log("RASTER TO NETCDF CONVERSION "
                  "(Optimized Parallel Processing)")
        self._log("=" * 80)
        self._log(f"Thickness folder: {self.thick_folder}")
        self._log(f"Output file: {self.output_nc}")
        self._log(f"Parallel workers: {self.max_workers}")

        if not os.path.exists(self.thick_folder):
            raise ValueError(f"Thickness folder does not exist: "
                             f"{self.thick_folder}")

        self._log("\nAuto-detecting time information from filenames...")
        auto_start, auto_end, auto_step, all_times = \
            self.auto_detect_time_range(self.thick_folder)

        if all_times:
            self._log(f"  Detected {len(all_times)} unique time values")
            self._log(f"  Time range: {auto_start} to {auto_end}")
            self._log(f"  Step: {auto_step}")
        else:
            self._log("  Could not auto-detect time information "
                      "from filenames")

        if self.progress_callback:
            self.progress_callback(5, "Starting conversion...")

        variable_specs = [
            (self.thick_folder, 'thk'),
            (self.surface_folder, 'surf'),
            (self.uvel_folder, 'uvel'),
            (self.vvel_folder, 'vvel'),
        ]
        variables = [(f, v) for f, v in variable_specs if f is not None]

        var_pct = 80 // len(variables)
        results = {}

        for i, (folder, var_name) in enumerate(variables):
            if not (folder and os.path.exists(folder)):
                if folder:
                    self._log(f"\n  Skipping {var_name} - folder does not "
                              f"exist: {folder}")
                continue

            self._log("\n" + "=" * 50)
            self._log(f"Processing {var_name.upper()}...")
            self._log("=" * 50)

            progress_base = 10 + i * var_pct
            if self.progress_callback and i > 0:
                if progress_base - self.last_progress >= 5:
                    self.progress_callback(progress_base,
                                           f"Processing {var_name}...")
                    self.last_progress = progress_base

            data, metadata, times = self.process_variable_parallel(
                folder, var_name, progress_base, var_pct
            )
            if data is not None:
                results[var_name] = {
                    'data': data, 'metadata': metadata, 'times': times,
                }
            else:
                self._log(f"  WARNING: No valid data found for {var_name}")

        if 'thk' not in results:
            raise ValueError(f"No thickness data found in {self.thick_folder}")

        if self.progress_callback:
            self.progress_callback(90, "Creating NetCDF file...")

        thick_data = results['thk']['data']
        thick_times = results['thk']['times']
        transform = results['thk']['metadata']['transform']

        n_time = len(thick_times)
        height = thick_data.shape[1]
        width = thick_data.shape[2]

        self._log("\nOutput dimensions:")
        self._log(f"  Time steps: {n_time}")
        self._log(f"  Height: {height}")
        self._log(f"  Width: {width}")

        x_coords = (np.arange(width, dtype=np.float64) * transform[0]
                    + transform[2] + transform[0] / 2)
        y_coords = (np.arange(height, dtype=np.float64) * transform[4]
                    + transform[5] + transform[4] / 2)

        crs_str = (str(results['thk']['metadata']['crs'])
                   if results['thk']['metadata']['crs'] else "Unknown")

        ds = xr.Dataset()

        var_attrs = {
            'thk': {
                'units': 'm',
                'long_name': 'ice thickness',
                'standard_name': 'land_ice_thickness',
            },
            'surf': {
                'units': 'm',
                'long_name': 'ice surface elevation',
                'standard_name': 'surface_altitude',
            },
            'uvel': {
                'units': 'm a-1',
                'long_name': 'ice velocity in x-direction',
                'standard_name': 'land_ice_surface_x_velocity',
            },
            'vvel': {
                'units': 'm a-1',
                'long_name': 'ice velocity in y-direction',
                'standard_name': 'land_ice_surface_y_velocity',
            },
        }

        for var_name, attrs in var_attrs.items():
            if var_name not in results:
                continue
            ds[var_name] = (['time', 'y', 'x'], results[var_name]['data'])
            ds[var_name].attrs = dict(attrs)
            ds[var_name].attrs['_FillValue'] = self.fill_value

        ds.coords['x'] = x_coords
        ds.coords['y'] = y_coords
        ds.coords['time'] = thick_times

        ds.x.attrs = {'long_name': 'Easting', 'units': 'meter'}
        ds.y.attrs = {'long_name': 'Northing', 'units': 'meter'}
        ds.time.attrs = {'long_name': 'Time', 'units': 'years'}

        ds.attrs = {
            'title': 'GLIMAT Converted Dataset '
                     '(Optimized Parallel Processing)',
            'Conventions': 'CF-1.8',
            'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            'crs': crs_str,
            'note': 'Original resolution preserved - no resampling applied',
            'time_detection': 'Auto-detected from filenames',
            'parallel_workers': self.max_workers,
        }

        self._log(f"\nSaving to {self.output_nc}...")

        try:
            if self.use_compression:
                # Guard against 0-sized chunks for very small rasters
                chunk_y = max(1, height // 4)
                chunk_x = max(1, width // 4)
                encoding = {
                    var: {
                        'zlib': True,
                        'complevel': 4,
                        'chunksizes': (1, chunk_y, chunk_x),
                    }
                    for var in ds.data_vars
                }
                ds.to_netcdf(self.output_nc, engine='netcdf4',
                             encoding=encoding)
            else:
                ds.to_netcdf(self.output_nc, engine='netcdf4')
        except Exception as e:
            self._log(f"  netcdf4 failed: {e}, falling back to scipy...")
            ds.to_netcdf(self.output_nc, engine='scipy')

        elapsed_time = time.time() - start_time

        if not os.path.exists(self.output_nc):
            raise RuntimeError("Failed to create NetCDF file")

        file_size = os.path.getsize(self.output_nc) / (1024 * 1024)
        self._log("\n" + "=" * 80)
        self._log("CONVERSION COMPLETED SUCCESSFULLY!")
        self._log("=" * 80)
        self._log(f"Output file: {self.output_nc}")
        self._log(f"File size: {file_size:.2f} MB")
        self._log(f"Time steps: {n_time}")
        self._log(f"Time range: {thick_times[0]} to {thick_times[-1]}")
        self._log(f"Grid size: {width} x {height}")
        self._log(f"Total runtime: {elapsed_time:.2f} seconds")
        self._log(f"Speedup: Optimized parallel processing with "
                  f"{self.max_workers} workers")
        self._log("=" * 80)

        if self.progress_callback:
            self.progress_callback(100, "Conversion complete!")

        return self.output_nc


# ---------------------------------------------------------------------- #
# Optional: batch processing of multiple converters
# ---------------------------------------------------------------------- #
class BatchRasterConverter:
    """Process multiple converters in parallel."""

    def __init__(self, max_workers=None):
        self.max_workers = max_workers or mp.cpu_count()

    def convert_batch(self, converter_configs):
        """
        Convert multiple raster datasets in parallel.

        Parameters
        ----------
        converter_configs : list of dict
            Each dict contains the keyword arguments for
            RasterToNetCDFConverter.
        """
        def run_converter(config):
            return RasterToNetCDFConverter(**config).convert()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(run_converter, cfg)
                       for cfg in converter_configs]
            results = [f.result() for f in futures]

        return results


# ---------------------------------------------------------------------- #
# Example usage
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    converter = RasterToNetCDFConverter(
        thick_folder="/path/to/thickness/tifs",
        output_nc="/path/to/output.nc",
        surface_folder="/path/to/surface/tifs",   # optional
        uvel_folder="/path/to/uvel/tifs",         # optional
        vvel_folder="/path/to/vvel/tifs",         # optional
        start_idx=0,       # optional - auto-detected if None
        end_idx=100,       # optional - auto-detected if None
        step=1,            # optional - auto-detected if None
        use_compression=True,
        max_workers=4,     # optional - defaults to CPU count - 1
    )

    output_file = converter.convert()
    print(f"Conversion complete: {output_file}")