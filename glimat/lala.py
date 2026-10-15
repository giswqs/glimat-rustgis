"""
/*********************************************************************************
* LALA - Likelihood of Accordant Lineations Analysis                             * 
* ===================================================                            *
* LALA provides a statistically rigorous measure of the log-likelihood of a      * 
* supplied simulation through comparison of simulation output with both the      *
* location and direction of observed lineations.                                 *
*                                                                                * 
* Revised based on:                                                              *
*    Archer et al. (2023). Assessing ice sheet models against the landform       *
*    record: The Likelihood of Accordant Lineations Analysis (LALA) tool.        * 
*    Earth Surface Processes and Landforms 48, 2754-2771.                        *
*    https://doi.org/10.1002/esp.5658                                            *
*                                                                                *   
* This file is part of GLIMAT Plugin.                                            *
*                                                                                *
* Author: Yingkui Li                                                             *
* Department of Geography & Sustainability                                       *
* University of Tennessee, Knoxville, TN 37996                                   * 
* Email: yli32@utk.edu                                                           *
* Start date: 06/26/2026                                                         *
* Finalized date: 09/17/2026                                                     *
* Copyright(C): 2026 Yingkui Li                                                  * 
*                                                                                *
* Version: 1.0.0                                                                 *
*********************************************************************************/
"""
import warnings
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import vonmises as vm
from shapely.geometry import Point, box

matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class LALA_Analyzer:
    """
    Likelihood of Accordant Lineations Analysis (LALA).

    Compares simulated ice-flow direction and extent against mapped
    lineations using a marked inhomogeneous Poisson point process model.
    """

    def __init__(self, pism_dataset, shapefile_path, output_subdirs, prefix="",
                 time_indices=None, thk_threshold=10,
                 velocity_threshold=1.0,
                 grounded_ice=2,
                 kappa=10, p_value=0.01,
                 u_var='uvelbase', v_var='vvelbase',
                 thk_var='thk', mask_var='mask',
                 progress_callback=None, cancel_checker=None):
        self.pism_dataset = pism_dataset
        self.thk_threshold = thk_threshold
        self.velocity_threshold = velocity_threshold
        self.grounded_ice = grounded_ice
        self.output_subdirs = output_subdirs
        self.shapefile_path = shapefile_path
        self.prefix = prefix
        self.time_indices = time_indices
        self.console = None
        self.crs = None

        self.u_var = u_var
        self.v_var = v_var
        self.thk_var = thk_var
        self.mask_var = mask_var
        self.kappa = kappa
        self.p_value = p_value

        self.progress_callback = progress_callback
        self.cancel_checker = cancel_checker
        self._is_cancelled = False

    # ====================================================================== #
    # Logging / cancellation / progress
    # ====================================================================== #
    def set_console(self, console_callback):
        """Set a console callback for debug messages."""
        self.console = console_callback

    def _log(self, message):
        if self.console:
            self.console(message)
        print(message)

    def _progress(self, percent, message):
        """Safe progress callback (no-op if no callback was supplied)."""
        if self.progress_callback:
            self.progress_callback(percent, message)

    def cancel(self):
        """Cancel the analysis."""
        self._is_cancelled = True
        self._log("\n  Cancellation requested for LALA analysis...")

    def is_cancelled(self):
        """Check whether the analysis has been cancelled."""
        if self.cancel_checker:
            return self.cancel_checker()
        return self._is_cancelled

    # ====================================================================== #
    # Initialization helpers
    # ====================================================================== #
    def _setup_coordinates(self):
        """Set up coordinate system and grid shape from the NetCDF dataset."""
        if 'x' in self.pism_dataset.dims and 'y' in self.pism_dataset.dims:
            self.x = self.pism_dataset.x.values
            self.y = self.pism_dataset.y.values
            self.netcdf_bounds = [
                float(self.x.min()), float(self.y.min()),
                float(self.x.max()), float(self.y.max()),
            ]
            self.model_shape = (len(self.y), len(self.x))
        else:
            first_var = list(self.pism_dataset.data_vars.values())[0]
            self.model_shape = first_var.shape[-2:]
            self.netcdf_bounds = [0, 0, self.model_shape[1], self.model_shape[0]]

        self._log(f"\nNetCDF bounds: {self.netcdf_bounds}")
        self._log(f"Model shape: {self.model_shape}")

    def _load_field_observations(self, shapefile_path):
        """Load field observations and derive point flow-lineation data."""
        self._log("\nLoading and deriving point flow-lineation data...")
        self.field_gdf = self.load_field_observations(shapefile_path)

        if self.field_gdf.crs is not None:
            self.crs = self.field_gdf.crs

        (self.lineation_x,
         self.lineation_y,
         self.lineation_direction) = \
            self.extract_line_midpoints_and_directions(self.field_gdf)

        number_lineations = len(self.lineation_x)
        self._log(f"Number of lineations: {number_lineations}")

        # Create point shapefile with direction attribute
        self._log("\nCreating point shapefile with lineation midpoints...")

        points = [Point(self.lineation_x[i], self.lineation_y[i])
                  for i in range(number_lineations)]

        points_gdf = gpd.GeoDataFrame(
            {
                'Lineation_ID': np.arange(number_lineations),
                'Direction': self.lineation_direction,
                'X_Coord': self.lineation_x,
                'Y_Coord': self.lineation_y,
                'geometry': points,
            },
            crs=self.crs,
        )

        point_path = (self.output_subdirs['shape_files']
                      / f"{self.prefix}_lineation_midpoints.shp")
        points_gdf.to_file(point_path)
        self._log(f"    Lineation midpoint point file saved to: {point_path}")
        self._log(f"    Number of points: {len(points_gdf)}")
        self._log(f"    Direction range: "
                  f"{self.lineation_direction.min():.1f}° to "
                  f"{self.lineation_direction.max():.1f}°")

        csv_path = (self.output_subdirs['csv']
                    / f"{self.prefix}_lineation_midpoints.csv")
        points_gdf.drop('geometry', axis=1).to_csv(csv_path, index=False)
        self._log(f"    Lineation midpoint CSV saved to: {csv_path}")

    def _setup_time_indices(self):
        """Set up time indices for analysis."""
        if 'time' in self.pism_dataset.dims:
            if self.time_indices is None:
                self.time_indices = list(range(len(self.pism_dataset.time)))
            else:
                self.time_indices = [i for i in self.time_indices
                                     if i < len(self.pism_dataset.time)]
            self._log(f"\nTime steps available: "
                      f"{len(self.pism_dataset.time)}")
            self._log(f"Analyzing {len(self.time_indices)} time indices")
        else:
            self.time_indices = [0]

    # ====================================================================== #
    # Core LALA primitives
    # ====================================================================== #
    def calc_angles(self, uvel, vvel):
        """Calculate the modeled angles."""
        return np.arctan2(vvel, uvel)

    def possible_formation(self, thickness_condition, velocity_condition,
                           grounded_ice, thickness, velocity, mask):
        """Find the locations that meet the lineation formation criteria."""
        thickness_binary = thickness > thickness_condition
        velocity_binary = velocity > velocity_condition
        mask_binary = mask == grounded_ice
        formation_possible_study = (thickness_binary
                                    & velocity_binary
                                    & mask_binary)
        A = np.sum(formation_possible_study)
        return formation_possible_study, A

    def likelihood_location(self, lam, A, lam_star, formation_possible_pre):
        """Likelihood of lineation formation at a given location."""
        return lam * A + lam_star * formation_possible_pre

    def nu_t(self, t, row_idx, col_idx, lam, direction,
             lineation_direction, flow, kappa):
        """
        Directional likelihood at a single (time, row, col) point.

        Notes
        -----
        `row_idx` and `col_idx` correspond to the y- and x-indices of the
        model grid (NumPy array order). The direction is evaluated
        symmetrically to account for the ±180° ambiguity of lineations.
        """
        return lam * (
            0.5 * vm.pdf(direction[t, row_idx, col_idx]
                         - lineation_direction[flow], kappa)
            + 0.5 * vm.pdf(direction[t, row_idx, col_idx]
                           - lineation_direction[flow] + np.pi, kappa)
        )

    # ====================================================================== #
    # Data loading / coordinate conversion
    # ====================================================================== #
    def load_field_observations(self, shapefile_path, target_bounds=None):
        """Load field observations from line features."""
        self._log(f"Loading field line features: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)

        self._log(f"  Original shapefile CRS: {gdf.crs}")
        self._log(f"  Original bounds: {gdf.total_bounds}")
        self._log(f"  Number of features: {len(gdf)}")

        if self.crs is not None and gdf.crs != self.crs:
            self._log("  Reprojecting shapefile to target CRS")
            gdf = gdf.to_crs(self.crs)

        if target_bounds is not None:
            clip_box = box(*target_bounds)
            original_count = len(gdf)
            gdf = gdf.clip(clip_box)
            self._log(f"  Features before clipping: {original_count}, "
                      f"after clipping: {len(gdf)}")

        return gdf

    def extract_line_midpoints_and_directions(self, gdf):
        """
        Extract midpoint and direction from line features.

        Direction is reported as an azimuth (0° = north, clockwise) via
        `arctan2(dx, dy)`. Note this differs from `calc_angles`, which uses
        the standard math convention (`arctan2(v, u)`); ensure both are
        consistent with what the LALA likelihood expects.
        """
        x_coords, y_coords, directions = [], [], []

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

                coords = list(line.coords)
                start, end = coords[0], coords[-1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                if dx == 0 and dy == 0:
                    continue

                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                angle_rad = np.arctan2(dx, dy)
                angle_deg = np.degrees(angle_rad) % 360

                x_coords.append(mid_x)
                y_coords.append(mid_y)
                directions.append(angle_deg)

        return (np.array(x_coords),
                np.array(y_coords),
                np.array(directions))

    def coords_to_grid_indices(self, x_coords, y_coords, grid_x, grid_y,
                               flip_y=True):
        """Convert geographic coordinates to grid indices."""
        x_indices = np.zeros(len(x_coords), dtype=int)
        y_indices = np.zeros(len(y_coords), dtype=int)

        for i in range(len(x_coords)):
            x_indices[i] = np.argmin(np.abs(grid_x - x_coords[i]))

        if flip_y:
            y_values = grid_y[::-1]
            for i in range(len(y_coords)):
                y_indices[i] = np.argmin(np.abs(y_values - y_coords[i]))
        else:
            for i in range(len(y_coords)):
                y_indices[i] = np.argmin(np.abs(grid_y - y_coords[i]))

        return x_indices, y_indices

    # ====================================================================== #
    # Main analysis
    # ====================================================================== #
    def runLALA(self):
        """Run the LALA analysis."""
        try:
            self._setup_coordinates()
            self._load_field_observations(self.shapefile_path)
            self._setup_time_indices()

            self._log("  LALA Analysis initialized")
            self._log(f"  Time steps to analyze: {len(self.time_indices)}")
            self._log(f"  Velocity variables: U='{self.u_var}', "
                      f"V='{self.v_var}'")
            self._log(f"  Velocity threshold: {self.velocity_threshold} m/a")

            uvel, vvel, velocity, thickness, mask = self._load_model_data()

            self._progress(40, "Calculating directions...")
            direction = self.calc_angles(uvel, vvel)

            lineation_x_idx, lineation_y_idx = self.coords_to_grid_indices(
                self.lineation_x, self.lineation_y,
                self.x, self.y, flip_y=True,
            )

            (lam, lam_star,
             likelihood_location,
             formation_possible_study,
             A) = self._calculate_lambdas_and_possible(thickness, velocity, mask)

            if np.isinf(lam) or np.isnan(lam):
                self._log("WARNING: lambda is invalid. Setting to 0.")
                lam = 0.0

            (nu, nu_by_time,
             times_possible, time_counts) = self._calculate_nu_scores(
                direction, lineation_x_idx, lineation_y_idx,
                lam, lam_star, formation_possible_study,
            )

            if nu is None:  # cancellation
                return None

            self._save_results(
                nu, nu_by_time, times_possible, time_counts,
                lineation_x_idx, lineation_y_idx,
                lam, lam_star, likelihood_location, A,
            )

            valid_lineations = np.where(time_counts > 0)[0]
            if len(valid_lineations) > 0:
                self._generate_visualizations(
                    nu_by_time, times_possible, time_counts,
                    lineation_x_idx, lineation_y_idx,
                )
            else:
                self._log("\nWARNING: No valid lineations with "
                          "possible formation times.")
                self._log("Skipping visualizations.")

            results = self._compile_results(
                nu, lineation_x_idx, lineation_y_idx,
                lam, lam_star, likelihood_location, A,
                time_counts, nu_by_time, times_possible,
                direction.shape[0],
            )

            self._progress(100, "Analysis complete!")
            self._log("Analysis complete!")
            return results

        except Exception as e:
            import traceback
            self._log(f"Error: {e}\n\n{traceback.format_exc()}")
            return None

    def _load_model_data(self):
        """Load model variables from the dataset."""
        self._progress(30, "Loading model data...")
        uvel = self.pism_dataset[self.u_var]
        vvel = self.pism_dataset[self.v_var]
        velocity = np.sqrt(uvel ** 2 + vvel ** 2)
        thickness = self.pism_dataset[self.thk_var]
        mask = self.pism_dataset[self.mask_var]
        self._progress(35, "Model data loaded")
        return uvel, vvel, velocity, thickness, mask

    def _calculate_lambdas_and_possible(self, thickness, velocity, mask):
        """Compute lambda, lambda_star, and formation-possible areas."""
        self._progress(45, "Calculating formation possible...")

        formation_possible_study, A = self.possible_formation(
            self.thk_threshold, self.velocity_threshold, self.grounded_ice,
            thickness, velocity, mask,
        )

        formation_possible_pre = velocity.shape[1] * velocity.shape[2]

        if A == 0:
            self._log("\nWARNING: No areas meet the formation conditions!")
            self._log(f"  Thickness threshold: {self.thk_threshold}")
            self._log(f"  Velocity threshold: {self.velocity_threshold}")
            self._log(f"  Grounded ice mask value: {self.grounded_ice}")
            self._log("  This could mean:")
            self._log("    1. The thresholds are too strict")
            self._log("    2. The mask values don't match (check mask values)")
            self._log("    3. The model data has no grounded ice")
            A = 1e-10
            self._log(f"  Setting A to {A} to avoid division by zero")

        # LALA mixing weights
        lam_star = (self.p_value * 1) / formation_possible_pre
        lam = (1 - formation_possible_pre * lam_star) / A

        if np.isinf(lam) or np.isnan(lam):
            self._log(f"WARNING: lambda is {lam}, setting to 0")
            lam = 0.0

        likelihood_location = self.likelihood_location(
            lam, A, lam_star, formation_possible_pre
        )

        self._log(f"lambda: {lam:.6f}, lambda_star: {lam_star:.6f}")
        self._log(f"Likelihood location: {likelihood_location:.6f}")

        return lam, lam_star, likelihood_location, formation_possible_study, A

    def _calculate_nu_scores(self, direction, lineation_x_idx,
                             lineation_y_idx, lam, lam_star,
                             formation_possible_study):
        """Compute nu scores for all lineations across all times."""
        self._progress(50, "Calculating nu scores...")

        n_lineations = len(self.lineation_x)
        n_times = direction.shape[0]

        nu = np.zeros(n_lineations, dtype=float)
        nu_by_time = np.full((n_times, n_lineations), np.nan, dtype=float)
        times_possible = np.zeros((n_times, n_lineations), dtype=bool)
        time_counts = np.zeros(n_lineations, dtype=int)

        for i in range(n_lineations):
            if self.is_cancelled():
                self._log("\n  Analysis cancelled by user.")
                return None, None, None, None

            xi = lineation_x_idx[i]
            yi = lineation_y_idx[i]

            possible = formation_possible_study[:, yi, xi]
            times_possible[:, i] = possible
            time_counts[i] = np.sum(possible)

            if time_counts[i] == 0:
                nu[i] = lam_star / (2 * np.pi)
            else:
                nu_i = 0.0
                for t in range(n_times):
                    if possible[t]:
                        nu_t_val = self.nu_t(
                            t, yi, xi, lam, direction,
                            self.lineation_direction, i, self.kappa,
                        )
                        nu_by_time[t, i] = nu_t_val
                        nu_i += nu_t_val
                nu[i] = nu_i

            if i % 10 == 0:
                progress = 50 + int((i / n_lineations) * 30)
                self._progress(progress,
                               f"Processing lineation {i + 1}/{n_lineations}")

        return nu, nu_by_time, times_possible, time_counts

    def _calculate_final_nu(self, nu, time_counts, nu_by_time, lam_star):
        """Final nu values, including lam_star for no-time lineations."""
        n_lineations = len(nu)
        nu_total = np.zeros(n_lineations, dtype=float)
        for i in range(n_lineations):
            if time_counts[i] > 0:
                nu_total[i] = np.nansum(nu_by_time[:, i])
            else:
                nu_total[i] = lam_star / (2 * np.pi)
        return nu_total

    # ====================================================================== #
    # Saving
    # ====================================================================== #
    def _save_results(self, nu, nu_by_time, times_possible, time_counts,
                      lineation_x_idx, lineation_y_idx,
                      lam, lam_star, likelihood_location, A):
        """Save all results to CSV files."""
        self._progress(80, "Saving results...")

        n_lineations = len(self.lineation_x)

        # 1. Main scores
        df_scores = pd.DataFrame({
            'Flowset': np.arange(n_lineations),
            'Score': nu,
            'Direction_Field': self.lineation_direction,
            'X_Index': lineation_x_idx,
            'Y_Index': lineation_y_idx,
            'X_Coord': self.lineation_x,
            'Y_Coord': self.lineation_y,
        })
        csv_path = self.output_subdirs['csv'] / f"{self.prefix}_flowset_scores.csv"
        df_scores.to_csv(csv_path, index=False)

        # 2. Nu by time
        df_nu_time = pd.DataFrame(nu_by_time)
        df_nu_time.columns = [f'Lineation_{i}' for i in range(n_lineations)]
        csv_path = self.output_subdirs['csv'] / f"{self.prefix}_nu_by_time.csv"
        df_nu_time.to_csv(csv_path, index_label='Time_Step')

        # 3. Summary
        valid_lineations_mask = time_counts > 0
        nu_time_avg = np.full(n_lineations, np.nan, dtype=float)
        max_time_indices = np.full(n_lineations, -1, dtype=int)
        max_nu_values = np.full(n_lineations, np.nan, dtype=float)

        for i in range(n_lineations):
            if time_counts[i] > 0:
                nu_time_avg[i] = np.nanmean(nu_by_time[:, i])
                max_idx = np.nanargmax(nu_by_time[:, i])
                max_time_indices[i] = max_idx
                max_nu_values[i] = nu_by_time[max_idx, i]

        summary_df = pd.DataFrame({
            'Lineation_ID': np.arange(n_lineations),
            'Time_Count': time_counts,
            'Time_Avg_Nu': nu_time_avg,
            'Max_Nu': max_nu_values,
            'Max_Time_Index': max_time_indices,
            'X_Index': lineation_x_idx,
            'Y_Index': lineation_y_idx,
            'Direction': self.lineation_direction,
        })
        csv_path = (self.output_subdirs['csv']
                    / f"{self.prefix}_lineation_summary.csv")
        summary_df.to_csv(csv_path, index=False)

        # 4. Final scores
        nu_total = self._calculate_final_nu(nu, time_counts, nu_by_time, lam_star)
        flowset_final = pd.DataFrame({
            'Flowset': np.arange(n_lineations),
            'Score': nu_total,
            'Has_Possible_Times': valid_lineations_mask,
            'Time_Count': time_counts,
            'Direction_Field': self.lineation_direction,
            'X_Index': lineation_x_idx,
            'Y_Index': lineation_y_idx,
            'X_Coord': self.lineation_x,
            'Y_Coord': self.lineation_y,
        })
        csv_path = (self.output_subdirs['csv']
                    / f"{self.prefix}_flowset_scores_final.csv")
        flowset_final.to_csv(csv_path, index=False)

        self._log(f"Results saved to: {csv_path}")
        self._progress(85, "Results saved")

    # ====================================================================== #
    # Visualizations
    # ====================================================================== #
    def _generate_visualizations(self, nu_by_time, times_possible,
                                 time_counts, lineation_x_idx, lineation_y_idx):
        """Generate all visualizations."""
        self._progress(90, "Generating visualizations...")

        self._log("\n" + "=" * 60)
        self._log("GENERATING VISUALIZATIONS")
        self._log("=" * 60)
        self._log(f"nu_by_time shape: {nu_by_time.shape}")
        self._log(f"times_possible shape: {times_possible.shape}")
        self._log(f"time_counts shape: {time_counts.shape}")

        n_times = nu_by_time.shape[0]
        n_lineations = len(time_counts)
        valid_lineations = np.where(time_counts > 0)[0]

        self._log(f"Valid lineations: {len(valid_lineations)} "
                  f"out of {n_lineations}")

        plots_dir = self.output_subdirs['plots']
        self._log(f"Plots directory: {plots_dir}")
        if not plots_dir.exists():
            self._log("Creating plots directory...")
            plots_dir.mkdir(parents=True, exist_ok=True)

        if len(valid_lineations) == 0:
            self._log("No lineations have valid formation times. "
                      "Cannot create visualizations.")
            self._log("=" * 60 + "\n")
            return

        saved_files = []

        self._log("\nCreating heatmap...")
        path = self._create_heatmap(nu_by_time, valid_lineations)
        if path:
            saved_files.append(path)

        self._log("\nCreating time series plot...")
        path = self._create_time_series_plot(
            nu_by_time, times_possible, valid_lineations, n_times
        )
        if path:
            saved_files.append(path)

        self._log("\nCreating histogram...")
        path = self._create_time_count_histogram(time_counts, valid_lineations)
        if path:
            saved_files.append(path)

        self._log("\nCreating spatial distribution plot...")
        path = self._create_spatial_distribution_plot(
            self.lineation_x, self.lineation_y, valid_lineations
        )
        if path:
            saved_files.append(path)

        if saved_files:
            self._log(f"\nSaved {len(saved_files)} visualization files:")
            for f in saved_files:
                self._log(f"  - {f}")
        else:
            self._log("\nNo visualization files were saved.")

        self._log("=" * 60 + "\n")
        self._progress(95, "Visualizations generated")

    def _save_figure(self, fig, path):
        """Save a figure and return the path (or None on failure)."""
        self._log(f"    Saving to: {path}")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        if path.exists():
            self._log(f"    Saved: {path.name} "
                      f"({path.stat().st_size / 1024:.1f} KB)")
            return str(path)
        self._log(f"    File was not created: {path}")
        return None

    def _create_heatmap(self, nu_by_time, valid_lineations):
        """Heatmap of nu scores over time."""
        try:
            self._log(f"  Creating heatmap with "
                      f"{len(valid_lineations)} valid lineations...")

            plot_data = nu_by_time[:, valid_lineations]
            if np.all(np.isnan(plot_data)):
                self._log("    All data is NaN, skipping heatmap")
                return None

            fig, ax = plt.subplots(figsize=(14, 8))
            im = ax.imshow(plot_data.T, aspect='auto', cmap='viridis',
                           origin='lower', interpolation='nearest')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Lineation ID (subset)')
            ax.set_title(f'Nu Scores by Time and Lineation '
                         f'({len(valid_lineations)} valid lineations)')
            plt.colorbar(im, ax=ax, label='Nu Score')
            plt.tight_layout()

            plt_path = (self.output_subdirs['plots']
                        / f"{self.prefix}_nu_heatmap.png")
            return self._save_figure(fig, plt_path)

        except Exception as e:
            self._log(f"    Error creating heatmap: {e}")
            return None

    def _create_time_series_plot(self, nu_by_time, times_possible,
                                 valid_lineations, n_times):
        """Time-series plot for valid lineations."""
        try:
            n_plot = len(valid_lineations)
            if n_plot == 0:
                return None

            self._log(f"  Creating time series plot for "
                      f"{n_plot} lineations...")

            fig, ax = plt.subplots(figsize=(12, 6))
            plotted = False

            for idx in range(n_plot):
                i = valid_lineations[idx]
                valid_mask = times_possible[:, i]
                if np.sum(valid_mask) > 0:
                    ax.scatter(np.arange(n_times)[valid_mask],
                               nu_by_time[valid_mask, i], marker='o')
                    plotted = True

            if not plotted:
                self._log("    No valid data points to plot")
                plt.close(fig)
                return None

            ax.set_xlabel('Time Step')
            ax.set_ylabel('Nu Score')
            ax.set_title(f'Nu Time Series for {n_plot} Valid Lineations')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plt_path = (self.output_subdirs['plots']
                        / f"{self.prefix}_nu_time_series_subset.png")
            return self._save_figure(fig, plt_path)

        except Exception as e:
            self._log(f"    Error creating time series: {e}")
            return None

    def _create_time_count_histogram(self, time_counts, valid_lineations):
        """Histogram of time counts."""
        try:
            if len(valid_lineations) == 0:
                return None

            self._log("  Creating time count histogram...")

            valid_counts = time_counts[valid_lineations]
            if len(valid_counts) == 0 or np.all(valid_counts == 0):
                self._log("    No valid time counts to plot")
                return None

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(valid_counts,
                    bins=min(30, len(np.unique(valid_counts))),
                    edgecolor='black', alpha=0.7)
            ax.set_xlabel('Number of Possible Time Steps per Lineation')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Lineations by Time Count')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plt_path = (self.output_subdirs['plots']
                        / f"{self.prefix}_time_count_distribution.png")
            return self._save_figure(fig, plt_path)

        except Exception as e:
            self._log(f"    Error creating histogram: {e}")
            return None

    def _create_spatial_distribution_plot(self, lineation_x, lineation_y,
                                          valid_lineations):
        """Spatial distribution of valid vs. invalid lineations."""
        try:
            self._log("  Creating spatial distribution plot...")

            fig, ax = plt.subplots(figsize=(10, 8))

            n_lineations = len(lineation_x)
            invalid_mask = np.ones(n_lineations, dtype=bool)
            invalid_mask[valid_lineations] = False

            invalid_x = lineation_x[invalid_mask]
            invalid_y = lineation_y[invalid_mask]
            if len(invalid_x) > 0:
                ax.scatter(invalid_x, invalid_y, c='red', s=5,
                           alpha=0.3, label='No possible times')

            if len(valid_lineations) > 0:
                valid_x = lineation_x[valid_lineations]
                valid_y = lineation_y[valid_lineations]
                ax.scatter(valid_x, valid_y, c='blue', s=10,
                           alpha=0.7, label='Has possible times')

            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Y (meters)')
            ax.set_title('Spatial Distribution of Lineations with/without '
                         'Possible Formation Times')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            plt_path = (self.output_subdirs['plots']
                        / f"{self.prefix}_lineation_spatial_distribution.png")
            return self._save_figure(fig, plt_path)

        except Exception as e:
            self._log(f"    Error creating spatial plot: {e}")
            return None

    # ====================================================================== #
    # Final result compilation
    # ====================================================================== #
    def _compile_results(self, nu, lineation_x_idx, lineation_y_idx,
                         lam, lam_star, likelihood_location, A,
                         time_counts, nu_by_time, times_possible, n_times):
        """Compile the final results dictionary."""
        n_lineations = len(nu)
        valid_lineations_mask = time_counts > 0
        valid_indices = np.where(valid_lineations_mask)[0]

        if len(valid_indices) > 0:
            nu_total = self._calculate_final_nu(
                nu, time_counts, nu_by_time, lam_star
            )
            valid_nu = nu_total[valid_lineations_mask]
            final_likelihood = np.sum(np.log(valid_nu)) - likelihood_location
        else:
            final_likelihood = np.nan

        self._log("\n" + "=" * 60)
        if len(valid_indices) > 0:
            self._log(f"Final likelihood "
                      f"(using {len(valid_indices)} valid lineations): "
                      f"{final_likelihood:.6f}")
        else:
            self._log("ERROR: No lineations have possible formation times!")
        self._log("=" * 60)

        lineations_with_no_times = np.where(time_counts == 0)[0]

        self._log("\nDIAGNOSTIC SUMMARY")
        self._log("=" * 60)
        self._log(f"Total lineations: {n_lineations}")
        self._log(f"Lineations with possible formation times: "
                  f"{len(valid_indices)}")
        self._log(f"Lineations with NO possible formation times: "
                  f"{len(lineations_with_no_times)}")
        self._log(f"Total possible lineation-time pairs: "
                  f"{int(np.sum(times_possible)):,}")

        if len(lineations_with_no_times) > 0:
            self._log(f"\n{len(lineations_with_no_times)} lineations have "
                      f"no possible formation times.")
            self._log("This could mean:")
            self._log("  1. The lineations are outside the model domain")
            self._log("  2. The thickness/velocity conditions are too strict")
            self._log("  3. The mask value for grounded ice doesn't match")
            self._log("  4. The lineations are in areas that never meet "
                      "the conditions")

        return {
            'final_likelihood': final_likelihood,
            'nu': nu,
            'lineation_x': self.lineation_x,
            'lineation_y': self.lineation_y,
            'lineation_direction': self.lineation_direction,
            'lineation_x_idx': lineation_x_idx,
            'lineation_y_idx': lineation_y_idx,
            'number_lineations': n_lineations,
            'A': A,
            'lam': lam,
            'lam_star': lam_star,
            'likelihood_location': likelihood_location,
            'time_steps': n_times,
        }