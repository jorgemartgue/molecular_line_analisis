Molecular-Line Analysis Pipeline for ALMA Observations of W51
==============================================================

Overview
--------

This repository contains a modular Python pipeline developed for a Master's
Thesis in Astrophysics. It is designed to analyse molecular-line emission in
ALMA spectral cubes of the W51 high-mass star-forming region.

The pipeline identifies and measures molecular transitions, estimates
excitation temperatures and column densities with rotational diagrams,
generates LTE synthetic spectra including optical-depth effects, and refines
the physical parameters through chi-squared fitting. The analysis can be
performed both on spectra averaged over DS9 regions and pixel by pixel to
produce spatial maps.

The current configuration is tailored to ALMA Band 3 and Band 6 observations
of W51 IRS2 and W51-E, but the modular structure can be adapted to other
regions, molecules, and spectral setups.


Main capabilities
-----------------

* Read ALMA FITS spectral cubes and extract data inside DS9 regions.
* Calculate continuum levels and spectral noise from fixed intervals or
  percentile-based estimators.
* Query and locally cache molecular transitions from Splatalogue, using CDMS
  and/or JPL line lists.
* Apply molecule- and region-dependent filters based on upper-state energy,
  Einstein A coefficient, line strength, quantum-number structure, and custom
  frequency exclusions.
* Calibrate systemic velocity and line FWHM from selected transitions.
* Produce pixel-by-pixel velocity and FWHM calibration maps.
* Measure integrated intensities and uncertainties for selected transitions.
* Calculate rotational diagrams under the LTE approximation.
* Produce excitation-temperature and column-density maps, including their
  uncertainties.
* Detect and replace invalid or anomalous map pixels using robust local
  statistics.
* Generate optically thin and opacity-aware LTE synthetic spectra.
* Perform iterative chi-squared fitting of excitation temperature and column
  density for averaged spectra and individual pixels.
* Save chi-squared maps of temperature, column density, uncertainties,
  minimum chi-squared, and the opacity of the transition with the largest line
  strength.
* Cache intermediate products to avoid unnecessary recalculation.


Analysis workflow
-----------------

The main workflow is:

1. Select an observing region and load its ALMA cubes.
2. Calculate or load the continuum and RMS noise.
3. Select a molecule and obtain its Splatalogue catalogue.
4. Calibrate the molecular velocity and FWHM.
5. Filter and measure suitable molecular transitions.
6. Construct the rotational diagram and estimate Tex and Ncol.
7. Generate an LTE synthetic spectrum including optical depth.
8. Refine Tex and Ncol through iterative chi-squared fitting.
9. Optionally repeat the analysis pixel by pixel and save FITS maps.

Most stages can load previously saved results or be explicitly recalculated.
The main script is interactive and allows several molecules to be processed
sequentially without reloading the selected region.


Repository modules
------------------

TFM_main_miriam.py
    Interactive entry point that coordinates the complete workflow.

TFM_config.py
    Observational windows, filesystem paths, source properties, molecular
    definitions, calibration lines, and region-dependent filters.

TFM_io_cubes.py
    Reading and spatial extraction of spectral cubes.

TFM_load_region.py
    Region selection and loading of averaged and full cubes.

TFM_continuum.py
    Continuum and RMS-noise estimation for averaged spectra and pixels.

TFM_splatalogue_tools.py
    Splatalogue queries, catalogue filtering, local caching, and reuse.

TFM_line_search.py
    Spectral-line extraction, Gaussian fitting, velocities, FWHM values, and
    integrated intensities.

TFM_calibration.py
    Molecular velocity and FWHM calibration, globally and pixel by pixel.

TFM_filtering.py
    Physical, spectroscopic, structural, and observational line filtering.

TFM_filtered_lines.py
    Loading, calculation, and storage of filtered molecular-line tables.

TFM_rotational_diagram.py
    Partition functions and rotational-diagram calculations.

TFM_diagrot_pipeline.py
    Rotational-diagram orchestration, FITS maps, plots, and robust map
    correction.

TFM_synthetic_model.py
    LTE synthetic spectra, radiative-transfer calculations, line opacity, and
    residual spectra.

TFM_synthetic_pipeline.py
    Construction and plotting of individual or combined synthetic models.

TFM_chi2_fit.py
    Chi-squared grids and iterative fitting of Tex and Ncol.

TFM_chi2_pipeline.py
    Averaged-spectrum chi-squared workflow and result management.

TFM_chi2_maps.py
    Pixel-by-pixel chi-squared fitting and FITS-map generation.

TFM_momentum_maps.py
    Integrated-intensity and moment-map utilities.

TFM_runtime.py
    Translation of table-based settings into runtime configurations.

TFM_storage.py
    Input/output helpers for ECSV tables, FITS maps, dictionaries, and cached
    results.


Requirements
------------

The code requires Python 3 and the following packages:

* NumPy
* Matplotlib
* Astropy
* Astroquery
* spectral-cube
* regions

An internet connection is required when a molecular catalogue or partition
function is retrieved for the first time. Cached Splatalogue catalogues are
stored locally and can be reused in later executions.


Installation
------------

Clone the repository and create a virtual environment:

    git clone <repository-url>
    cd <repository-directory>

    python3 -m venv .venv
    source .venv/bin/activate

Install the Python dependencies:

    python -m pip install --upgrade pip
    python -m pip install numpy matplotlib astropy astroquery spectral-cube regions


Input data
----------

The observational data are not included in this repository. The pipeline
expects:

* ALMA spectral cubes in FITS format.
* Valid spectral and celestial WCS information.
* Beam information compatible with conversion to brightness temperature.
* DS9 region files defining the sources to be analysed.

Cube filenames must contain the identifiers declared in the observational
window definitions in TFM_config.py.


Configuration
-------------

Before running the pipeline, edit TFM_config.py.

At minimum, update:

1. rutabase, which defines the root directory of the project.
2. The directories containing the FITS cubes.
3. The DS9 region-file paths.
4. The observational windows and their frequency limits.
5. The active region and its systemic velocity.
6. MOLECULE_CONFIG, including catalogue identifiers and molecular constants.
7. REGION_LINE_CONFIG, including calibration transitions, excluded lines, and
   region-dependent filtering parameters.
8. SPEC_SINT_LINE_FILTERS and the chi-squared fitting order when required.

The version currently supplied contains absolute paths for the original local
environment. These must be changed before running the code on another system.


Suggested directory structure
-----------------------------

    project_root/
    |-- TFM_main_miriam.py
    |-- TFM_config.py
    |-- TFM_*.py
    |-- reprojected/              # W51 IRS2 FITS cubes
    |-- reprojectedv2/            # W51-E FITS cubes
    |-- regiones/                 # DS9 region files
    |-- tables/
    |   |-- config/
    |   |-- catalogos_splatalogue/
    |   |-- calibracion/
    |   |-- calibracion_pixeles/
    |   |-- continuos/
    |   |-- filtradas/
    |   |-- filtradas_pixeles/
    |   |-- diagrot/
    |   |-- chi2/
    |   `-- spec_sint/
    |-- maps/
    |   |-- diagrot/
    |   `-- chi2/
    `-- figures/
        |-- diagrot/
        |-- synthetic_model/
        `-- maps/

Output directories are created automatically when possible.


Usage
-----

After configuring the paths and molecular settings, run:

    python TFM_main_miriam.py

The program will ask whether cached products should be recalculated and will
allow a molecule to be selected by name or number. At the end of each molecular
analysis, another molecule can be selected without restarting the program.

The two principal operating modes are configured per region:

medio
    Uses the spectrum averaged over the selected DS9 region.

pixeles
    Performs the rotational-diagram and chi-squared calculations independently
    for the spatial pixels inside the selected region.


Main outputs
------------

Tabular and cached products are written mainly to tables/ as ECSV, pickle, or
FITS files:

* Cached molecular catalogues.
* Continuum measurements.
* Calibration results.
* Pixel-by-pixel velocity and FWHM calibration maps.
* Filtered transition tables.
* Rotational-diagram results.
* Averaged-spectrum chi-squared results.

Spatial products are written to maps/ as FITS files:

* Excitation temperature, Tex.
* Column density, Ncol.
* Uncertainties in Tex and Ncol.
* Minimum chi-squared.
* Opacity of the transition with the highest line strength.

Diagnostic and publication-oriented plots are written to figures/.


Scientific assumptions and limitations
--------------------------------------

* Rotational diagrams assume local thermodynamic equilibrium (LTE).
* The initial rotational-diagram calculation assumes optically thin emission.
* The synthetic model and chi-squared refinement include line opacity.
* A single excitation temperature and column density are fitted for each
  molecular component and spatial pixel.
* Line blending, catalogue completeness, continuum estimation, calibration-line
  selection, and region-dependent filters can significantly affect the result.
* Pixel-level correction uses local robust statistics and should be inspected
  before scientific interpretation.
* This is research software under active development and has been tailored to a
  specific ALMA dataset. Results should be validated for every new dataset.


Data products and reproducibility
---------------------------------

Large ALMA FITS cubes and derived products should normally be excluded from Git
using .gitignore. Configuration tables, region definitions, and lightweight
metadata required to reproduce the analysis may be version controlled when
their distribution is permitted.

Recommended items to exclude include virtual environments, Python cache files,
large FITS cubes, generated figures, cached maps, and local temporary files.


Acknowledgements
----------------

This project makes use of Astropy, Astroquery, spectral-cube, Splatalogue, the
Cologne Database for Molecular Spectroscopy (CDMS), and the Jet Propulsion
Laboratory molecular spectroscopy catalogue (JPL).

When using this code or its results, please cite the relevant software,
catalogues, observational data, and scientific publications associated with
the analysis.
