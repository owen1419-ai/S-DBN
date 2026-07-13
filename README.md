# S-DBN: Self-Supervised Regional TEC Reconstruction over China

This repository provides the core code accompanying the manuscript submitted to **Geo-spatial Information Science (GSIS)**:

**A Self-Supervised Mask-Enhanced Dual-Branch Convolutional Neural Network for Reconstructing Regional TEC over China**

The study develops a self-supervised regional ionospheric modeling framework, named **S-DBN** (Self-supervised mask-enhanced Dual-Branch convolutional neural Network), for reconstructing high-resolution vertical total electron content (VTEC) over China. Instead of using external ionospheric products such as CODE GIM as training labels, S-DBN uses real GNSS-derived observations as supervision through a mask-enhanced self-supervised learning strategy. A high-resolution observation branch and a low-resolution IRI background branch are jointly optimized to combine local observational structures with large-scale physical prior information.

In the manuscript, the method is evaluated using 2023 GNSS observations from 241 continuous stations of the Crustal Movement Observation Network of China (CMONOC) and surrounding IGS stations. The reconstructed regional TEC maps have a spatial resolution of 0.2 deg and a temporal resolution of 10 min. The annual mean RMSE of S-DBN is 1.50 TECU, outperforming DB-CNN, SCHA, and CODE GIM in the reported experiments.

## Repository Contents

```text
.
├── stec2grids_new.py              # GNSS STEC/VTEC preprocessing, gridding, IRI interpolation, and mask generation
├── TECCompletionModel_smooth.py   # S-DBN model, training, validation, testing, and visualization
├── const.py
└── README.md
```

## Code Overview

### `stec2grids_new.py`

This script prepares the gridded TEC samples used by S-DBN. Its main functions include:

- Converting geographic longitude to solar local time longitude (`slon`), which aligns TEC structures with solar illumination.
- Converting GNSS slant TEC observations to VTEC using the mapping function information stored in the input observation files.
- Removing outliers by fitting spherical-cap-harmonic-like basis functions and rejecting observations with residuals exceeding the configured threshold.
- Aggregating pierce-point VTEC observations into regular latitude-`slon` grids.
- Interpolating IRI background fields to 10 min epochs and cropping them to the study region.
- Generating self-supervised masks, including `base_mask`, `train_mask`, `val_mask`, and `test_mask`, using a configurable erased-grid window.

The script saves three files for each epoch:

```text
YYYY-MM-DDTHH:MM_iri.obj       # low-resolution IRI background field
YYYY-MM-DDTHH:MM_tec_slon.obj  # gridded GNSS-derived TEC observations
YYYY-MM-DDTHH:MM_mask.obj      # self-supervised masks
```

### `TECCompletionModel_smooth.py`

This script implements the S-DBN training and inference workflow. The model contains:

- A **background branch** that extracts large-scale prior information from low-resolution IRI input.
- An **observation branch** that receives two high-resolution channels: the masked TEC observations and the corresponding observation mask.
- A **feature fusion module** that concatenates features from both branches and reconstructs a complete TEC field through convolutional and residual layers.
- A **combined loss function** consisting of masked mean squared error and a smoothness regularization term.

The script supports:

- Dataset loading from the gridded `.obj` files produced by `stec2grids_new.py`.
- Training with Adam optimization, learning-rate scheduling, early stopping, and checkpoint saving.
- Testing from the best checkpoint.
- Saving prediction results and diagnostic figures.

## Method Workflow

1. Estimate STEC from GNSS observations using un-differenced and un-combined PPP processing.
2. Convert STEC to VTEC under the single-layer ionospheric model.
3. Transform longitude to solar local time longitude (`slon`).
4. Grid the VTEC observations at the configured spatial resolution.
5. Generate random self-supervised masks over the valid observation grid.
6. Interpolate IRI background fields to each 10 min epoch.
7. Train S-DBN using masked observations as input and the original observations as supervision.
8. Reconstruct continuous regional TEC maps and evaluate masked validation/test areas.

## Requirements

The code was developed in Python and depends on the following main packages:

```text
numpy
pandas
xarray
scipy
matplotlib
tqdm
torch
```

The scripts also import local project modules:

```text
const.py
ppgnss
```

These modules are used for constants, GNSS utilities, and object serialization/deserialization. Before running the scripts, please make sure these local utilities are available in your Python path and that the constants in `const.py` match your study region and experiment settings.

## Data and Paths

The released code expects locally prepared `.obj` files and uses paths that were configured for the authors' computing environment. Please update the paths in the scripts before running:

- In `stec2grids_new.py`, configure the IRI file path, STEC input directory, grid output directory, year, day-of-year range, study-region bounds, resolution, and mask size.
- In `TECCompletionModel_smooth.py`, configure the gridded data directory, training/testing mode, year, day-of-year range, batch size, smoothness parameters, and checkpoint/output directories.

The intended gridded-data directory contains matched files with the following naming convention:

```text
YYYY-MM-DDTHH:MM_iri.obj
YYYY-MM-DDTHH:MM_tec_slon.obj
YYYY-MM-DDTHH:MM_mask.obj
```

## Running the Code

### 1. Generate gridded TEC samples and masks

After preparing STEC observations and IRI background data, adjust the paths and constants, then run:

```bash
python stec2grids_new.py
```

This produces IRI background fields, gridded TEC observations, and self-supervised masks for each 10 min epoch.

### 2. Train or test S-DBN

Edit the `trainning` flag in `TECCompletionModel_smooth.py`:

```python
trainning = True   # train the model
trainning = False  # test using checkpoints_final/best_model.pth
```

Then run:

```bash
python TECCompletionModel_smooth.py
```

During training, checkpoints are saved to `checkpoints_final/`, and training curves are saved to `results_final/`. During testing, reconstructed TEC maps and visualization figures are saved under `results_final/`.

## Data Availability

The satellite observation data used in this study were provided by the China Earthquake Networks Center, Crustal Movement Observation Network of China (CMONOC). Due to the data policy of CMONOC, the raw observational data are **not publicly available**.

The CODE GIM data are freely available from:

```text
http://ftp.unibe.ch/aiub/CODE/
```

The Dst index data were obtained from NASA's Space Physics Data Facility:

```text
https://omniweb.gsfc.nasa.gov/form/dx1.html
```

Processed ionospheric VTEC data and model reconstruction results supporting the findings of the study are available from the corresponding author upon reasonable request:

```text
Liang Zhang
lzhang2019@whu.edu.cn
```

## Citation

If you use this code, please cite the manuscript associated with this repository. The formal citation information will be updated after publication in **Geo-spatial Information Science**.

## Notes

This repository is intended to provide the core implementation of the proposed S-DBN workflow for academic review and research reuse. Because the raw CMONOC observations cannot be redistributed, users need to prepare their own authorized GNSS observation data or request processed data from the corresponding author where appropriate.
