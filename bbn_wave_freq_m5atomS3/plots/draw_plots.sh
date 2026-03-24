#!/bin/bash -e

python3 wave_dir_plots.py
python3 wave_sim_plots.py
python3 wave_spectrum_plots.py
python3 freq_track_plots.py
python3 sea_reg_plots.py
python3 qmekf_plots.py
python3 w3d_plots.py
python3 reg_spectra_plots.py
python3 fusion_diag_plots.py

python3 adaptive_wave_detrender_plot.py
python3 adaptive_wave_detrender_sim_plot.py

python3 calibrate_imu_plots.py
