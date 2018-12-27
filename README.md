# Filtering for Heston and VGSA Process
Applications of Kalman Filters and Particle Filter (sequential Monte Carlo method) on stochastic processes such as Heston and Variance Gamma Stochastic Arrival (VGSA)

The repository contains implementations and examples for filtering on Heston and VGSA and simulation for diffusion and stochastic time processes (VG, VGSA). `kalman_filter_example.ipynb` provides an introduction to Kalman Filters.

[more details to be added]

Note: The project is still in progress.

# Setup
We suggest setting up a virtual environment to run the provided codes:
```
virtualenv -p python3 venv
source venv/bin/activate
```
Run `python -r requirements.txt` to install all required packages. Run `jupyter notebook` to run notebooks.

# Testing
Simulation can be found in `simulate.py` for Heston, VG, and VGSA process with notebook examples.

The implemented filtering process can be found in `kf.py` and `particle_filter.py`. Currently, only filtering for Heston has been added. We also have notebook examples on simulated data in `ekf_heston.ipynb`, `ukf_heston.ipynb` and `pf_heston_example.ipynb`.

# Acknowledgement
We would like to thank Prof. Hirsa for his guidance on this project.
