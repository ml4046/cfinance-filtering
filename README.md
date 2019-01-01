# Filtering for Heston and VGSA Process
Applications of Kalman Filters and Particle Filter (sequential Monte Carlo method) on stochastic processes such as Heston and Variance Gamma Stochastic Arrival (VGSA)

This repository contains implementations and examples for filtering on Heston and VGSA and simulation for diffusion and stochastic time processes (VG, VGSA). `kalman_filter_example.ipynb` provides an introduction to Kalman Filters. We also provide code on how to simulate diffusion and jump processes via. Heston, VG, and VGSA process.

We provide two case studies for filtering: 1. Filtering synthetic data based on a known set of parameters 2. Filtering real market data for selected NASDAQ stocks and oil prices. 

# Setup
We suggest setting up a virtual environment to run the provided codes:
```
virtualenv -p python3 venv
source venv/bin/activate
```
Run `python -r requirements.txt` to install all required packages. Run `jupyter notebook` to run notebooks. The code has been tested for Python 3.6.x

# Examples and testing
We provide notebook examples on filtering and simulating processes based on Heston, VG (Variance Gamma), and VGSA in the notebook directory. The examples contain filtering for synthetic data and real market data from Yahoo Finance for AAPL, AMZN, MSFT, and FB. In addition, we compared estimated state for Sinopec (SNP) and Crude Oil prices.

The implementations for filtering and simulation can be found in the filtering directory.

# Acknowledgement
We would like to thank Prof. Hirsa for his guidance on this project through his lectures and sample codes. Derivations of the algorithms were provided by Prof. Hirsa and implementations are based on [1].

# References
[1] A. Hirsa. Computational Methods in Finance. Chapman and Hall/CRC Financial Mathematics Series. CRC Press, 2016.
