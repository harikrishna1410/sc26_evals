This repository contains source files for EnsembleLauncher paper at SC26.

The repository is organized as follows:

- env: Contains environment configuration files for setting up the necessary software and dependencies.
    - el: Contains virtual env specific to the EnsembleLauncher.
    - parsl: Contains virtual env specific to Parsl. Used in weak and strong scaling experiments.
    - dask: Contains virtual env specific to Dask. Used in weak and strong scaling experiments.
    - parsl_dask: Contains virtual env specific to both Parsl and Dask. Used in microbenchmark experiments.
- experiments: Contains scripts and data related to the experiments conducted for the paper.