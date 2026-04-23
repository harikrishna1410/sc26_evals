#!/bin/bash

# Exit immediately if any command fails
set -e

# Create the directory for virtual environments
mkdir -p "$HOME/.venv"

# Create the virtual environment
python3 -m venv "$HOME/.venv/parsl_dask" --system-site-packages

# Activate the virtual environment
source "$HOME/.venv/parsl_dask/bin/activate"

##Install dask and parsl
python -m pip install -r requirements.txt

# Change directory to the package
cd ../el/ensemble_launcher

# Install the package. 
# Because the environment is active, 'python' now points to the venv.
python -m pip install ".[mcp]"


# Return to the previous directory
cd ..