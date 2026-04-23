#!/bin/bash

# Exit immediately if any command fails
set -e

# Create the directory for virtual environments
mkdir -p "$HOME/.venv"

# Create the virtual environment
python3 -m venv "$HOME/.venv/dask" --system-site-packages

# Activate the virtual environment
source "$HOME/.venv/dask/bin/activate"

##Install dask and parsl
python -m pip install -r requirements.txt