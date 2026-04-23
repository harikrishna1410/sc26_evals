#!/bin/bash

# Exit immediately if any command fails
set -e

# Create the directory for virtual environments
mkdir -p "$HOME/.venv"

# Create the virtual environment
python3 -m venv "$HOME/.venv/el" --system-site-packages

# Activate the virtual environment
source "$HOME/.venv/el/bin/activate"

# Change directory to the package
cd ensemble_launcher

# Install the package. 
# Because the environment is active, 'python' now points to the venv.
python -m pip install ".[mcp]"

# Return to the previous directory
cd ..