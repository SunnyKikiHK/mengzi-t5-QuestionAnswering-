#!/bin/bash

# --- Configuration ---
ENV_NAME="my_ai_env"
PYTHON_VERSION="3.10"

# --- Check for Conda ---
# This block ensures conda is available in the script shell
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
if ! command -v conda &> /dev/null; then
    echo "Error: Conda could not be found. Ensure it is installed and initialized."
    exit 1
fi

# --- Create the Environment ---
echo "Creating Conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# --- Activate the Environment ---
echo "Activating environment..."
conda activate $ENV_NAME

# --- Install Dependencies ---
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    
    # Using pip to install from the text file
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found! Skipping package installation."
fi

echo ""
echo "========================================================"
echo "Setup Complete!"
echo "To use this environment, run: conda activate $ENV_NAME"
echo "========================================================"