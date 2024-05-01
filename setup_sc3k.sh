#!/bin/bash

# Check if the conda command is available
if ! command -v conda &> /dev/null; then
    echo "conda command not found. Please make sure conda is installed and added to your PATH."
    exit 1
fi

# Function to display script usage
display_usage() {
    echo "Usage: $0 <environment_name>"
    echo "Setup SC3K on a Macbook. Should work on other environments as well ideally."
    echo "Arguments:"
    echo "  <environment_name>   Name of the conda environment to create and activate."
}

# Check if the environment name is provided as an argument
if [ $# -eq 0 ]; then
    display_usage
    exit 1
fi

# Check if the help flag is provided
if [ "$1" == "--help" ]; then
    display_usage
    exit 0
fi

# Add conda forge as a channel
conda config --add channels conda-forge

# Set channel priority as strict
conda config --set channel_priority strict

# Create the conda environment
conda create -n "$1" --yes

# Activate the new environment
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate "$1"

# Display the activated environment
echo "New conda environment '$1' created and activated."

# Install the required packages
conda install -n "$1" -c open3d-admin -c conda-forge open3d --yes
conda install -n "$1" pytorch torchvision --yes
conda install hydra-core --yes
conda install omegaconf --yes
conda install tqdm --yes
conda install matplotlib --yes
conda install numpy --yes
conda install einops --yes
conda install tensorboard --yes
conda install pandas --yes
conda install seaborn --yes
conda install scikit-learn --yes
conda install scipy --yes
