#!/bin/bash

set -euo pipefail  # Strict error handling

# Ensure an installation path is provided
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <install_path>"
    exit 1
fi

INSTALL_PATH="$1"

# Validate install path (absolute path recommended)
if [[ "$INSTALL_PATH" != /* ]]; then
    echo "Error: Please provide an absolute path."
    exit 1
fi

# Detect system architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)   CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh" ;;
    aarch64)  CONDA_INSTALLER="Miniconda3-latest-Linux-aarch64.sh" ;;
    *)        echo "Error: Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Define Miniconda download URL
CONDA_URL="https://repo.anaconda.com/miniconda/$CONDA_INSTALLER"

# Create a temporary file for the installer
INSTALLER_TEMP="$(mktemp /tmp/miniconda-installer.XXXXXX).sh"
INSTALLER_TEMP="miniconda-installer.sh"

# Ensure the temporary file is removed on exit
trap 'rm -f "$INSTALLER_TEMP"' EXIT

# Download Miniconda installer
echo "Downloading Miniconda for $ARCH..."
curl -fsSL -o "$INSTALLER_TEMP" "$CONDA_URL"

# Verify the integrity of the downloaded file
if ! grep -q "Miniconda" "$INSTALLER_TEMP"; then
    echo "Error: Downloaded file does not appear to be a valid Miniconda installer."
    exit 1
fi

# Run the installer in a fresh bash environment
echo "Installing Miniconda to $INSTALL_PATH..."
env -i bash "$INSTALLER_TEMP" -b -p "$INSTALL_PATH"

# Verify installation
if [[ ! -x "$INSTALL_PATH/bin/conda" ]]; then
    echo "Error: Installation failed. Conda binary not found."
    exit 1
fi

echo "✅ Miniconda installed at: $INSTALL_PATH"
echo "🔹 To use Conda, run:"
echo "  source $INSTALL_PATH/bin/activate"
echo "🔹 To uninstall, remove the directory:"
echo "  rm -rf $INSTALL_PATH"

# Source conda:
source "${INSTALL_PATH}/bin/activate"
# Create a new environment:
conda create -y --name nmr-to-structure-lite python=3.9
# Activate the new environment:
conda activate nmr-to-structure-lite
# Install the required packages:
pip install -r "$(dirname "$0")/requirements.training.txt"
