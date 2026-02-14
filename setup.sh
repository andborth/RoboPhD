#!/bin/bash
# RoboPhD Setup Script
# Requires: conda (Anaconda or Miniconda)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="robophd"
PYTHON_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "${GREEN}==>${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

echo_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo_error "conda not found. Please install Anaconda or Miniconda first."
        echo "  - Miniconda: https://docs.conda.io/en/latest/miniconda.html"
        echo "  - Anaconda: https://www.anaconda.com/download"
        exit 1
    fi
    echo_step "Found conda: $(conda --version)"
}

# Verify we're in the repo
verify_repo() {
    if [ ! -d "$SCRIPT_DIR/.git" ] || [ ! -f "$SCRIPT_DIR/RoboPhD/researcher.py" ]; then
        echo_error "This script must be run from the RoboPhD repository root."
        echo "  git clone https://github.com/andborth/RoboPhD.git"
        echo "  cd RoboPhD"
        echo "  ./setup.sh"
        exit 1
    fi
}

# Create conda environment
create_environment() {
    # Check if environment already exists
    if conda env list | grep -q "^${ENV_NAME} "; then
        echo_warn "Conda environment '${ENV_NAME}' already exists."
        read -p "Do you want to remove and recreate it? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo_step "Removing existing environment..."
            conda env remove -n "$ENV_NAME" -y
        else
            echo_step "Using existing environment. Run 'conda env remove -n ${ENV_NAME}' to start fresh."
            return 0
        fi
    fi

    echo_step "Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
}

# Install CLI tools from conda-forge
install_cli_tools() {
    echo_step "Installing tree and jq from conda-forge..."
    conda install -n "$ENV_NAME" -c conda-forge tree jq -y
}

# Install Python dependencies
install_python_deps() {
    echo_step "Installing Python dependencies from requirements.txt..."

    # Use conda run to execute pip in the environment
    conda run -n "$ENV_NAME" pip install -r "$SCRIPT_DIR/requirements.txt"
}

# Print next steps
print_next_steps() {
    echo ""
    echo -e "${GREEN}Setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Activate the environment:"
    echo "     conda activate ${ENV_NAME}"
    echo ""
    echo "  2. Set your Anthropic API key:"
    echo "     export ANTHROPIC_API_KEY_FOR_ROBOPHD=\"your_key\""
    echo ""
    echo "  3. Download the BIRD dataset (~50GB):"
    echo "     ./benchmark_resources/download_bird.sh"
    echo ""
    echo "  4. Pre-compute ground truth (prevents timeout warnings):"
    echo "     python RoboPhD/tools/precompute_ground_truth.py"
    echo ""
    echo "  5. Run a quick test:"
    echo "     python RoboPhD/researcher.py \\"
    echo "       --num-iterations 2 \\"
    echo "       --config '{\"contexts_per_iteration\": 2, \"problems_per_context\": 10}'"
    echo ""
    echo "For more information, see:"
    echo "  - QUICKSTART.md for a 5-minute tutorial"
    echo "  - INSTALLATION.md for detailed setup instructions"
    echo "  - CLAUDE.md for comprehensive system documentation"
    echo ""
}

# Main
main() {
    echo "RoboPhD Setup Script"
    echo "===================="
    echo ""

    check_conda
    verify_repo
    create_environment
    install_cli_tools
    install_python_deps
    print_next_steps
}

main
