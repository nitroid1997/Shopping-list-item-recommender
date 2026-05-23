#!/usr/bin/env bash
# ============================================================================
# Shopping List Item Recommender — Setup Script
# ============================================================================
# This script checks for required dependencies and installs them.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
# ============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo " Shopping List Item Recommender — Setup"
echo "=============================================="
echo ""

# ------------------------------------------------------------------
# 1. Check Python version
# ------------------------------------------------------------------
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=7

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed.${NC}"
    echo "Please install Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt "$REQUIRED_PYTHON_MAJOR" ] || \
   { [ "$PYTHON_MAJOR" -eq "$REQUIRED_PYTHON_MAJOR" ] && [ "$PYTHON_MINOR" -lt "$REQUIRED_PYTHON_MINOR" ]; }; then
    echo -e "${RED}Error: Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ is required (found ${PYTHON_VERSION}).${NC}"
    exit 1
fi

echo -e "${GREEN}✔ Python ${PYTHON_VERSION} detected.${NC}"

# ------------------------------------------------------------------
# 2. Check / create virtual environment
# ------------------------------------------------------------------
VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment in ./${VENV_DIR} ..."
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}✔ Virtual environment created.${NC}"
else
    echo -e "${GREEN}✔ Virtual environment already exists.${NC}"
fi

echo ""
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# ------------------------------------------------------------------
# 3. Upgrade pip
# ------------------------------------------------------------------
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# ------------------------------------------------------------------
# 4. Install dependencies
# ------------------------------------------------------------------
echo ""
echo "Installing dependencies from requirements.txt ..."
pip install -r requirements.txt --quiet
echo -e "${GREEN}✔ All dependencies installed.${NC}"

# ------------------------------------------------------------------
# 5. Verify critical imports
# ------------------------------------------------------------------
echo ""
echo "Verifying critical imports..."

IMPORTS_OK=true

for pkg in numpy pandas matplotlib apyori mlxtend sklearn; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        echo -e "  ${GREEN}✔${NC} ${pkg}"
    else
        echo -e "  ${RED}✘${NC} ${pkg} — import failed"
        IMPORTS_OK=false
    fi
done

if [ "$IMPORTS_OK" = false ]; then
    echo ""
    echo -e "${RED}Some imports failed. Please check the errors above.${NC}"
    exit 1
fi

# ------------------------------------------------------------------
# 6. Check for dataset
# ------------------------------------------------------------------
echo ""
DATA_FILE="data/Market_Basket_Optimisation.csv"
if [ -f "$DATA_FILE" ]; then
    echo -e "${GREEN}✔ Dataset found at ${DATA_FILE}.${NC}"
else
    echo -e "${YELLOW}⚠ Dataset not found at ${DATA_FILE}.${NC}"
    echo "  The analysis mode requires this file."
    echo "  You can still run the prediction demo with: python main.py --predict"
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "=============================================="
echo -e "${GREEN} Setup complete!${NC}"
echo "=============================================="
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python main.py              # Full pipeline (analysis + prediction)"
echo "  python main.py --analyze    # Market basket analysis only"
echo "  python main.py --predict    # Interactive cart prediction only"
echo ""
