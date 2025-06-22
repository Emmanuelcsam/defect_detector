#!/bin/bash

# Pre-deployment check script
# Run this before deploying to ensure everything is ready

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=== Pre-Deployment Check for HPC ==="
echo

# Check expect installation
echo -n "Checking 'expect' installation... "
if command -v expect &> /dev/null; then
    echo -e "${GREEN}✓ Installed${NC}"
else
    echo -e "${RED}✗ Not installed${NC}"
    echo "  Install with: sudo apt-get install expect"
    exit 1
fi

# Check current directory structure
echo -e "\n${YELLOW}Checking project structure:${NC}"

required_files=(
    "app.py"
    "detection.py"
    "separation.py"
    "process.py"
    "config.json"
)

optional_files=(
    "data_acquisition.py"
    "requirements.txt"
)

missing=0

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (REQUIRED)"
        missing=$((missing + 1))
    fi
done

for file in "${optional_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${YELLOW}○${NC} $file (optional)"
    fi
done

# Check zones_methods directory
if [ -d "zones_methods" ]; then
    echo -e "  ${GREEN}✓${NC} zones_methods/"
    zone_count=$(find zones_methods -name "*.py" | wc -l)
    echo "    Found $zone_count Python scripts"
else
    echo -e "  ${RED}✗${NC} zones_methods/ (REQUIRED)"
    missing=$((missing + 1))
fi

# Check for images
if [ -d "image_batch" ]; then
    image_count=$(find image_batch -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.bmp" \) 2>/dev/null | wc -l)
    echo -e "  ${GREEN}✓${NC} image_batch/"
    echo "    Found $image_count images"
else
    echo -e "  ${YELLOW}○${NC} image_batch/ (create this and add images)"
fi

# Check config.json validity
if [ -f "config.json" ]; then
    echo -e "\n${YELLOW}Checking config.json:${NC}"
    if python -m json.tool config.json > /dev/null 2>&1; then
        echo -e "  ${GREEN}✓ Valid JSON format${NC}"
        
        # Check for required sections
        for section in "paths" "process_settings" "separation_settings" "detection_settings"; do
            if grep -q "\"$section\"" config.json; then
                echo -e "  ${GREEN}✓${NC} Has '$section' section"
            else
                echo -e "  ${RED}✗${NC} Missing '$section' section"
            fi
        done
    else
        echo -e "  ${RED}✗ Invalid JSON format${NC}"
    fi
fi

# Check Python dependencies locally (optional)
echo -e "\n${YELLOW}Checking Python environment (local):${NC}"
if command -v python3 &> /dev/null; then
    python_version=$(python3 --version 2>&1)
    echo -e "  ${GREEN}✓${NC} Python installed: $python_version"
    
    # Test imports
    echo "  Testing imports:"
    for module in cv2 numpy skimage scipy matplotlib sklearn; do
        if python3 -c "import $module" 2>/dev/null; then
            echo -e "    ${GREEN}✓${NC} $module"
        else
            echo -e "    ${YELLOW}○${NC} $module (will be installed on HPC)"
        fi
    done
else
    echo -e "  ${YELLOW}○${NC} Python not found locally (will use HPC Python)"
fi

# Network check
echo -e "\n${YELLOW}Checking network connectivity:${NC}"
if ping -c 1 -W 2 kuro.sciclone.wm.edu &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Can reach kuro.sciclone.wm.edu"
else
    echo -e "  ${YELLOW}!${NC} Cannot reach Kuro directly (may need VPN)"
fi

# Summary
echo -e "\n${YELLOW}=== Summary ===${NC}"
if [ $missing -eq 0 ]; then
    echo -e "${GREEN}✓ All required files present${NC}"
    echo -e "\n${GREEN}Ready for deployment!${NC}"
    echo -e "\nNext steps:"
    echo "1. Add images to image_batch/ directory (if needed)"
    echo "2. Run: ./hpc_interactive_manager.sh"
    echo "3. Select option 1 to upload and submit job"
else
    echo -e "${RED}✗ Missing $missing required files${NC}"
    echo -e "\nPlease ensure all required files are present before deploying."
fi

# Disk space check
echo -e "\n${YELLOW}Local disk space:${NC}"
df -h . | tail -1 | awk '{print "  Available: " $4 " (" $5 " used)"}'

# Create example requirements.txt if missing
if [ ! -f "requirements.txt" ]; then
    echo -e "\n${YELLOW}Creating requirements.txt...${NC}"
    cat > requirements.txt << EOF
opencv-python
numpy
scikit-image
scipy
matplotlib
scikit-learn
seaborn
gudhi
pillow
tqdm
pandas
EOF
    echo -e "${GREEN}✓ Created requirements.txt${NC}"
fi

echo -e "\n${YELLOW}Tip:${NC} Run this script in your project directory before deploying to HPC"