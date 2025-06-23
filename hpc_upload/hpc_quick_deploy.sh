#!/bin/bash

# Quick Deploy Script for Kuro - Simplified version
# For when you just want to get your code running quickly

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
USERNAME="ecsampson"
PASSWORD="TeenT!tans4"
HOST="kuro.sciclone.wm.edu"
SCRATCH="/sciclone/scr-lst/$USERNAME"

echo -e "${GREEN}=== Quick Deploy to Kuro ===${NC}"

# Get source directory
read -p "Path to detection software directory [.]: " SOURCE_DIR
SOURCE_DIR=${SOURCE_DIR:-.}

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Directory not found!"
    exit 1
fi

# Create project directory
PROJECT="fiber_optic_$(date +%Y%m%d_%H%M%S)"
REMOTE_DIR="$SCRATCH/$PROJECT"

echo -e "${YELLOW}Uploading to $REMOTE_DIR...${NC}"

# Create expect script for upload
cat > /tmp/quick_deploy.exp << EOF
#!/usr/bin/expect
set timeout 600

# Create remote directory
spawn ssh $USERNAME@$HOST "mkdir -p $REMOTE_DIR"
expect "password:"
send "$PASSWORD\r"
expect eof

# Upload files
spawn scp -r $SOURCE_DIR/* $USERNAME@$HOST:$REMOTE_DIR/
expect "password:"
send "$PASSWORD\r"
expect eof

# Create requirements.txt
spawn ssh $USERNAME@$HOST "cat > $REMOTE_DIR/requirements.txt << 'EOL'
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
EOL"
expect "password:"
send "$PASSWORD\r"
expect eof

# Create and submit job
spawn ssh $USERNAME@$HOST "cd $REMOTE_DIR && cat > run.slurm << 'EOL'
#!/bin/bash
#SBATCH --job-name=fiber_optic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

module load miniforge3
conda create -n fiber_optic python=3.9 -y
source activate fiber_optic
pip install -r requirements.txt

echo \"Starting pipeline...\"
python app.py
echo \"Pipeline completed!\"
EOL
sbatch run.slurm"
expect "password:"
send "$PASSWORD\r"
expect eof
EOF

chmod +x /tmp/quick_deploy.exp
/tmp/quick_deploy.exp | grep -E "(Submitted|error|Job)"
rm -f /tmp/quick_deploy.exp

echo -e "\n${GREEN}Deployment complete!${NC}"
echo -e "Project location: ${YELLOW}$REMOTE_DIR${NC}"
echo -e "\nUseful commands:"
echo -e "  Check status:  ssh $USERNAME@$HOST 'squeue -u $USERNAME'"
echo -e "  View output:   ssh $USERNAME@$HOST 'tail -f $REMOTE_DIR/output_*.log'"
echo -e "  Get results:   scp -r $USERNAME@$HOST:$REMOTE_DIR/results ./"