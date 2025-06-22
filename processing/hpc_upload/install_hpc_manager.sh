#!/usr/bin/env bash

# HPC Interactive Manager - Automated deploy & monitoring for Kuro cluster

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Cluster credentials & settings
USERNAME="ecsampson"
PASSWORD="TeenT!tans4"
HOST="kuro.sciclone.wm.edu"
SCRATCH="/sciclone/scr-lst/$USERNAME"
MAX_CORES=64
MAX_WALLTIME=48 # hours

# Utility: print header
show_header() {
  clear
  echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
  echo -e "${CYAN}║         HPC Interactive Manager for Kuro Cluster         ║${NC}"
  echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}\n"
}

# Utility: run a single SSH command via expect
remote_exec() {
  expect -c "
    set timeout 30
    spawn ssh $USERNAME@$HOST \"$1\"
    expect \"password:\"
    send \"$PASSWORD\\r\"
    expect eof
  " 2>/dev/null | grep -vE "spawn|password:" | sed '/^$/d'
}

# Check connectivity
check_connection() {
  echo -e "${YELLOW}Checking connection to Kuro...${NC}"
  if [[ $(remote_exec "echo OK") == OK ]]; then
    echo -e "${GREEN}✓ Connected to Kuro${NC}"; return 0
  else
    echo -e "${RED}✗ Cannot connect to Kuro${NC}"; return 1
  fi
}

# Upload detection software
upload_detection_software() {
  local src="$(cd "$1" && pwd)"
  local ts="$(date +%Y%m%d_%H%M%S)"
  local proj="$SCRATCH/fiber_optic_detector_$ts"
  echo -e "${YELLOW}Uploading software from $src...${NC}"
  # Verify required files
  for f in app.py detection.py separation.py process.py config.json; do
    if [[ -f "$src/$f" ]]; then
      echo -e "  ${GREEN}✓ $f${NC}"
    else
      echo -e "  ${RED}✗ $f${NC}"
    fi
  done
  # Create project directory on cluster
  remote_exec "mkdir -p $proj"
  # Transfer code
  if command -v rsync &>/dev/null; then
    expect -c "
      set timeout 600
      spawn rsync -avz --progress '$src/' '$USERNAME@$HOST:$proj/'
      expect \"password:\" { send \"$PASSWORD\\r\"; exp_continue }
      expect eof
    "
  else
    expect -c "
      set timeout 600
      spawn bash -c \"cd '$src' && tar czf - . | ssh $USERNAME@$HOST 'cd $proj && tar xzf -'\"
      expect \"password:\" { send \"$PASSWORD\\r\"; exp_continue }
      expect eof
    "
  fi
  echo -e "${GREEN}Upload complete to $proj${NC}"
  echo "$proj" > /tmp/hpc_proj.tmp
}

# Configure job parameters
configure_job() {
  echo -e "${CYAN}--- Job Configuration ---${NC}"
  read -p "Job name [fiber_optic]: " JOB; JOB=${JOB:-fiber_optic}
  read -p "Cores (max $MAX_CORES) [32]: " CORES; CORES=${CORES:-32}
  if (( CORES > MAX_CORES )); then CORES=$MAX_CORES; fi
  read -p "Walltime HH:MM:SS [4:00:00]: " WT; WT=${WT:-4:00:00}
  read -p "Memory (GB) [64]: " MEM; MEM=${MEM:-64}
  echo -e "Select environment setup:"
  echo -e "  [1] Create new conda env  [2] Use existing conda env  [3] System Python"
  read -p "Env option [3]: " OPT; OPT=${OPT:-3}
  case $OPT in
    1) read -p "New env name [fiber_env]: " ENV; ENV=${ENV:-fiber_env}; NEW_ENV=true;;
    2) read -p "Existing env name: " ENV; NEW_ENV=false;;
    3) ENV=""; NEW_ENV=false;;
  esac
  read -p "Do you have images to process? (y/n) [y]: " UPLOAD_IMAGES; UPLOAD_IMAGES=${UPLOAD_IMAGES:-y}
  UPLOAD_IMAGES=$(echo "$UPLOAD_IMAGES" | tr '[:upper:]' '[:lower:]')
  if [[ "$UPLOAD_IMAGES" == "y" ]]; then
    read -p "Path to images directory [./images]: " IMGPATH; IMGPATH=${IMGPATH:-./images}
  else
    IMGPATH=""
  fi
  read -p "Enable job monitoring after submission? (y/n) [y]: " MONITOR; MONITOR=${MONITOR:-y}
  read -p "Email notifications (NONE/BEGIN/END/FAIL/ALL) [END,FAIL]: " MAIL_TYPE; MAIL_TYPE=${MAIL_TYPE:-END,FAIL}
  read -p "Email address: " MAIL_USER
  echo -e "${CYAN}Configuration Summary:${NC}"
  echo "  Job name: $JOB"
  echo "  Cores: $CORES"
  echo "  Walltime: $WT"
  echo "  Memory: ${MEM}G"
  echo "  Env: ${ENV:-system}"
  echo "  Upload images: $UPLOAD_IMAGES"
  echo "  Images path: ${IMGPATH:-none}"
  echo "  Monitoring: $MONITOR"
  echo "  Email notif: $MAIL_TYPE"
  echo "  Email user: $MAIL_USER"
  read -p "Proceed with this configuration? (y/n): " -n1 OK; echo
  [[ "$OK" =~ [Yy] ]] || return 1
  cat >/tmp/hpc_job.conf <<EOF
JOB=$JOB
CORES=$CORES
WT=$WT
MEM=${MEM}G
ENV=$ENV
NEW_ENV=$NEW_ENV
UPLOAD_IMAGES=$UPLOAD_IMAGES
IMGPATH=$IMGPATH
MONITOR=$MONITOR
MAIL_TYPE=$MAIL_TYPE
MAIL_USER=$MAIL_USER
EOF
}

# Submit the job
submit_job() {
  source /tmp/hpc_job.conf
  PROJ=$(< /tmp/hpc_proj.tmp)
  echo -e "${YELLOW}Creating SLURM script...${NC}"
  cat >/tmp/run_job.slurm <<SCRIPT
#!/usr/bin/env bash
#SBATCH --job-name=$JOB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CORES
#SBATCH --time=$WT
#SBATCH --mem=$MEM
#SBATCH --output=${JOB}_%j.out
#SBATCH --error=${JOB}_%j.err
#SBATCH --mail-type=$MAIL_TYPE
#SBATCH --mail-user=$MAIL_USER

module load miniforge3

# Environment setup
SCRIPT
  if [[ "$NEW_ENV" == "true" ]]; then
    cat >>/tmp/run_job.slurm <<ENVSETUP
if ! conda env list | grep -q "^$ENV "; then
  conda create -n $ENV python=3.9 -y
fi
source activate $ENV
ENVSETUP
  elif [[ -n "$ENV" ]]; then
    cat >>/tmp/run_job.slurm <<ENVSETUP
source activate $ENV
ENVSETUP
  fi
  cat >>/tmp/run_job.slurm <<COMMON
# Install dependencies
pip install -r requirements.txt

# Change to project directory
cd $PROJ

# Start detection
echo "Starting detection..."
python app.py
COMMON

  echo -e "${BLUE}Transferring SLURM script to remote...${NC}"
  scp /tmp/run_job.slurm $USERNAME@$HOST:$PROJ/run_job.slurm
  echo -e "${YELLOW}Submitting job...${NC}"
  SBATCH_OUT=$(remote_exec "cd $PROJ && sbatch run_job.slurm")
  echo "$SBATCH_OUT"
  JOBID=$(echo "$SBATCH_OUT" | awk '{print $4}')
  echo -e "${GREEN}Job submitted (ID: $JOBID)${NC}"

  if [[ "$MONITOR" == "y" ]]; then
    echo -e "${CYAN}Tailing job output (press Ctrl+C to exit)...${NC}"
    ssh $USERNAME@$HOST "cd $PROJ && tail -f ${JOB}_${JOBID}.out"
  fi

  # Clean up temporary files
  rm -f /tmp/run_job.slurm /tmp/hpc_job.conf /tmp/hpc_proj.tmp
}

# Main logic
main() {
  show_header
  check_connection || exit 1
  echo -e "Upload Fiber Optic Detection Software"
  read -p "Enter path to detection software directory [.]: " PATHDIR; PATHDIR=${PATHDIR:-.}
  upload_detection_software "$PATHDIR"
  if configure_job; then
    submit_job
  else
    echo -e "${RED}Job configuration cancelled.${NC}"
  fi
}

main

