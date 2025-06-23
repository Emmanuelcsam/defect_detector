#!/bin/bash

# HPC Interactive Manager - Complete automated script for Kuro
# Enhanced with configurable SLURM generation and improved reliability

# Ensure we're running with bash
if [ -z "$BASH_VERSION" ]; then
    echo "Error: This script must be run with bash, not sh"
    echo "Run with: bash $0"
    exit 1
fi

# Color codes for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
USERNAME="ecsampson"
PASSWORD="TeenT!tans4"
HOST="kuro.sciclone.wm.edu"
SCRATCH="/sciclone/scr-lst/$USERNAME"
MAX_CORES=64
MAX_WALLTIME=48
DEFAULT_PARTITION="parallel"

# Function to display header
show_header() {
    clear
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         HPC Interactive Manager for Kuro Cluster         ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo
}

# Enhanced function to execute remote commands with better error handling
remote_exec() {
    local command="$1"
    local timeout="${2:-30}"
    
    local result=$(expect -c "
    set timeout $timeout
    log_user 0
    spawn ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \"$command\"
    expect {
        \"password:\" {
            send \"$PASSWORD\r\"
            exp_continue
        }
        \"Permission denied\" {
            exit 1
        }
        eof
    }
    catch wait result
    exit [lindex \$result 3]
    " 2>/dev/null)
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "$result" | grep -v "spawn\|password:" | sed '/^$/d'
    else
        return $exit_code
    fi
}

# Function to check connection with better error reporting
check_connection() {
    echo -e "${YELLOW}Checking connection to Kuro...${NC}"
    
    # Test SSH connection
    result=$(timeout 10 expect -c "
    set timeout 10
    spawn ssh -o StrictHostKeyChecking=no $USERNAME@$HOST \"echo 'CONNECTION_OK'\"
    expect {
        \"password:\" {
            send \"$PASSWORD\r\"
            expect \"CONNECTION_OK\"
            exit 0
        }
        \"Permission denied\" {
            exit 2
        }
        timeout {
            exit 1
        }
        eof {
            exit 3
        }
    }
    " 2>&1)
    
    exit_code=$?
    
    case $exit_code in
        0)
            if [[ $result == *"CONNECTION_OK"* ]]; then
                echo -e "${GREEN}✓ Connected to Kuro successfully${NC}"
                
                # Check scratch directory
                if remote_exec "test -d $SCRATCH && echo 'SCRATCH_OK'" | grep -q "SCRATCH_OK"; then
                    echo -e "${GREEN}✓ Scratch directory accessible${NC}"
                else
                    echo -e "${YELLOW}⚠ Creating scratch directory...${NC}"
                    remote_exec "mkdir -p $SCRATCH"
                fi
                return 0
            fi
            ;;
        1)
            echo -e "${RED}✗ Connection timeout${NC}"
            ;;
        2)
            echo -e "${RED}✗ Authentication failed${NC}"
            ;;
        *)
            echo -e "${RED}✗ Failed to connect to Kuro${NC}"
            ;;
    esac
    return 1
}

# Enhanced upload function with progress tracking
upload_detection_software() {
    local source_dir="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local project_name="fiber_optic_detector_$timestamp"
    local remote_dir="$SCRATCH/$project_name"
    
    echo -e "\n${YELLOW}Preparing to upload detection software...${NC}"
    
    # Validate source directory
    if [ ! -d "$source_dir" ]; then
        echo -e "${RED}Error: Source directory not found: $source_dir${NC}"
        return 1
    fi
    
    # Convert to absolute path
    source_dir=$(cd "$source_dir" && pwd)
    echo -e "${BLUE}Source directory: $source_dir${NC}"
    
    # Check for required files
    echo -e "${BLUE}Checking required files...${NC}"
    local required_files="app.py detection.py separation.py process.py config.json"
    local missing_files=""
    
    for file in $required_files; do
        if [ -f "$source_dir/$file" ]; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${RED}✗${NC} $file"
            missing_files="$missing_files $file"
        fi
    done
    
    # Check for zones_methods directory
    if [ -d "$source_dir/zones_methods" ]; then
        echo -e "  ${GREEN}✓${NC} zones_methods/"
        local zone_count=$(find "$source_dir/zones_methods" -name "*.py" 2>/dev/null | wc -l)
        echo -e "    Found $zone_count Python scripts"
    else
        echo -e "  ${RED}✗${NC} zones_methods/"
        missing_files="$missing_files zones_methods"
    fi
    
    if [ -n "$missing_files" ]; then
        echo -e "\n${RED}Missing required files:$missing_files${NC}"
        read -p "Continue anyway? (y/n): " continue_upload
        if [[ ! $continue_upload =~ ^[Yy]$ ]]; then
            return 1
        fi
    fi
    
    # Calculate total size
    local total_size=$(du -sh "$source_dir" 2>/dev/null | cut -f1)
    echo -e "\n${YELLOW}Total upload size: $total_size${NC}"
    
    # Create remote directory structure
    echo -e "${YELLOW}Creating remote directory: $remote_dir${NC}"
    if ! remote_exec "mkdir -p $remote_dir"; then
        echo -e "${RED}Failed to create remote directory${NC}"
        return 1
    fi
    
    # Enhanced upload with multiple methods
    echo -e "${YELLOW}Starting file transfer...${NC}"
    
    # Method 1: Try rsync if available
    if command -v rsync &> /dev/null; then
        echo -e "${BLUE}Using rsync for efficient transfer...${NC}"
        
        # Create progress monitoring script
        cat > /tmp/rsync_upload.exp << 'EOF'
#!/usr/bin/expect
set timeout -1
set source [lindex $argv 0]
set dest [lindex $argv 1]
set pass [lindex $argv 2]

spawn rsync -avz --progress --stats "$source/" "$dest/"
expect {
    "password:" {
        send "$pass\r"
        exp_continue
    }
    "100%" {
        exp_continue
    }
    eof
}
EOF
        chmod +x /tmp/rsync_upload.exp
        
        if /tmp/rsync_upload.exp "$source_dir" "$USERNAME@$HOST:$remote_dir" "$PASSWORD"; then
            echo -e "${GREEN}✓ Files transferred successfully${NC}"
        else
            echo -e "${YELLOW}Rsync failed, trying alternative method...${NC}"
            
            # Method 2: Tar over SSH
            tar -czf - -C "$source_dir" . | \
            expect -c "
            set timeout -1
            spawn ssh $USERNAME@$HOST \"cd $remote_dir && tar -xzf -\"
            expect \"password:\"
            send \"$PASSWORD\r\"
            expect eof
            " 2>/dev/null
        fi
        
        rm -f /tmp/rsync_upload.exp
    else
        # Fallback: Tar transfer
        echo -e "${BLUE}Using tar transfer...${NC}"
        tar -czf - -C "$source_dir" . | \
        expect -c "
        set timeout -1
        spawn ssh $USERNAME@$HOST \"cd $remote_dir && tar -xzf -\"
        expect \"password:\"
        send \"$PASSWORD\r\"
        expect eof
        " 2>/dev/null
    fi
    
    # Verify upload and create requirements.txt if missing
    echo -e "\n${YELLOW}Verifying upload and setting up environment...${NC}"
    
    local remote_files=$(remote_exec "find $remote_dir -type f | wc -l")
    echo -e "${GREEN}✓ Uploaded $remote_files files${NC}"
    
    # Create comprehensive requirements.txt if not exists
    if ! remote_exec "test -f $remote_dir/requirements.txt && echo 'EXISTS'" | grep -q "EXISTS"; then
        echo -e "${YELLOW}Creating requirements.txt...${NC}"
        remote_exec "cat > $remote_dir/requirements.txt << 'EOL'
# Core image processing
opencv-python>=4.5.0
numpy>=1.19.0
scikit-image>=0.18.0
scipy>=1.7.0
matplotlib>=3.3.0
seaborn>=0.11.0

# Machine learning
scikit-learn>=0.24.0

# Computational geometry
gudhi>=3.4.0

# Utilities
pillow>=8.0.0
tqdm>=4.60.0
pandas>=1.3.0
h5py>=3.0.0

# Additional libraries
imageio>=2.9.0
networkx>=2.6.0
EOL"
    fi
    
    # Store project directory
    echo "$remote_dir" > /tmp/hpc_project_dir.tmp
    
    # Generate initial SLURM template
    generate_slurm_template "$remote_dir"
    
    echo -e "${GREEN}✓ Upload complete! Project directory: $remote_dir${NC}"
    return 0
}

# New function to generate SLURM template
generate_slurm_template() {
    local project_dir="$1"
    
    echo -e "${YELLOW}Generating SLURM template...${NC}"
    
    remote_exec "cat > $project_dir/job_template.slurm << 'EOL'
#!/bin/bash
#---------------------------------------------------------------------------
# SLURM Job Script Template - Fiber Optic Detection Pipeline
# Generated: $(date)
#---------------------------------------------------------------------------

#SBATCH --job-name=fiber_optic
#SBATCH --partition=parallel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Optional directives (uncomment to use)
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=your.email@example.com
# #SBATCH --array=1-10
# #SBATCH --dependency=afterok:job_id
# #SBATCH --exclusive
# #SBATCH --constraint=gpu

#---------------------------------------------------------------------------
# Environment Setup
#---------------------------------------------------------------------------

echo \"========================================\"
echo \"Job Information\"
echo \"========================================\"
echo \"Job ID: \$SLURM_JOB_ID\"
echo \"Job Name: \$SLURM_JOB_NAME\"
echo \"Node: \$SLURM_NODELIST\"
echo \"CPUs: \$SLURM_CPUS_PER_TASK\"
echo \"Memory: \$SLURM_MEM_PER_NODE MB\"
echo \"Start Time: \$(date)\"
echo \"Working Directory: \$(pwd)\"
echo \"========================================\"

# Load required modules
module load miniforge3

# Python environment setup (choose one method)
# Method 1: Create/activate conda environment
if [ -n \"\$CONDA_ENV_NAME\" ]; then
    if ! conda env list | grep -q \"^\$CONDA_ENV_NAME \"; then
        echo \"Creating conda environment \$CONDA_ENV_NAME...\"
        conda create -n \$CONDA_ENV_NAME python=3.9 -y
    fi
    source activate \$CONDA_ENV_NAME
fi

# Method 2: Use virtual environment
# python -m venv venv
# source venv/bin/activate

# Install dependencies
echo \"Installing Python packages...\"
pip install --upgrade pip
pip install -r requirements.txt

# Environment information
echo \"\"
echo \"Environment Information:\"
echo \"Python: \$(which python)\"
echo \"Python version: \$(python --version)\"
echo \"Pip packages: \$(pip list | wc -l) packages installed\"
echo \"========================================\"

#---------------------------------------------------------------------------
# Main Execution
#---------------------------------------------------------------------------

# Set OpenMP threads
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

# Memory monitoring (optional)
monitor_memory() {
    while true; do
        echo \"[\$(date +%H:%M:%S)] Memory: \$(free -h | grep Mem | awk '{print \$3 \"/\" \$2}')\"
        sleep 60
    done
}
# Uncomment to enable: monitor_memory &

# Run the main application
echo \"Starting fiber optic detection pipeline...\"
python app.py

# Alternative execution modes:
# python app.py --batch-size 100 --workers \$SLURM_CPUS_PER_TASK
# python app.py --input image_batch --output results
# mpirun -np \$SLURM_NTASKS python app_mpi.py

#---------------------------------------------------------------------------
# Post-processing
#---------------------------------------------------------------------------

echo \"\"
echo \"========================================\"
echo \"Job completed at: \$(date)\"
echo \"Exit code: \$?\"

# Generate summary report
if [ -d results ]; then
    echo \"Results summary:\"
    find results -type f | wc -l | xargs echo \"  Total output files:\"
    du -sh results | xargs echo \"  Results size:\"
fi

# Cleanup (optional)
# rm -rf temp_*

echo \"========================================\"
EOL"
}

# Enhanced job configuration function
configure_job() {
    echo -e "\n${CYAN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                  Job Configuration                        ${NC}"
    echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
    
    # Store configuration in temp file
    rm -f /tmp/hpc_job_config.tmp
    
    # Basic job parameters
    echo -e "\n${MAGENTA}Basic Job Parameters:${NC}"
    
    read -p "Job name [fiber_optic]: " job_name
    job_name=${job_name:-fiber_optic}
    echo "JOB_NAME=\"$job_name\"" >> /tmp/hpc_job_config.tmp
    
    read -p "Queue/Partition [parallel]: " partition
    partition=${partition:-parallel}
    echo "PARTITION=\"$partition\"" >> /tmp/hpc_job_config.tmp
    
    # Compute resources
    echo -e "\n${MAGENTA}Compute Resources:${NC}"
    echo -e "${YELLOW}Kuro specifications: up to $MAX_CORES cores per node${NC}"
    
    read -p "Number of cores [32]: " cores
    cores=${cores:-32}
    if [ $cores -gt $MAX_CORES ]; then
        echo -e "${RED}Warning: Requested cores exceed maximum. Setting to $MAX_CORES${NC}"
        cores=$MAX_CORES
    fi
    echo "CORES=$cores" >> /tmp/hpc_job_config.tmp
    
    read -p "Number of nodes [1]: " nodes
    nodes=${nodes:-1}
    echo "NODES=$nodes" >> /tmp/hpc_job_config.tmp
    
    read -p "Tasks per node [1]: " tasks
    tasks=${tasks:-1}
    echo "TASKS=$tasks" >> /tmp/hpc_job_config.tmp
    
    # Time and memory
    echo -e "\n${MAGENTA}Time and Memory:${NC}"
    echo -e "${YELLOW}Maximum walltime: $MAX_WALLTIME hours${NC}"
    
    read -p "Walltime (HH:MM:SS) [4:00:00]: " walltime
    walltime=${walltime:-4:00:00}
    echo "WALLTIME=\"$walltime\"" >> /tmp/hpc_job_config.tmp
    
    read -p "Memory (GB) [64]: " memory
    memory=${memory:-64}
    echo "MEMORY=\"${memory}G\"" >> /tmp/hpc_job_config.tmp
    
    # Advanced options
    echo -e "\n${MAGENTA}Advanced Options:${NC}"
    
    read -p "Use GPU nodes? (y/n) [n]: " use_gpu
    if [[ $use_gpu =~ ^[Yy]$ ]]; then
        echo "USE_GPU=yes" >> /tmp/hpc_job_config.tmp
        read -p "Number of GPUs [1]: " num_gpus
        num_gpus=${num_gpus:-1}
        echo "NUM_GPUS=$num_gpus" >> /tmp/hpc_job_config.tmp
    fi
    
    read -p "Job array? (y/n) [n]: " use_array
    if [[ $use_array =~ ^[Yy]$ ]]; then
        read -p "Array range (e.g., 1-10): " array_range
        echo "ARRAY_RANGE=\"$array_range\"" >> /tmp/hpc_job_config.tmp
    fi
    
    read -p "Exclusive node access? (y/n) [n]: " exclusive
    if [[ $exclusive =~ ^[Yy]$ ]]; then
        echo "EXCLUSIVE=yes" >> /tmp/hpc_job_config.tmp
    fi
    
    # Python environment
    echo -e "\n${MAGENTA}Python Environment:${NC}"
    echo "1) Create new conda environment"
    echo "2) Use existing conda environment"
    echo "3) Create virtual environment"
    echo "4) Use system Python"
    read -p "Select option [1]: " env_option
    env_option=${env_option:-1}
    
    case $env_option in
        1)
            read -p "New environment name [fiber_optic_env]: " env_name
            env_name=${env_name:-fiber_optic_env}
            echo "ENV_NAME=\"$env_name\"" >> /tmp/hpc_job_config.tmp
            echo "ENV_TYPE=conda_new" >> /tmp/hpc_job_config.tmp
            ;;
        2)
            read -p "Existing environment name: " env_name
            echo "ENV_NAME=\"$env_name\"" >> /tmp/hpc_job_config.tmp
            echo "ENV_TYPE=conda_existing" >> /tmp/hpc_job_config.tmp
            ;;
        3)
            echo "ENV_TYPE=venv" >> /tmp/hpc_job_config.tmp
            ;;
        4)
            echo "ENV_TYPE=system" >> /tmp/hpc_job_config.tmp
            ;;
    esac
    
    # Notification settings
    echo -e "\n${MAGENTA}Notification Settings:${NC}"
    echo "Email notification types:"
    echo "  NONE - No email notifications"
    echo "  BEGIN - Job start"
    echo "  END - Job completion"
    echo "  FAIL - Job failure"
    echo "  ALL - All events"
    read -p "Notification type [END,FAIL]: " email_notify
    email_notify=${email_notify:-END,FAIL}
    echo "EMAIL_NOTIFY=\"$email_notify\"" >> /tmp/hpc_job_config.tmp
    
    if [[ $email_notify != "NONE" ]]; then
        read -p "Email address: " email_address
        echo "EMAIL_ADDRESS=\"$email_address\"" >> /tmp/hpc_job_config.tmp
    fi
    
    # Application parameters
    echo -e "\n${MAGENTA}Application Parameters:${NC}"
    read -p "Process image batch? (y/n) [y]: " process_images
    process_images=${process_images:-y}
    echo "PROCESS_IMAGES=\"$process_images\"" >> /tmp/hpc_job_config.tmp
    
    if [[ $process_images =~ ^[Yy]$ ]]; then
        read -p "Local image directory (or 'skip' to use uploaded): " local_image_dir
        if [[ $local_image_dir != "skip" ]] && [ -d "$local_image_dir" ]; then
            local_image_dir=$(cd "$local_image_dir" && pwd)
            local img_count=$(find "$local_image_dir" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) 2>/dev/null | wc -l)
            echo -e "${GREEN}Found $img_count images${NC}"
            echo "LOCAL_IMAGE_DIR=\"$local_image_dir\"" >> /tmp/hpc_job_config.tmp
            echo "UPLOAD_IMAGES=yes" >> /tmp/hpc_job_config.tmp
        fi
    fi
    
    read -p "Batch size for processing [100]: " batch_size
    batch_size=${batch_size:-100}
    echo "BATCH_SIZE=$batch_size" >> /tmp/hpc_job_config.tmp
    
    read -p "Number of worker processes [auto]: " workers
    workers=${workers:-auto}
    echo "WORKERS=\"$workers\"" >> /tmp/hpc_job_config.tmp
    
    # Monitoring
    read -p "Enable real-time monitoring after submission? (y/n) [y]: " monitor
    monitor=${monitor:-y}
    echo "MONITOR=\"$monitor\"" >> /tmp/hpc_job_config.tmp
    
    # Display configuration summary
    display_job_summary
    
    read -p $'\nProceed with this configuration? (y/n): ' confirm
    [[ $confirm =~ ^[Yy]$ ]]
}

# Function to display job configuration summary
display_job_summary() {
    # Load configuration
    source /tmp/hpc_job_config.tmp
    
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                Configuration Summary                      ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    
    echo -e "\n${MAGENTA}Job Settings:${NC}"
    echo -e "  Job name:        $JOB_NAME"
    echo -e "  Partition:       $PARTITION"
    echo -e "  Nodes:           $NODES"
    echo -e "  Cores/node:      $CORES"
    echo -e "  Tasks/node:      $TASKS"
    echo -e "  Walltime:        $WALLTIME"
    echo -e "  Memory:          $MEMORY"
    
    [ -n "$USE_GPU" ] && echo -e "  GPUs:            $NUM_GPUS"
    [ -n "$ARRAY_RANGE" ] && echo -e "  Array:           $ARRAY_RANGE"
    [ -n "$EXCLUSIVE" ] && echo -e "  Exclusive:       Yes"
    
    echo -e "\n${MAGENTA}Environment:${NC}"
    echo -e "  Type:            $ENV_TYPE"
    [ -n "$ENV_NAME" ] && echo -e "  Name:            $ENV_NAME"
    
    echo -e "\n${MAGENTA}Application:${NC}"
    echo -e "  Batch size:      $BATCH_SIZE"
    echo -e "  Workers:         $WORKERS"
    [ "$UPLOAD_IMAGES" = "yes" ] && echo -e "  Upload images:   Yes"
    
    if [[ $EMAIL_NOTIFY != "NONE" ]]; then
        echo -e "\n${MAGENTA}Notifications:${NC}"
        echo -e "  Type:            $EMAIL_NOTIFY"
        echo -e "  Email:           $EMAIL_ADDRESS"
    fi
}

# Enhanced job submission with comprehensive SLURM generation
submit_job() {
    local project_dir=$(cat /tmp/hpc_project_dir.tmp)
    
    # Load configuration
    source /tmp/hpc_job_config.tmp
    
    echo -e "\n${YELLOW}Preparing job submission...${NC}"
    
    # Upload images if needed
    if [[ $UPLOAD_IMAGES == "yes" ]]; then
        upload_images_batch "$LOCAL_IMAGE_DIR" "$project_dir"
    fi
    
    # Generate comprehensive SLURM script
    echo -e "${YELLOW}Generating SLURM batch script...${NC}"
    
    # Create SLURM script content
    cat > /tmp/slurm_script.tmp << EOF
#!/bin/bash
#---------------------------------------------------------------------------
# SLURM Job Script - Generated by HPC Interactive Manager
# Project: $project_dir
# Generated: $(date)
#---------------------------------------------------------------------------

# Basic job parameters
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --nodes=$NODES
#SBATCH --ntasks=$TASKS
#SBATCH --cpus-per-task=$CORES
#SBATCH --time=$WALLTIME
#SBATCH --mem=$MEMORY

# Output files
#SBATCH --output=${JOB_NAME}_%j.out
#SBATCH --error=${JOB_NAME}_%j.err
EOF

    # Add optional parameters
    if [ "$EXCLUSIVE" = "yes" ]; then
        echo "#SBATCH --exclusive" >> /tmp/slurm_script.tmp
    fi
    
    if [ -n "$USE_GPU" ]; then
        echo "#SBATCH --gres=gpu:$NUM_GPUS" >> /tmp/slurm_script.tmp
        echo "#SBATCH --constraint=gpu" >> /tmp/slurm_script.tmp
    fi
    
    if [ -n "$ARRAY_RANGE" ]; then
        echo "#SBATCH --array=$ARRAY_RANGE" >> /tmp/slurm_script.tmp
    fi
    
    if [[ $EMAIL_NOTIFY != "NONE" ]]; then
        echo "#SBATCH --mail-type=$EMAIL_NOTIFY" >> /tmp/slurm_script.tmp
        echo "#SBATCH --mail-user=$EMAIL_ADDRESS" >> /tmp/slurm_script.tmp
    fi
    
    # Add the rest of the script
    cat >> /tmp/slurm_script.tmp << 'EOF'

#---------------------------------------------------------------------------
# Environment Setup
#---------------------------------------------------------------------------

# Job information
echo "========================================"
echo "Job Information"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "========================================"

# Load required modules
module purge
module load miniforge3
EOF

    # Add GPU module if needed
    if [ -n "$USE_GPU" ]; then
        echo "module load cuda" >> /tmp/slurm_script.tmp
    fi
    
    # Python environment setup based on configuration
    case $ENV_TYPE in
        conda_new|conda_existing)
            cat >> /tmp/slurm_script.tmp << EOF

# Conda environment setup
EOF
            if [ "$ENV_TYPE" = "conda_new" ]; then
                cat >> /tmp/slurm_script.tmp << EOF
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Creating conda environment $ENV_NAME..."
    conda create -n $ENV_NAME python=3.9 -y
fi
EOF
            fi
            cat >> /tmp/slurm_script.tmp << EOF
source activate $ENV_NAME
EOF
            ;;
        venv)
            cat >> /tmp/slurm_script.tmp << 'EOF'

# Virtual environment setup
if [ ! -d venv ]; then
    python -m venv venv
fi
source venv/bin/activate
EOF
            ;;
        system)
            cat >> /tmp/slurm_script.tmp << 'EOF'

# Using system Python
EOF
            ;;
    esac
    
    # Add package installation and main execution
    cat >> /tmp/slurm_script.tmp << EOF

# Install/update packages
echo "Installing Python packages..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir

# Display environment info
echo ""
echo "Python: \$(which python)"
echo "Python version: \$(python --version)"
echo "Installed packages: \$(pip list | wc -l)"
echo "========================================"

#---------------------------------------------------------------------------
# Main Execution
#---------------------------------------------------------------------------

# Set environment variables
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
EOF

    if [ -n "$USE_GPU" ]; then
        echo "export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID" >> /tmp/slurm_script.tmp
    fi
    
    cat >> /tmp/slurm_script.tmp << EOF

# Change to project directory
cd $project_dir

# Resource monitoring (background)
monitor_resources() {
    while true; do
        echo "[\$(date +%T)] CPU: \$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}')% | Memory: \$(free -h | grep Mem | awk '{print \$3"/"\$2}')"
        sleep 300
    done
}
monitor_resources > resource_monitor.log 2>&1 &
MONITOR_PID=\$!

# Run the main application
echo ""
echo "Starting fiber optic detection pipeline..."
echo "========================================"

python app.py\\
EOF

    # Add command line arguments if configured
    if [ "$WORKERS" = "auto" ]; then
        echo " --workers \$SLURM_CPUS_PER_TASK\\" >> /tmp/slurm_script.tmp
    elif [[ $WORKERS =~ ^[0-9]+$ ]]; then
        echo " --workers $WORKERS\\" >> /tmp/slurm_script.tmp
    fi
    
    echo " --batch-size $BATCH_SIZE\\" >> /tmp/slurm_script.tmp
    
    if [[ $PROCESS_IMAGES =~ ^[Yy]$ ]]; then
        echo " --input image_batch --output results" >> /tmp/slurm_script.tmp
    else
        echo "" >> /tmp/slurm_script.tmp
    fi
    
    # Add post-processing
    cat >> /tmp/slurm_script.tmp << 'EOF'

#---------------------------------------------------------------------------
# Post-processing and Cleanup
#---------------------------------------------------------------------------

# Kill monitoring process
kill $MONITOR_PID 2>/dev/null

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job Summary"
echo "========================================"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"

# Generate results summary
if [ -d results ]; then
    echo ""
    echo "Results Summary:"
    echo "  Output files: $(find results -type f | wc -l)"
    echo "  Total size: $(du -sh results | cut -f1)"
    
    # List result types
    echo "  File types:"
    find results -type f | sed 's/.*\.//' | sort | uniq -c | while read count ext; do
        echo "    .$ext: $count files"
    done
fi

# Create job summary file
cat > job_summary_${SLURM_JOB_ID}.txt << EOSUMMARY
Job Summary Report
==================
Job ID: $SLURM_JOB_ID
EOF
    
    echo "Job Name: $JOB_NAME" >> /tmp/slurm_script.tmp
    
    cat >> /tmp/slurm_script.tmp << 'EOF'
Start Time: $SLURM_JOB_START
End Time: $(date)
Exit Code: $EXIT_CODE
Nodes: $SLURM_NODELIST
EOF
    
    echo "CPUs Used: $CORES" >> /tmp/slurm_script.tmp
    echo "Memory Requested: $MEMORY" >> /tmp/slurm_script.tmp
    
    cat >> /tmp/slurm_script.tmp << 'EOF'

Resource Usage:
$(tail -5 resource_monitor.log)

Output Summary:
$([ -d results ] && find results -type f | wc -l || echo "0") files generated
EOSUMMARY

echo ""
echo "Job summary saved to: job_summary_${SLURM_JOB_ID}.txt"
echo "========================================"

# Exit with captured code
exit $EXIT_CODE
EOF
    
    # Upload the SLURM script
    echo -e "${YELLOW}Uploading SLURM script...${NC}"
    
    # Copy script to remote
    expect -c "
    set timeout 60
    spawn scp /tmp/slurm_script.tmp $USERNAME@$HOST:$project_dir/run_job.slurm
    expect \"password:\"
    send \"$PASSWORD\r\"
    expect eof
    " 2>/dev/null
    
    # Make script executable
    remote_exec "chmod +x $project_dir/run_job.slurm"
    
    # Submit the job
    echo -e "\n${YELLOW}Submitting job to queue...${NC}"
    
    local submit_output=$(remote_exec "cd $project_dir && sbatch run_job.slurm")
    
    if [[ $submit_output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        local job_id=${BASH_REMATCH[1]}
        echo -e "${GREEN}✓ Job submitted successfully!${NC}"
        echo -e "${GREEN}Job ID: $job_id${NC}"
        echo -e "${GREEN}Project: $project_dir${NC}"
        
        # Save job information
        cat > /tmp/last_job_info.tmp << EOINFO
JOB_ID=$job_id
PROJECT_DIR=$project_dir
JOB_NAME=$JOB_NAME
SUBMIT_TIME=$(date)
EOINFO
        
        # Start monitoring if requested
        if [[ $MONITOR =~ ^[Yy]$ ]]; then
            sleep 2
            monitor_job_enhanced $job_id "$project_dir"
        else
            echo -e "\n${YELLOW}Job Management Commands:${NC}"
            echo -e "  Status:     squeue -j $job_id"
            echo -e "  Output:     tail -f $project_dir/${JOB_NAME}_${job_id}.out"
            echo -e "  Error:      tail -f $project_dir/${JOB_NAME}_${job_id}.err"
            echo -e "  Cancel:     scancel $job_id"
            echo -e "  Efficiency: seff $job_id (after completion)"
        fi
    else
        echo -e "${RED}✗ Failed to submit job${NC}"
        echo -e "Output: $submit_output"
        
        # Try to diagnose the issue
        echo -e "\n${YELLOW}Checking for common issues...${NC}"
        
        # Check script syntax
        local syntax_check=$(remote_exec "cd $project_dir && bash -n run_job.slurm 2>&1")
        if [ -n "$syntax_check" ]; then
            echo -e "${RED}Script syntax error: $syntax_check${NC}"
        fi
        
        return 1
    fi
    
    # Cleanup temp files
    rm -f /tmp/slurm_script.tmp
}

# Function to upload images in batch
upload_images_batch() {
    local source_dir="$1"
    local project_dir="$2"
    
    echo -e "\n${YELLOW}Uploading image batch...${NC}"
    
    # Create image directory
    remote_exec "mkdir -p $project_dir/image_batch"
    
    # Find all image files
    local image_count=$(find "$source_dir" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tif" -o -iname "*.tiff" \) 2>/dev/null | wc -l)
    
    if [ $image_count -eq 0 ]; then
        echo -e "${YELLOW}No images found in $source_dir${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Found $image_count images to upload${NC}"
    
    # Calculate total size
    local total_size=$(du -sh "$source_dir" 2>/dev/null | cut -f1)
    echo -e "${BLUE}Total size: $total_size${NC}"
    
    # Upload using tar streaming for efficiency
    echo -e "${YELLOW}Uploading images...${NC}"
    
    # Create tar on the fly and stream to remote
    tar -czf - -C "$source_dir" . | \
    expect -c "
    set timeout -1
    spawn ssh $USERNAME@$HOST \"cd $project_dir/image_batch && tar -xzf -\"
    expect \"password:\"
    send \"$PASSWORD\r\"
    expect eof
    " | while read line; do
        echo -ne "\r${BLUE}Transfer in progress...${NC}                    "
    done
    
    # Verify upload
    local uploaded_count=$(remote_exec "find $project_dir/image_batch -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) | wc -l")
    
    echo -e "\n${GREEN}✓ Uploaded $uploaded_count images${NC}"
    
    # Create image manifest
    echo -e "${YELLOW}Creating image manifest...${NC}"
    remote_exec "cd $project_dir && find image_batch -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.bmp' -o -iname '*.tif' -o -iname '*.tiff' \) > image_manifest.txt"
    
    return 0
}

# Enhanced job monitoring with real-time updates
monitor_job_enhanced() {
    local job_id=$1
    local project_dir=$2
    
    echo -e "\n${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              Real-time Job Monitoring                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}Job ID: $job_id | Press Ctrl+C to stop monitoring${NC}\n"
    
    # Initialize monitoring variables
    local prev_status=""
    local start_time=$(date +%s)
    local line_count=0
    
    # Create monitoring loop
    while true; do
        # Get current job status
        local current_status=$(remote_exec "squeue -j $job_id -h -o '%T|%M|%l|%S|%e' 2>/dev/null")
        
        if [ -z "$current_status" ]; then
            # Job completed
            echo -e "\n${GREEN}✓ Job completed!${NC}"
            break
        fi
        
        # Parse status fields
        IFS='|' read -r state time_used time_limit start_time end_time <<< "$current_status"
        
        # Update display if status changed
        if [ "$current_status" != "$prev_status" ]; then
            prev_status="$current_status"
            
            # Clear previous line
            echo -ne "\033[K"
            
            # Color code based on state
            case $state in
                RUNNING)
                    echo -ne "\r${GREEN}● RUNNING${NC}"
                    ;;
                PENDING)
                    echo -ne "\r${YELLOW}● PENDING${NC}"
                    ;;
                *)
                    echo -ne "\r${BLUE}● $state${NC}"
                    ;;
            esac
            
            # Display time info
            echo -ne " | Time: $time_used/$time_limit"
            
            # Get queue position if pending
            if [ "$state" = "PENDING" ]; then
                local queue_pos=$(remote_exec "squeue -u $USERNAME --sort=+i -h | grep -n $job_id | cut -d: -f1")
                [ -n "$queue_pos" ] && echo -ne " | Queue position: $queue_pos"
            fi
            
            echo -ne " | $(date +%H:%M:%S)    "
        fi
        
        # Check for output file and display tail if running
        if [ "$state" = "RUNNING" ] && [ -f /tmp/hpc_job_config.tmp ]; then
            source /tmp/hpc_job_config.tmp
            local out_file="$project_dir/${JOB_NAME}_${job_id}.out"
            local new_lines=$(remote_exec "tail -n +$((line_count + 1)) $out_file 2>/dev/null")
            
            if [ -n "$new_lines" ]; then
                echo -e "\n${CYAN}Output:${NC}"
                echo "$new_lines" | tail -10
                line_count=$(remote_exec "wc -l < $out_file 2>/dev/null || echo 0")
            fi
        fi
        
        sleep 5
    done
    
    # Job completed - show summary
    echo -e "\n${CYAN}Retrieving job summary...${NC}"
    
    # Get job efficiency
    local efficiency=$(remote_exec "seff $job_id 2>/dev/null")
    if [ -n "$efficiency" ]; then
        echo -e "\n${MAGENTA}Job Efficiency Report:${NC}"
        echo "$efficiency"
    fi
    
    # Check for errors
    if [ -f /tmp/hpc_job_config.tmp ]; then
        source /tmp/hpc_job_config.tmp
        local err_file="$project_dir/${JOB_NAME}_${job_id}.err"
        local error_lines=$(remote_exec "wc -l < $err_file 2>/dev/null || echo 0")
        
        if [ "$error_lines" -gt 0 ]; then
            echo -e "\n${YELLOW}⚠ Error file contains $error_lines lines${NC}"
            read -p "View error file? (y/n): " view_errors
            if [[ $view_errors =~ ^[Yy]$ ]]; then
                remote_exec "tail -50 $err_file"
            fi
        fi
    fi
    
    # Check for results
    echo -e "\n${CYAN}Checking for results...${NC}"
    
    if remote_exec "test -d $project_dir/results && echo 'EXISTS'" | grep -q "EXISTS"; then
        local result_count=$(remote_exec "find $project_dir/results -type f | wc -l")
        local result_size=$(remote_exec "du -sh $project_dir/results | cut -f1")
        
        echo -e "${GREEN}✓ Found $result_count result files (Total: $result_size)${NC}"
        
        # Show result summary
        echo -e "\n${MAGENTA}Result file types:${NC}"
        remote_exec "find $project_dir/results -type f -name '*.*' | sed 's/.*\\.//' | sort | uniq -c"
        
        read -p $'\nDownload results? (y/n): ' download
        if [[ $download =~ ^[Yy]$ ]]; then
            download_results_enhanced "$project_dir"
        fi
    else
        echo -e "${YELLOW}No results directory found${NC}"
    fi
}

# Enhanced results download with progress
download_results_enhanced() {
    local project_dir="$1"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local job_name="unknown"
    
    # Try to get job name from config
    if [ -f /tmp/hpc_job_config.tmp ]; then
        source /tmp/hpc_job_config.tmp
        job_name=$JOB_NAME
    fi
    
    local local_dir="./results_${job_name}_$timestamp"
    
    echo -e "\n${YELLOW}Preparing to download results...${NC}"
    
    # Create local directory structure
    mkdir -p "$local_dir"/{results,logs,metadata}
    
    # Download results
    echo -e "${YELLOW}Downloading result files...${NC}"
    
    expect -c "
    set timeout -1
    spawn bash -c \"ssh $USERNAME@$HOST 'cd $project_dir && tar -czf - results/' | tar -xzf - -C $local_dir\"
    expect \"password:\"
    send \"$PASSWORD\r\"
    expect eof
    "
    
    # Download logs
    echo -e "${YELLOW}Downloading log files...${NC}"
    
    for log_pattern in "*.out" "*.err" "job_summary_*.txt" "resource_monitor.log"; do
        expect -c "
        set timeout 300
        spawn scp -q \"$USERNAME@$HOST:$project_dir/$log_pattern\" \"$local_dir/logs/\" 
        expect \"password:\"
        send \"$PASSWORD\r\"
        expect eof
        " 2>/dev/null
    done
    
    # Download configuration files
    echo -e "${YELLOW}Downloading configuration...${NC}"
    
    for config_file in "run_job.slurm" "config.json" "requirements.txt" "image_manifest.txt"; do
        expect -c "
        set timeout 60
        spawn scp -q \"$USERNAME@$HOST:$project_dir/$config_file\" \"$local_dir/metadata/\" 
        expect \"password:\"
        send \"$PASSWORD\r\"
        expect eof
        " 2>/dev/null
    done
    
    # Create download summary
    cat > "$local_dir/download_summary.txt" << EOSUMMARY
Download Summary
================
Downloaded: $(date)
Source: $project_dir
Job Name: $job_name

Contents:
---------
Results: $(find "$local_dir/results" -type f 2>/dev/null | wc -l) files
Logs: $(find "$local_dir/logs" -type f 2>/dev/null | wc -l) files
Metadata: $(find "$local_dir/metadata" -type f 2>/dev/null | wc -l) files

Total Size: $(du -sh "$local_dir" | cut -f1)
EOSUMMARY
    
    echo -e "\n${GREEN}✓ Download complete!${NC}"
    echo -e "${GREEN}Location: $local_dir${NC}"
    
    # Display summary
    cat "$local_dir/download_summary.txt"
    
    # Offer to open results directory
    if command -v xdg-open &> /dev/null; then
        read -p $'\nOpen results directory? (y/n): ' open_dir
        [[ $open_dir =~ ^[Yy]$ ]] && xdg-open "$local_dir"
    elif command -v open &> /dev/null; then
        read -p $'\nOpen results directory? (y/n): ' open_dir
        [[ $open_dir =~ ^[Yy]$ ]] && open "$local_dir"
    fi
}

# Function to check cluster status
check_cluster_status() {
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}                    Cluster Status                          ${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
    
    # Check user's jobs
    echo -e "\n${MAGENTA}Your Jobs:${NC}"
    local my_jobs=$(remote_exec "squeue -u $USERNAME -o '%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R' 2>/dev/null")
    if [ -n "$my_jobs" ]; then
        echo "$my_jobs"
    else
        echo "No active jobs"
    fi
    
    # Check queue status
    echo -e "\n${MAGENTA}Queue Summary:${NC}"
    remote_exec "squeue -o '%.9P %.5D %.4c %.8m %.8T %C' | awk 'NR==1 || \$5!=\"IDLE\"' | sort | uniq -c"
    
    # Check node availability
    echo -e "\n${MAGENTA}Node Status:${NC}"
    remote_exec "sinfo -o '%.12P %.5D %.10T %.4c %.8m %.8G %.20N' 2>/dev/null" | head -10
    
    # Check recent completed jobs
    echo -e "\n${MAGENTA}Recent Completed Jobs:${NC}"
    remote_exec "sacct -u $USERNAME --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS --starttime=\$(date -d '24 hours ago' +%Y-%m-%d) 2>/dev/null | head -10"
    
    # Disk usage
    echo -e "\n${MAGENTA}Disk Usage:${NC}"
    remote_exec "df -h $SCRATCH | tail -1"
    
    local disk_usage=$(remote_exec "du -sh $SCRATCH 2>/dev/null | cut -f1")
    echo "Your usage: $disk_usage"
}

# Function to clean up old files
cleanup_old_files() {
    echo -e "\n${CYAN}Cleanup Utility${NC}"
    
    # Show current usage
    echo -e "\n${YELLOW}Current disk usage:${NC}"
    remote_exec "du -sh $SCRATCH/* 2>/dev/null | sort -hr | head -20"
    
    # Find old directories
    echo -e "\n${YELLOW}Directories older than 30 days:${NC}"
    local old_dirs=$(remote_exec "find $SCRATCH -maxdepth 1 -type d -mtime +30 -name 'fiber_optic_detector_*' 2>/dev/null")
    
    if [ -z "$old_dirs" ]; then
        echo "No old directories found"
        return
    fi
    
    echo "$old_dirs"
    
    read -p $'\nDelete these directories? (y/n): ' confirm_delete
    if [[ $confirm_delete =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Deleting old directories...${NC}"
        remote_exec "find $SCRATCH -maxdepth 1 -type d -mtime +30 -name 'fiber_optic_detector_*' -exec rm -rf {} \; 2>/dev/null"
        echo -e "${GREEN}✓ Cleanup complete${NC}"
        
        # Show new usage
        echo -e "\n${YELLOW}Updated disk usage:${NC}"
        remote_exec "du -sh $SCRATCH/* 2>/dev/null | sort -hr | head -10"
    fi
}

# Download results (basic version)
download_results() {
    local project_dir=$1
    local local_dir="./results_$(date +%Y%m%d_%H%M%S)"
    
    echo -e "\n${YELLOW}Downloading results to $local_dir...${NC}"
    mkdir -p "$local_dir"
    
    # Download using tar method for reliability
    cat > /tmp/download_results.exp << 'EOEXP'
#!/usr/bin/expect
set timeout 1200
log_user 0
spawn bash -c "ssh $::env(USERNAME)@$::env(HOST) 'cd $::env(PROJECT_DIR) && tar czf - results/' | tar xzf - -C $::env(LOCAL_DIR)"
expect {
    "password:" {
        send "$::env(PASSWORD)\r"
        exp_continue
    }
    eof
}
EOEXP
    
    chmod +x /tmp/download_results.exp
    PROJECT_DIR=$project_dir LOCAL_DIR=$local_dir /tmp/download_results.exp
    rm -f /tmp/download_results.exp
    
    # Also get logs
    mkdir -p "$local_dir/logs"
    for logtype in "out" "err"; do
        cat > /tmp/download_log_$logtype.exp << 'EOEXP'
#!/usr/bin/expect
set timeout 300
log_user 0
spawn scp "$::env(USERNAME)@$::env(HOST):$::env(PROJECT_DIR)/*.$::env(LOGTYPE)" "$::env(LOCAL_DIR)/logs/"
expect {
    "password:" {
        send "$::env(PASSWORD)\r"
        exp_continue
    }
    eof
}
EOEXP
        chmod +x /tmp/download_log_$logtype.exp
        PROJECT_DIR=$project_dir LOCAL_DIR=$local_dir LOGTYPE=$logtype /tmp/download_log_$logtype.exp 2>/dev/null
        rm -f /tmp/download_log_$logtype.exp
    done
    
    echo -e "${GREEN}Results downloaded to $local_dir${NC}"
    
    # Show summary
    if [ -d "$local_dir" ]; then
        local file_count=$(find "$local_dir" -type f | wc -l)
        echo -e "${GREEN}Downloaded $file_count files${NC}"
    fi
}

# Enhanced upload custom directory function
upload_custom_directory() {
    local custom_dir="$1"
    
    if [ ! -d "$custom_dir" ]; then
        read -p "Enter the full path to the directory you want to upload: " custom_dir
        if [ ! -d "$custom_dir" ]; then
            echo -e "${RED}Error: Directory not found: $custom_dir${NC}"
            return 1
        fi
    fi
    
    custom_dir=$(cd "$custom_dir" && pwd)
    local dir_name=$(basename "$custom_dir")
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local remote_dir="$SCRATCH/${dir_name}_$timestamp"
    
    echo -e "\n${YELLOW}Analyzing directory...${NC}"
    
    # Get directory statistics
    local file_count=$(find "$custom_dir" -type f | wc -l)
    local dir_count=$(find "$custom_dir" -type d | wc -l)
    local total_size=$(du -sh "$custom_dir" | cut -f1)
    
    echo -e "${BLUE}Directory: $custom_dir${NC}"
    echo -e "${BLUE}Files: $file_count | Directories: $dir_count | Size: $total_size${NC}"
    
    # Show file type breakdown
    echo -e "\n${MAGENTA}File types:${NC}"
    find "$custom_dir" -type f | sed 's/.*\\.//' | sort | uniq -c | sort -nr | head -10
    
    read -p $'\nProceed with upload? (y/n): ' proceed
    if [[ ! $proceed =~ ^[Yy]$ ]]; then
        return 1
    fi
    
    # Create remote directory
    echo -e "\n${YELLOW}Creating remote directory: $remote_dir${NC}"
    remote_exec "mkdir -p $remote_dir"
    
    # Upload with progress
    echo -e "${YELLOW}Uploading files...${NC}"
    
    if command -v rsync &> /dev/null && [ $file_count -lt 1000 ]; then
        # Use rsync for smaller directories
        expect -c "
        set timeout -1
        spawn rsync -avz --progress \"$custom_dir/\" \"$USERNAME@$HOST:$remote_dir/\"
        expect \"password:\"
        send \"$PASSWORD\r\"
        expect eof
        "
    else
        # Use tar for large directories
        echo -e "${BLUE}Using compressed transfer for $file_count files...${NC}"
        
        tar -czf - -C "$custom_dir" . | \
        expect -c "
        set timeout -1
        spawn ssh $USERNAME@$HOST \"cd $remote_dir && tar -xzf -\"
        expect \"password:\"
        send \"$PASSWORD\r\"
        expect eof
        "
    fi
    
    # Verify upload
    echo -e "\n${YELLOW}Verifying upload...${NC}"
    local uploaded_files=$(remote_exec "find $remote_dir -type f | wc -l")
    local uploaded_size=$(remote_exec "du -sh $remote_dir | cut -f1")
    
    echo -e "${GREEN}✓ Uploaded $uploaded_files files ($uploaded_size) to:${NC}"
    echo -e "${GREEN}  $remote_dir${NC}"
    
    # Check for common project files
    echo -e "\n${YELLOW}Checking for project files...${NC}"
    
    local project_files="Makefile CMakeLists.txt setup.py requirements.txt package.json Cargo.toml go.mod pom.xml build.gradle"
    local found_project_files=""
    
    for pf in $project_files; do
        if remote_exec "test -f $remote_dir/$pf && echo 'FOUND'" | grep -q "FOUND"; then
            found_project_files="$found_project_files $pf"
        fi
    done
    
    if [ -n "$found_project_files" ]; then
        echo -e "${GREEN}Found project files:$found_project_files${NC}"
        
        read -p "Would you like to create a job submission script? (y/n): " create_job
        if [[ $create_job =~ ^[Yy]$ ]]; then
            echo "$remote_dir" > /tmp/hpc_project_dir.tmp
            configure_job && submit_job
        fi
    else
        echo -e "${YELLOW}No standard project files found${NC}"
        
        read -p "Create a basic job script anyway? (y/n): " create_basic
        if [[ $create_basic =~ ^[Yy]$ ]]; then
            echo "$remote_dir" > /tmp/hpc_project_dir.tmp
            configure_job && submit_job
        fi
    fi
}

# Check dependencies with more detailed information
check_dependencies() {
    local missing_deps=""
    
    echo -e "${YELLOW}Checking dependencies...${NC}"
    
    # Check for expect
    if ! command -v expect &> /dev/null; then
        missing_deps="expect"
    fi
    
    # Check for optional but useful tools
    local optional_tools="rsync pv bc"
    local missing_optional=""
    
    for tool in $optional_tools; do
        if ! command -v $tool &> /dev/null; then
            missing_optional="$missing_optional $tool"
        fi
    done
    
    # Report findings
    if [ -n "$missing_deps" ]; then
        echo -e "${RED}Error: Required dependencies missing: $missing_deps${NC}"
        echo
        echo "Installation instructions:"
        echo "  Ubuntu/Debian: sudo apt-get install $missing_deps"
        echo "  RHEL/CentOS:   sudo yum install $missing_deps"
        echo "  macOS:         brew install $missing_deps"
        return 1
    fi
    
    if [ -n "$missing_optional" ]; then
        echo -e "${YELLOW}Optional tools not found:$missing_optional${NC}"
        echo "Some features may be limited. Install for full functionality."
    fi
    
    echo -e "${GREEN}✓ All required dependencies found${NC}"
    return 0
}

# Main menu with enhanced options
main_menu() {
    while true; do
        show_header
        
        # Check for active jobs
        local active_jobs=$(remote_exec "squeue -u $USERNAME -h 2>/dev/null | wc -l")
        # Ensure active_jobs is a number
        if [ -z "$active_jobs" ] || ! [[ "$active_jobs" =~ ^[0-9]+$ ]]; then
            active_jobs=0
        fi
        
        if [ "$active_jobs" -gt 0 ]; then
            echo -e "${GREEN}● You have $active_jobs active job(s)${NC}\n"
        fi
        
        echo -e "${YELLOW}Main Menu:${NC}"
        echo
        echo "  1) Upload and submit fiber optic detection job"
        echo "  2) Upload custom directory"
        echo "  3) Monitor existing job"
        echo "  4) Download results from previous job"
        echo "  5) Check cluster status"
        echo "  6) View disk usage and cleanup"
        echo "  7) Advanced job configuration"
        echo "  8) Exit"
        echo
        read -p "Select option (1-8): " choice
        
        case $choice in
            1)
                show_header
                echo -e "${CYAN}Fiber Optic Detection Pipeline${NC}\n"
                
                read -p "Enter path to detection software [.]: " detect_dir
                detect_dir=${detect_dir:-.}
                
                if upload_detection_software "$detect_dir"; then
                    if configure_job; then
                        submit_job
                    fi
                fi
                
                read -p $'\nPress Enter to continue...'
                ;;
                
            2)
                show_header
                echo -e "${CYAN}Upload Custom Directory${NC}\n"
                
                upload_custom_directory
                
                read -p $'\nPress Enter to continue...'
                ;;
                
            3)
                show_header
                echo -e "${CYAN}Job Monitoring${NC}\n"
                
                # Show active jobs
                local user_jobs=$(remote_exec "squeue -u $USERNAME -o '%.18i %.30j %.8T %.10M' 2>/dev/null")
                if [ -n "$user_jobs" ]; then
                    echo -e "${MAGENTA}Your active jobs:${NC}"
                    echo "$user_jobs"
                    echo
                fi
                
                # Check for last job
                if [ -f /tmp/last_job_info.tmp ]; then
                    source /tmp/last_job_info.tmp
                    echo -e "${BLUE}Last submitted job: $JOB_ID ($JOB_NAME)${NC}"
                fi
                
                read -p "Enter job ID to monitor: " monitor_id
                if [ -n "$monitor_id" ]; then
                    # Get project directory for the job
                    local job_dir=$(remote_exec "scontrol show job $monitor_id 2>/dev/null | grep WorkDir | cut -d= -f2")
                    if [ -n "$job_dir" ]; then
                        monitor_job_enhanced "$monitor_id" "$job_dir"
                    else
                        echo -e "${RED}Job $monitor_id not found${NC}"
                    fi
                fi
                
                read -p $'\nPress Enter to continue...'
                ;;
                
            4)
                show_header
                echo -e "${CYAN}Download Results${NC}\n"
                
                # Show recent directories
                echo -e "${MAGENTA}Recent project directories:${NC}"
                remote_exec "ls -ltd $SCRATCH/fiber_optic_detector_* 2>/dev/null | head -10"
                
                echo
                read -p "Enter project directory path: " project_path
                
                if [ -n "$project_path" ]; then
                    if remote_exec "test -d $project_path && echo 'EXISTS'" | grep -q "EXISTS"; then
                        # Load job config if available
                        if [ -f /tmp/last_job_info.tmp ]; then
                            source /tmp/last_job_info.tmp
                            if [ "$PROJECT_DIR" = "$project_path" ]; then
                                # Restore job config for proper download
                                download_results_enhanced "$project_path"
                            else
                                # Basic download without config
                                download_results "$project_path"
                            fi
                        else
                            download_results "$project_path"
                        fi
                    else
                        echo -e "${RED}Directory not found: $project_path${NC}"
                    fi
                fi
                
                read -p $'\nPress Enter to continue...'
                ;;
                
            5)
                show_header
                check_cluster_status
                read -p $'\nPress Enter to continue...'
                ;;
                
            6)
                show_header
                echo -e "${CYAN}Disk Usage Management${NC}\n"
                
                echo -e "${MAGENTA}Storage Summary:${NC}"
                remote_exec "df -h | grep -E '(Filesystem|scr-lst)'"
                
                echo -e "\n${MAGENTA}Your disk usage breakdown:${NC}"
                remote_exec "du -sh $SCRATCH/* 2>/dev/null | sort -hr | head -20"
                
                echo -e "\n${MAGENTA}Options:${NC}"
                echo "1) Clean up old files (>30 days)"
                echo "2) Delete specific directory"
                echo "3) Return to main menu"
                
                read -p "Select option: " cleanup_choice
                
                case $cleanup_choice in
                    1) cleanup_old_files ;;
                    2)
                        read -p "Enter directory to delete: " del_dir
                        if remote_exec "test -d $del_dir && echo 'EXISTS'" | grep -q "EXISTS"; then
                            remote_exec "du -sh $del_dir"
                            read -p "Confirm deletion? (y/n): " confirm
                            if [[ $confirm =~ ^[Yy]$ ]]; then
                                remote_exec "rm -rf $del_dir"
                                echo -e "${GREEN}✓ Deleted${NC}"
                            fi
                        else
                            echo -e "${RED}Directory not found${NC}"
                        fi
                        ;;
                esac
                
                read -p $'\nPress Enter to continue...'
                ;;
                
            7)
                show_header
                echo -e "${CYAN}Advanced Job Configuration${NC}\n"
                
                echo "1) Create custom SLURM script"
                echo "2) Submit pre-existing SLURM script"
                echo "3) Interactive job session"
                echo "4) Job array configuration"
                
                read -p "Select option: " adv_choice
                
                case $adv_choice in
                    1)
                        echo "Opening nano to create custom SLURM script..."
                        local custom_script="/tmp/custom_job_$$.slurm"
                        nano "$custom_script"
                        
                        if [ -f "$custom_script" ]; then
                            read -p "Upload and submit this script? (y/n): " submit_custom
                            if [[ $submit_custom =~ ^[Yy]$ ]]; then
                                local remote_script="$SCRATCH/custom_job_$(date +%Y%m%d_%H%M%S).slurm"
                                
                                expect -c "
                                spawn scp \"$custom_script\" \"$USERNAME@$HOST:$remote_script\"
                                expect \"password:\"
                                send \"$PASSWORD\r\"
                                expect eof
                                "
                                
                                local submit_out=$(remote_exec "cd $SCRATCH && sbatch $(basename $remote_script)")
                                echo "$submit_out"
                            fi
                            rm -f "$custom_script"
                        fi
                        ;;
                    2)
                        read -p "Enter path to SLURM script: " slurm_script
                        if [ -f "$slurm_script" ]; then
                            local remote_script="$SCRATCH/$(basename $slurm_script)"
                            
                            expect -c "
                            spawn scp \"$slurm_script\" \"$USERNAME@$HOST:$remote_script\"
                            expect \"password:\"
                            send \"$PASSWORD\r\"
                            expect eof
                            "
                            
                            local submit_out=$(remote_exec "cd $SCRATCH && sbatch $(basename $remote_script)")
                            echo "$submit_out"
                        else
                            echo -e "${RED}Script not found${NC}"
                        fi
                        ;;
                    3)
                        echo -e "${YELLOW}Requesting interactive session...${NC}"
                        echo "This will open an SSH session with an interactive job."
                        read -p "Number of cores [4]: " int_cores
                        int_cores=${int_cores:-4}
                        read -p "Time (HH:MM:SS) [1:00:00]: " int_time
                        int_time=${int_time:-1:00:00}
                        
                        echo -e "${BLUE}Connecting...${NC}"
                        expect -c "
                        spawn ssh $USERNAME@$HOST
                        expect \"password:\"
                        send \"$PASSWORD\r\"
                        expect \"$ \"
                        send \"srun --pty --cpus-per-task=$int_cores --time=$int_time /bin/bash\r\"
                        interact
                        "
                        ;;
                esac
                
                read -p $'\nPress Enter to continue...'
                ;;
                
            8)
                echo -e "\n${GREEN}Thank you for using HPC Interactive Manager!${NC}"
                echo -e "${YELLOW}Cleaning up temporary files...${NC}"
                
                # Cleanup
                rm -f /tmp/hpc_*.tmp /tmp/last_job*.tmp /tmp/*.exp
                
                echo -e "${GREEN}Goodbye!${NC}\n"
                exit 0
                ;;
                
            *)
                echo -e "${RED}Invalid option. Please select 1-8.${NC}"
                sleep 2
                ;;
        esac
    done
}

# Main execution function
main() {
    show_header
    
    # Check dependencies
    if ! check_dependencies; then
        exit 1
    fi
    
    # Test connection
    echo
    if ! check_connection; then
        echo -e "\n${RED}Cannot establish connection to Kuro cluster.${NC}"
        echo "Please check:"
        echo "  1. Network connectivity"
        echo "  2. VPN connection (if required)"
        echo "  3. SSH service status"
        echo "  4. Credentials"
        exit 1
    fi
    
    # Check for updates or messages
    if remote_exec "test -f /etc/motd && echo 'EXISTS'" | grep -q "EXISTS"; then
        echo -e "\n${CYAN}System Messages:${NC}"
        remote_exec "cat /etc/motd 2>/dev/null | head -10"
    fi
    
    # Start main menu
    sleep 1
    main_menu
}

# Signal handlers for clean exit
trap 'echo -e "\n${YELLOW}Interrupted. Cleaning up...${NC}"; rm -f /tmp/hpc_*.tmp /tmp/*.exp /tmp/slurm_script.tmp; exit 130' INT TERM

# Run the main function
main
