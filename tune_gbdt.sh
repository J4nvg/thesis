#!/bin/bash
#SBATCH -p GPUExtended          # Use GPUExtended since tuning takes a while
#SBATCH -N 1                    # Request 1 Node
#SBATCH --cpus-per-task=64      # Request all 64 CPU cores
#SBATCH -t 1-00:00:00           # Set time limit to 1 day (Adjust if needed)
#SBATCH -o slurm_gbdt_%j.out  # Standard output log (%j adds the job ID)
#SBATCH -e slurm_gbdt_%j.err  # Standard error log

# Initialize Conda (based on GPU4EDU tutorial)
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

# Activate your specific environment
conda activate thesis

python _regression_GBDT.py
