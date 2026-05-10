#!/bin/bash
#SBATCH -p GPU                  # Use the standard GPU partition
#SBATCH -N 1                    # Request 1 Node
#SBATCH --gres=gpu:1            # CRITICAL: Request 1 GPU
#SBATCH --cpus-per-task=4
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
