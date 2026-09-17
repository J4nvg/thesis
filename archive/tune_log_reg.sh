#!/bin/bash
#SBATCH -p GPU                  # Use the standard GPU partition
#SBATCH -N 1                    # Request 1 Node
#SBATCH --gres=gpu:1            # CRITICAL: Request 1 GPU 
#SBATCH --cpus-per-task=4
#SBATCH -o slurm_logreg_%j.out
#SBATCH -e slurm_logreg_%j.err

# Initialize Conda
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

conda activate thesis
python _log_regression.py
