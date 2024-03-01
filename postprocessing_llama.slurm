#!/bin/bash -l
### job name
#SBATCH --job-name=postprocessing_llama

### 
#SBATCH --mail-user=ferrazzi@math.unipd.it

### Standard output and standard error for our job
#SBATCH --error=postprocessing_llama.err
#SBATCH --output=postprocessing_llama.out

### queue/partition choosed
#SBATCH --partition=testing
### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=32G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=96:00:00

### GPU request
#SBATCH --gres=gpu
####SBATCH --constraint=A6000
####SBATCH --constraint=vgpu5-1

### Some useful informative commands
echo -n 'Date: '
date
echo -n 'Directory: '
pwd
echo -n 'This job will be executed on th following nodes: '
echo ${SLURM_NODELIST}
echo

SHELL=/bin/bash

### Jobs execution commands
conda activate gpu_venv_lates
which python

python --version
python  postprocessing_llama.py
