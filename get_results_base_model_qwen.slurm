#!/bin/bash -l
### job name
#SBATCH --job-name=qwen_get_results_base_model_qwen

### 
#SBATCH --mail-user=ferrazzi@math.unipd.it

### Standard output and standard error for our job
#SBATCH --error=get_results_base_model_qwen.err
#SBATCH --output=get_results_base_model_qwen.out

### queue/partition choosed
#SBATCH --partition=testing
### Number of tasks
#SBATCH --ntasks=1

### RAM requirement
#SBATCH --mem=32G

### Time limit for our job (ten minutes here: HH:MM:SS)
#SBATCH --time=24:00:00

### GPU request
#SBATCH --gres=gpu
#SBATCH --constraint=A6000
#####SBATCH --constraint=vgpu2-1

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
python  get_results_base_model_qwen.py
