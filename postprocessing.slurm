#!/bin/bash -l
### job name
#SBATCH --job-name=postprocessing
### 
#SBATCH --mail-user=ferrazzi@math.unipd.it

### Standard output and standard error for our job
#SBATCH --error=postprocessing.err
#SBATCH --output=postprocessing.out

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
####SBATCH --constraint=vgpu1-1

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
ENV_PATH='/home/ferrazzi/.conda/envs/gpu_venv_lates/bin'

${ENV_PATH}/python --version
${ENV_PATH}/python  postprocessing.py
echo -n 'Date: '
date
