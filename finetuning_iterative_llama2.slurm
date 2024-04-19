#!/bin/bash -l
### job name
#SBATCH --job-name=llama_finetuning_iterative

### 
#SBATCH --mail-user=ferrazzi@math.unipd.it

### Standard output and standard error for our job
#SBATCH --error=finetuning_iterative_llama2.err
#SBATCH --output=finetuning_iterative_llama2.out

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
####SBATCH --constraint=vgpu9-0
#SBATCH --exclusive 
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
echo -n 'Transformers: '
conda list -n gpu_venv_lates -f transformers
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

python --version
python  finetuning_iterative_llama2.py
