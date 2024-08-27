#!/bin/bash
#PBS -l select=1:ncpus=4:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=3:00:00
#PBS -N p-dcm_fitting_all_vars_sim
#PBS -o /rds/general/user/ag1523/home/fyp-main/pytorch_models/new_logs
#PBS -e /rds/general/user/ag1523/home/fyp-main/pytorch_models/new_logs


eval "$(~/anaconda3/bin/conda shell.bash hook)"
source activate pytorch_env
echo ${PBS_O_WORKDIR}
cd ${PBS_O_WORKDIR}

python3 time_domain_model_simulated.py




