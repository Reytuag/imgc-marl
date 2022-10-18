#!/bin/bash

#SBATCH -A imi@cpu
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --input=none
#SBATCH --mail-user=mayalen.etcheverry@inria.fr
#SBATCH -o run_experiment.slurm.out
#SBATCH -e run_experiment.slurm.err
#SBATCH --array=0-4


python train_rllib.py --environment goal_lines configs/goal_lines_double_condition.yml /gpfswork/rech/imi/${USER}/elias_experiments/double_condition_3

