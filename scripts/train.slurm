#!/bin/bash

#############################
# les directives Slurm vont ici:

# Your job name (displayed by the queue)
#SBATCH -J JOB_NAME
# walltime (hh:mm::ss)
#SBATCH -t 16:00:00

# Specify the number of nodes(nodes=) and the number of cores per nodes(tasks-pernode=) to be used
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-4
#SBATCH --output=/gpfs/home/emasquil/logs/LOG_DIR/slurm-%A_%a.out
# change working directory

# fin des directives PBS
#############################

# useful informations to print
echo "#############################"
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################"

#############################

# What you actually want to launch

python train_rllib.py --environment large_goal_lines configs/CONFIG_YML /gpfs/home/emasquil/runs/RUN_DIR

# all done
echo "Job finished"
