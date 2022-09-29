#!/bin/bash

#############################
# les directives Slurm vont ici:

# We consider 8 CPUs, 6 for training and 2 for evaluating 
# The walltime we usually set for a two agents population is: 
# 16 hours for 1000 iterations (commonly used for 6 landmarks)
# 6 hours for 300 iterations (commonly used for 3 landmarks)
# When considering bigger populations, scale the walltime properly (e.g 4 agents 3 landmarks = 12 hours and 600 iterations)


# Your job name (displayed by the queue)
#SBATCH -J JOB_NAME
# walltime (hh:mm::ss)
#SBATCH -t 16:00:00

# Specify the number of nodes(nodes=) and the number of cores per nodes(tasks-pernode=) to be used
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --array=0-4
# Set the directory to store logs
#SBATCH --output=/DIRECTORY_TO_STORE_LOGS/slurm-%A_%a.out
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
# Launch the job, change the environment if needed (3 or 6 landmarks), it can be: goal_lines or large_goal_lines
python train_population.py --environment large_goal_lines configs/CONFIG_YML /DIRECTORY_TO_STORE_RESULTS

# all done
echo "Job finished"
