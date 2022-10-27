#!/bin/bash

#############################
# les directives Slurm vont ici:

# We consider 8 CPUs, 6 for training and 2 for evaluating 
# The walltime we usually set for a two agents population is: 
# 16 hours for 1000 iterations (commonly used for 6 landmarks)
# 6 hours for 300 iterations (commonly used for 3 landmarks)
# When considering bigger populations, scale the walltime properly (e.g 4 agents 3 landmarks = 12 hours and 600 iterations)


# Your job name (displayed by the queue)
#SBATCH -J imgc_marl
# walltime (hh:mm::ss)
#SBATCH -t 19:59:00
#SBATCH -A imi@cpu
# Specify the number of nodes(nodes=) and the number of cores per nodes(tasks-pernode=) to be used
#SBATCH -N 1
#SBATCH --array=0-4
#SBATCH --cpus-per-task=12
# Set the directory to store logs
#SBATCH -o /gpfsscratch/rech/imi/uzw47io/elias_expe/2_agents/all/modified_reward/6_landmarks_reward2/25align/train-%A_%a.slurm.out
#SBATCH -e /gpfsscratch/rech/imi/uzw47io/elias_expe/2_agents/all/modified_reward/6_landmarks_reward2/25align/train-%A_%a.slurm.err

# change working directory

export TMPDIR=$JOBSCRATCH
# fin des directives PBS
#############################

# useful informations to print
echo "#############################"
echo "User:" ${USER}
echo "SLURM_JOBID:" $SLURM_JOBID
echo "SLURM_SUBMIT_DIR:" $SLURM_SUBMIT_DIR
echo "SLURM_JOB_NODELIST:" $SLURM_JOB_NODELIST
echo "#############################"

#############################

# What you actually want to launch
# Launch the job, change the environment if needed (3 or 6 landmarks), it can be: goal_lines or large_goal_lines
python train_population.py --environment large_goal_lines configs/large_goal_lines_2agents_50align_all_modified.yml ${SCRATCH}/elias_expe/2_agents/all/modified_reward/6_landmarks_reward2/25align/

# all done
echo "Job finished"
