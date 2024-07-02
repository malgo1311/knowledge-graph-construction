#!/bin/bash

#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=18GB  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 16:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --mail-type=BEGIN

/bin/true

module load miniconda/22.11.1-1

conda activate hw1
python3 /work/pi_dhruveshpate_umass_edu/project_19/aishwarya/696DS-named-entity-extraction-and-linking-for-KG-construction/code/openai/openai_sd_apr25_rel.py