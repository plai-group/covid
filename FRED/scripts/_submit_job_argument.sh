#!/bin/bash

#SBATCH -t 48:00:00
#SBATCH --account=rrg-kevinlb
#SBATCH --mem=16G
#SBATCH --mail-user=saeidnp@cs.ubc.ca
#SBATCH --mail-type=ALL

export LC_CTYPE=en_US.UTF-8
export LC_ALL=en_US.UTF-8
echo ${root_dir}
cd ${root_dir}

module load singularity/3.5

# mount Singlularity cache and tmp directories locally
# taken from here: https://docs.computecanada.ca/wiki/Singularity
# mkdir -p /scratch/$USER/singularity/{cache,tmp}
#export SINGULARITY_CACHEDIR="/scratch/$USER/singularity/cache"
#export SINGULARITY_TMPDIR="/scratch/$USER/singularity/tmp"

SINGULARITY_IMAGE_PATH="/project/def-fwood/$USER/singularity_images/covid.sif"

echo '# hostname = '`hostname`

if [ -z $SLURM_ARRAY_TASK_ID ]
then
    singularity run -B $SCRATCH -B $SLURM_TMPDIR -B $PWD:/workdir $SINGULARITY_IMAGE_PATH python $args out_level_2=${exp_name} tmp_directory=${SLURM_TMPDIR}
else
    singularity run -B $SCRATCH $SINGULARITY_IMAGE_PATH python $args out_level_2=${exp_name} out_level_3=sim${SLURM_ARRAY_TASK_ID} tmp_directory=${SLURM_TMPDIR}
fi
