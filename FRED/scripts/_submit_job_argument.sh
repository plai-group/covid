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

echo '# hostname = '`hostname`

if [ -z $SLURM_ARRAY_TASK_ID ]
then
    singularity exec --nv -B /project/def-fwood/saeidnp/rejection -B /scratch/saeidnp /project/def-fwood/saeidnp/singularity_images/rejection.sif python $args

else
    singularity exec --nv -B /project/def-fwood/saeidnp/rejection -B /scratch/saeidnp /project/def-fwood/saeidnp/singularity_images/rejection.sif python $args level_2=${exp_name} level_3=${SLURM_ARRAY_TASK_ID}
fi
