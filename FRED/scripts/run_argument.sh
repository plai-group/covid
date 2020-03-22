#!/bin/bash
export now=$(date +"%Y_%m_%d_%H_%M_%S" )
echo $now

root_dir='/project/def-fwood/saeidnp/rejection'

declare -A options
declare -a flags
slurm_mem="16G"
slurm_time="24:00:00"
slurm_gpu=0
slurm_cores=1
slurm_nodes=1
array_flag=false
array_first=0
array_last=2
parse_args() {
  if [[ $# -ge 1 ]]; then
    case $1 in
      --exp_name)
        exp_name=$2 num_shift=2;;
      --mem)
        slurm_mem=$2 num_shift=2;;
      --time)
        slurm_time=$2 num_shift=2;;
      --gpu)
        slurm_gpu=$2 num_shift=2;;
      --cores)
        slurm_cores=$2 num_shift=2;;
      --nodes)
        slurm_nodes=$2 num_shift=2;;
      --array)
        array_flag=true num_shift=1;;
      --first)
        array_first=$2 num_shift=2;;
      --last)
        array_last=$2 num_shift=2;;
      *)
        args="$args $1" num_shift=1;;
    esac
  else
    num_shift=0
  fi
}
num_shift=-1
while [[ $num_shift -ne 0 ]]; do
  parse_args $@
  shift $num_shift
done
# ----- Done parsing arguments -----

if [ -z $exp_name ]
then
  echo "Experiment name not provided"
  exit 1
fi
echo "done"

export args=$args
export root_dir=$root_dir
export exp_name=$exp_name

echo "arguments = $args"

echo 'submitting job: '$exp_name

## SLURM arguments
slurm_args="--export=ALL -J $exp_name -t ${slurm_time} --mem ${slurm_mem} --nodes ${slurm_nodes} --ntasks-per-node ${slurm_cores}"

# Report file
report_out=${root_dir}/scripts/Reports/'results-%j-%x.out'
report_err=${root_dir}/scripts/Reports/'results-%j-%x.err'
if ${array_flag}
then
  report_out=${root_dir}/scripts/Reports/'results-%A_%a-%x.out'
  report_err=${root_dir}/scripts/Reports/'results-%A_%a-%x.err'
fi
slurm_args="${slurm_args} -o ${report_out} -e ${report_err}"

# GPU
if [ ${slurm_gpu} -gt 0 ]
then
  slurm_args="${slurm_args} --gres=gpu:${slurm_gpu}"
fi

# Job array
if ${array_flag}
then
  slurm_args="${slurm_args} --array=${array_first}-${array_last}"
fi

echo "sbatch ${slurm_args} _submit_job_argument.sh"
bash -c "sbatch ${slurm_args} _submit_job_argument.sh"
