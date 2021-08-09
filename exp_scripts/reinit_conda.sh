#!/bin/bash
if [ $# -lt 2 ]; then
  echo "$#"
  echo "$0 <conda_env_path> <avalanche_source_path>"
  exit 1
else
  echo  "$@"
fi

conda_env_path=$1
avalanche_source_path=$2
conda_yml_file="${avalanche_source_path}/environment-dev.yml"

eval "$(conda shell.bash hook)"
conda init bash

re_init_conda_env ()
{
  echo "Removing conda env $conda_env_path"
  conda remove  --prefix $conda_env_path --all

  echo "Creating conda env $conda_env_path from config $conda_yml_file"
  conda env create --prefix $conda_env_path --file $conda_yml_file

  echo "Updating conda env $conda_env_path from config $conda_yml_file"
  conda env update --prefix $conda_env_path --file $conda_yml_file  --prune

  conda activate $conda_env_path
  conda env list

  #pip3 install -e $avalanche_source_path
  pip install -U "$avalanche_source_path"
}

re_init_conda_env
