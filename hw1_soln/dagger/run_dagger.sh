#!/usr/bin/env bash

set -e
set -u
set -o pipefail

num_iters=$1
num_rollouts=$2
model_path=$3
expert_policy_path=$4
mean_path=$5
train_data_path=$6
model_data_path=$7

# python run_model_dagger.py --model_path $model_path --mean_path $mean_path --model_data_path $model_data_path \
# --num_rollouts $num_rollouts --train_data_path $train_data_path

# python run_expert_dagger.py $expert_policy_path --model_data_path $model_data_path --train_path $train_data_path

for (( i=1; i<=${num_iters}; i++ ))
do
	python run_model_dagger.py --model_path $model_path --mean_path $mean_path --model_data_path $model_data_path \
	--num_rollouts $num_rollouts --train_path $train_data_path

	python run_expert_dagger.py $expert_policy_path --model_data_path $model_data_path --train_path $train_data_path
done
