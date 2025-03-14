#!/bin/bash

dataset=("halfcheetah-medium-expert-v2" "halfcheetah-medium-replay-v2")
for n_diffusion_steps in 8 4
do	
	for dataset_i in "${!dataset[@]}"
	do
		OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=1 python scripts/plan_guided.py --dataset "${dataset[$dataset_i]}" --logbase log/ --n_diffusion_steps "$n_diffusion_steps" --seed 0
	done
done