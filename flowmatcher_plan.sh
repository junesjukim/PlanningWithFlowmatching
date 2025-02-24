#!/bin/bash
# Create output directories if they don't exist
mkdir -p output
mkdir -p output/flowmatcher_plan_guideX

n_diffusion_steps=8

# Loop over seed values from 0 to 149
for seed in {0..149}
do
  # GPU 장치 배열 정의
  declare -a GPU_DEVICES=(0 1)
  # 데이터셋 배열 정의  
  declare -a DATASETS=(
    "walker2d-medium-replay-v2"
    "hopper-medium-replay-v2"
  )

  # 각 GPU에서 작업 실행
  pids=()
  for i in "${!GPU_DEVICES[@]}"; do
    OMP_NUM_THREADS=24 CUDA_VISIBLE_DEVICES=${GPU_DEVICES[$i]} python scripts/plan_guided.py \
      --dataset ${DATASETS[$i]} \
      --logbase logs \
      --diffusion_loadpath "f:flowmatching/flowmatcher_H4_T${n_diffusion_steps}_S0" \
      --value_loadpath "f:values/flowmatching_H4_T${n_diffusion_steps}_S0_d0.99" \
      --horizon 4 \
      --n_diffusion_steps ${n_diffusion_steps} \
      --seed $seed \
      --prefix 'plans/guideX' \
      --discount 0.99 > output/flowmatcher_plan_guideX/output_${GPU_DEVICES[$i]}_seed_${seed}.log 2>&1 &

    pids+=($!)
    echo "Started job for seed $seed on GPU ${GPU_DEVICES[$i]}"
  done

  # Wait for all background jobs to finish before moving to the next seed
  wait "${pids[@]}"

  echo "Completed jobs for seed $seed"
done

echo "All jobs have been completed."
