#!/bin/bash

# Script to first process all datasets, then train on all datasets using 2 GPUs in parallel
# Runs 2 trainings in parallel (one per GPU), starts next training when one finishes

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
N_DEMO=100
CONFIG_NAME="train_maniflow_pointcloud_workspace.yaml"

# All datasets (including stack_d1 which is already processed)
DATASETS=(
    "stack_d1"
    "coffee_d2"
    "coffee_preparation_d1"
    "hammer_cleanup_d1"
    "kitchen_d1"
    "mug_cleanup_d1"
    "nut_assembly_d0"
    "pick_place_d0"
    "square_d2"
    "stack_three_d1"
    "threading_d2"
    "three_piece_assembly_d2"
)

# Available GPUs
GPUS=(0 1)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ManiFlow Full Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 1: Process all datasets${NC}"
echo -e "${BLUE}Step 2: Train on all datasets${NC}"
echo -e "${BLUE}========================================${NC}\n"

# ============================================
# Step 1: Process all datasets
# ============================================
echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Starting dataset processing${NC}"
echo -e "${BLUE}Running process_all_datasets.sh...${NC}\n"

if bash process_all_datasets.sh; then
    echo -e "\n${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Dataset processing completed successfully!${NC}\n"
else
    echo -e "\n${RED}[$(date '+%Y-%m-%d %H:%M:%S')] Step 1: Dataset processing failed!${NC}"
    exit 1
fi

# ============================================
# Step 2: Train on all datasets
# ============================================
echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] Step 2: Starting training on all datasets${NC}"
echo -e "${BLUE}Total datasets: ${#DATASETS[@]}${NC}"
echo -e "${BLUE}Training configuration: ${CONFIG_NAME}${NC}"
echo -e "${BLUE}Number of demos: ${N_DEMO}${NC}"
echo -e "${BLUE}GPUs: ${GPUS[@]} (${#GPUS[@]} GPUs in parallel)${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Track training status
successful_trainings=()
failed_trainings=()
declare -A running_jobs  # Map of PID to dataset name
declare -A job_gpu       # Map of PID to GPU id
dataset_index=0

# Function to start a training job
start_training() {
    local dataset=$1
    local gpu=$2

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Starting training: ${dataset}${NC}"
    echo -e "${BLUE}GPU: cuda:${gpu}${NC}"
    echo -e "${BLUE}Progress: $((${#successful_trainings[@]} + ${#failed_trainings[@]} + 1))/${#DATASETS[@]}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Start training in background
    python train.py \
        --config-name="${CONFIG_NAME}" \
        task_name="${dataset}" \
        n_demo=${N_DEMO} \
        training.device="cuda:${gpu}" &

    local pid=$!
    running_jobs[$pid]=$dataset
    job_gpu[$pid]=$gpu

    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Started ${dataset} on GPU ${gpu} (PID: ${pid})${NC}\n"
}

# Function to wait for any job to finish and return the GPU
wait_for_any_job() {
    while true; do
        for pid in "${!running_jobs[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                # Job finished
                wait $pid
                local exit_code=$?
                local dataset=${running_jobs[$pid]}
                local gpu=${job_gpu[$pid]}

                if [ $exit_code -eq 0 ]; then
                    successful_trainings+=("${dataset}")
                    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully: ${dataset} (GPU ${gpu})${NC}\n"
                else
                    failed_trainings+=("${dataset}")
                    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] Training failed: ${dataset} (GPU ${gpu})${NC}\n"
                fi

                # Clean up
                unset running_jobs[$pid]
                unset job_gpu[$pid]

                # Return the freed GPU
                echo $gpu
                return 0
            fi
        done
        sleep 2
    done
}

# Start initial trainings (one per GPU)
available_gpus=("${GPUS[@]}")
for gpu in "${available_gpus[@]}"; do
    if [ $dataset_index -lt ${#DATASETS[@]} ]; then
        start_training "${DATASETS[$dataset_index]}" "$gpu"
        dataset_index=$((dataset_index + 1))
    fi
done

# Process remaining datasets
while [ $dataset_index -lt ${#DATASETS[@]} ]; do
    # Wait for any training to finish and get the freed GPU
    freed_gpu=$(wait_for_any_job)

    # Start next training on the freed GPU
    if [ $dataset_index -lt ${#DATASETS[@]} ]; then
        start_training "${DATASETS[$dataset_index]}" "$freed_gpu"
        dataset_index=$((dataset_index + 1))
    fi
done

# Wait for all remaining jobs to finish
while [ ${#running_jobs[@]} -gt 0 ]; do
    wait_for_any_job > /dev/null
done

# ============================================
# Final Summary
# ============================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Training Pipeline Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Successful trainings (${#successful_trainings[@]}/${#DATASETS[@]}):${NC}"
for dataset in "${successful_trainings[@]}"; do
    echo -e "${GREEN}  ✓ ${dataset}${NC}"
done

if [ ${#failed_trainings[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed trainings (${#failed_trainings[@]}/${#DATASETS[@]}):${NC}"
    for dataset in "${failed_trainings[@]}"; do
        echo -e "${RED}  ✗ ${dataset}${NC}"
    done
fi

echo -e "${BLUE}========================================${NC}"

# Exit with error if any training failed
if [ ${#failed_trainings[@]} -gt 0 ]; then
    echo -e "${RED}Pipeline completed with errors.${NC}"
    exit 1
else
    echo -e "${GREEN}All trainings completed successfully!${NC}"
    exit 0
fi
