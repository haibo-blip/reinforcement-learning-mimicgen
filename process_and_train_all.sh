#!/bin/bash

# Script to first process all datasets, then train on all datasets using 3 GPUs sequentially
# Each training runs on a single GPU at a time

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
GPUS=(0 1 2)

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
echo -e "${BLUE}GPUs: ${GPUS[@]}${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Track training status
successful_trainings=()
failed_trainings=()
gpu_index=0

# Train each dataset sequentially
for dataset in "${DATASETS[@]}"; do
    # Select GPU in round-robin fashion
    gpu=${GPUS[$gpu_index]}
    gpu_index=$(( (gpu_index + 1) % ${#GPUS[@]} ))

    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Training: ${dataset}${NC}"
    echo -e "${BLUE}GPU: cuda:${gpu}${NC}"
    echo -e "${BLUE}Progress: $((${#successful_trainings[@]} + ${#failed_trainings[@]} + 1))/${#DATASETS[@]}${NC}"
    echo -e "${BLUE}========================================${NC}"

    # Run training
    if python train.py \
        --config-name="${CONFIG_NAME}" \
        task_name="${dataset}" \
        n_demo=${N_DEMO} \
        training.device="cuda:${gpu}"; then

        successful_trainings+=("${dataset}")
        echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] Training completed successfully: ${dataset}${NC}\n"
    else
        failed_trainings+=("${dataset}")
        echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] Training failed: ${dataset}${NC}\n"

        # Ask whether to continue or stop
        echo -e "${YELLOW}Do you want to continue with remaining datasets? (y/n)${NC}"
        read -r -t 30 response || response="y"  # Default to yes after 30 seconds

        if [[ ! "$response" =~ ^[Yy]$ ]]; then
            echo -e "${RED}Stopping training pipeline.${NC}"
            break
        fi
    fi
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
