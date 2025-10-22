#!/bin/bash

# Script to process all datasets with dataset_states_to_obs.py and robomimic_dataset_conversion.py
# Runs multiple datasets in parallel to save time

set -e  # Exit on error

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base directory
BASE_DIR="equi_diffpo/data/robomimic/datasets"

# List of all datasets (excluding stack_d1 which is already processed)
DATASETS=(
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

# Number of datasets to process in parallel (adjust based on CPU capacity)
# With 32 cores, we can run 2 datasets in parallel (24 workers each fits well)
MAX_PARALLEL=1

# Function to process a single dataset
process_dataset() {
    local dataset=$1
    local input_file="${BASE_DIR}/${dataset}/${dataset}.hdf5"
    local voxel_file="${BASE_DIR}/${dataset}/${dataset}_voxel.hdf5"
    local output_file="${BASE_DIR}/${dataset}/${dataset}_voxel_abs.hdf5"

    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Starting processing for ${dataset}${NC}"

    # Step 1: Convert states to observations
    echo -e "${GREEN}[${dataset}] Step 1/2: Running dataset_states_to_obs.py${NC}"
    if python equi_diffpo/scripts/dataset_states_to_obs.py \
        --input "${input_file}" \
        --output "${voxel_file}" \
        --num_workers=24; then
        echo -e "${GREEN}[${dataset}] Step 1/2: Completed successfully${NC}"
    else
        echo -e "${RED}[${dataset}] Step 1/2: Failed!${NC}"
        return 1
    fi

    # Step 2: Convert dataset format
    echo -e "${GREEN}[${dataset}] Step 2/2: Running robomimic_dataset_conversion.py${NC}"
    if python equi_diffpo/scripts/robomimic_dataset_conversion.py \
        -i "${voxel_file}" \
        -o "${output_file}" \
        -n 12; then
        echo -e "${GREEN}[${dataset}] Step 2/2: Completed successfully${NC}"
    else
        echo -e "${RED}[${dataset}] Step 2/2: Failed!${NC}"
        return 1
    fi

    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] Finished processing for ${dataset}${NC}"
    return 0
}

# Export function so it can be used by parallel
export -f process_dataset
export BASE_DIR GREEN BLUE RED NC

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Dataset Processing Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Total datasets to process: ${#DATASETS[@]}${NC}"
echo -e "${BLUE}Parallel jobs: ${MAX_PARALLEL}${NC}"
echo -e "${BLUE}Workers per job (step 1): 24${NC}"
echo -e "${BLUE}Workers per job (step 2): 12${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check if GNU parallel is installed
if command -v parallel &> /dev/null; then
    echo -e "${GREEN}Using GNU parallel for parallel processing${NC}\n"

    # Use GNU parallel to process datasets
    printf '%s\n' "${DATASETS[@]}" | parallel -j ${MAX_PARALLEL} --line-buffer process_dataset {}

else
    echo -e "${BLUE}GNU parallel not found, using simple parallel processing${NC}\n"

    # Simple parallel processing using background jobs
    running_jobs=0
    failed_datasets=()

    for dataset in "${DATASETS[@]}"; do
        # Wait if we've reached max parallel jobs
        while [ $running_jobs -ge $MAX_PARALLEL ]; do
            sleep 5
            running_jobs=$(jobs -r | wc -l)
        done

        # Start processing in background
        (
            if ! process_dataset "$dataset"; then
                echo "$dataset" >> /tmp/failed_datasets.txt
            fi
        ) &

        running_jobs=$((running_jobs + 1))
    done

    # Wait for all background jobs to complete
    wait

    # Check for failed datasets
    if [ -f /tmp/failed_datasets.txt ]; then
        echo -e "\n${RED}========================================${NC}"
        echo -e "${RED}Failed datasets:${NC}"
        cat /tmp/failed_datasets.txt
        echo -e "${RED}========================================${NC}"
        rm /tmp/failed_datasets.txt
        exit 1
    fi
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}All datasets processed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
