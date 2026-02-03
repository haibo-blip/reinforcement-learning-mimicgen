## Installation
1.  Install the following apt packages for mujoco:
    ```bash
    sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
    ```
1. Install gfortran (dependancy for escnn) 
    ```bash
    sudo apt install -y gfortran
    ```

1. Install [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) (strongly recommended) or Anaconda
1. Clone this repo
    ```bash
    git clone https://github.com/pointW/equidiff.git
    cd equidiff
    ```
1. Install environment:
    Use Mambaforge (strongly recommended):
    ```bash
    mamba env create -f conda_environment.yaml
    conda activate equidiff
    ```
    or use Anaconda (not recommended): 
    ```bash
    conda env create -f conda_environment.yaml
    conda activate equidiff
    ```
1. Install mimicgen:
    ```bash
    cd ..
    git clone https://github.com/NVlabs/mimicgen_environments.git
    cd mimicgen_environments
    # This project was developed with Mimicgen v0.1.0. The latest version should work fine, but it is not tested
    git checkout 081f7dbbe5fff17b28c67ce8ec87c371f32526a9
    pip install -e .
    cd ../equidiff
    ```
1. Make sure mujoco version is 2.3.2 (required by mimicgen)
    ```bash
    pip list | grep mujoco
    ```
1. Installing missing package
    ```bash
    bash install_missing_packages.sh
    ```


## Dataset
### Download Dataset
Download dataset from MimicGen's hugging face: https://huggingface.co/datasets/amandlek/mimicgen_datasets/tree/main/core  
Make sure the dataset is kept under `/path/to/equidiff/data/robomimic/datasets/[dataset]/[dataset].hdf5`

### Generating Voxel and Point Cloud Observation

```bash
# Template
python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/[dataset]/[dataset].hdf5 --output data/robomimic/datasets/[dataset]/[dataset]_voxel.hdf5 --num_workers=[n_worker]
# Replace [dataset] and [n_worker] with your choices.
# E.g., use 24 workers to generate point cloud and voxel observation for stack_d1
python equi_diffpo/scripts/dataset_states_to_obs.py --input data/robomimic/datasets/stack_d1/stack_d1.hdf5 --output data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 --num_workers=24
```

### Convert Action Space in Dataset
The downloaded dataset has a relative action space. To train with absolute action space, the dataset needs to be converted accordingly
```bash
# Template
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i equi_diffpo/data/robomimic/datasets/[dataset]/[dataset].hdf5 -o equi_diffpo/data/robomimic/datasets/[dataset]/[dataset]_abs.hdf5 -n [n_worker]
# Replace [dataset] and [n_worker] with your choices.
# E.g., convert stack_d1 (non-voxel) with 12 workers
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i equi_diffpo/data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 -o equi_diffpo/data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5 -n 12
# E.g., convert stack_d1_voxel (voxel) with 12 workers
python equi_diffpo/scripts/robomimic_dataset_conversion.py -i equi_diffpo/data/robomimic/datasets/stack_d1/stack_d1_voxel.hdf5 -o equi_diffpo/data/robomimic/datasets/stack_d1/stack_d1_voxel_abs.hdf5 -n 12
```

## Training 
### Pretrain Regular Maniflow
To pretrain regular maniflow in Nut Assembly task:
```bash
# Make sure you have the non-voxel converted dataset with absolute action space from the previous step 
bash pretrain.sh
```
modify the para as needed --config-name=train_maniflow_pointcloud_workspace task_name=[task] n_demo=[n_demo]
### Pretrain Equi Maniflow
To pretrain Equi maniflow in Nut Assembly task:
```bash
# Make sure you have the non-voxel converted dataset with absolute action space from the previous step 
bash pretrain_equi.sh
```
modify the para as needed --config-name=train_maniflow_pointcloud_workspace task_name=[task] n_demo=[n_demo]



