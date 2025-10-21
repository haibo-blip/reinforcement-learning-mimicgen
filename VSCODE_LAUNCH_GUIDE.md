# VSCode Launch Configuration Guide

This guide explains the VSCode launch configurations for testing ManiFlow policies.

## Available Configurations

### 1. ManiFlow PointCloud - stack_d1 (Debug)
**Purpose**: Quick debugging of point cloud ManiFlow policy
**Use when**: Testing changes, debugging issues, verifying setup

**Settings**:
- Task: `stack_d1`
- Demos: `10` (small dataset for quick iteration)
- Debug mode: `True`
- Epochs: `2`
- Train steps: `5` per epoch
- Batch size: `4` (small for fast execution)
- Logging: `offline` (no wandb upload)

**Expected runtime**: ~2-5 minutes

### 2. ManiFlow PointCloud - stack_d1 (Full Training)
**Purpose**: Full training run for point cloud policy
**Use when**: Running actual experiments

**Settings**:
- Task: `stack_d1`
- Demos: `200` (full dataset)
- Batch size: `64`
- DiTX layers: `12`
- Heads: `8`
- Embedding dim: `768`
- Logging: `wandb online` to project `maniflow_pointcloud_stack_d1`

**Expected runtime**: Several hours depending on GPU

### 3. ManiFlow Image - stack_d1 (Debug)
**Purpose**: Quick debugging of image ManiFlow policy
**Use when**: Testing changes, debugging issues, verifying setup

**Settings**:
- Task: `stack_d1`
- Demos: `10`
- Debug mode: `True`
- Epochs: `2`
- Train steps: `5` per epoch
- Batch size: `4`
- Vision encoder: `ResNet18`
- Logging: `offline`

**Expected runtime**: ~2-5 minutes

### 4. ManiFlow Image - stack_d1 (Full Training)
**Purpose**: Full training run for image policy
**Use when**: Running actual experiments

**Settings**:
- Task: `stack_d1`
- Demos: `200`
- Batch size: `64`
- Vision encoder: `ResNet18`
- DiTX layers: `12`
- Logging: `wandb online` to project `maniflow_image_stack_d1`

**Expected runtime**: Several hours depending on GPU

### 5. Test ManiFlow Import
**Purpose**: Run the import test suite
**Use when**: Verifying ManiFlow integration is working

**Expected runtime**: ~30 seconds

## How to Use

### Using VSCode Debug Panel

1. Open VSCode
2. Go to Run and Debug panel (Ctrl+Shift+D)
3. Select configuration from dropdown
4. Press F5 or click green play button

### Using Command Palette

1. Press Ctrl+Shift+P
2. Type "Debug: Select and Start Debugging"
3. Choose your configuration

## Debugging Tips

### Setting Breakpoints
1. Click left margin of code line to set breakpoint
2. Run debug configuration
3. Execution pauses at breakpoint
4. Use debug toolbar to step through code

### Inspecting Variables
- **Variables panel**: Shows local variables
- **Watch panel**: Add expressions to monitor
- **Debug console**: Execute Python expressions

### Common Breakpoint Locations

**For debugging training loop**:
```python
# In train_maniflow_*_workspace.py
for batch_idx, batch in enumerate(tepoch):  # Line ~171
    loss = self.model.compute_loss(batch)    # Line ~178
```

**For debugging policy**:
```python
# In maniflow_image_policy.py or maniflow_pointcloud_policy.py
def compute_loss(self, batch):               # Check loss computation
def predict_action(self, obs_dict):          # Check inference
```

**For debugging encoder**:
```python
# In timm_obs_encoder.py
def forward(self, obs_dict):                 # Check image encoding
```

## Customizing Configurations

### Change GPU Device
Edit `training.device`:
```json
"args": [
    "training.device=cuda:1"  // Use GPU 1 instead of 0
]
```

### Change Batch Size
```json
"args": [
    "dataloader.batch_size=32"  // Reduce if OOM
]
```

### Change Number of Demos
```json
"args": [
    "n_demo=50"  // Use 50 demonstrations
]
```

### Enable WandB in Debug Mode
```json
"args": [
    "logging.mode=online"  // Change from offline
]
```

### Change Model Architecture
```json
"args": [
    "policy.n_layer=6",      // Reduce layers
    "policy.n_head=4",       // Reduce heads
    "policy.n_emb=512"       // Reduce embedding dim
]
```

## Environment Variables

All configurations set:
- **PYTHONPATH**: `${workspaceFolder}` - Ensures imports work
- **HYDRA_FULL_ERROR**: `1` - Shows full error traces from Hydra

## Output Locations

### Debug Runs
```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_maniflow_{pointcloud|image}_stack_d1/
├── checkpoints/latest.ckpt
├── logs.json.txt
└── wandb/ (if logging.mode=online)
```

### Logs
- **Console output**: Shows in VSCode integrated terminal
- **JSON logs**: `logs.json.txt` in output directory
- **WandB**: Online dashboard (if enabled)

## Troubleshooting

### Configuration Not Found
Make sure you're using Python debugger (`debugpy`):
```bash
pip install debugpy
```

### Module Not Found
Check that `PYTHONPATH` is set correctly in the configuration.

### CUDA Out of Memory
Use debug configuration with smaller batch sizes first:
- Debug configs use `batch_size=4`
- Full training uses `batch_size=64`

Reduce if needed:
```json
"dataloader.batch_size=16"
```

### Hydra Errors
The `HYDRA_FULL_ERROR=1` environment variable shows full stack traces.
Check that config files exist:
- `equi_diffpo/config/train_maniflow_pointcloud_workspace.yaml`
- `equi_diffpo/config/train_maniflow_image_workspace.yaml`
- `equi_diffpo/config/task/mimicgen_pc_abs.yaml`
- `equi_diffpo/config/task/mimicgen_abs.yaml`

### Dataset Not Found
Ensure dataset exists at:
- Point cloud: `data/robomimic/datasets/stack_d1/stack_d1_pc_abs.hdf5`
- Image: `data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5`

## Quick Test Workflow

1. **First**: Run "Test ManiFlow Import" to verify integration
2. **Second**: Run debug configuration for your policy type
3. **Third**: If debug works, run full training configuration

## Performance Monitoring

### During Debug Run
Watch for:
- Import errors in first few seconds
- Dataset loading messages
- First batch training completes
- Loss values are reasonable (not NaN)

### During Full Training
Monitor:
- Training loss decreasing
- Validation loss stabilizing
- Test success rate (in rollouts)
- GPU utilization (use `nvidia-smi`)

## Example: Debugging a Training Issue

1. Set breakpoint in `train_maniflow_pointcloud_workspace.py` at:
   ```python
   loss = self.model.compute_loss(batch)  # Line ~178
   ```

2. Run "ManiFlow PointCloud - stack_d1 (Debug)"

3. When breakpoint hits, inspect:
   - `batch` variable - check data shapes
   - Step into `compute_loss()` to debug policy
   - Check `obs_dict` and `action` tensors

4. Use Debug Console:
   ```python
   batch['obs'].keys()           # See observation keys
   batch['action'].shape         # Check action shape
   torch.isnan(loss).any()       # Check for NaN
   ```

## Command Line Equivalents

Each VSCode configuration corresponds to a command:

### Debug Point Cloud
```bash
python -m equi_diffpo.workspace.train_maniflow_pointcloud_workspace \
    task_name=stack_d1 n_demo=10 training.debug=True
```

### Full Point Cloud Training
```bash
python -m equi_diffpo.workspace.train_maniflow_pointcloud_workspace \
    task_name=stack_d1 n_demo=200
```

### Debug Image
```bash
python -m equi_diffpo.workspace.train_maniflow_image_workspace \
    task_name=stack_d1 n_demo=10 training.debug=True
```

### Full Image Training
```bash
python -m equi_diffpo.workspace.train_maniflow_image_workspace \
    task_name=stack_d1 n_demo=200
```
