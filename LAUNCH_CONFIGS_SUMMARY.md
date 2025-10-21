# VSCode Launch Configurations - Quick Reference

## Created File
📁 `.vscode/launch.json` - Contains 5 debug/launch configurations

## Configurations Overview

| # | Name | Type | Purpose | Runtime | Demos |
|---|------|------|---------|---------|-------|
| 1 | **ManiFlow PointCloud - stack_d1 (Debug)** | Point Cloud | Quick test | 2-5 min | 10 |
| 2 | **ManiFlow PointCloud - stack_d1 (Full Training)** | Point Cloud | Full train | Hours | 200 |
| 3 | **ManiFlow Image - stack_d1 (Debug)** | Image | Quick test | 2-5 min | 10 |
| 4 | **ManiFlow Image - stack_d1 (Full Training)** | Image | Full train | Hours | 200 |
| 5 | **Test ManiFlow Import** | Test | Verify setup | 30 sec | - |

## Quick Start

### In VSCode:
1. Press `Ctrl+Shift+D` (Run and Debug panel)
2. Select configuration from dropdown
3. Press `F5` to start

### Recommended Testing Order:
```
1. Test ManiFlow Import          ✅ Verify integration works
   ↓
2. ManiFlow PointCloud (Debug)   🧪 Quick test point cloud
   ↓  OR
   ManiFlow Image (Debug)        🧪 Quick test image
   ↓
3. Full Training                 🚀 Run actual experiment
```

## Configuration Details

### 🔧 Debug Configurations (Point Cloud & Image)
```yaml
Settings:
  - n_demo: 10              # Small dataset
  - training.debug: True    # Debug mode enabled
  - num_epochs: 2           # Only 2 epochs
  - max_train_steps: 5      # 5 batches per epoch
  - batch_size: 4           # Small batch
  - logging.mode: offline   # No wandb upload

Use for:
  ✓ Testing changes quickly
  ✓ Debugging errors
  ✓ Verifying setup
  ✗ NOT for actual experiments
```

### 🚀 Full Training Configurations (Point Cloud & Image)
```yaml
Settings:
  - n_demo: 200             # Full dataset
  - batch_size: 64          # Normal batch size
  - n_layer: 12             # Full DiTX model
  - n_head: 8
  - n_emb: 768
  - logging: wandb online   # Full logging

Use for:
  ✓ Running experiments
  ✓ Training final models
  ✓ Comparing approaches
  ✗ NOT for quick testing
```

## Point Cloud vs Image Differences

| Aspect | Point Cloud Config | Image Config |
|--------|-------------------|--------------|
| **Task file** | `mimicgen_pc_abs.yaml` | `mimicgen_abs.yaml` |
| **Dataset** | `stack_d1_pc_abs.hdf5` | `stack_d1_abs.hdf5` |
| **Encoder** | DP3Encoder | TimmObsEncoder (ResNet18) |
| **Input** | `[1024, 6]` point cloud | `[3, 84, 84]` RGB images |
| **Workspace** | `train_maniflow_pointcloud_workspace.py` | `train_maniflow_image_workspace.py` |

## Key Arguments Explained

| Argument | Debug Value | Full Value | Purpose |
|----------|-------------|------------|---------|
| `task_name` | stack_d1 | stack_d1 | Which task to train on |
| `n_demo` | 10 | 200 | Number of demonstrations |
| `training.debug` | True | (not set) | Enable debug mode |
| `training.device` | cuda:0 | cuda:0 | Which GPU to use |
| `dataloader.batch_size` | 4 | 64 | Batch size |
| `logging.mode` | offline | (default: online) | WandB logging |
| `policy.n_layer` | (not set) | 12 | Transformer layers |

## Common Customizations

### Change GPU
```json
"training.device=cuda:1"
```

### Change Task
```json
"task_name=coffee"  // Or any other task
```

### Reduce Batch Size (if OOM)
```json
"dataloader.batch_size=32"
```

### Smaller Model (faster training)
```json
"policy.n_layer=6",
"policy.n_head=4",
"policy.n_emb=512"
```

### Change Vision Encoder (Image only)
```json
"policy.obs_encoder.model_name=resnet34"  // or resnet50
```

## Debug Workflow

```
┌─────────────────────────────────────┐
│ 1. Select Debug Configuration      │
│    (PointCloud or Image Debug)      │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 2. Press F5 to Start                │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 3. Watch Console for:               │
│    ✓ Import success                 │
│    ✓ Dataset loaded                 │
│    ✓ First batch trains             │
│    ✓ Loss is not NaN                │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ 4. If successful:                   │
│    → Run Full Training Config       │
│    If errors:                       │
│    → Set breakpoints and debug      │
└─────────────────────────────────────┘
```

## Output Locations

### Console Output
Shows in VSCode integrated terminal:
- Import messages
- Training progress
- Loss values
- Success rates

### Files Created
```
data/outputs/YYYY.MM.DD/HH.MM.SS_train_maniflow_{type}_stack_d1/
├── checkpoints/
│   ├── latest.ckpt                              # Latest checkpoint
│   └── epoch=XXXX-test_mean_score=X.XXX.ckpt   # Best checkpoints
├── logs.json.txt                                # JSON training logs
└── wandb/                                       # WandB logs (if online)
```

## Debugging Tips

### Set Breakpoints
Click left margin in code to set breakpoint:
```python
# Good places for breakpoints:
loss = self.model.compute_loss(batch)    # Check loss computation
action_dict = policy.predict_action(obs) # Check inference
```

### Debug Console Commands
During debugging, try:
```python
batch['obs'].keys()              # Check observation keys
batch['action'].shape            # Check shapes
loss.item()                      # Check loss value
torch.cuda.memory_summary()      # Check GPU memory
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| **Module not found** | Check PYTHONPATH is set correctly |
| **CUDA out of memory** | Use debug config or reduce batch size |
| **Dataset not found** | Check dataset path exists |
| **Config file error** | Check YAML files exist and are valid |
| **Import errors** | Run "Test ManiFlow Import" first |

## Environment Variables Set

All configs automatically set:
- `PYTHONPATH=${workspaceFolder}` - For imports
- `HYDRA_FULL_ERROR=1` - Show full error traces

## Performance Expectations

### Debug Mode (10 demos, 5 steps/epoch)
- **Point Cloud**: ~2 min on RTX 3090
- **Image**: ~2 min on RTX 3090
- **GPU Memory**: ~2-4 GB

### Full Training (200 demos, 64 batch size)
- **Point Cloud**: ~4-6 hours on RTX 3090
- **Image**: ~4-6 hours on RTX 3090
- **GPU Memory**: ~8-12 GB

## Next Steps After Setup

1. ✅ **Verify**: Run "Test ManiFlow Import"
2. 🧪 **Test**: Run debug configuration
3. 📊 **Monitor**: Check loss decreases
4. 🚀 **Train**: Run full training
5. 📈 **Evaluate**: Check test success rate in WandB

## Command Line Alternatives

If you prefer terminal over VSCode debugger:

**Debug Point Cloud:**
```bash
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=stack_d1 n_demo=10 training.debug=True training.device=cuda:0
```

**Full Point Cloud Training:**
```bash
python equi_diffpo/workspace/train_maniflow_pointcloud_workspace.py \
    task_name=stack_d1 n_demo=200 training.device=cuda:0
```

**Debug Image:**
```bash
python equi_diffpo/workspace/train_maniflow_image_workspace.py \
    task_name=stack_d1 n_demo=10 training.debug=True training.device=cuda:0
```

**Full Image Training:**
```bash
python equi_diffpo/workspace/train_maniflow_image_workspace.py \
    task_name=stack_d1 n_demo=200 training.device=cuda:0
```

---

📚 **See Also:**
- `VSCODE_LAUNCH_GUIDE.md` - Detailed debugging guide
- `MANIFLOW_TRAINING_GUIDE.md` - Complete training documentation
