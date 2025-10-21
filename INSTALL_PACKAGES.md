# Package Installation Guide for ManiFlow Integration

## Quick Answer

**There is NO "dit" package to install.** DiT stands for "Diffusion Transformer" - it's the architecture name, not a separate package.

All the DiT/DiTX code is already copied to `equi_diffpo/model/diffusion/ditx.py`.

## Required Packages

You only need to install these packages (all available via pip):

### Option 1: Automatic (Recommended)

```bash
cd /home/haibo/Desktop/Developer/equidiff
conda activate equidiff
./setup_maniflow_policy.sh
```

This will automatically install:
- `timm` (for vision encoders)
- `einops` (for tensor operations)
- `gdown` (optional, for downloading pre-trained weights)

### Option 2: Manual Installation

If you prefer to install manually:

```bash
conda activate equidiff

# Required packages
pip install timm      # Vision transformer models
pip install einops    # Tensor operations

# Optional but recommended
pip install gdown     # For downloading R3M weights
```

## Package Details

### 1. timm (required)
- **What:** PyTorch Image Models library
- **Why:** Provides ResNet, ViT, EfficientNet encoders
- **Used in:** `equi_diffpo/model/vision_2d/timm_obs_encoder.py`
- **Install:** `pip install timm`

### 2. einops (required)
- **What:** Tensor operations library
- **Why:** Used for tensor rearrangement in DiTX blocks
- **Used in:** `equi_diffpo/model/diffusion/ditx_block.py`
- **Install:** `pip install einops`

### 3. gdown (optional)
- **What:** Google Drive downloader
- **Why:** Download R3M pre-trained weights if using R3M encoder
- **Used in:** Optional, only if you use `model_name='r3m'`
- **Install:** `pip install gdown`

## Already Installed in Equidiff

These packages should already be in your equidiff conda environment:

- ✅ PyTorch
- ✅ torchvision
- ✅ numpy
- ✅ h5py
- ✅ zarr
- ✅ hydra-core
- ✅ tqdm
- ✅ termcolor
- ✅ wandb
- ✅ opencv (cv2)

## Verification

After installation, verify everything works:

```bash
# Test imports
python -c "import timm; print('timm:', timm.__version__)"
python -c "import einops; print('einops OK')"

# Test DiTX import
python -c "from equi_diffpo.model.diffusion.ditx import DiTX; print('DiTX OK')"

# Test policy import
python -c "from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy; print('Policy OK')"
```

Or run the comprehensive test:

```bash
python test_maniflow_import.py
```

## Common Errors & Solutions

### Error: "No module named 'timm'"
**Solution:** `pip install timm`

### Error: "No module named 'einops'"
**Solution:** `pip install einops`

### Error: "No module named 'dit'"
**Solution:** There is no "dit" package. DiT is the architecture name. All code is already in `equi_diffpo/model/diffusion/ditx.py`

### Error: "cannot import name 'RmsNorm' from 'timm.models.vision_transformer'"
**Solution:** Update timm: `pip install --upgrade timm`

### Error: "No module named 'torch'"
**Solution:** You're not in the equidiff conda environment. Run: `conda activate equidiff`

## Version Requirements

Minimum versions:
- `timm >= 0.9.0` (for RmsNorm support)
- `einops >= 0.4.0`
- `torch >= 2.0.0` (already in equidiff)

Check your versions:
```bash
pip show timm einops torch
```

## What About "DiT"?

**DiT** = **Di**ffusion **T**ransformer
- It's an architecture, not a package
- DiTX is ManiFlow's variant of DiT
- The code is in `equi_diffpo/model/diffusion/ditx.py`
- No separate installation needed!

## Summary

**To install everything:**
```bash
conda activate equidiff
pip install timm einops gdown
```

**To verify:**
```bash
python test_maniflow_import.py
```

That's it! No "dit" package exists or is needed.
