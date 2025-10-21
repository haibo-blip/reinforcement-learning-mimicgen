#!/usr/bin/env python3
"""
Diagnostic script to identify missing imports for ManiFlow integration
Run this to see exactly what's missing
"""

import sys
import os

print("=" * 70)
print("ManiFlow Import Diagnostic Tool")
print("=" * 70)
print()

# Check if we're in the right directory
if not os.path.exists('equi_diffpo'):
    print("❌ Error: Not in equidiff directory")
    print("   Please cd to /home/haibo/Desktop/Developer/equidiff")
    sys.exit(1)

print("✓ In correct directory")
print()

# Test each import step by step
tests = [
    ("PyTorch", "import torch"),
    ("NumPy", "import numpy"),
    ("timm", "import timm"),
    ("einops", "import einops"),
    ("termcolor", "from termcolor import cprint"),
    ("zarr", "import zarr"),
    ("h5py", "import h5py"),
    ("Normalizer", "from equi_diffpo.model.common.normalizer import LinearNormalizer"),
    ("pytorch_util", "from equi_diffpo.common.pytorch_util import dict_apply"),
    ("model_util", "from equi_diffpo.common.model_util import print_params"),
    ("Positional Embedding", "from equi_diffpo.model.diffusion.positional_embedding import SinusoidalPosEmb"),
    ("DiTX Block", "from equi_diffpo.model.diffusion.ditx_block import DiTXBlock, AdaptiveLayerNorm"),
    ("DiTX Model", "from equi_diffpo.model.diffusion.ditx import DiTX"),
    ("sample_util", "from equi_diffpo.model.common.sample_util import *"),
    ("Timm Encoder", "from equi_diffpo.model.vision_2d.timm_obs_encoder import TimmObsEncoder"),
    ("PointNet", "from equi_diffpo.model.vision_3d.pointnet_extractor import DP3Encoder"),
    ("Base Policy", "from equi_diffpo.policy.base_policy import BasePolicy"),
    ("Image Policy", "from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy"),
    ("Pointcloud Policy", "from equi_diffpo.policy.maniflow.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy"),
]

print("Testing imports...")
print("-" * 70)

failed_imports = []
for name, import_statement in tests:
    try:
        exec(import_statement)
        print(f"✓ {name:30s} OK")
    except Exception as e:
        print(f"✗ {name:30s} FAILED")
        failed_imports.append((name, import_statement, str(e)))

print()
print("=" * 70)

if not failed_imports:
    print("✅ SUCCESS! All imports working.")
    print()
    print("You can now use ManiFlow policies!")
    sys.exit(0)
else:
    print(f"❌ {len(failed_imports)} import(s) failed")
    print("=" * 70)
    print()

    for name, statement, error in failed_imports:
        print(f"Failed: {name}")
        print(f"  Import: {statement}")
        print(f"  Error: {error}")
        print()

    # Provide specific solutions
    print("=" * 70)
    print("Solutions:")
    print("=" * 70)

    error_messages = [e[2] for e in failed_imports]

    if any("torch" in e for e in error_messages):
        print("⚠️  PyTorch not found")
        print("   → Activate conda environment: conda activate equidiff")
        print()

    if any("timm" in e for e in error_messages):
        print("⚠️  timm not installed")
        print("   → Install: pip install timm")
        print()

    if any("einops" in e for e in error_messages):
        print("⚠️  einops not installed")
        print("   → Install: pip install einops")
        print()

    if any("dit" in e.lower() for e in error_messages):
        print("⚠️  'dit' mentioned in error")
        print("   → Note: There is no 'dit' package")
        print("   → DiTX code is in equi_diffpo/model/diffusion/ditx.py")
        print("   → Check the full error message above for the real issue")
        print()

    print("Quick fix:")
    print("  conda activate equidiff")
    print("  pip install timm einops")
    print()

    sys.exit(1)
