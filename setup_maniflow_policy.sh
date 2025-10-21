#!/bin/bash
# Setup script for integrating ManiFlow policy into equidiff project
# Run this after activating the equidiff conda environment

set -e  # Exit on error

echo "========================================="
echo "ManiFlow Policy Integration for Equidiff"
echo "========================================="
echo ""

# Check if we're in the right directory
if [ ! -d "equi_diffpo" ]; then
    echo "Error: Please run this script from the equidiff root directory"
    exit 1
fi

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment detected. Please activate equidiff environment first:"
    echo "  conda activate equidiff"
    exit 1
fi

echo "Current conda environment: $CONDA_DEFAULT_ENV"
echo ""

# Install missing Python packages
echo "Step 1: Installing missing Python packages..."
echo "---------------------------------------------"

# Check if torch is available (should be in equidiff env)
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    echo "Error: PyTorch not found. Please check your conda environment."
    exit 1
}

# Install required packages
echo "Installing timm (for vision encoders)..."
pip install timm --quiet

echo "Installing einops (for tensor operations)..."
pip install einops --quiet

# Optional but recommended packages
echo "Installing additional recommended packages..."
pip install gdown --quiet  # For downloading R3M weights if needed

echo ""
echo "Step 2: Verifying installations..."
echo "-----------------------------------"

# Test imports
python << EOF
import sys
try:
    # Test core imports
    import torch
    print("✓ PyTorch imported successfully")

    import timm
    print("✓ timm imported successfully")

    import einops
    print("✓ einops imported successfully")

    from equi_diffpo.model.common.normalizer import LinearNormalizer
    print("✓ LinearNormalizer imported successfully")

    from equi_diffpo.model.diffusion.ditx import DiTX
    print("✓ DiTX model imported successfully")

    from equi_diffpo.policy.maniflow.maniflow_image_policy import ManiFlowTransformerImagePolicy
    print("✓ ManiFlowTransformerImagePolicy imported successfully")

    from equi_diffpo.policy.maniflow.maniflow_pointcloud_policy import ManiFlowTransformerPointcloudPolicy
    print("✓ ManiFlowTransformerPointcloudPolicy imported successfully")

    print("\n✅ All imports successful!")

except Exception as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)
EOF

echo ""
echo "========================================="
echo "✅ Setup complete!"
echo "========================================="
echo ""
echo "Copied files structure:"
echo "  equi_diffpo/policy/maniflow/"
echo "    ├── maniflow_image_policy.py"
echo "    └── maniflow_pointcloud_policy.py"
echo ""
echo "  equi_diffpo/model/diffusion/"
echo "    ├── ditx.py"
echo "    ├── ditx_block.py"
echo "    ├── positional_embedding.py"
echo "    └── mask_generator.py"
echo ""
echo "  equi_diffpo/model/vision_2d/"
echo "    └── (timm encoders)"
echo ""
echo "  equi_diffpo/model/vision_3d/"
echo "    └── (pointcloud encoders)"
echo ""
echo "Next steps:"
echo "1. Test importing the policies in Python"
echo "2. Adapt robomimic_image_runner.py to use ManiFlow policies"
echo "3. Create config files for your tasks"
echo ""
