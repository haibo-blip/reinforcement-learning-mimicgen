#!/bin/bash
# Quick script to install missing packages for ManiFlow integration
# Run this if setup_maniflow_policy.sh fails due to missing packages

echo "Installing missing packages for ManiFlow integration..."
echo "======================================================="
echo ""

# Check conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected."
    echo "Please run: conda activate equidiff"
    exit 1
fi

echo "Current environment: $CONDA_DEFAULT_ENV"
echo ""

# Install packages
echo "Installing timm..."
pip install timm

echo ""
echo "Installing einops..."
pip install einops

echo ""
echo "Installing gdown (optional, for R3M weights)..."
pip install gdown

echo ""
echo "======================================================="
echo "✅ Installation complete!"
echo ""
echo "Verify with:"
echo "  python -c 'import timm, einops; print(\"OK\")'"
echo ""
echo "Or run the full test:"
echo "  python test_maniflow_import.py"
