#!/bin/bash
# ManiFlow RL Training Starter Script
# Quick commands to get RL training running

set -e

echo "🎯 ManiFlow RL Training Starter"
echo "================================"

# Check if we're in the right directory
if [[ ! -f "train_maniflow_rl.py" ]]; then
    echo "❌ Please run this script from the maniflow_mimicgen directory"
    exit 1
fi

echo "📁 Current directory: $(pwd)"

# Show available commands
echo ""
echo "🚀 Available Training Commands:"
echo "================================"

echo ""
echo "1. Basic Training (default config):"
echo "   python train_maniflow_rl.py"

echo ""
echo "2. With Custom Dataset & Checkpoint:"
echo "   python train_maniflow_rl.py \\"
echo "       task.dataset_path=/path/to/dataset.hdf5 \\"
echo "       policy.checkpoint=/path/to/checkpoint.ckpt"

echo ""
echo "3. Quick Test Run (minimal config):"
echo "   python train_maniflow_rl.py \\"
echo "       training.total_timesteps=1000 \\"
echo "       training.num_envs=2 \\"
echo "       training.num_steps_per_rollout=20 \\"
echo "       use_wandb=false"

echo ""
echo "4. Production Training (extended):"
echo "   python train_maniflow_rl.py \\"
echo "       training.total_timesteps=5000000 \\"
echo "       training.num_envs=16 \\"
echo "       training.learning_rate=1e-4 \\"
echo "       training.batch_size=1024"

echo ""
echo "5. Environment-Specific Training:"
echo "   python train_maniflow_rl.py \\"
echo "       task.env_runner.env_meta.env_name=PickPlace \\"
echo "       wandb_run_name=maniflow_pickplace_v1"

echo ""
echo "📋 Configuration Files:"
echo "================================"
echo "- Main config: config/train_maniflow_pointcloud_rl.yaml"
echo "- Logs: See outputs/ directory after training"
echo "- Checkpoints: outputs/*/checkpoints/"

echo ""
echo "🔧 Setup Validation:"
echo "================================"
echo "Run: python test_rl_setup.py"

echo ""
echo "📚 Documentation:"
echo "================================"
echo "Full guide: README_RL_TRAINING.md"

echo ""
read -p "Do you want to start a quick test run? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting quick test run..."
    python train_maniflow_rl.py \
        training.total_timesteps=1000 \
        training.num_envs=2 \
        training.num_steps_per_rollout=20 \
        training.log_interval=1 \
        use_wandb=false
else
    echo "👍 Ready to go! Use the commands above to start training."
fi