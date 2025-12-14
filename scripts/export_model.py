#!/usr/bin/env python3
"""Export trained model for inference.

Usage:
    python scripts/export_model.py checkpoints/policy_step_100.pt --output models/policy.onnx
"""

import argparse
import torch
import torch.onnx
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from gossip_rl.training.networks import ActorCritic


def load_checkpoint(checkpoint_path: str, obs_dim: int, action_dim: int) -> ActorCritic:
    """Load a trained model from checkpoint."""
    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[256, 256],
        continuous=False,
        shared_backbone=False,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Loaded checkpoint from {checkpoint_path}")
    print(f"  Step: {checkpoint['step']}")
    print(f"  Model version: {checkpoint['model_version']}")
    
    return model


def export_to_onnx(model: ActorCritic, output_path: str, obs_dim: int):
    """Export model to ONNX format."""
    dummy_input = torch.randn(1, obs_dim)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action', 'value'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'},
            'value': {0: 'batch_size'},
        }
    )
    
    print(f"✓ Exported model to {output_path}")


def export_to_torchscript(model: ActorCritic, output_path: str, obs_dim: int):
    """Export model to TorchScript format."""
    dummy_input = torch.randn(1, obs_dim)
    
    # Trace the model
    traced = torch.jit.trace(model, dummy_input)
    traced.save(output_path)
    
    print(f"✓ Exported model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export trained policy model')
    parser.add_argument('checkpoint', help='Path to checkpoint file')
    parser.add_argument('--output', '-o', default='models/policy.onnx',
                        help='Output path (default: models/policy.onnx)')
    parser.add_argument('--format', choices=['onnx', 'torchscript', 'pytorch'],
                        default='onnx', help='Export format')
    parser.add_argument('--obs-dim', type=int, default=4,
                        help='Observation dimension (default: 4 for CartPole)')
    parser.add_argument('--action-dim', type=int, default=2,
                        help='Action dimension (default: 2 for CartPole)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Load the model
    model = load_checkpoint(args.checkpoint, args.obs_dim, args.action_dim)
    
    # Export
    if args.format == 'onnx':
        export_to_onnx(model, args.output, args.obs_dim)
    elif args.format == 'torchscript':
        export_to_torchscript(model, args.output, args.obs_dim)
    elif args.format == 'pytorch':
        torch.save(model.state_dict(), args.output)
        print(f"✓ Saved PyTorch state dict to {args.output}")
    
    print("\n📦 Model export complete!")
    print("   Use this model for inference in production.")


if __name__ == '__main__':
    main()
