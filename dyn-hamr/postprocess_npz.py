"""
Post-process DynHAMR npz: vertex-space outlier detection + parameter interpolation.
Runs MANO forward to detect frames where hand mesh jumps, then fixes
the source parameters (trans, root_orient, pose_body) via interpolation.
Skips boundary frames to avoid artifacts.

Usage:
    python postprocess_npz.py --input <input.npz> --output <output.npz>
    python postprocess_npz.py --input <input.npz>  # overwrites in-place
"""
import argparse
import numpy as np
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))
from body_model import MANO
from body_model.utils import run_mano

MANO_DIR = os.path.join(os.path.dirname(__file__), '..', '_DATA', 'data', 'mano')
MANO_MEAN = os.path.join(os.path.dirname(__file__), '..', '_DATA', 'data', 'mano_mean_params.npz')


def postprocess_npz(input_path, output_path=None, margin=30, thresh=3.0, device='cuda'):
    if output_path is None:
        output_path = input_path

    print(f"Loading {input_path}")
    d = dict(np.load(input_path, allow_pickle=True))
    B, T = d['trans'].shape[:2]
    print(f"  {B} hands, {T} frames")

    # Run MANO forward
    print("Running MANO forward pass...")
    hand_model = MANO(
        model_path=MANO_DIR, batch_size=B * T, pose2rot=True,
        gender='neutral', num_hand_joints=15, mean_params=MANO_MEAN,
        create_body_pose=False,
    ).to(device)

    with torch.no_grad():
        out = run_mano(
            hand_model,
            torch.tensor(d['trans'], dtype=torch.float32, device=device),
            torch.tensor(d['root_orient'], dtype=torch.float32, device=device),
            torch.tensor(d['pose_body'].reshape(B, T, -1), dtype=torch.float32, device=device),
            torch.tensor(d['is_right'], dtype=torch.float32, device=device),
            torch.tensor(d['betas'], dtype=torch.float32, device=device),
        )
    verts = out['vertices'].cpu().numpy()  # (B, T, V, 3)

    # Detect and fix per hand
    param_keys = ['trans', 'root_orient', 'pose_body']
    total_fixed = 0

    for b in range(B):
        # Max vertex displacement per frame
        vert_diff = verts[b, 1:] - verts[b, :-1]  # (T-1, V, 3)
        vel = np.linalg.norm(vert_diff, axis=-1).max(axis=-1)  # (T-1,)

        if len(vel) <= 2 * margin:
            continue

        med = np.median(vel)
        mad = max(np.median(np.abs(vel - med)), 1e-6)
        threshold = med + thresh * mad * 1.4826

        # Only detect in middle region
        outliers = set()
        for i in range(len(vel)):
            if vel[i] > threshold and margin <= i < len(vel) - margin:
                outliers.add(i)
                outliers.add(min(i + 1, T - 1))

        if not outliers:
            continue

        print(f"  Hand {b}: {len(outliers)} vertex-space outlier frames, interpolating params...")
        total_fixed += len(outliers)

        for key in param_keys:
            arr = d[key][b].copy()
            flat = arr.reshape(T, -1)

            for f in sorted(outliers):
                left = f - 1
                while left >= margin and left in outliers:
                    left -= 1
                right = f + 1
                while right < T - margin and right in outliers:
                    right += 1

                has_left = left >= margin and left not in outliers
                has_right = right < T - margin and right not in outliers

                if has_left and has_right:
                    alpha = (f - left) / max(right - left, 1)
                    flat[f] = (1 - alpha) * flat[left] + alpha * flat[right]

            d[key][b] = flat.reshape(arr.shape)

    np.savez(output_path, **d)
    print(f"Fixed {total_fixed} frames total. Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', default=None)
    parser.add_argument('--margin', type=int, default=30, help='Skip first/last N frames (default: 30 = 1 second at 30fps)')
    parser.add_argument('--thresh', type=float, default=3.0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    postprocess_npz(args.input, args.output, args.margin, args.thresh, args.device)
