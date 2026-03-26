"""
Post-process DynHAMR npz output: outlier detection, interpolation, and smoothing.
Operates on the npz file directly, producing a cleaned version.

Usage:
    python postprocess_npz.py --input <input.npz> --output <output.npz>
    python postprocess_npz.py --input <input.npz>  # overwrites in-place
"""
import argparse
import numpy as np
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))
from body_model import MANO
from body_model.utils import run_mano
from scipy.ndimage import gaussian_filter1d


MANO_DIR = os.path.join(os.path.dirname(__file__), '..', '_DATA', 'data', 'mano')
MANO_MEAN = os.path.join(os.path.dirname(__file__), '..', '_DATA', 'data', 'mano_mean_params.npz')


def detect_outliers_mad(signal, thresh=3.0):
    """Detect outlier indices from a 1D signal using median + MAD."""
    med = np.median(signal)
    mad = max(np.median(np.abs(signal - med)), 1e-6)
    return signal > med + thresh * mad * 1.4826


def interpolate_outliers(arr, outlier_frames, T, safe_interp_dist=3.0):
    """
    Interpolate outlier frames using nearest good neighbors.
    If two anchors are too far apart in value (safe_interp_dist), use nearest-neighbor
    copy instead of linear interpolation to avoid axis-angle wrap-around artifacts.
    """
    flat = arr.reshape(T, -1).copy()
    for f in sorted(outlier_frames):
        if f < 0 or f >= T:
            continue
        left = f - 1
        while left >= 0 and left in outlier_frames:
            left -= 1
        right = f + 1
        while right < T and right in outlier_frames:
            right += 1

        has_left = left >= 0 and left not in outlier_frames
        has_right = right < T and right not in outlier_frames

        if has_left and has_right:
            anchor_dist = np.linalg.norm(flat[right] - flat[left])
            if anchor_dist > safe_interp_dist:
                # Anchors too far apart (likely axis-angle wrap) - use nearest copy
                if (f - left) <= (right - f):
                    flat[f] = flat[left]
                else:
                    flat[f] = flat[right]
            else:
                alpha = (f - left) / max(right - left, 1)
                flat[f] = (1 - alpha) * flat[left] + alpha * flat[right]
        elif has_left:
            flat[f] = flat[left]
        elif has_right:
            flat[f] = flat[right]
    return flat.reshape(arr.shape)


def run_mano_forward(data, device='cuda'):
    """Run MANO forward pass to get vertices and joints."""
    B, T = data['trans'].shape[:2]
    hand_model = MANO(
        model_path=MANO_DIR,
        batch_size=B * T,
        pose2rot=True,
        gender='neutral',
        num_hand_joints=15,
        mean_params=MANO_MEAN,
        create_body_pose=False,
    ).to(device)

    with torch.no_grad():
        out = run_mano(
            hand_model,
            torch.tensor(data['trans'], dtype=torch.float32, device=device),
            torch.tensor(data['root_orient'], dtype=torch.float32, device=device),
            torch.tensor(data['pose_body'].reshape(B, T, -1), dtype=torch.float32, device=device),
            torch.tensor(data['is_right'], dtype=torch.float32, device=device),
            torch.tensor(data['betas'], dtype=torch.float32, device=device),
        )
    return out['vertices'].cpu().numpy(), out['joints'].cpu().numpy()


def postprocess_npz(input_path, output_path=None, sigma=2.0, thresh=3.0, device='cuda'):
    """
    Full post-processing pipeline for DynHAMR npz:
    1. Parameter-space outlier detection (trans + orient + pose velocity)
    2. Linear interpolation of outlier frames
    3. Gaussian smoothing
    4. MANO forward pass
    5. Vertex-space outlier detection (max vertex displacement)
    6. Back-solve: for vertex-space outliers, interpolate parameters again
    7. Final Gaussian smoothing pass
    """
    if output_path is None:
        output_path = input_path

    print(f"Loading {input_path}")
    d = dict(np.load(input_path, allow_pickle=True))
    B, T = d['trans'].shape[:2]
    print(f"  {B} hands, {T} frames")

    param_keys = ["trans", "root_orient", "pose_body"]

    # ===== Pass 1: Parameter-space outlier detection + interpolation =====
    print("Pass 1: Parameter-space outlier detection...")
    for b in range(B):
        outlier_frames = set()

        if T > 10:
            for key in param_keys:
                flat = d[key][b].reshape(T, -1)
                vel = np.linalg.norm(np.diff(flat, axis=0), axis=1)
                outliers = detect_outliers_mad(vel, thresh)
                for i in range(len(outliers)):
                    if outliers[i]:
                        outlier_frames.add(i)
                        outlier_frames.add(i + 1)

        if outlier_frames:
            print(f"  Hand {b}: {len(outlier_frames)} param-space outliers")
            for key in param_keys:
                d[key][b] = interpolate_outliers(d[key][b], outlier_frames, T)

    # ===== Gaussian smooth pass 1 =====
    print("Gaussian smooth (sigma={})...".format(sigma))
    for b in range(B):
        for key in param_keys:
            flat = d[key][b].reshape(T, -1)
            for dim in range(flat.shape[1]):
                flat[:, dim] = gaussian_filter1d(flat[:, dim], sigma=sigma, mode='nearest')
            d[key][b] = flat.reshape(d[key][b].shape)

    # ===== Pass 2: Vertex-space outlier detection =====
    print("Pass 2: MANO forward pass + vertex-space outlier detection...")
    verts, joints = run_mano_forward(d, device)

    for b in range(B):
        # Max vertex displacement per frame
        vert_diff = verts[b, 1:] - verts[b, :-1]  # (T-1, V, 3)
        vel = np.linalg.norm(vert_diff, axis=-1).max(axis=-1)  # (T-1,)

        if len(vel) > 10:
            outliers = detect_outliers_mad(vel, thresh)
            outlier_frames = set()
            for i in range(len(outliers)):
                if outliers[i]:
                    outlier_frames.add(i)
                    outlier_frames.add(i + 1)

            if outlier_frames:
                print(f"  Hand {b}: {len(outlier_frames)} vertex-space outliers")
                for key in param_keys:
                    d[key][b] = interpolate_outliers(d[key][b], outlier_frames, T)

    # ===== Final Gaussian smooth =====
    print("Final Gaussian smooth...")
    for b in range(B):
        for key in param_keys:
            flat = d[key][b].reshape(T, -1)
            for dim in range(flat.shape[1]):
                flat[:, dim] = gaussian_filter1d(flat[:, dim], sigma=sigma, mode='nearest')
            d[key][b] = flat.reshape(d[key][b].shape)

    # ===== Save =====
    np.savez(output_path, **d)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input npz path')
    parser.add_argument('--output', default=None, help='Output npz path (default: overwrite input)')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian smoothing sigma')
    parser.add_argument('--thresh', type=float, default=3.0, help='MAD outlier threshold')
    parser.add_argument('--device', default='cuda', help='torch device')
    args = parser.parse_args()

    postprocess_npz(args.input, args.output, args.sigma, args.thresh, args.device)
