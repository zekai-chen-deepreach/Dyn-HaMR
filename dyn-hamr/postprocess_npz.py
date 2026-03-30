"""
Post-process DynHAMR npz: vertex-space outlier detection + parameter interpolation.
Runs MANO forward to detect frames where hand mesh jumps, then fixes
the source parameters (trans, root_orient, pose_body) via interpolation.
Uses SLERP for rotation parameters to avoid axis-angle interpolation artifacts near 180°.
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


def _aa_to_rotmat(aa):
    """Axis-angle (3,) -> rotation matrix (3,3)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.eye(3)
    axis = aa / angle
    K = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rotmat_to_aa(R):
    """Rotation matrix (3,3) -> axis-angle (3,)."""
    cos_angle = (np.trace(R) - 1) / 2
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        return np.zeros(3)
    # Extract axis from skew-symmetric part
    axis = np.array([R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0],
                      R[1, 0] - R[0, 1]])
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:
        # angle ≈ π, need eigenvector of R + I
        eigvals, eigvecs = np.linalg.eigh(R + np.eye(3))
        axis = eigvecs[:, np.argmax(eigvals)]
        return axis * angle
    return axis / axis_norm * angle


def _slerp_aa(aa1, aa2, alpha):
    """SLERP between two axis-angle rotations via rotation matrices."""
    R1 = _aa_to_rotmat(aa1)
    R2 = _aa_to_rotmat(aa2)
    # R_diff = R1^T @ R2, then interpolate: R1 @ R_diff^alpha
    R_diff = R1.T @ R2
    aa_diff = _rotmat_to_aa(R_diff)
    angle_diff = np.linalg.norm(aa_diff)
    if angle_diff < 1e-8:
        return aa1.copy()
    # Scale the rotation by alpha
    aa_interp = aa_diff * alpha
    R_interp = R1 @ _aa_to_rotmat(aa_interp)
    return _rotmat_to_aa(R_interp)


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

        # Rotation keys use SLERP; others use linear interpolation
        rot_keys = {'root_orient', 'pose_body'}

        for key in param_keys:
            arr = d[key][b].copy()  # (T, ...)
            use_slerp = key in rot_keys

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
                    if use_slerp:
                        # SLERP each 3-component rotation vector independently
                        flat_l = arr[left].reshape(-1, 3)
                        flat_r = arr[right].reshape(-1, 3)
                        flat_f = np.empty_like(flat_l)
                        for j in range(len(flat_l)):
                            flat_f[j] = _slerp_aa(flat_l[j], flat_r[j], alpha)
                        arr[f] = flat_f.reshape(arr[f].shape)
                    else:
                        flat = arr.reshape(T, -1)
                        flat[f] = (1 - alpha) * flat[left] + alpha * flat[right]
                        arr = flat.reshape(arr.shape)

            d[key][b] = arr

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
