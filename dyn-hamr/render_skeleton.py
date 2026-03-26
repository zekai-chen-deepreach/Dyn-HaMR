"""
Render MANO hand skeleton overlay on source video from DynHAMR npz results.

Usage:
    python render_skeleton.py --npz_path <path_to_npz> --video_path <path_to_video> --output_path <output.mp4>
"""

import argparse
import cv2
import numpy as np


# MANO 21 joints after joint_map remapping:
#  0: wrist
#  1-4: thumb (MCP, PIP, DIP, tip)
#  5-8: index (MCP, PIP, DIP, tip)
#  9-12: middle (MCP, PIP, DIP, tip)
# 13-16: ring (MCP, PIP, DIP, tip)
# 17-20: pinky (MCP, PIP, DIP, tip)

# Skeleton connections: each finger chain from wrist
MANO_SKELETON = [
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Colors per finger (BGR)
FINGER_COLORS = {
    'thumb':  (0, 255, 255),   # yellow
    'index':  (0, 165, 255),   # orange
    'middle': (0, 255, 0),     # green
    'ring':   (255, 0, 0),     # blue
    'pinky':  (255, 0, 255),   # magenta
}

def get_bone_color(j1, j2):
    """Get color for a bone based on which finger it belongs to."""
    for j in (j1, j2):
        if j in (1, 2, 3, 4):
            return FINGER_COLORS['thumb']
        elif j in (5, 6, 7, 8):
            return FINGER_COLORS['index']
        elif j in (9, 10, 11, 12):
            return FINGER_COLORS['middle']
        elif j in (13, 14, 15, 16):
            return FINGER_COLORS['ring']
        elif j in (17, 18, 19, 20):
            return FINGER_COLORS['pinky']
    return (255, 255, 255)


def project_joints(joints_3d, cam_R, cam_t, intrins):
    """
    Project 3D joints to 2D pixel coordinates.

    joints_3d: (J, 3) world coordinates
    cam_R: (3, 3) rotation matrix
    cam_t: (3,) translation vector
    intrins: (4,) [fx, fy, cx, cy]

    Returns: (J, 2) pixel coordinates
    """
    # World to camera: p_cam = R @ p_world + t
    joints_cam = (cam_R @ joints_3d.T).T + cam_t  # (J, 3)

    # Perspective projection
    fx, fy, cx, cy = intrins
    x = joints_cam[:, 0] / joints_cam[:, 2]
    y = joints_cam[:, 1] / joints_cam[:, 2]

    px = fx * x + cx
    py = fy * y + cy

    return np.stack([px, py], axis=-1)  # (J, 2)


def draw_skeleton(img, joints_2d, color_override=None, joint_radius=4, bone_thickness=2, alpha=0.7):
    """
    Draw hand skeleton on image.

    joints_2d: (J, 2) pixel coordinates
    """
    overlay = img.copy()
    h, w = img.shape[:2]

    # Draw bones
    for j1, j2 in MANO_SKELETON:
        if j1 >= len(joints_2d) or j2 >= len(joints_2d):
            continue
        pt1 = tuple(joints_2d[j1].astype(int))
        pt2 = tuple(joints_2d[j2].astype(int))

        # Skip if out of frame
        if (pt1[0] < -100 or pt1[0] > w + 100 or pt1[1] < -100 or pt1[1] > h + 100 or
            pt2[0] < -100 or pt2[0] > w + 100 or pt2[1] < -100 or pt2[1] > h + 100):
            continue

        color = color_override if color_override else get_bone_color(j1, j2)
        cv2.line(overlay, pt1, pt2, color, bone_thickness, cv2.LINE_AA)

    # Draw joints
    for j_idx, pt in enumerate(joints_2d):
        pt_int = tuple(pt.astype(int))
        if pt_int[0] < -100 or pt_int[0] > w + 100 or pt_int[1] < -100 or pt_int[1] > h + 100:
            continue

        # Wrist is white, others follow finger color
        if j_idx == 0:
            jcolor = (255, 255, 255)
        else:
            jcolor = color_override if color_override else get_bone_color(j_idx, j_idx)

        cv2.circle(overlay, pt_int, joint_radius, jcolor, -1, cv2.LINE_AA)
        cv2.circle(overlay, pt_int, joint_radius, (0, 0, 0), 1, cv2.LINE_AA)

    # Blend
    result = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz_path', required=True, help='Path to DynHAMR world_results.npz')
    parser.add_argument('--video_path', required=True, help='Path to source video')
    parser.add_argument('--output_path', required=True, help='Output video path')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--joint_radius', type=int, default=4)
    parser.add_argument('--bone_thickness', type=int, default=2)
    args = parser.parse_args()

    # Load npz
    data = np.load(args.npz_path, allow_pickle=True)

    # Extract fields - shape: (B, T, ...)
    trans = data['trans']           # (B, T, 3)
    root_orient = data['root_orient']  # (B, T, 3)
    pose_body = data['pose_body']  # (B, T, 15, 3)
    betas = data['betas']          # (B, 10)
    is_right = data['is_right']    # (B, T)
    cam_R = data['cam_R']          # (B, T, 3, 3)
    cam_t = data['cam_t']          # (B, T, 3)
    intrins = data['intrins']      # (4,)

    B, T = trans.shape[:2]
    print(f"Loaded {B} tracks, {T} frames")
    print(f"Intrinsics: {intrins}")

    # We need MANO forward pass to get joints. Use torch + MANO.
    import sys
    sys.path.insert(0, '/home/deepreach/Zekai_code/Dyn-HaMR/dyn-hamr')
    import torch
    from body_model import MANO
    from body_model.utils import run_mano

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mano_cfg = {
        'model_path': '/home/deepreach/Zekai_code/Dyn-HaMR/_DATA/data/mano',
        'gender': 'neutral',
        'num_hand_joints': 15,
        'mean_params': '/home/deepreach/Zekai_code/Dyn-HaMR/_DATA/data/mano_mean_params.npz',
        'create_body_pose': False,
    }
    hand_model = MANO(batch_size=B * T, pose2rot=True, **mano_cfg).to(device)

    # Run MANO to get joints
    t_trans = torch.tensor(trans, dtype=torch.float32).to(device)
    t_root = torch.tensor(root_orient, dtype=torch.float32).to(device)
    t_pose = torch.tensor(pose_body, dtype=torch.float32).reshape(B, T, -1).to(device)
    t_is_right = torch.tensor(is_right, dtype=torch.float32).to(device)
    t_betas = torch.tensor(betas, dtype=torch.float32).to(device)

    with torch.no_grad():
        mano_out = run_mano(hand_model, t_trans, t_root, t_pose, t_is_right, t_betas)

    joints_3d = mano_out['joints'].cpu().numpy()  # (B, T, J, 3)
    J = joints_3d.shape[2]
    print(f"MANO joints: {J} per hand")

    # Open video
    cap = cv2.VideoCapture(args.video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = args.fps if args.fps else video_fps
    print(f"Video: {W}x{H}, {total_frames} frames, {video_fps} fps")
    print(f"Output fps: {out_fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_path, fourcc, out_fps, (W, H))

    # Use cam_R and cam_t from track 0 (same for all tracks)
    for t in range(min(T, total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        R = cam_R[0, t]   # (3, 3)
        tvec = cam_t[0, t]  # (3,)

        # Draw skeleton for each track
        for b in range(B):
            joints_w = joints_3d[b, t]  # All 21 MANO joints
            joints_2d = project_joints(joints_w, R, tvec, intrins)

            # Left hand = cyan tones, right hand = per-finger colors
            if is_right[b, 0] > 0.5:
                frame = draw_skeleton(frame, joints_2d, joint_radius=args.joint_radius,
                                    bone_thickness=args.bone_thickness, alpha=1.0)
            else:
                frame = draw_skeleton(frame, joints_2d, color_override=(255, 200, 0),
                                    joint_radius=args.joint_radius,
                                    bone_thickness=args.bone_thickness, alpha=1.0)

        writer.write(frame)

        if (t + 1) % 100 == 0:
            print(f"  Rendered {t + 1}/{min(T, total_frames)} frames")

    writer.release()
    cap.release()

    # Re-encode with ffmpeg for better compatibility
    import subprocess
    tmp_path = args.output_path + '.tmp.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-i', args.output_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an',
        tmp_path
    ], capture_output=True)
    import os
    os.replace(tmp_path, args.output_path)

    print(f"Saved skeleton video to {args.output_path}")


if __name__ == '__main__':
    main()
