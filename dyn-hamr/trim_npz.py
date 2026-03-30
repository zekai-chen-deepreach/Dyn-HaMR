"""
Trim NPZ by removing frames where wrist velocity exceeds threshold.
Splits into multiple segments if anomalous frames occur in the middle.
Generates aligned video clips for each segment from the pipeline output images.

Usage:
    python trim_npz.py --input_dir <pipeline_output_dir> --output_dir <output_dir>
    python trim_npz.py --input_dir <pipeline_output_dir> --output_dir <output_dir> --max_vel 5.0

Input:  pipeline output directory containing:
    <seq>_postprocessed.npz, images/, shot_idcs.json

Output per segment:
    <seq>_seg001.npz          # trimmed NPZ
    <seq>_seg001.mp4          # aligned video clip
"""
import argparse
import glob
import json
import numpy as np
import os
import subprocess


def trim_npz(input_dir, output_dir, max_vel=5.0, fps=30, min_segment_seconds=1.0):
    min_segment_frames = int(min_segment_seconds * fps)

    # Find npz
    npz_files = glob.glob(os.path.join(input_dir, '*_postprocessed.npz'))
    if not npz_files:
        print(f"ERROR: No *_postprocessed.npz in {input_dir}")
        return []
    npz_path = npz_files[0]
    seq = os.path.basename(npz_path).replace('_postprocessed.npz', '')

    # Find images
    img_dir = os.path.join(input_dir, 'images')
    if not os.path.isdir(img_dir):
        print(f"ERROR: No images/ in {input_dir}")
        return []
    all_images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    n_images = len(all_images)

    # Find shot offset (npz frame 0 = image index shot_start)
    shot_path = os.path.join(input_dir, 'shot_idcs.json')
    shot_start = 0
    if os.path.isfile(shot_path):
        with open(shot_path) as f:
            shots = json.load(f)
        # Find first non-zero or infer from npz vs image count
    # Infer shot_start from frame count mismatch
    d = dict(np.load(npz_path, allow_pickle=True))
    trans = d['trans']
    B, T = trans.shape[:2]
    shot_start = (n_images - T) // 2  # centered crop assumption
    # More robust: check the render log's START value if available
    # For now use centered crop which matches DynHAMR's default behavior

    print(f"Loading {npz_path}")
    print(f"  {B} hands, {T} npz frames, {n_images} images, shot_start={shot_start}")

    # Velocity per hand (units/s)
    vel = np.linalg.norm(np.diff(trans, axis=1), axis=2) * fps

    for b in range(B):
        hand = 'LEFT' if d['is_right'][b, 0] < 0.5 else 'RIGHT'
        v = vel[b]
        n_bad = np.sum(v > max_vel)
        print(f"  {hand}: vel max={v.max():.2f}, mean={v.mean():.2f}, >{max_vel}: {n_bad} frames")

    # Bad if ANY hand exceeds threshold
    bad = np.any(vel > max_vel, axis=0)
    bad_frames = np.zeros(T, dtype=bool)
    for i in range(len(bad)):
        if bad[i]:
            bad_frames[i] = True
            bad_frames[i + 1] = True

    # Find contiguous good segments
    segments = []
    start = None
    for i in range(T):
        if not bad_frames[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_segment_frames:
                    segments.append((start, i))
                start = None
    if start is not None and T - start >= min_segment_frames:
        segments.append((start, T))

    if not segments:
        print("WARNING: No good segments found!")
        return []

    print(f"\n{len(segments)} segment(s):")
    os.makedirs(output_dir, exist_ok=True)
    output_files = []

    for i, (s, e) in enumerate(segments):
        seg_name = f"{seq}_seg{i+1:03d}"

        # 1. Save trimmed NPZ
        seg = {}
        for key, val in d.items():
            if isinstance(val, np.ndarray) and val.ndim >= 2 and val.shape[1] == T:
                seg[key] = val[:, s:e]
            else:
                seg[key] = val

        npz_out = os.path.join(output_dir, f"{seg_name}.npz")
        np.savez(npz_out, **seg)

        # 2. Generate aligned video clip from images
        vid_out = os.path.join(output_dir, f"{seg_name}.mp4")
        tmp_frames = os.path.join(output_dir, f".tmp_frames_{i}")
        os.makedirs(tmp_frames, exist_ok=True)

        idx = 1
        for npz_frame in range(s, e):
            img_index = shot_start + npz_frame  # 0-indexed into all_images
            if img_index < len(all_images):
                src = all_images[img_index]
                dst = os.path.join(tmp_frames, f"{idx:06d}.jpg")
                os.symlink(os.path.abspath(src), dst)
                idx += 1

        subprocess.run([
            'ffmpeg', '-y', '-framerate', str(fps),
            '-i', os.path.join(tmp_frames, '%06d.jpg'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', vid_out
        ], capture_output=True)

        # Cleanup temp
        for f in glob.glob(os.path.join(tmp_frames, '*.jpg')):
            os.unlink(f)
        os.rmdir(tmp_frames)

        n_frames = e - s
        print(f"  {seg_name}: frames {s}-{e} ({n_frames/fps:.1f}s)")
        print(f"    -> {npz_out}")
        print(f"    -> {vid_out}")
        output_files.append((npz_out, vid_out))

    total_kept = sum(e - s for s, e in segments)
    print(f"\nKept {total_kept}/{T} frames ({total_kept/fps:.1f}s), "
          f"removed {T-total_kept} ({(T-total_kept)/fps:.1f}s)")
    return output_files


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True, help='Pipeline output directory')
    p.add_argument('--output_dir', required=True, help='Output directory for segments')
    p.add_argument('--max_vel', type=float, default=5.0, help='Max wrist velocity threshold')
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--min_segment', type=float, default=1.0, help='Min segment length in seconds')
    a = p.parse_args()
    trim_npz(a.input_dir, a.output_dir, a.max_vel, a.fps, a.min_segment)
