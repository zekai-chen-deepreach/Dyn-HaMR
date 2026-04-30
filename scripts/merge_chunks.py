#!/usr/bin/env python3
"""
Merge Dyn-HaMR chunk outputs into a single unified result.

Given a base sequence name (e.g., GX010024) and the output log directory,
finds all chunk directories (e.g., GX010024-p1-all-shot-0-0--1, GX010024-p2-...),
and merges:
  1. NPZ result files (concatenate along time axis)
  2. MP4 visualization videos (ffmpeg concat)
  3. OBJ mesh files (copy + renumber frames)

Usage:
    python merge_chunks.py --seq GX010024 --log-dir outputs/logs/video-custom/2026-03-01
    python merge_chunks.py --seq GX010024 --log-dir outputs/logs/video-custom/2026-03-01 --num-parts 3
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def _aa2mat(aa):
    """Axis-angle (3,) -> rotation matrix (3, 3)."""
    return Rotation.from_rotvec(aa).as_matrix()


def _mat2aa(mat):
    """Rotation matrix (3, 3) -> axis-angle (3,)."""
    return Rotation.from_matrix(mat).as_rotvec()


def align_chunks(all_data):
    """Align chunk world frames so boundaries are continuous.

    Each chunk's world coordinate system is defined by its camera. This function
    computes a rigid transform (R_corr, t_corr) from the camera extrinsics at
    chunk boundaries, then applies it to all world-space quantities in chunk B
    (both hands' trans/root_orient AND the shared cam_R/cam_t).

    The correction is derived from the camera (not individual hands) so that
    reprojection is preserved: p_cam = cam_R @ p_world + cam_t is invariant.

    Args:
        all_data: list of NpzFile or dicts with keys including
                  'trans', 'root_orient', 'cam_R', 'cam_t'.
    Returns:
        list of mutable dicts with aligned arrays.
    """
    mutable = []
    for d in all_data:
        mutable.append({k: np.array(d[k]) for k in d.keys()})

    for i in range(1, len(mutable)):
        prev = mutable[i - 1]
        curr = mutable[i]

        # Derive world-frame correction from camera extrinsics (shared across hands).
        # cam_R/cam_t are world-to-camera: p_cam = cam_R @ p_world + cam_t
        # At the boundary, both chunks' cameras see the same scene, so:
        #   cam_R_A @ p_A + cam_t_A = cam_R_B @ p_B + cam_t_B
        # With p_A = R_corr @ p_B + t_corr:
        #   R_corr = cam_R_A^T @ cam_R_B
        #   t_corr = cam_R_A^T @ (cam_t_B - cam_t_A)
        cam_R_A = prev["cam_R"][0, -1]   # (3, 3) - last frame of prev chunk
        cam_t_A = prev["cam_t"][0, -1]   # (3,)
        cam_R_B = curr["cam_R"][0, 0]    # (3, 3) - first frame of curr chunk
        cam_t_B = curr["cam_t"][0, 0]    # (3,)

        R_corr = cam_R_A.T @ cam_R_B
        t_corr = cam_R_A.T @ (cam_t_B - cam_t_A)

        # MANO applies an x-flip for left hands BEFORE the root rotation:
        #   joints_world = flip_x(R_orient @ mesh_local) + trans
        # where flip_x negates x for left hands (is_right=0).
        # To preserve reprojection, the root_orient correction for left hands
        # must account for this: R_orient' = F @ R_corr @ F @ R_orient
        # where F = diag(-1, 1, 1). For right hands, F = I so R_orient' = R_corr @ R_orient.
        F = np.diag([-1.0, 1.0, 1.0])

        B = curr["trans"].shape[0]
        T = curr["trans"].shape[1]

        for t in range(T):
            # Transform both hands' world-space poses
            for b in range(B):
                curr["trans"][b, t] = R_corr @ curr["trans"][b, t] + t_corr
                is_right_b = curr["is_right"][b, min(t, curr["is_right"].shape[1] - 1)]
                R_corr_hand = R_corr if is_right_b > 0.5 else F @ R_corr @ F
                R_frame = _aa2mat(curr["root_orient"][b, t])
                curr["root_orient"][b, t] = _mat2aa(R_corr_hand @ R_frame)

            # Transform camera extrinsics (same for both hands, update hand 0 then copy)
            # cam_R' = cam_R @ R_corr^T
            # cam_t' = cam_t - cam_R' @ t_corr
            new_cam_R = curr["cam_R"][0, t] @ R_corr.T
            new_cam_t = curr["cam_t"][0, t] - new_cam_R @ t_corr
            for b in range(B):
                curr["cam_R"][b, t] = new_cam_R
                curr["cam_t"][b, t] = new_cam_t

        # Per-hand translation snap: after world-frame alignment, each hand may
        # still have a position offset due to DynHaMR estimation differences
        # between chunks. Shift each hand's trajectory so the boundary is
        # continuous. This preserves relative motion within the chunk.
        for b in range(B):
            hand_offset = prev["trans"][b, -1] - curr["trans"][b, 0]
            offset_mag = np.linalg.norm(hand_offset)
            if offset_mag > 0.01:  # only snap if offset is significant
                label = "right" if curr["is_right"][b, 0] > 0.5 else "left"
                print(f"    Snapping {label} hand in chunk {i+1} "
                      f"(offset={offset_mag:.4f})")
                curr["trans"][b] += hand_offset

    print(f"  Aligned {len(mutable)} chunks at {len(mutable) - 1} boundary(ies)")
    return mutable


def find_chunk_dirs(log_dir, seq_name):
    """Find chunk output directories in order: {seq}-p1-all-..., {seq}-p2-all-..., etc.
    Also searches sibling date directories in case chunks crossed midnight."""
    pattern = re.compile(rf"^{re.escape(seq_name)}-p(\d+)-all-shot-0-0--1$")
    chunk_dirs = []

    # Search the given log_dir and sibling date directories
    parent = os.path.dirname(log_dir)
    search_dirs = [log_dir]
    if parent and os.path.isdir(parent):
        for sibling in sorted(os.listdir(parent)):
            sibling_path = os.path.join(parent, sibling)
            if sibling_path != log_dir and os.path.isdir(sibling_path):
                search_dirs.append(sibling_path)

    seen_parts = set()
    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        for entry in sorted(os.listdir(sdir)):
            m = pattern.match(entry)
            if m and os.path.isdir(os.path.join(sdir, entry)):
                part_num = int(m.group(1))
                full_path = os.path.join(sdir, entry)
                # Only use this chunk if it has actual results (not an empty leftover)
                has_results = (
                    glob.glob(f"{full_path}/smooth_fit/*_world_results.npz")
                    or glob.glob(f"{full_path}/*.mp4")
                )
                if has_results and part_num not in seen_parts:
                    chunk_dirs.append((part_num, full_path))
                    seen_parts.add(part_num)

    chunk_dirs.sort(key=lambda x: x[0])
    return [d for _, d in chunk_dirs]


def get_final_iteration(res_dir):
    """Get the highest iteration number from result files in a phase directory."""
    res_files = sorted(glob.glob(f"{res_dir}/*_world_results.npz"))
    if not res_files:
        return None
    iterations = set()
    for f in res_files:
        parts = os.path.basename(f).split("_")
        # Format: {seq}_{iter}_world_results.npz
        # iter is 6 digits
        for p in parts:
            if p.isdigit() and len(p) == 6:
                iterations.add(p)
    return sorted(iterations)[-1] if iterations else None


def merge_npz_files(chunk_dirs, merged_dir, phase="smooth_fit"):
    """Merge NPZ result files from chunks by concatenating along time axis."""
    phase_dir = os.path.join(merged_dir, phase)
    os.makedirs(phase_dir, exist_ok=True)

    # Find the final iteration from the first chunk
    first_phase_dir = os.path.join(chunk_dirs[0], phase)
    if not os.path.isdir(first_phase_dir):
        print(f"  Phase dir {first_phase_dir} not found, skipping NPZ merge")
        return None

    final_iter = get_final_iteration(first_phase_dir)
    if final_iter is None:
        print(f"  No result files found in {first_phase_dir}")
        return None

    print(f"  Merging NPZ files for phase={phase}, iteration={final_iter}")

    # Collect result files from all chunks for the final iteration
    chunk_results = []
    for cdir in chunk_dirs:
        cphase_dir = os.path.join(cdir, phase)
        # Find the result file matching this iteration
        pattern = f"{cphase_dir}/*_{final_iter}_world_results.npz"
        matches = glob.glob(pattern)
        if not matches:
            print(f"  WARNING: No result file for iteration {final_iter} in {cphase_dir}")
            return None
        chunk_results.append(matches[0])

    # Load all chunk results and align world frames at boundaries
    all_data = [np.load(f) for f in chunk_results]
    keys = list(all_data[0].keys())
    print(f"  NPZ keys: {keys}")

    if len(all_data) > 1:
        all_data = align_chunks(all_data)

    # Determine which keys have a time dimension (axis 1) to concatenate
    # Arrays with shape (B, T, ...) need concatenation along T
    # Arrays with shape (B,) or (N,) or scalar are kept from first chunk
    # "betas" is MANO shape params (B, 10), NOT time-varying
    NON_TEMPORAL_KEYS = {"betas", "world_scale", "intrins"}

    merged = {}
    seq_name = os.path.basename(merged_dir).split("-all-")[0]

    for key in keys:
        arrays = [d[key] for d in all_data]
        shape0 = arrays[0].shape

        if key in NON_TEMPORAL_KEYS:
            merged[key] = arrays[0]
            print(f"    {key}: {shape0} (kept from first chunk, non-temporal)")
        elif len(shape0) >= 2 and shape0[1] > 1:
            # Likely (B, T, ...) - concatenate along T (axis 1)
            try:
                merged[key] = np.concatenate(arrays, axis=1)
                print(f"    {key}: {[a.shape for a in arrays]} -> {merged[key].shape} (concatenated)")
            except ValueError as e:
                print(f"    {key}: concat failed ({e}), using first chunk")
                merged[key] = arrays[0]
        else:
            # Scalar, (B,), (1,), (4,), (B, 1, ...) - keep from first chunk
            merged[key] = arrays[0]
            print(f"    {key}: {shape0} (kept from first chunk)")

    # Save merged result
    out_path = os.path.join(phase_dir, f"{seq_name}_{final_iter}_world_results.npz")
    np.savez(out_path, **merged)
    print(f"  Saved merged NPZ to {out_path}")

    for d in all_data:
        if hasattr(d, "close"):
            d.close()

    return final_iter


def merge_mp4_files(chunk_dirs, merged_dir, seq_name, final_iter, ffmpeg="ffmpeg", phase="smooth_fit"):
    """Merge MP4 visualization videos using ffmpeg concat."""
    views = ["src_cam", "above", "side", "front"]
    video_types = []

    # Input video
    video_types.append(("input", f"_input.mp4"))

    # Phase visualization videos
    if final_iter:
        for view in views:
            video_types.append((f"{phase}_{view}", f"_{phase}_final_{final_iter}_{view}.mp4"))
        video_types.append((f"{phase}_grid", f"_{phase}_grid.mp4"))

    for label, suffix in video_types:
        # Collect chunk videos
        chunk_videos = []
        for cdir in chunk_dirs:
            chunk_seq = os.path.basename(cdir).split("-all-")[0]
            vid_path = os.path.join(cdir, f"{chunk_seq}{suffix}")
            if os.path.isfile(vid_path):
                chunk_videos.append(vid_path)
            else:
                print(f"  WARNING: {vid_path} not found, skipping {label}")
                break
        else:
            if not chunk_videos:
                continue

            # Create concat list file
            list_path = os.path.join(merged_dir, f"_concat_{label}.txt")
            with open(list_path, "w") as f:
                for vp in chunk_videos:
                    f.write(f"file '{vp}'\n")

            out_path = os.path.join(merged_dir, f"{seq_name}{suffix}")
            cmd = [
                ffmpeg, "-y", "-loglevel", "warning",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                out_path
            ]
            print(f"  Merging {label}: {len(chunk_videos)} chunks -> {out_path}")
            subprocess.run(cmd, check=True)
            os.remove(list_path)

    print(f"  MP4 merge complete")


def merge_mesh_files(chunk_dirs, merged_dir, seq_name, final_iter, phase="smooth_fit", max_frames=600):
    """Merge OBJ mesh files from chunks, renumbering frames."""
    if not final_iter:
        return

    out_mesh_dir = os.path.join(merged_dir, phase, f"{seq_name}_{final_iter}_meshes")
    os.makedirs(out_mesh_dir, exist_ok=True)

    frame_offset = 0
    for i, cdir in enumerate(chunk_dirs):
        chunk_seq = os.path.basename(cdir).split("-all-")[0]
        chunk_mesh_dir = os.path.join(cdir, phase, f"{chunk_seq}_{final_iter}_meshes")

        if not os.path.isdir(chunk_mesh_dir):
            print(f"  WARNING: Mesh dir not found: {chunk_mesh_dir}")
            # Still advance offset
            frame_offset += max_frames
            continue

        # Find all OBJ files in this chunk
        obj_files = sorted(glob.glob(f"{chunk_mesh_dir}/*.obj"))
        if not obj_files:
            frame_offset += max_frames
            continue

        # Parse frame numbers to find max frame in this chunk
        max_frame_in_chunk = 0
        for obj_file in obj_files:
            basename = os.path.basename(obj_file)
            # Format: {frame:06d}_{hand_id}.obj
            parts = basename.replace(".obj", "").split("_")
            frame_num = int(parts[0])
            hand_id = parts[1]
            max_frame_in_chunk = max(max_frame_in_chunk, frame_num)

            new_frame = frame_num + frame_offset
            new_name = f"{new_frame:06d}_{hand_id}.obj"
            shutil.copy2(obj_file, os.path.join(out_mesh_dir, new_name))

        print(f"  Chunk {i+1}: copied {len(obj_files)} meshes (frame offset={frame_offset})")
        frame_offset += max_frame_in_chunk + 1

    total = len(glob.glob(f"{out_mesh_dir}/*.obj"))
    print(f"  Mesh merge complete: {total} total OBJ files in {out_mesh_dir}")


def copy_metadata(chunk_dirs, merged_dir):
    """Copy metadata. cameras.json from first chunk; track_info.json: concat per-track vis_mask across chunks."""
    src = chunk_dirs[0]
    cam_src = os.path.join(src, "cameras.json")
    if os.path.isfile(cam_src):
        shutil.copy2(cam_src, os.path.join(merged_dir, "cameras.json"))
        print(f"  Copied cameras.json")

    merged_track = None
    for cd in chunk_dirs:
        tp = os.path.join(cd, "track_info.json")
        if not os.path.isfile(tp):
            continue
        with open(tp, "r") as f:
            ti = json.load(f)
        if merged_track is None:
            merged_track = ti
            continue
        for tid, td in ti.get("tracks", {}).items():
            if tid not in merged_track["tracks"]:
                merged_track["tracks"][tid] = td
                continue
            for k, v in td.items():
                if isinstance(v, list) and isinstance(merged_track["tracks"][tid].get(k), list):
                    merged_track["tracks"][tid][k] = merged_track["tracks"][tid][k] + v
    if merged_track is not None:
        with open(os.path.join(merged_dir, "track_info.json"), "w") as f:
            json.dump(merged_track, f)
        sample_tid = next(iter(merged_track["tracks"]))
        sample_vm = merged_track["tracks"][sample_tid].get("vis_mask", [])
        print(f"  Merged track_info.json (track {sample_tid} vis_mask len={len(sample_vm)})")

    hydra_src = os.path.join(src, ".hydra")
    hydra_dst = os.path.join(merged_dir, ".hydra")
    if os.path.isdir(hydra_src) and not os.path.isdir(hydra_dst):
        shutil.copytree(hydra_src, hydra_dst)
        print(f"  Copied .hydra/")


def merge_sequence(seq_name, log_dir, max_frames=600, ffmpeg="ffmpeg"):
    """Merge all chunk outputs for a given sequence."""
    chunk_dirs = find_chunk_dirs(log_dir, seq_name)
    if len(chunk_dirs) < 2:
        print(f"Found {len(chunk_dirs)} chunk(s) for {seq_name}, nothing to merge")
        return False

    print(f"\nMerging {len(chunk_dirs)} chunks for {seq_name}:")
    for cd in chunk_dirs:
        print(f"  {os.path.basename(cd)}")

    merged_dir = os.path.join(log_dir, f"{seq_name}-all-shot-0-0--1")
    os.makedirs(merged_dir, exist_ok=True)
    print(f"Output: {merged_dir}")

    # 1. Copy metadata from first chunk
    print("\n[1/4] Copying metadata...")
    copy_metadata(chunk_dirs, merged_dir)

    # 2. Merge NPZ files
    print("\n[2/4] Merging NPZ result files...")
    final_iter = merge_npz_files(chunk_dirs, merged_dir, phase="smooth_fit")

    # 3. Merge MP4 videos
    print("\n[3/4] Merging MP4 visualization videos...")
    merge_mp4_files(chunk_dirs, merged_dir, seq_name, final_iter, ffmpeg=ffmpeg)

    # 4. Merge OBJ meshes
    print("\n[4/4] Merging OBJ mesh files...")
    merge_mesh_files(chunk_dirs, merged_dir, seq_name, final_iter, max_frames=max_frames)

    print(f"\nMerge complete for {seq_name}: {merged_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge Dyn-HaMR chunk outputs")
    parser.add_argument("--seq", required=True, help="Base sequence name (e.g., GX010024)")
    parser.add_argument("--log-dir", required=True, help="Log directory containing chunk outputs")
    parser.add_argument("--max-frames", type=int, default=600, help="Max frames per chunk (for mesh renumbering)")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg binary")
    args = parser.parse_args()

    success = merge_sequence(args.seq, args.log_dir, max_frames=args.max_frames, ffmpeg=args.ffmpeg)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
