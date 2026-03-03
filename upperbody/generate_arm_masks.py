#!/usr/bin/env python3
"""Generate arm masks using SAM3 video predictor.

Runs in the `sam3` conda environment. Segments arms from video frames using
SAM3's text-prompted video segmentation, then saves binary masks as NPZ.

Usage:
    conda run -n sam3 python generate_arm_masks.py \
        --frames-dir /path/to/frames/ \
        --output /path/to/arm_masks.npz \
        [--prompt "arm"] \
        [--resize 0.25]
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch


def sorted_frame_paths(frames_dir):
    """Return sorted list of image paths in the frames directory."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    paths.sort()
    return paths


def generate_masks_text_prompt(frames_dir, prompt, device="cuda"):
    """Segment arms using SAM3 text prompt on video frames.

    Uses offload_video_to_cpu=True to keep frame tensors on CPU and avoid OOM.

    Args:
        frames_dir: directory of JPEG/PNG frames (sorted alphabetically)
        prompt: text prompt (e.g. "arm")
        device: torch device

    Returns:
        masks: dict mapping frame_idx -> binary mask (H, W) at original res
        num_frames: total number of frames
    """
    from sam3.model_builder import build_sam3_video_model

    print("Building SAM3 video model...")
    model = build_sam3_video_model()
    model = model.to(device).eval()

    # Init state with offload_video_to_cpu=True to avoid OOM on long videos
    print(f"Loading frames from {frames_dir} (offloading to CPU)...")
    with torch.inference_mode():
        inference_state = model.init_state(
            resource_path=frames_dir,
            offload_video_to_cpu=True,
        )

    num_frames = inference_state["num_frames"]
    print(f"Loaded {num_frames} frames")

    # Add text prompt on frame 0
    with torch.inference_mode():
        frame_idx, outputs_f0 = model.add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=prompt,
        )
    # add_prompt returns already-postprocessed output
    n_objs = len(outputs_f0.get("out_obj_ids", []))
    print(f"Added text prompt '{prompt}' on frame 0 -> {n_objs} object(s)")

    if n_objs == 0:
        print("WARNING: No objects detected with text prompt.")
        return {}, 0

    # Propagate forward in chunks to avoid OOM on long videos
    # max_frame_num_to_track limits how many frames are tracked at once
    chunk_size = 300  # frames per propagation chunk
    masks_dict = {}
    frame_count = 0

    with torch.inference_mode():
        for fid, outputs in model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=chunk_size,
            reverse=False,
        ):
            if outputs is not None:
                binary_masks = outputs.get("out_binary_masks")
                if binary_masks is not None and len(binary_masks) > 0:
                    if isinstance(binary_masks, torch.Tensor):
                        binary_masks = binary_masks.cpu().numpy()
                    combined = binary_masks.any(axis=0)
                    masks_dict[fid] = combined
                else:
                    masks_dict[fid] = None
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Propagated {frame_count} frames...")

    print(f"First chunk: {frame_count} frames")

    # Continue propagation in chunks for remaining frames
    while frame_count < num_frames:
        # Re-prompt at the last successfully tracked frame to continue
        last_tracked = max(masks_dict.keys())
        print(f"  Continuing from frame {last_tracked}...")

        # Reset and re-add prompt to free accumulated tracking state
        model.reset_state(inference_state)
        torch.cuda.empty_cache()

        frame_idx, _ = model.add_prompt(
            inference_state=inference_state,
            frame_idx=last_tracked,
            text_str=prompt,
        )

        chunk_count = 0
        for fid, outputs in model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=last_tracked,
            max_frame_num_to_track=chunk_size,
            reverse=False,
        ):
            if fid not in masks_dict and outputs is not None:
                binary_masks = outputs.get("out_binary_masks")
                if binary_masks is not None and len(binary_masks) > 0:
                    if isinstance(binary_masks, torch.Tensor):
                        binary_masks = binary_masks.cpu().numpy()
                    combined = binary_masks.any(axis=0)
                    masks_dict[fid] = combined
                else:
                    masks_dict[fid] = None
                chunk_count += 1
            frame_count += 1

        print(f"  Chunk added {chunk_count} new frames (total: {len(masks_dict)})")
        if chunk_count == 0:
            break  # No new frames added, we're done

    print(f"Total: {len(masks_dict)} frames with masks")

    return masks_dict, num_frames


def generate_masks_point_prompt(frames_dir, wrist_points, device="cuda"):
    """Fallback: use projected wrist positions as point prompts.

    Args:
        frames_dir: directory of frames
        wrist_points: (N, 2) array of (x, y) wrist positions on frame 0
        device: torch device

    Returns:
        masks: dict mapping frame_idx -> binary mask (H, W)
        num_frames: total number of frames
    """
    from sam3.model_builder import build_sam3_video_predictor

    predictor = build_sam3_video_predictor()

    session = predictor.handle_request({
        "type": "start_session",
        "resource_path": frames_dir,
    })
    session_id = session["session_id"]

    # Add point prompts on frame 0
    points = wrist_points.tolist()
    point_labels = [1] * len(points)  # 1 = foreground

    result = predictor.handle_request({
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": 0,
        "points": points,
        "point_labels": point_labels,
    })

    outputs_f0 = result["outputs"]
    n_objs = len(outputs_f0.get("out_obj_ids", []))
    print(f"Point prompt: detected {n_objs} object(s) on frame 0")

    masks_dict = {}
    frame_count = 0
    for result in predictor.handle_stream_request({
        "type": "propagate_in_video",
        "session_id": session_id,
        "propagation_direction": "both",
    }):
        frame_idx = result["frame_index"]
        outputs = result["outputs"]
        binary_masks = outputs.get("out_binary_masks")

        if binary_masks is not None and len(binary_masks) > 0:
            combined = binary_masks.any(axis=0)
            masks_dict[frame_idx] = combined
        else:
            masks_dict[frame_idx] = None

        frame_count += 1

    predictor.handle_request({"type": "close_session", "session_id": session_id})
    return masks_dict, frame_count


def masks_to_npz(masks_dict, num_frames, resize, orig_h, orig_w, output_path):
    """Convert masks dict to NPZ file.

    Args:
        masks_dict: dict mapping frame_idx -> (H, W) bool mask or None
        num_frames: total frame count
        resize: resize factor (e.g. 0.25)
        orig_h, orig_w: original frame dimensions
        output_path: path to save NPZ

    Saves:
        masks: (T, H_out, W_out) uint8
        frame_indices: (T,) int
        orig_h, orig_w: original resolution
        resize_factor: the resize factor used
    """
    h_out = int(orig_h * resize)
    w_out = int(orig_w * resize)

    # Collect all frame indices in sorted order
    all_indices = sorted(masks_dict.keys())

    # Build full-sequence mask array
    masks_arr = np.zeros((num_frames, h_out, w_out), dtype=np.uint8)
    valid_count = 0

    for idx in all_indices:
        mask = masks_dict[idx]
        if mask is None:
            continue

        mask_uint8 = mask.astype(np.uint8) * 255

        if resize != 1.0:
            mask_resized = cv2.resize(
                mask_uint8, (w_out, h_out), interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask_uint8

        masks_arr[idx] = (mask_resized > 127).astype(np.uint8)
        valid_count += 1

    frame_indices = np.arange(num_frames, dtype=np.int32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        masks=masks_arr,
        frame_indices=frame_indices,
        orig_h=orig_h,
        orig_w=orig_w,
        resize_factor=resize,
    )

    print(f"Saved {valid_count}/{num_frames} valid masks to {output_path}")
    print(f"  Shape: ({num_frames}, {h_out}, {w_out}), dtype=uint8")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Generate arm masks using SAM3 video predictor"
    )
    parser.add_argument(
        "--frames-dir", required=True,
        help="Directory containing video frames (JPEG/PNG, sorted alphabetically)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output NPZ path for arm masks",
    )
    parser.add_argument(
        "--prompt", default="arm",
        help="Text prompt for SAM3 segmentation (default: 'arm')",
    )
    parser.add_argument(
        "--resize", type=float, default=0.25,
        help="Resize factor for output masks (default: 0.25 = quarter resolution)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for SAM3 model (default: cuda)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.frames_dir):
        print(f"Error: frames directory not found: {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    frame_paths = sorted_frame_paths(args.frames_dir)
    if len(frame_paths) == 0:
        print(f"Error: no image files found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    # Read first frame to get original dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"Error: cannot read {frame_paths[0]}", file=sys.stderr)
        sys.exit(1)
    orig_h, orig_w = first_frame.shape[:2]
    print(f"Found {len(frame_paths)} frames, resolution: {orig_w}x{orig_h}")

    t_start = time.time()

    # Generate masks with text prompt
    print(f"\nRunning SAM3 with text prompt: '{args.prompt}'")
    masks_dict, num_frames = generate_masks_text_prompt(
        args.frames_dir, args.prompt, device=args.device
    )

    if num_frames == 0:
        print("Error: no frames processed", file=sys.stderr)
        sys.exit(1)

    # Check mask coverage
    valid_masks = sum(1 for v in masks_dict.values() if v is not None)
    coverage = valid_masks / num_frames * 100
    print(f"\nMask coverage: {valid_masks}/{num_frames} frames ({coverage:.1f}%)")

    if valid_masks == 0:
        print("WARNING: No valid masks generated. The output will be all-zero masks.")

    # Save to NPZ
    masks_to_npz(masks_dict, num_frames, args.resize, orig_h, orig_w, args.output)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.1f}s ({elapsed / num_frames:.2f}s/frame)")


if __name__ == "__main__":
    main()
