#!/usr/bin/env python3
"""Generate separate left/right arm masks using SAM3.

Runs in the `sam3` conda env. Produces an NPZ with 'left_masks' and 'right_masks'.

Usage:
    conda run -n sam3 python generate_lr_arm_masks.py \
        --frames-dir /path/to/frames/ \
        --output /path/to/lr_arm_masks.npz \
        --resize 0.25
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
    exts = ("*.jpg", "*.jpeg", "*.png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    paths.sort()
    return paths


def propagate_prompt(model, inference_state, prompt, chunk_size=300):
    """Add text prompt and propagate in chunks. Returns dict {frame_idx: mask}."""
    model.reset_state(inference_state)
    torch.cuda.empty_cache()

    num_frames = inference_state["num_frames"]

    with torch.inference_mode():
        frame_idx, outputs = model.add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            text_str=prompt,
        )

    n_objs = len(outputs.get("out_obj_ids", []))
    print(f"  Prompt '{prompt}': {n_objs} object(s) on frame 0")
    if n_objs == 0:
        return {}

    masks_dict = {}
    frame_count = 0

    with torch.inference_mode():
        for fid, out in model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=chunk_size,
            reverse=False,
        ):
            if out is not None:
                bm = out.get("out_binary_masks")
                if bm is not None and len(bm) > 0:
                    if isinstance(bm, torch.Tensor):
                        bm = bm.cpu().numpy()
                    masks_dict[fid] = bm.any(axis=0)
            frame_count += 1

    # Continue in chunks
    while frame_count < num_frames:
        last = max(masks_dict.keys())
        model.reset_state(inference_state)
        torch.cuda.empty_cache()

        with torch.inference_mode():
            model.add_prompt(
                inference_state=inference_state,
                frame_idx=last,
                text_str=prompt,
            )

        chunk_new = 0
        with torch.inference_mode():
            for fid, out in model.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=last,
                max_frame_num_to_track=chunk_size,
                reverse=False,
            ):
                if fid not in masks_dict and out is not None:
                    bm = out.get("out_binary_masks")
                    if bm is not None and len(bm) > 0:
                        if isinstance(bm, torch.Tensor):
                            bm = bm.cpu().numpy()
                        masks_dict[fid] = bm.any(axis=0)
                    chunk_new += 1
                frame_count += 1

        if chunk_new == 0:
            break

    print(f"  '{prompt}': {len(masks_dict)} frames with masks")
    return masks_dict


def masks_dict_to_array(masks_dict, num_frames, resize, orig_h, orig_w):
    h_out = int(orig_h * resize)
    w_out = int(orig_w * resize)
    arr = np.zeros((num_frames, h_out, w_out), dtype=np.uint8)
    for idx, mask in masks_dict.items():
        if mask is None:
            continue
        m = mask.astype(np.uint8) * 255
        if resize != 1.0:
            m = cv2.resize(m, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
        arr[idx] = (m > 127).astype(np.uint8)
    return arr


def main():
    parser = argparse.ArgumentParser(description="Generate left/right arm masks with SAM3")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resize", type=float, default=0.25)
    parser.add_argument("--left-prompt", default="left arm")
    parser.add_argument("--right-prompt", default="right arm")
    args = parser.parse_args()

    frame_paths = sorted_frame_paths(args.frames_dir)
    if not frame_paths:
        print(f"No frames in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    first = cv2.imread(frame_paths[0])
    orig_h, orig_w = first.shape[:2]
    print(f"Found {len(frame_paths)} frames, {orig_w}x{orig_h}")

    from sam3.model_builder import build_sam3_video_model
    print("Building SAM3 model...")
    model = build_sam3_video_model()
    model = model.cuda().eval()

    print("Loading frames (offload to CPU)...")
    with torch.inference_mode():
        inference_state = model.init_state(
            resource_path=args.frames_dir,
            offload_video_to_cpu=True,
        )
    num_frames = inference_state["num_frames"]
    print(f"Loaded {num_frames} frames")

    t0 = time.time()

    # Left arm
    print("\n--- Left arm ---")
    left_dict = propagate_prompt(model, inference_state, args.left_prompt)

    # Right arm
    print("\n--- Right arm ---")
    right_dict = propagate_prompt(model, inference_state, args.right_prompt)

    # Convert to arrays
    left_arr = masks_dict_to_array(left_dict, num_frames, args.resize, orig_h, orig_w)
    right_arr = masks_dict_to_array(right_dict, num_frames, args.resize, orig_h, orig_w)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        left_masks=left_arr,
        right_masks=right_arr,
        orig_h=orig_h,
        orig_w=orig_w,
        resize_factor=args.resize,
    )

    elapsed = time.time() - t0
    left_valid = (left_arr.max(axis=(1, 2)) > 0).sum()
    right_valid = (right_arr.max(axis=(1, 2)) > 0).sum()
    fsize = os.path.getsize(args.output) / 1024 / 1024
    print(f"\nSaved to {args.output} ({fsize:.1f} MB)")
    print(f"  Left arm: {left_valid}/{num_frames} frames")
    print(f"  Right arm: {right_valid}/{num_frames} frames")
    print(f"  Shape: ({num_frames}, {left_arr.shape[1]}, {left_arr.shape[2]})")
    print(f"  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
