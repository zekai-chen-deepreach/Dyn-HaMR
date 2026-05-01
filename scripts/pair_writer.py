"""
Pair writer for the HaMeR ↔ DynHaMR refinement dataset.

For one chunk, reads:
  - HaMeR per-frame raw pkl     (test/dynhamr/hamer_out/<seq>/<seq>.pkl)
  - DynHaMR smooth_fit NPZ      (logs/.../<seq>-all-shot-0-0--1/smooth_fit/<seq>_000060_world_results.npz)
  - SLERP outlier mask          (computed during postprocess on this chunk)

Writes one self-contained NPZ pair to:
  <dataset_root>/shards/<lo>-<hi>/<pair_id>.npz
And appends one row to <dataset_root>/manifest.parquet (or jsonl fallback).

Schema (V3, 2026-04-30):
  metadata      : pair_id, source_video, chunk_idx, src_frame_start/end, fps_src, n_frames, img_size
  hamer.*       : raw per-frame detections in HaMeR cam frame
                  rotation matrices (3,3) for global_orient and 15 finger joint poses
                  trans is SLAHMR-style pseudo-3D [tx,ty,tz] meters
                  bbox/det_valid track per-frame detection state
  dyn.*         : DynHaMR refined in chunk-local world frame
                  rotation matrices (converted from axis-angle)
                  cam_R/cam_t deduplicated to (T,3,3)/(T,3) (single VIPE camera)
                  outlier_fixed bool mask (where SLERP touched)
"""
import argparse
import json
import os
import pickle
import sys
import time
import uuid

import numpy as np


def axis_angle_to_matrix_np(aa):
    """(..., 3) axis-angle → (..., 3, 3) rotation matrix. Vectorized Rodrigues."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)  # (..., 1)
    safe = np.where(theta < 1e-8, np.ones_like(theta), theta)
    k = aa / safe                                       # (..., 3) unit axis
    K = np.zeros(aa.shape[:-1] + (3, 3), dtype=np.float64)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] =  k[..., 0]
    sin_t = np.sin(theta)[..., None]
    cos_t = np.cos(theta)[..., None]
    eye = np.broadcast_to(np.eye(3), aa.shape[:-1] + (3, 3))
    R = eye + sin_t * K + (1 - cos_t) * (K @ K)
    R = np.where(theta[..., None] < 1e-8, eye, R)
    return R.astype(np.float32)


def load_hamer_pkl(pkl_path, n_frames, image_dir, img_size_wh):
    """
    Returns dict with per-track per-frame raw HaMeR detections.

    HaMeR pkl is keyed by image path → list of detections (one per detected hand in that frame).
    Each detection has handedness; we sort detections per frame so track 0 = is_right=0 (left),
    track 1 = is_right=1 (right). Missing tracks → det_valid=False, fields filled with NaN/zeros.
    """
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    W, H = img_size_wh
    # Build expected per-frame key list. HaMeR keys by absolute image path.
    img_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")])
    img_paths = img_paths[:n_frames]
    assert len(img_paths) == n_frames, f"image dir has {len(img_paths)} jpgs, expected {n_frames}"

    B = 2  # left, right
    out = {
        "is_right":      np.array([0, 1], dtype=np.int8),               # (2,)
        "det_valid":     np.zeros((B, n_frames), dtype=bool),
        "bbox":          np.full((B, n_frames, 4), np.nan, dtype=np.float32),
        "global_orient": np.tile(np.eye(3, dtype=np.float32), (B, n_frames, 1, 1)),  # (B,T,3,3)
        "pose_body":     np.tile(np.eye(3, dtype=np.float32), (B, n_frames, 15, 1, 1)),  # (B,T,15,3,3)
        "betas":         np.zeros((B, n_frames, 10), dtype=np.float32),
        "trans":         np.full((B, n_frames, 3), np.nan, dtype=np.float32),
        "joints2d":      np.zeros((B, n_frames, 21, 3), dtype=np.float32),
    }

    n_missing_keys = 0
    for t, img_path in enumerate(img_paths):
        if img_path not in results:
            n_missing_keys += 1
            continue
        frame = results[img_path]
        manos = frame.get("mano", []) or []
        cams  = frame.get("cam_trans", []) or []
        bboxes = frame.get("bbox", []) or []
        kps   = frame.get("extra_data", []) or []
        if len(manos) == 0:
            continue
        for d_idx, mano in enumerate(manos):
            is_right = int(mano["is_right"])
            if is_right not in (0, 1):
                continue
            b = is_right
            out["det_valid"][b, t] = True
            out["global_orient"][b, t] = mano["global_orient"].squeeze().astype(np.float32)  # (3,3)
            out["pose_body"][b, t]    = mano["hand_pose"].astype(np.float32)                 # (15,3,3)
            out["betas"][b, t]        = mano["betas"].astype(np.float32)                     # (10,)
            if d_idx < len(cams):
                out["trans"][b, t] = np.asarray(cams[d_idx], dtype=np.float32)
            if d_idx < len(bboxes):
                out["bbox"][b, t]  = np.asarray(bboxes[d_idx], dtype=np.float32)
            if d_idx < len(kps):
                kp = np.asarray(kps[d_idx], dtype=np.float32)  # (21, 3) [x,y,conf]
                out["joints2d"][b, t] = kp

    if n_missing_keys > 0:
        print(f"  hamer: {n_missing_keys}/{n_frames} frames missing key in pkl (treated as no det)")

    return out


def load_dyn_smooth_npz(npz_path, n_frames):
    """Load DynHaMR smooth_fit NPZ for one chunk, convert axis-angle → rotation matrix."""
    d = np.load(npz_path, allow_pickle=True)

    root_aa = d["root_orient"][:, :n_frames]   # (B, T, 3) — assert chunk-local
    pose_aa = d["pose_body"][:, :n_frames]     # (B, T, 15, 3)
    trans   = d["trans"][:, :n_frames].astype(np.float32)
    betas   = d["betas"].astype(np.float32)
    is_r    = d["is_right"][:, 0].astype(np.int8)  # (B,) constant per track
    cam_R   = d["cam_R"][0, :n_frames].astype(np.float32)  # dedup: take track 0
    cam_t   = d["cam_t"][0, :n_frames].astype(np.float32)
    intrins = d["intrins"].astype(np.float32)

    # Sanity: cam_R/cam_t identical across tracks
    assert np.array_equal(d["cam_R"][0], d["cam_R"][1]), "cam_R differs across hands!"
    assert np.array_equal(d["cam_t"][0], d["cam_t"][1]), "cam_t differs across hands!"

    return {
        "is_right":     is_r,
        "root_orient":  axis_angle_to_matrix_np(root_aa),    # (B, T, 3, 3)
        "pose_body":    axis_angle_to_matrix_np(pose_aa),    # (B, T, 15, 3, 3)
        "trans":        trans,
        "betas":        betas,
        "cam_R":        cam_R,
        "cam_t":        cam_t,
        "intrins":      intrins,
    }


def compute_outlier_mask(raw_npz_path, post_npz_path, n_frames, eps=1e-6):
    """Compare raw chunk NPZ vs post-processed (SLERP) version → bool mask of touched frames."""
    raw  = np.load(raw_npz_path)
    post = np.load(post_npz_path)
    diff = np.abs(raw["trans"][:, :n_frames] - post["trans"][:, :n_frames]).sum(axis=-1)  # (B, T)
    return (diff > eps)


def write_pair(args):
    out_dir = args.out_root
    os.makedirs(out_dir, exist_ok=True)
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    img_size_wh = (int(args.img_w), int(args.img_h))
    focal_assumed = float(max(img_size_wh) * 5000.0 / 256.0)

    # Determine pair_id from source_video + chunk_idx
    src_stem = os.path.splitext(os.path.basename(args.source_video))[0]
    pair_id = f"{src_stem}__chunk_{args.chunk_idx:04d}"

    # Decide shard
    shard_size = args.shard_size
    pair_global_idx = args.pair_global_idx
    lo = (pair_global_idx // shard_size) * shard_size
    hi = lo + shard_size - 1
    shard_subdir = os.path.join(shards_dir, f"{lo:05d}-{hi:05d}")
    os.makedirs(shard_subdir, exist_ok=True)
    out_npz = os.path.join(shard_subdir, f"{pair_global_idx:05d}_{pair_id}.npz")

    # Load HaMeR raw
    print(f"[pair_writer] loading hamer pkl: {args.hamer_pkl}")
    hamer = load_hamer_pkl(args.hamer_pkl, args.n_frames, args.image_dir, img_size_wh)

    # Load DynHaMR refined (post-SLERP)
    print(f"[pair_writer] loading dyn smooth_fit: {args.dyn_post_npz}")
    dyn = load_dyn_smooth_npz(args.dyn_post_npz, args.n_frames)

    # Compute outlier_fixed mask
    if args.dyn_raw_npz and os.path.isfile(args.dyn_raw_npz):
        dyn["outlier_fixed"] = compute_outlier_mask(args.dyn_raw_npz, args.dyn_post_npz, args.n_frames)
    else:
        dyn["outlier_fixed"] = np.zeros((2, args.n_frames), dtype=bool)

    # Compose payload
    payload = {
        # metadata
        "pair_id":         pair_id,
        "source_video":    os.path.basename(args.source_video),
        "chunk_idx":       np.int32(args.chunk_idx),
        "src_frame_start": np.int32(args.src_frame_start),
        "src_frame_end":   np.int32(args.src_frame_start + args.n_frames),
        "fps_src":         np.float32(args.fps_src),
        "n_frames":        np.int32(args.n_frames),
        "img_size":        np.array([img_size_wh[0], img_size_wh[1]], dtype=np.int32),
        # hamer
        "hamer.is_right":             hamer["is_right"],
        "hamer.det_valid":            hamer["det_valid"],
        "hamer.bbox":                 hamer["bbox"],
        "hamer.global_orient":        hamer["global_orient"],
        "hamer.pose_body":            hamer["pose_body"],
        "hamer.betas":                hamer["betas"],
        "hamer.trans":                hamer["trans"],
        "hamer.joints2d":             hamer["joints2d"],
        "hamer.focal_length_assumed": np.float32(focal_assumed),
        # dyn
        "dyn.is_right":      dyn["is_right"],
        "dyn.root_orient":   dyn["root_orient"],
        "dyn.pose_body":     dyn["pose_body"],
        "dyn.trans":         dyn["trans"],
        "dyn.betas":         dyn["betas"],
        "dyn.cam_R":         dyn["cam_R"],
        "dyn.cam_t":         dyn["cam_t"],
        "dyn.intrins":       dyn["intrins"],
        "dyn.outlier_fixed": dyn["outlier_fixed"],
    }

    np.savez_compressed(out_npz, **payload)
    size_mb = os.path.getsize(out_npz) / 1024 / 1024
    print(f"[pair_writer] wrote {out_npz} ({size_mb:.1f} MB)")

    # Manifest row (jsonl, one per pair — easy to convert to parquet later)
    manifest_path = os.path.join(out_dir, "manifest.jsonl")
    det_rate = float(hamer["det_valid"].mean())
    outlier_ratio = float(payload["dyn.outlier_fixed"].mean())
    trans_norms = np.linalg.norm(dyn["trans"], axis=-1)  # (B, T)
    mean_trans = float(trans_norms.mean())
    max_jump = float(np.max(np.linalg.norm(np.diff(dyn["trans"], axis=1), axis=-1))) if args.n_frames > 1 else 0.0

    row = {
        "pair_id": pair_id,
        "shard_path": os.path.relpath(out_npz, out_dir),
        "source_video": os.path.basename(args.source_video),
        "chunk_idx": int(args.chunk_idx),
        "n_frames": int(args.n_frames),
        "src_frame_start": int(args.src_frame_start),
        "src_frame_end": int(args.src_frame_start + args.n_frames),
        "fps_src": float(args.fps_src),
        "img_w": img_size_wh[0],
        "img_h": img_size_wh[1],
        "hamer_det_rate": det_rate,
        "outlier_fix_ratio": outlier_ratio,
        "mean_trans_norm_dyn": mean_trans,
        "max_jump_dyn": max_jump,
        "size_mb": float(f"{size_mb:.2f}"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(manifest_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[pair_writer] manifest += 1 row → {manifest_path}")
    return out_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-video", required=True, help="path to original full-length source video (for pair_id)")
    ap.add_argument("--chunk-idx", type=int, required=True, help="chunk index within source video (0-based)")
    ap.add_argument("--src-frame-start", type=int, required=True, help="first source frame this chunk covers")
    ap.add_argument("--n-frames", type=int, required=True)
    ap.add_argument("--fps-src", type=float, required=True)
    ap.add_argument("--img-w", type=int, required=True)
    ap.add_argument("--img-h", type=int, required=True)
    ap.add_argument("--hamer-pkl", required=True)
    ap.add_argument("--image-dir", required=True, help="dir with jpg frames for this chunk (HaMeR pkl uses these as keys)")
    ap.add_argument("--dyn-post-npz", required=True, help="post-SLERP NPZ for this chunk")
    ap.add_argument("--dyn-raw-npz", default=None, help="pre-SLERP NPZ for this chunk (for outlier mask)")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pair-global-idx", type=int, required=True, help="global pair index for shard placement")
    ap.add_argument("--shard-size", type=int, default=1000)
    args = ap.parse_args()
    write_pair(args)


if __name__ == "__main__":
    main()
