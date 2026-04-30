"""
Realign a chunk-merged DynHAMR NPZ to source-video frame indices.

Why this is needed:
  The chunked pipeline calls ffmpeg with `-ss <n*20s> -frames:v 600` on a
  29.97 fps source. Each chunk slot in the NPZ holds 601 frames (or 156 for
  the last) but spans only 20s × 29.97 ≈ 599.4 source frames. So consecutive
  chunks in the NPZ overlap with the prior chunk by ~1.6 frames in source-time.
  Concatenated, the NPZ has more frames than the source (by ~chunk_count × 0.6).

This script reverses the mapping: for each NPZ chunk it samples back to the
exact source-frame indices it covers, producing an NPZ with exactly N_src
frames in the same order as the source video.

Usage:
    python realign_npz_to_source.py \
        --in  test.npz \
        --out test_aligned.npz \
        --src-frames 36719 --src-fps 29.97002997 \
        --chunk-frames 601 --chunk-stride-sec 20.0 \
        --num-chunks 62
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src-frames", type=int, required=True, help="Source video total frame count")
    ap.add_argument("--src-fps", type=float, required=True, help="Source video fps (e.g., 29.97002997 for 30000/1001)")
    ap.add_argument("--chunk-frames", type=int, default=601, help="Frames per chunk in NPZ (max)")
    ap.add_argument("--chunk-stride-sec", type=float, default=20.0, help="ffmpeg -ss stride between chunks (seconds)")
    ap.add_argument("--num-chunks", type=int, required=True)
    args = ap.parse_args()

    d = np.load(args.inp, allow_pickle=True)
    keys = list(d.files)
    arrays = {k: d[k] for k in keys}

    B, T_npz = arrays["trans"].shape[:2]
    print(f"Input NPZ: B={B} tracks, T={T_npz} frames")
    print(f"Target: {args.src_frames} source frames")
    print(f"Chunk layout: {args.num_chunks} chunks × ≤{args.chunk_frames} frames, stride {args.chunk_stride_sec}s @ {args.src_fps} fps")

    # Compute per-chunk source-frame coverage:
    #   chunk n covers source frames [round(n*stride*fps), round(n*stride*fps) + chunk_len_in_npz)
    # In NPZ, chunk n occupies indices [n*chunk_frames, n*chunk_frames + chunk_len_in_npz).
    # We pick exactly one NPZ frame per source frame index covered (de-duplicating overlaps with chunk n-1).
    src_indices_picked = []  # list of (src_frame_idx, npz_frame_idx)
    last_src = -1
    for n in range(args.num_chunks):
        npz_chunk_start = n * args.chunk_frames
        if n == args.num_chunks - 1:
            npz_chunk_len = T_npz - npz_chunk_start
        else:
            npz_chunk_len = args.chunk_frames

        src_chunk_start = int(round(n * args.chunk_stride_sec * args.src_fps))
        # Pair NPZ frame i (within chunk) → source frame src_chunk_start + i
        for i in range(npz_chunk_len):
            src_idx = src_chunk_start + i
            npz_idx = npz_chunk_start + i
            if src_idx >= args.src_frames:
                break  # chunk extends past end of source (shouldn't happen often)
            if src_idx <= last_src:
                # Overlap with previous chunk → skip this NPZ frame, prefer the earlier chunk's frame
                continue
            src_indices_picked.append((src_idx, npz_idx))
            last_src = src_idx

    print(f"Picked {len(src_indices_picked)} (src, npz) pairs (target {args.src_frames})")
    if len(src_indices_picked) != args.src_frames:
        print(f"  WARN: count mismatch — coverage gap or overrun")

    # Sanity: src_indices_picked should be sorted and cover all source frames once
    src_arr = np.array([s for s, _ in src_indices_picked], dtype=np.int64)
    npz_arr = np.array([n for _, n in src_indices_picked], dtype=np.int64)
    assert np.all(np.diff(src_arr) >= 1), "src indices not strictly increasing"
    print(f"  src range: [{src_arr.min()}, {src_arr.max()}]  npz range: [{npz_arr.min()}, {npz_arr.max()}]")

    # If there are missing source frames, pad by repeating the last available NPZ frame
    if len(src_arr) < args.src_frames:
        missing = set(range(args.src_frames)) - set(src_arr.tolist())
        print(f"  Filling {len(missing)} missing source frames by nearest neighbor")
        full_src = np.arange(args.src_frames)
        full_npz = np.empty(args.src_frames, dtype=np.int64)
        # For each src frame, find nearest mapped src index
        sorted_pairs = sorted(zip(src_arr.tolist(), npz_arr.tolist()))
        ss = np.array([p[0] for p in sorted_pairs])
        nn = np.array([p[1] for p in sorted_pairs])
        for i in range(args.src_frames):
            j = np.searchsorted(ss, i)
            if j >= len(ss):
                j = len(ss) - 1
            elif j > 0 and abs(ss[j-1] - i) <= abs(ss[j] - i):
                j -= 1
            full_npz[i] = nn[j]
        npz_arr = full_npz
        src_arr = full_src

    # Re-index temporal arrays
    out = {}
    for k, a in arrays.items():
        if a.ndim >= 2 and a.shape[1] == T_npz:
            out[k] = a[:, npz_arr]
            print(f"  {k}: {a.shape} → {out[k].shape}")
        else:
            out[k] = a
            print(f"  {k}: {a.shape} (kept as-is)")

    np.savez(args.out, **out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
