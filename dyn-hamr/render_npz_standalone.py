"""
Render a merged (post-processed) DynHAMR NPZ + source-video frames to mp4
without going through the chunked dataset/.hydra config plumbing.

Usage:
    python render_npz_standalone.py \
        --npz /path/to/test.npz \
        --frames-dir /path/to/_frames \
        --out /path/to/test_mesh.mp4 \
        [--fps 30] [--start 0] [--end -1]
"""
import argparse
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from body_model import MANO
from util.tensor import get_device, move_to
from vis.output import prep_result_vis, animate_scene
from vis.viewer import init_viewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--frames-dir", required=True, help="Dir with %06d.jpg starting at 1")
    ap.add_argument("--out", required=True, help="Output mp4 path")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--start", type=int, default=0, help="Start frame index (NPZ space)")
    ap.add_argument("--end", type=int, default=-1, help="End frame index (exclusive); -1 = use all bg frames")
    ap.add_argument("--mano-dir", default=os.path.expanduser("~/Zekai_code/Dyn-HaMR/_DATA/data"))
    ap.add_argument("--temporal-smooth", action="store_true", default=False,
                    help="Apply temporal smoothing inside prep_result_vis. Off → render the NPZ values exactly.")
    ap.add_argument("--chunk-size", type=int, default=600,
                    help="Render in chunks of this many frames; concat at the end. 0 = single pass.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    B, T_npz = d["trans"].shape[:2]
    print(f"NPZ: B={B} tracks, T={T_npz} frames")

    # Match NPZ length to available bg frames
    bg_frames = sorted([
        os.path.join(args.frames_dir, f)
        for f in os.listdir(args.frames_dir)
        if f.endswith(".jpg")
    ])
    n_bg = len(bg_frames)
    print(f"Background frames available: {n_bg}")

    end = args.end if args.end > 0 else min(T_npz, n_bg)
    start = max(0, args.start)
    end = min(end, T_npz, n_bg)
    T = end - start
    assert T > 0, f"empty range: start={start} end={end}"
    print(f"Rendering frames [{start}, {end}) → T={T}")

    intrins = d["intrins"]  # (4,) [fx, fy, cx, cy]
    fx, fy, cx, cy = [float(abs(x)) for x in intrins]
    img_W = int(round(2 * cx))
    img_H = int(round(2 * cy))
    img_size = (img_W, img_H)
    print(f"img_size={img_size}, intrins=({fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f})")

    mano_cfg = {
        "model_path":      os.path.join(args.mano_dir, "mano"),
        "gender":          "neutral",
        "num_hand_joints": 15,
        "mean_params":     os.path.join(args.mano_dir, "mano_mean_params.npz"),
        "create_body_pose": False,
    }
    intrins_for_viewer = torch.tensor([fx, fy, cx, cy])

    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    base_no_ext = os.path.splitext(os.path.basename(args.out))[0]

    chunk_size = args.chunk_size if args.chunk_size > 0 else T
    chunks = []
    for cs in range(start, end, chunk_size):
        ce = min(cs + chunk_size, end)
        chunks.append((cs, ce))
    print(f"Will render {len(chunks)} chunk(s) of up to {chunk_size} frames each")

    chunk_mp4s = []

    for ci, (cs, ce) in enumerate(chunks):
        Tc = ce - cs
        print(f"\n=== Chunk {ci+1}/{len(chunks)}: frames [{cs},{ce}) → T={Tc} ===")
        bg_paths = bg_frames[cs:ce]

        def sl(a):
            return a[:, cs:ce] if a.ndim >= 2 and a.shape[1] == T_npz else a

        res = {
            "trans":       torch.from_numpy(sl(d["trans"]).astype(np.float32)),
            "root_orient": torch.from_numpy(sl(d["root_orient"]).astype(np.float32)),
            "pose_body":   torch.from_numpy(sl(d["pose_body"]).astype(np.float32)),
            "is_right":    torch.from_numpy(sl(d["is_right"]).astype(np.float32)),
            "betas":       torch.from_numpy(d["betas"].astype(np.float32)),
            "cam_R":       torch.from_numpy(sl(d["cam_R"]).astype(np.float32)),
            "cam_t":       torch.from_numpy(sl(d["cam_t"]).astype(np.float32)),
        }
        vis_mask = torch.ones(B, Tc, dtype=torch.float32)
        track_id = torch.arange(B, dtype=torch.long)

        device = get_device(0)
        res = move_to(res, device)
        vis_mask = vis_mask.to(device)
        track_id = track_id.to(device)

        hand_model = MANO(batch_size=B*Tc, pose2rot=True, **mano_cfg).to(device)
        scene_dict = prep_result_vis(
            res, vis_mask, track_id, hand_model,
            temporal_smooth=args.temporal_smooth, smooth_trans=False,
        )

        vis = init_viewer(
            img_size, intrins_for_viewer,
            vis_scale=1.0, bg_paths=bg_paths, fps=args.fps,
        )

        chunk_stem = os.path.join(out_dir, f"{base_no_ext}_chunk{ci:03d}")
        animate_scene(
            vis, scene_dict, chunk_stem,
            seq_name=f"{base_no_ext}_chunk{ci:03d}",
            render_views=["src_cam"],
            render_bg=True,
            render_cam=False,
            render_ground=False,
        )
        vis.close()
        del vis, hand_model, scene_dict, res
        import gc; gc.collect()
        torch.cuda.empty_cache()

        chunk_mp4 = f"{chunk_stem}_src_cam.mp4"
        if not os.path.isfile(chunk_mp4):
            print(f"  WARN: chunk mp4 missing: {chunk_mp4}")
            continue
        chunk_mp4s.append(chunk_mp4)

    if not chunk_mp4s:
        print("ERROR: no chunk mp4s rendered.")
        sys.exit(1)

    if len(chunk_mp4s) == 1:
        os.replace(chunk_mp4s[0], args.out)
        print(f"Single chunk → {args.out}")
        return

    concat_list = os.path.join(out_dir, f"{base_no_ext}_concat.txt")
    with open(concat_list, "w") as f:
        for m in chunk_mp4s:
            f.write(f"file '{os.path.abspath(m)}'\n")
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_list, "-c", "copy", args.out,
    ]
    print(f"Concatenating {len(chunk_mp4s)} chunks → {args.out}")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    for m in chunk_mp4s:
        os.remove(m)
    os.remove(concat_list)
    print(f"DONE → {args.out}")


if __name__ == "__main__":
    main()
