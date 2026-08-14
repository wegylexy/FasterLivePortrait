# -*- coding: utf-8 -*-
"""
Dedicated CI/production entrypoint matching the non-Faster fork's own
inference.py CLI contract exactly: same flags, single already-compressed
output file, no crop/debug side-by-side variant. See CLAUDE.md "Actual use
case" for the compatibility rationale.

Use run.py instead for local development/debugging - it keeps the
side-by-side crop comparison output, supports the animal model, still
images, pickled driving templates, realtime webcam mode, etc. This script
is intentionally narrower: human, video-source, TensorRT-only, one output.

python inference.py -s <portrait> -d <driving> -o <dir> \
    --flag-normalize-lip --animation-region lip --driving-multiplier 1.2
"""
import os
import argparse
import subprocess
import time
import platform
import cv2
import numpy as np
from omegaconf import OmegaConf

from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio
from src.utils.nvdec_capture import open_video_capture
from run import apply_cli_overrides, resolve_save_dir

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CFG = os.environ.get("FLP_DEFAULT_CFG", "configs/trt_infer_lip_ci.yaml")
if not os.path.isabs(DEFAULT_CFG):
    # Resolve relative to this script's location, not the process cwd - CI
    # runners (e.g. GitLab CI using this image as a job `image:`) override
    # the container's cwd to their own checkout path instead of honoring
    # the image's WORKDIR, so a plain relative path resolves against the
    # wrong directory.
    DEFAULT_CFG = os.path.join(REPO_ROOT, DEFAULT_CFG)


def main(args):
    infer_cfg = OmegaConf.load(DEFAULT_CFG)
    infer_cfg.infer_params.flag_pasteback = True
    apply_cli_overrides(infer_cfg, args)

    print("loading pipeline / TensorRT engines...", flush=True)
    t0 = time.time()
    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"preparing source: {args.src_image}", flush=True)
    t0 = time.time()
    ret = pipe.prepare_source(args.src_image, realtime=False)
    if not ret:
        print(f"no face in {args.src_image}! exit!", flush=True)
        exit(1)
    print(f"source ready in {time.time() - t0:.1f}s", flush=True)

    use_nvdec = getattr(infer_cfg.infer_params, "flag_use_nvdec", False)
    vcap = open_video_capture(args.dri_video, use_nvdec=use_nvdec)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = resolve_save_dir(args)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vsave_path = os.path.join(save_dir,
                              f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}.mp4")
    vout = cv2.VideoWriter(vsave_path, fourcc, fps, (w, h))

    total_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    infer_times = []
    frame_ind = 0
    start_t = time.time()
    last_log_t = start_t
    # CI log collectors (e.g. GitLab) treat a long silent stdout gap as a
    # stalled job and kill it, and Python fully block-buffers stdout when
    # it isn't a tty - print+flush a heartbeat every 30s so progress is
    # visible regardless of per-frame speed or clip length, without
    # spamming the log on long runs (e.g. ~60 lines for a 30-minute job).
    HEARTBEAT_SECS = 30
    while vcap.isOpened():
        ret, frame = vcap.read()
        if not ret:
            break
        t0 = time.time()
        first_frame = frame_ind == 0
        src_idx = frame_ind if pipe.is_source_video else 0
        _, _, out_org, _ = pipe.run(frame, pipe.src_imgs[src_idx], pipe.src_infos[src_idx],
                                    first_frame=first_frame)
        frame_ind += 1
        if out_org is None:
            print(f"no face in driving frame:{frame_ind}", flush=True)
            continue
        infer_times.append(time.time() - t0)
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        vout.write(out_org)
        now = time.time()
        if now - last_log_t >= HEARTBEAT_SECS:
            avg_ms = (now - start_t) / frame_ind * 1000
            print(f"progress: frame {frame_ind}/{total_frames or '?'} "
                  f"({now - start_t:.1f}s elapsed, {avg_ms:.0f} ms/frame avg)", flush=True)
            last_log_t = now
    vcap.release()
    vout.release()
    print(f"render done: {frame_ind} frames in {time.time() - start_t:.1f}s", flush=True)

    # -b:v 1M, -c:a copy: matches the non-Faster fork's own CLI output
    # contract exactly (fixed bitrate, audio copied not re-encoded) - the
    # downstream pipeline needs no separate re-compression step.
    # -rc cbr must be explicit here: h264_nvenc with -b:v but no explicit
    # -rc mode hangs indefinitely instead of erroring on this ffmpeg/driver
    # combo (same underlying issue as the -cq deadlock noted in CLAUDE.md /
    # NVENC_ARGS elsewhere - any -b:v/-maxrate without an explicit -rc mode
    # is unsafe here, not just the -cq combination).
    vsave_final = os.path.splitext(vsave_path)[0] + "-final.mp4"
    mux_args = [FFMPEG, "-i", vsave_path]
    if video_has_audio(args.dri_video):
        mux_args += ["-i", args.dri_video, "-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
    mux_args += ["-c:v", "h264_nvenc", "-rc", "cbr", "-b:v", "1M", vsave_final, "-y"]
    print(f"muxing: {' '.join(mux_args)}", flush=True)
    mux_t0 = time.time()
    ret = subprocess.call(mux_args)
    if ret != 0:
        raise RuntimeError(f"ffmpeg mux failed with exit code {ret}: {' '.join(mux_args)}")
    print(f"mux done in {time.time() - mux_t0:.1f}s", flush=True)
    os.remove(vsave_path)
    print(vsave_final, flush=True)
    print("inference median time: {} ms/frame, mean time: {} ms/frame".format(
        np.median(infer_times) * 1000 if infer_times else 0,
        np.mean(infer_times) * 1000 if infer_times else 0), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FasterLivePortrait CI-compatible inference (single-file output)')
    parser.add_argument('-s', '--src_image', required=True, type=str, help='source image or video')
    parser.add_argument('-d', '--dri_video', required=True, type=str, help='driving video')
    parser.add_argument('-o', '--output_dir', required=False, type=str, default=None, help='output directory')
    parser.add_argument('--flag-normalize-lip', dest='flag_normalize_lip', action='store_true', default=None,
                        help='normalize lip (overrides --cfg)')
    parser.add_argument('--animation-region', dest='animation_region', required=False, type=str, default=None,
                        choices=['exp', 'pose', 'lip', 'eyes', 'all'], help='animation region (overrides --cfg)')
    parser.add_argument('--driving-multiplier', dest='driving_multiplier', required=False, type=float, default=None,
                        help='driving multiplier (overrides --cfg)')
    parser.add_argument('--video-chunk-size', required=False, type=int, default=None,
                        help='no-op here - this fork streams frames instead of chunking, kept only for '
                             'drop-in CLI compatibility with the non-Faster fork')
    args, unknown = parser.parse_known_args()
    # Resolve user-supplied paths against the invoking cwd *before* chdir'ing
    # below - CI runners (e.g. GitLab CI using this image as a job `image:`)
    # override the container's cwd to their own checkout path instead of
    # honoring the image's WORKDIR, and every relative path baked into the
    # shipped config (checkpoint .trt files, mask_crop_path, DEFAULT_CFG
    # itself) assumes cwd == repo root. Making src/dri/output absolute here
    # first lets us safely chdir to REPO_ROOT for everything else below.
    args.src_image = os.path.abspath(args.src_image)
    args.dri_video = os.path.abspath(args.dri_video)
    if args.output_dir:
        args.output_dir = os.path.abspath(args.output_dir)
    os.chdir(REPO_ROOT)
    main(args)
