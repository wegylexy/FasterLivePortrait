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

DEFAULT_CFG = os.environ.get("FLP_DEFAULT_CFG", "configs/trt_infer_lip_ci.yaml")


def main(args):
    infer_cfg = OmegaConf.load(DEFAULT_CFG)
    infer_cfg.infer_params.flag_pasteback = True
    apply_cli_overrides(infer_cfg, args)

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=False)
    ret = pipe.prepare_source(args.src_image, realtime=False)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)

    use_nvdec = getattr(infer_cfg.infer_params, "flag_use_nvdec", False)
    vcap = open_video_capture(args.dri_video, use_nvdec=use_nvdec)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = resolve_save_dir(args)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vsave_path = os.path.join(save_dir,
                              f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}.mp4")
    vout = cv2.VideoWriter(vsave_path, fourcc, fps, (w, h))

    infer_times = []
    frame_ind = 0
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
            print(f"no face in driving frame:{frame_ind}")
            continue
        infer_times.append(time.time() - t0)
        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
        vout.write(out_org)
    vcap.release()
    vout.release()

    # -b:v 1M, -c:a copy: matches the non-Faster fork's own CLI output
    # contract exactly (fixed bitrate, audio copied not re-encoded) - the
    # downstream pipeline needs no separate re-compression step.
    vsave_final = os.path.splitext(vsave_path)[0] + "-final.mp4"
    mux_args = [FFMPEG, "-i", vsave_path]
    if video_has_audio(args.dri_video):
        mux_args += ["-i", args.dri_video, "-map", "0:v", "-map", "1:a", "-c:a", "copy", "-shortest"]
    mux_args += ["-b:v", "1M", "-c:v", "h264_nvenc", vsave_final, "-y"]
    subprocess.call(mux_args)
    os.remove(vsave_path)
    print(vsave_final)
    print("inference median time: {} ms/frame, mean time: {} ms/frame".format(
        np.median(infer_times) * 1000 if infer_times else 0,
        np.mean(infer_times) * 1000 if infer_times else 0))


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
    main(args)
