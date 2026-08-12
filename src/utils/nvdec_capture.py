# -*- coding: utf-8 -*-
"""
Drop-in replacement for cv2.VideoCapture that decodes via ffmpeg's NVDEC
(h264_cuvid/hevc_cuvid) hardware decoders instead of CPU. Implements just the
subset of the cv2.VideoCapture interface this codebase actually uses
(.read(), .set(CAP_PROP_POS_FRAMES, ...), .get(...), .release()) so it can be
substituted wherever cv2.VideoCapture is used for driving/source video decode.

Falls back to raising RuntimeError on construction if ffmpeg/ffprobe can't be
found or the video's codec has no cuvid decoder - callers should catch this
and fall back to cv2.VideoCapture.
"""
import shutil
import subprocess

import cv2
import ffmpeg
import numpy as np

_CODEC_TO_CUVID = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vc1": "vc1_cuvid",
    "av1": "av1_cuvid",
}


class NvdecVideoCapture:
    def __init__(self, path):
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg/ffprobe not found on PATH")

        self._path = path
        probe = ffmpeg.probe(path)
        video_streams = [s for s in probe["streams"] if s["codec_type"] == "video"]
        if not video_streams:
            raise RuntimeError(f"no video stream found in {path}")
        stream = video_streams[0]

        codec_name = stream["codec_name"]
        if codec_name not in _CODEC_TO_CUVID:
            raise RuntimeError(f"no cuvid decoder mapping for codec {codec_name}")
        self._cuvid_decoder = _CODEC_TO_CUVID[codec_name]

        self._width = int(stream["width"])
        self._height = int(stream["height"])
        num, den = stream["r_frame_rate"].split("/")
        self._fps = float(num) / float(den) if float(den) != 0 else 0.0
        if "nb_frames" in stream and stream["nb_frames"] not in (None, "N/A"):
            self._frame_count = int(stream["nb_frames"])
        else:
            duration = float(probe["format"].get("duration", 0.0))
            self._frame_count = int(round(duration * self._fps)) if self._fps > 0 else 0

        self._frame_bytes = self._width * self._height * 3
        self._proc = None
        self._read_cursor = 0
        self._start_proc(0)

    def _start_proc(self, start_frame):
        if self._proc is not None:
            self._proc.stdout.close()
            self._proc.kill()
            self._proc.wait()
        start_sec = start_frame / self._fps if self._fps > 0 else 0.0
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-hwaccel", "cuda", "-c:v", self._cuvid_decoder]
        if start_sec > 0:
            cmd += ["-ss", f"{start_sec:.6f}"]
        cmd += ["-i", self._path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-fps_mode", "passthrough", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._read_cursor = start_frame

    def read(self):
        if self._proc is None:
            return False, None
        raw = self._proc.stdout.read(self._frame_bytes)
        if len(raw) < self._frame_bytes:
            stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            if stderr.strip():
                print(f"NVDEC ffmpeg process ended early: {stderr.strip()}")
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self._height, self._width, 3))
        self._read_cursor += 1
        return True, frame

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._start_proc(int(value))
            return True
        return False

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self._read_cursor)
        return 0.0

    def release(self):
        if self._proc is not None:
            self._proc.stdout.close()
            self._proc.kill()
            self._proc.wait()
            self._proc = None

    def isOpened(self):
        return self._proc is not None


def open_video_capture(path, use_nvdec=True):
    """Try NVDEC first, fall back to cv2.VideoCapture on any failure."""
    if use_nvdec:
        try:
            return NvdecVideoCapture(path)
        except Exception as e:
            print(f"NVDEC decode unavailable for {path} ({e}), falling back to CPU decode")
    return cv2.VideoCapture(path)
