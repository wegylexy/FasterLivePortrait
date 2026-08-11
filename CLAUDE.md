# FasterLivePortrait fork — status and notes

This is `wegylexy/FasterLivePortrait`, forked 2026-08-10 to replace the
`wegylexy/LivePortrait` fork (sibling folder `../LivePortrait`, `docker` branch)
as the base project, since FasterLivePortrait uses ONNX/TensorRT inference
instead of raw PyTorch and is significantly faster.

## Actual use case

Drive an already-recorded source video (natural head motion) with a
separately-produced Wav2Lip output video as the driving signal, using
`animation_region: "lip"` so only the mouth is retargeted and the source's own
head motion/lighting/background stay untouched. This avoids the paste-back
seam and black-corner artifacts that show up when animating a full head pose
onto a still image instead.

Known-good settings (matching the real CI's `inference.py` flags):
`--flag-normalize-lip --animation-region lip --driving-multiplier 1.2`.
See `configs/trt_infer_lip_ci.yaml`.

JoyVASA (audio-driven, no separate lip-sync step) was tried as an alternative
to the Wav2Lip+FasterLivePortrait combo — **re-tested this session after the
relative-motion bug fix below, still worse than Wav2Lip, deprioritized.**
First retest used a video source (`source_video5.mp4`) by mistake, which
defeats JoyVASA's actual point (no source video needed — audio alone drives
all motion including head movement) and also triggered a real fps-labeling
bug: `run_pickle_driving` (`gradio_live_portrait_pipeline.py`) used the
*source video's own* fps to label the rendered output whenever
`is_source_video` was true, instead of `dri_motion_infos["output_fps"]` (the
fps the driving motion was actually generated at — 25fps for JoyVASA vs. the
24fps source video), causing the render to run ~4% slower than the audio and
the lip sync to drift increasingly out of time over the clip. Fixed by always
trusting `output_fps` (confirmed via `run.py`'s pickle-export and JoyVASA's
own `gen_motion_sequence` that every driving pickle sets it correctly,
regardless of source type). Re-tested afterward both with a still image
(correct usage) and with the video source again (to confirm the fix) — fps
labeling is now correct in both cases (audio/video duration match to within
0.01s) but the lip movement is still noticeably worse/more delayed than the
Wav2Lip baseline even with correct sync, so the fps bug wasn't the whole
story — likely an intrinsic latency/quality gap in JoyVASA's own generation
(e.g. diffusion denoising lag, or the causal chunk-conditioning described
below), not something worth chasing further right now. Worth a re-test if
priorities change, but not the direction to invest in for the primary
use case.

## Fixed this session

1. **Streaming source video** (`src/pipelines/faster_live_portrait_pipeline.py`,
   `_LazySourceFrames`/`_LazySourceInfos`) — `prepare_source`'s video branch used
   to decode+preprocess every frame of the source video upfront, holding it all
   in memory. Now decodes/processes one frame at a time, cached, with automatic
   wraparound looping when the driving clip is longer than the source
   (replaces the old `max_frame = min(dframe, len(self.src_imgs))` truncation
   in `gradio_live_portrait_pipeline.py`).

2. **NVENC + ffmpeg fixes**, applied to every ffmpeg call site in `run.py`,
   `api.py`, and `gradio_live_portrait_pipeline.py` (was previously libx264-only,
   or NVENC-only in the JoyVASA audio path — now consistent everywhere via the
   module-level `NVENC_ARGS` constant in each file):
   - `-b:v`/`-maxrate` combined with `-cq` deadlocks `h264_nvenc` on this
     ffmpeg/driver combo (hangs at a fixed frame count instead of erroring) —
     use `-rc vbr -cq 19` alone.
   - The native `aac` encoder deadlocks on 16kHz mono input in this ffmpeg
     build — always resample audio to 44100 before encoding (`-ar 44100`).
   - The distro/pip-packaged ffmpeg predates this driver's NVENC API; a
     current static build (BtbN) is required — see `Dockerfile.slim`.

3. **`is_source_video` relative-motion formula bug** (the real quality gap vs.
   the non-Faster reference) — in `faster_live_portrait_pipeline.py`'s `_run()`,
   the video-source path for `exp`/`lip`/`eyes` regions was substituting the
   driving frame's raw/smoothed *absolute* expression value directly
   (`delta_new[:, lip_idx, :] = x_d_exp_smooth[...]`), discarding the source's
   own baseline expression and never subtracting the driving clip's own
   frame-0 baseline. Confirmed against the original PyTorch LivePortrait's
   actual code (sibling `../LivePortrait` repo) that the correct formula is a
   relative delta: `source's own baseline + (driving's current frame minus
   driving's own frame-0 baseline)`. Same bug existed in `R_new` (rotation) for
   the `pose`/`all` region — fixed the same way. **Confirmed present in current
   upstream FasterLivePortrait itself** (checked `master`/`dev`/`anim`/`win`
   branches via a temporary `upstream` remote to `warmshao/FasterLivePortrait`
   — no branch fixes this), not something introduced locally.
   - Before the fix: frame 0 had a visible mouth gap instead of fully closed,
     and mouth-open amplitude undershot the reference badly at fast transients
     (e.g. barely parted vs. clearly open with teeth at the same timestamp).
   - After the fix: frame 0 matches the reference exactly; amplitude is much
     closer (not byte-identical — the reference's Kalman *smoother* sees the
     whole sequence offline; this pipeline's `OneEuroFilter` is causal/online,
     see "Not yet done" below).

4. **Slim Docker image** (`Dockerfile.slim`) — replaces the 39GB
   `shaoguo/faster_liveportrait:v3` base (no longer needed: JoyVASA's heavy
   deps are gone from the critical path, and the custom `grid_sample` TensorRT
   plugin ships precompiled in the downloaded checkpoints, so it doesn't need
   compiling from source). Built from `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
   + pip-installed `tensorrt==8.6.1` (same ABI as the already-built `.trt`
   engines' `8.6.1.6`, confirmed to load them without regenerating) + a
   BtbN static ffmpeg build (NVENC *and* NVDEC — `h264_cuvid` confirmed
   available). ~31.4GB vs. the original ~39GB. Two real gotchas hit while
   building it, both fixed by doing the fix *in the same RUN layer* as the
   problem (a later separate layer doesn't reclaim the earlier layer's bytes,
   so fixing forward in a new layer wastes disk space for no benefit):
   - `mediapipe` (and others) pull in plain CPU-only `onnxruntime` as a
     transitive dep, which shares `onnxruntime-gpu`'s import namespace and
     silently wins if it installs after — `pip uninstall -y onnxruntime &&
     pip install onnxruntime-gpu==1.17.0` at the end of the same layer.
   - Something in that same resolution pulls `numpy==2.x`, breaking the
     compiled-extension ABI every pinned-1.x wheel (onnxruntime, opencv,
     pycuda, ...) was built against — pin `numpy==1.26.4` again, also at the
     very end of the same layer.
   - Also needed: `torchaudio` (imported unconditionally by
     `gradio_live_portrait_pipeline.py` even though unused on this path), and
     `transformers<4.50` + `huggingface_hub[cli]==0.23.5` pinned (newer
     `transformers` needs `torch>=2.5`, we're pinned to `2.1.2` for `cu118`;
     newer `huggingface_hub` 5.x drops `HfFolder`, which `gradio` still
     imports) — same version-pinning gotchas hit earlier with `Dockerfile.nvenc`.
   - Rejected `nvcr.io/nvidia/tensorrt:26.07-py3` and similar recent NGC tags:
     confirmed NGC's TensorRT container versioning jumped from 8.6.3 to
     10.0.1.6 at tag `24.05` — anything from around there onward ships
     TensorRT ≥10.x, which is incompatible with the precompiled `grid_sample`
     plugin (README explicitly warns TensorRT ≥10.x doesn't work with it).
     `nvcr.io/nvidia/tensorrt:24.01-py3` (TensorRT `8.6.1.6` exactly, ~4GB) is
     a plausible smaller alternative base worth trying later, but would need
     re-verifying `torch`/`onnxruntime-gpu` wheel compatibility against its
     CUDA 12.3 instead of the current 11.8.

`Dockerfile.nvenc` (the earlier, `shaoguo`-based attempt) is kept for
reference/fallback but superseded by `Dockerfile.slim`.

## Not yet done

- **NVDEC hardware decode — tried, measured worse, disabled by default.**
  Implemented `src/utils/nvdec_capture.py` (`NvdecVideoCapture` /
  `open_video_capture()`), a drop-in `cv2.VideoCapture`-shaped wrapper around
  `ffmpeg -hwaccel cuda -c:v h264_cuvid -i ... -f rawvideo -pix_fmt bgr24
  -fps_mode passthrough -` (note: `-vsync` is removed in this ffmpeg build,
  must use `-fps_mode passthrough` instead), wired into all four
  `cv2.VideoCapture` call sites (`_LazySourceFrames`, `run_video_driving` in
  both `gradio_live_portrait_pipeline.py` and `api.py`, `run.py`) behind a
  `infer_params.flag_use_nvdec` config flag. **Measured 4x slower than CPU
  decode** on the actual production clip (`source_video5.mp4` +
  `driving_wav2lip_t0.mp4`, 240 frames): 211.85s with NVDEC vs. 48.87s with
  `cv2.VideoCapture`. Root cause not fully isolated but likely per-frame
  subprocess-pipe overhead (spawning ffmpeg, piping raw BGR frames back
  through a Python `subprocess.PIPE` read) plus GPU decoder/inference
  contention on the same card, given decode was never actually the
  bottleneck here (see the read-ahead note below — model inference already
  dominates). `flag_use_nvdec` now **defaults to `False`** everywhere; the
  code path is kept (functionally correct, confirmed via manual `ffmpeg`
  invocation and a full pipeline run) as an opt-in in case a different
  workload (e.g. much larger source videos where decode share of total time
  is higher) benefits, but don't expect it to help on typical clip lengths
  without further profiling of where the 4x actually goes.
- **Background-thread read-ahead** for video decode (overlap decode with GPU
  model inference) — the old LP fork's actual design for chunked reading.
  Given the NVDEC finding above (decode isn't the bottleneck; model
  inference dominates), this is unlikely to buy much either — deprioritized.
- **Offline (Kalman) smoothing** instead of causal `OneEuroFilter` for
  `is_source_video` — the original does a full-sequence, two-pass (extract all
  driving motion coefficients, `pykalman.KalmanFilter.smooth()` once on the
  whole sequence, then render) instead of smoothing frame-by-frame in real
  time. This is the remaining, smaller gap after the formula-bug fix above
  (a causal filter inherently lags/blunts fast transients; a smoother doesn't
  need to, since it has "future" context via its backward pass). Chunking (à
  la the old fork's `read_video_chunks`/`--video-chunk-size`) should bound
  memory for the *decode/extraction* pass only — never chunk the smoothing
  itself, confirmed from the original's code that it always smooths the full
  concatenated sequence in one call, regardless of how many decode chunks fed
  into it (chunking the smoother would reintroduce the same kind of seam
  artifact found in JoyVASA's fixed-window chunking, see below).
- `scale`/`t` (translation) for `is_source_video` in `animation_region: "all"`
  mode currently just holds the source's own value unconditionally
  (`faster_live_portrait_pipeline.py`, `_run()`) — not verified whether this
  matches the original's intent or is a similar bug; doesn't affect the
  `lip`-only path actually used.

## JoyVASA notes (secondary path, not the main use case)

If revisiting JoyVASA: it generates motion in fixed, non-overlapping 100-frame
(4s) chunks (`n_motions=100` baked into the checkpoint's training args,
`src/pipelines/joyvasa_audio_to_motion_pipeline.py`), conditioning each chunk
on the previous chunk's last 10 frames via attention (soft nudge, not a hard
continuity constraint) — this can produce a visible pose jump right at each
4-second boundary. Mitigated (not root-caused) in
`_smooth_chunk_seams()` in that same file: cancels the jump at each seam and
fades the correction back out over 15 frames rather than carrying a permanent
offset. `configs/trt_infer_lip.yaml`/`trt_infer_lip_ci.yaml` set
`animation_region: "lip"` for this path too, for the same paste-back-seam
reasons as the primary Wav2Lip use case.

## Environment gotchas

- **Docker Desktop / WSL2 disk**: the WSL2 virtual disk (`.vhdx`) is
  thin-provisioned and grows automatically but does **not** shrink back when
  Docker reports images/cache freed internally — `docker system df` numbers
  freed don't reflect on the actual Windows host disk until the vhdx is
  compacted separately. Ran the host disk down to single-digit GB free twice
  this session between large builds/pulls; check `df -h /` before any
  build/pull, not just `docker system df`.
- **NVENC needs `NVIDIA_DRIVER_CAPABILITIES` to include `video`** — `--gpus
  all` alone only grants `compute,utility`. Without it, `h264_nvenc` fails to
  load `libnvidia-encode.so.1`.
- **NVDEC (`h264_cuvid` etc.) confirmed available** in the BtbN ffmpeg build
  used here — not yet wired into the actual decode path (see above).
- A bleeding-edge host driver (610.74 at time of writing) means the
  *distro-packaged* ffmpeg (built years ago against an old NVENC API) fails
  with `Cannot get the preset configuration: unsupported param` — a current
  static ffmpeg build is required, not just any ffmpeg with `--enable-nvenc`
  in its compile flags.
- Docker on Windows/git-bash mangles absolute container-side paths in `-v`
  flags (MSYS path conversion rewrites e.g. `/root/...` as a Windows path) —
  prefix with `MSYS_NO_PATHCONV=1`.
- Don't bind-mount a single file at a path nested *inside* another
  already-mounted directory on the container side (e.g. mounting one file to
  `/root/app/tmp/audio.wav` while `/root/app/tmp` itself is also mounted) —
  Docker Desktop silently created an empty placeholder instead of the real
  file when this was tried. Mount to a separate, non-nested container path
  instead.
