# FasterLivePortrait fork — status and notes

This is `wegylexy/FasterLivePortrait`, forked 2026-08-10 to replace the
`wegylexy/LivePortrait` fork (sibling folder `../LivePortrait`, `docker` branch)
as the base project, since FasterLivePortrait uses ONNX/TensorRT inference
instead of raw PyTorch and is significantly faster.

**Measured this session** (same source/driving clip, same
`--flag-normalize-lip --animation-region lip --driving-multiplier 1.2`, 240
frames, one GPU, CPU-decode in both cases): FasterLivePortrait (this fork)
finished in **48.9–55.3s** vs. **306.1s** for the non-Faster PyTorch fork —
about **5.5–6.3x faster** — while also using less RAM (~1.2–1.5 GiB vs.
~3.4–3.5 GiB) and less VRAM (~2.3 GB vs. ~7.8 GB). The one place it costs
more is CPU core usage during the render (~7.7–8.3 cores sustained here vs.
~1.4 for the non-Faster fork), which is expected — TensorRT's own
pre/post-processing threads — but for a fraction of the wall-clock time.

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

**Added `inference.py`, a dedicated CI/production entrypoint matching the
non-Faster fork's `inference.py` CLI *and output* contract exactly** —
`run.py` was made CLI-compatible first (`-s`/`-d`/`-o` short flags,
`--flag-normalize-lip`/`--animation-region`/`--driving-multiplier` as
`--cfg` overrides, `--video-chunk-size` accepted as a documented no-op since
this fork streams instead of chunking), but its *output* still differs
fundamentally: `run.py` always writes `crop.mp4`/`org.mp4` plus `-audio`
variants plus a `.pkl` motion template (4-5 files) for its own local-dev/
debug purposes, whereas the non-Faster fork's CI always gets exactly one
compressed file. Giving `run.py` the name `inference.py` (e.g. via a
symlink) would have implied a byte-for-byte drop-in swap that wasn't
actually true and the CI script would still have needed fixing (its
`mv temp/*.mp4 "$a"` breaks with multiple matches) — so `inference.py` is a
separate, narrower script instead: human/video-source/TensorRT-only, single
output file, `-b:v 1M -c:a copy` baked in (matching the non-Faster fork's
own downstream compress step exactly, verified their driving video's audio
is already AAC/48kHz/mono by the time it reaches this stage — a prior
pipeline stage re-encodes it to that format, making `-c:a copy` safe). CI
migration is a one-line `image:` swap plus the script name (`inference.py`
stays the same) and one added `--cfg` flag pointing at
`configs/trt_infer_lip_ci.yaml` (or rely on this image's own
`FLP_DEFAULT_CFG` env default and omit `--cfg` entirely) — **this claim was
initially wrong** (see "Fixed this session" #7 below: checkpoints weren't
actually baked into the image until that point, so the "one-line swap" only
became true after that fix landed). `run.py` remains the local-dev/debug
entrypoint (side-by-side crop comparison, animal model, still images,
pickled driving templates, realtime webcam mode) — use that, not
`inference.py`, for anything other than the CI path.

Two real, pre-existing bugs surfaced while first testing `run.py`/`api.py`'s
`run_with_video`/`run_with_pkl` directly this session — this was the first
time either was actually exercised end-to-end all session (everything
before used `GradioLivePortraitPipeline` via ad hoc scripts instead), so
neither bug was ever a regression from that session's other work:
- Both always indexed `pipe.src_imgs[0]`/`pipe.src_infos[0]` regardless of
  driving frame index, so a video source's own head motion never advanced
  past frame 0 — fixed to index by `frame_ind` when `pipe.is_source_video`.
- `run.py` unconditionally did `infer_cfg.infer_params.flag_pasteback =
  args.paste_back`, and `--paste_back` defaults to `False` — so any
  invocation without `--paste_back` silently disabled paste-back regardless
  of the `--cfg` file's own `flag_pasteback: True` setting, making the
  entire output just the raw, unmodified source video. Fixed to only
  override when `--paste_back` is explicitly passed (same pattern as the
  other CLI overrides).

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
   - Also rejected `nvcr.io/nvidia/tensorrt:24.01-py3` (TensorRT `8.6.1.6`
     exactly): its cuDNN is `8.9.7`, but `onnxruntime-gpu`'s CUDA 12 wheels
     require cuDNN 9 — no CUDA-12-with-cuDNN-8 combo exists for it. At 12GB
     uncompressed it also isn't smaller than what we needed anyway once our
     deps go on top.

5. **Slim image shrunk further, 31.4GB → 23.7GB, via a real multi-stage
   build** (still `Dockerfile.slim` — this replaced the single-stage version
   from point 4 above). `pycuda`'s C-extension is the only thing that
   actually needs `nvcc`, so it's compiled in an `nvidia/cuda:...-devel`
   *builder* stage and just the built wheel is copied into a final
   `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04` stage — dropping the full
   CUDA devel toolkit (headers, static libs, nvcc itself) from the shipped
   image. Also bumped `torch`/`torchvision`/`torchaudio` to `2.7.1`/`0.22.1`/
   `2.7.1` (the newest versions that still publish a `cu118` wheel —
   `download.pytorch.org/whl/cu118` goes up to `2.7.1`), which also lifted
   the old `transformers<4.50` pin (only needed because torch was `<2.5`).
   Verified pixel-identical output and matching speed/RAM/VRAM vs. the old
   31.4GB image on the same test clip.
   - **A `pytorch/pytorch:*-runtime` (conda-based) image was tried first and
     rejected** — even pinning it to the exact same `torch==2.1.2` this fork
     used to install via pip, it was still ~16-19% slower on identical
     inference work (measured after ruling out host memory pressure as a
     confound — see below). The conda environment itself, not the torch
     version, was the cause. A plain `nvidia/cuda-runtime` base (no conda) at
     the same size class doesn't have this penalty.
   - **That base also ships an old bundled ffmpeg (`4.3`, no NVENC `-rc`
     support) earlier in `PATH` than a custom install** — silently broke the
     final audio mux (`subprocess.call`'s return code isn't checked, so the
     pipeline reported "DONE" even though the muxed `-audio.mp4` was never
     actually created). Not an issue for the final `nvidia/cuda-runtime`
     base (no conda ffmpeg to conflict with), but worth remembering if a
     conda-based base is ever revisited.
   - **Host machine RAM being nearly exhausted (from accumulated Docker
     Desktop/WSL2 overhead across many pulls/builds in one session) measurably
     inflated timing results** — a "26% slower" reading dropped to "~19%
     slower" just from restarting Docker Desktop before re-measuring under
     otherwise identical conditions. Any suspicious timing regression found
     mid-session should be re-checked after confirming host RAM isn't
     pegged, before trusting it as real.
   - **The WSL2 vhdx-doesn't-shrink gotcha (see Environment gotchas below)
     got hit hard and repeatedly during this exploration** — `docker system
     df` reporting large "reclaimable" totals does not mean the host disk
     reflects it; `docker builder prune -af` in particular had accumulated
     31GB+ of unpruned BuildKit cache across several back-to-back multi-stage
     builds before being caught. Prune build cache explicitly and often when
     iterating on multi-stage Dockerfiles, not just images.

6. **The multi-stage builder from point 5 turned out to be unnecessary —
   `pycuda` is entirely unused dead weight.** A repo-wide grep finds zero
   imports of it anywhere (`requirements.txt` lists it, but nothing in
   `src/` or `run.py`/`api.py` actually imports it — TensorRT execution goes
   through the `tensorrt` Python API's own context/buffer handling, not
   `pycuda`). Since `pycuda`'s C-extension compile was the *only* reason the
   image needed `nvcc`/the `-devel` builder stage, dropping it collapses
   `Dockerfile.slim` back to a single `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
   stage. Also audited the actual `run.py` (human, `predict_type: trt`,
   lip-only) import chain file-by-file and dropped everything not on it:
   `torchvision` (only pulled in by the animal/`XPose` path's lazy import),
   `torchaudio`/`gradio`/`huggingface_hub`/`transformers`/`pykalman`/
   `soundfile` (all JoyVASA/gradio-pipeline-only, never imported by `run.py`),
   and the `uvicorn`/`fastapi`/`pydantic` added briefly for `api.py`
   compatibility (`api.py` is a separate long-running server, not part of
   the CI's one-shot CLI use case). `mediapipe` still has to stay even
   though the CI config never selects it — `src/models/__init__.py` imports
   `MediaPipeFaceModel` unconditionally at module load time. Net result:
   23.1GB (barely smaller than the 23.7GB multi-stage version — the base
   image dominates size more than these packages did — but meaningfully
   simpler and more correct as a dependency list). Two real omissions caught
   only by actually running `run.py` end-to-end (none of this session's
   other testing ever exercised `run.py` or `api.py` directly, only
   `GradioLivePortraitPipeline` via ad hoc test scripts): `colorama` (used by
   `run.py` itself) and `omegaconf`/`tqdm`/`Pillow` (missed in the first
   from-scratch rewrite pass, caught by a systematic file-by-file import
   audit before the second rebuild rather than another blind rebuild-retest
   cycle).

`Dockerfile.nvenc` (the earlier, `shaoguo`-based attempt) is kept for
reference/fallback but superseded by `Dockerfile.slim`.

7. **Checkpoints were never actually baked into `Dockerfile.slim`, in any
   prior session — confirmed by grepping the Dockerfile (no `checkpoint`
   reference at all) and finding `checkpoints` explicitly listed in
   `.dockerignore`.** This was caught only because the user re-ran the real
   GitLab CI job after the cwd fix (#6 above / "Environment gotchas") and
   hit a second failure: `libgrid_sample_3d_plugin.so` (and every other
   `./checkpoints/...` path) not found, since `os.chdir(REPO_ROOT)` now
   correctly pointed at `/workspace`, but `/workspace/checkpoints` never
   existed in the image. All prior local testing of this image almost
   certainly relied on a bind-mount (`-v .../checkpoints:/workspace/checkpoints`
   — consistent with the Windows/git-bash `-v` path-mangling gotcha already
   documented below), which isn't something a CI job's `image:` gets for
   free. The earlier "CI migration is a one-line `image:` swap" claim above
   was wrong until this was fixed.
   - Fixed by actually building the TensorRT engines: downloaded the
     human/lip-path ONNX subset (`appearance_feature_extractor`,
     `face_2dpose_106_static`, `landmark`, `motion_extractor`,
     `retinaface_det_static`, `stitching`, `stitching_eye`, `stitching_lip`,
     `warping_spade-fix`) plus the precompiled `libgrid_sample_3d_plugin.so`
     from `huggingface.co/warmshao/FasterLivePortrait` (confirmed the plugin
     really does ship precompiled there, matching the earlier claim in
     "Actual use case" above), then ran `scripts/all_onnx2trt.sh` inside a
     container built from `Dockerfile.slim` itself with `--gpus all` (so the
     engines are built with the exact same `tensorrt==8.6.1` that will later
     load them) — 9 `.trt` files, ~418MB total, fp16 except
     `motion_extractor` at fp32 per the script.
   - **First attempt (Git LFS) failed at push time, not at conversion —
     worth remembering for any future large-binary-in-repo decision**:
     committed the `.trt` files + plugin `.so` via Git LFS (`.gitattributes`,
     `.gitignore`/`.dockerignore` changed from a blanket `checkpoints`
     exclusion to `checkpoints/*` plus explicit negations for the needed
     files), but `git push` failed outright: `batch response: @wegylexy can
     not upload new objects to public fork wegylexy/FasterLivePortrait`.
     **GitHub blocks pushing new Git LFS objects into forks entirely**,
     regardless of plan/quota, since LFS storage bills against the upstream
     repo owner rather than the fork — this is a hard restriction, not a
     cost one, confirmed only by actually attempting the push (this wasn't
     predictable by inspecting the repo beforehand). Reverted the LFS setup
     (`.gitattributes` removed, `.gitignore`/`.dockerignore` reverted to
     the original blanket `checkpoints` exclusion) rather than keep dead
     LFS config around.
   - **Actual fix: published the 10 files as a tarball GitHub Release
     asset** (`gh release create checkpoints-liveportrait-onnx-trt-v1`,
     ~374MB compressed, sha256 `26f6237fd3735ff9f072e0120c1370d7a96e5e6698d7dd1d1e68c918a0f7c8ad`)
     instead — Release assets aren't Git LFS and aren't subject to the
     fork restriction, free, 2GB/file limit. `Dockerfile.slim` adds a `RUN
     wget ... && sha256sum -c ... && tar -xzf ...` step (after `COPY .
     /workspace`) that downloads and verifies the tarball into
     `checkpoints/liveportrait_onnx/` at build time. No GPU needed for that
     step, so it works fine on GitHub Actions' `ubuntu-latest` runner too.
   - **Considered and rejected**: a GitLab Runner `config.toml` `volumes`
     mount (host-side, not a pipeline-YAML change, so it would have
     satisfied "don't touch my pipeline" too) — rejected because it's
     runner-wide (applies to every job on that runner across all
     projects/pipelines, not just this one) and makes the image
     non-standalone if that runner config ever changes; the Release-asset
     approach keeps the image self-sufficient regardless of runner setup.
   - **Engine portability caveat**: TensorRT engines are tied to the
     GPU/driver they're built on, unlike ONNX. Built here on an RTX 4070
     (driver 610.74); the user confirmed the actual GitLab runner is an RTX
     4090 with the latest Studio driver — both Ada Lovelace (compute
     capability 8.9), so expected to be compatible, but if a future runner
     ever moves to a different GPU generation, the released tarball would
     need rebuilding for that hardware and re-uploading under a new release
     tag, not just re-downloading.
   - **Verified end-to-end twice before declaring this done** (after wrongly
     declaring the cwd fix alone "done" earlier the same session): once
     against the Git-LFS-staged local checkpoints before discovering the
     fork-push block, and again against the final Release-asset-based image
     built from a completely clean context (43KB build context — confirmed
     nothing local leaked in). Both times: built the image, ran a container
     with *no volume mounts at all* and `-w /builds/ai/edu/bible` (the exact
     path from the user's traceback) to genuinely reproduce the GitLab CI
     cwd override, using real sample assets (`assets/examples/source/s0.jpg`
     + `assets/examples/driving/d0.mp4`). Produced an identical real
     151KB/3.12s output video both times, confirming the cwd fix and the
     baked-in checkpoints work together standalone.
   - **`docker-publish.yml` (GitHub Actions) still can't build these
     engines itself** — it runs on a plain `ubuntu-latest` runner with no
     GPU, so it was never going to be able to do the ONNX→TensorRT
     conversion regardless of this fix. That's the whole reason the engines
     are pre-built and fetched via `wget`/`RUN` rather than generated by a
     Dockerfile step — no GPU required for a download+extract. Also worth
     watching: that workflow already had a comment flagging the ~23GB image
     as marginal against the runner's ~14–26GB disk budget *before* this
     change added another ~420MB — hasn't been confirmed to still fit, only
     that the logic is correct.

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

- **GitLab CI overrides the job image's `WORKDIR`/cwd to its own checkout
  path** (`/builds/<namespace>/<project>`) when this image is used as a job
  `image:` — confirmed by the user, who runs `inference.py` this way. Broke
  `inference.py`'s `DEFAULT_CFG` (`configs/trt_infer_lip_ci.yaml`, a plain
  relative path resolved against cwd instead of `/workspace`) and would have
  broken the checkpoint loading right after it too, since
  `trt_infer_lip_ci.yaml`'s own `model_path`/`mask_crop_path` entries are
  *also* plain `./checkpoints/...`-style relative paths resolved against cwd
  wherever they're opened (`src/models/predictor.py`'s
  `os.path.exists(model_path)`/`InferenceSession(model_path, ...)`), not
  against the config file's own location. Fixed in `inference.py` by
  resolving `-s`/`-d`/`-o` to absolute paths *first* (so they still work
  relative to whatever cwd the CI job actually invokes from) and only then
  `os.chdir()`-ing to the script's own directory (`REPO_ROOT`) before loading
  the config — so every relative path baked into the shipped config resolves
  against the image's bundled `/workspace`, regardless of what cwd the CI
  runner set. `run.py` (local-dev entrypoint) doesn't need this since it's
  always invoked from within the checked-out repo, where cwd already equals
  repo root.
- **Docker Desktop / WSL2 disk**: the WSL2 virtual disk (`.vhdx`) is
  thin-provisioned and grows automatically but does **not** shrink back when
  Docker reports images/cache freed internally — `docker system df` numbers
  freed don't reflect on the actual Windows host disk until the vhdx is
  compacted separately. Ran the host disk down to single-digit GB free
  several times this session between large builds/pulls; check `df -h /`
  before any build/pull, not just `docker system df`.
  - **Hit true zero free disk once** (from a multi-stage build failing
    mid-install after several back-to-back builds/pulls without pruning in
    between) — at 0 bytes free, even basic commands (`ls`, `du | head`)
    failed with write errors, since apparently even stdout buffering needs
    disk headroom in this environment. Recovery: `docker builder prune -af`
    (build cache had silently grown to 30GB+ unpruned) plus clearing this
    project's own `tmp/` test-output accumulation reclaimed *some* space,
    but real host free space stayed near-zero afterward regardless —
    consistent with Windows' page/swap file having grown into the same
    disk-full condition and not releasing until the underlying memory
    pressure eased (no reboot needed this time; it recovered gradually on
    its own over the following ~15-20 minutes as Docker Desktop settled).
    **Lesson: prune build cache (`docker builder prune -af`), not just
    images, proactively between iterative multi-stage-Dockerfile rebuilds —
    don't wait until disk is critically low to check.**
- **NVENC needs `NVIDIA_DRIVER_CAPABILITIES` to include `video`** — `--gpus
  all` alone only grants `compute,utility`. Without it, `h264_nvenc` fails to
  load `libnvidia-encode.so.1`.
- **NVDEC implemented (`src/utils/nvdec_capture.py`), disabled by default** —
  see "Not yet done" above for why (measured 4x slower than CPU decode due
  to GPU decoder/inference contention, not decode speed itself).
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
