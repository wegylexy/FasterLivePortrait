#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
ONNX -> TensorRT engine, for TensorRT 11.

TensorRT 11 networks are strongly typed: each layer runs in the precision the
ONNX graph declares, and the FP16 builder flag is gone. So an fp16 engine is
built from an ONNX graph converted to mixed precision first, with NVIDIA
ModelOpt AutoCast (keeps numerically sensitive nodes in FP32, keeps the graph's
inputs/outputs FP32 so callers don't change).

TensorRT 11's built-in GridSample takes 5D input, so warping_spade.onnx (the
stock export, standard GridSample) needs no plugin. The ONNX spec only defines
5D GridSample from opset 20, so such graphs are converted to opset 20 first.

Requires: tensorrt>=11, onnx, nvidia-modelopt[onnx] (fp16 only).
"""
import os
import sys
import logging
import argparse
import tempfile

import onnx
import tensorrt as trt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("EngineBuilder")

GRID_SAMPLE_ND_OPSET = 20


def prepare_onnx(onnx_path, precision):
    """Returns the path of an ONNX graph ready for a strongly typed TensorRT build."""
    model = onnx.load(onnx_path)
    changed = False
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)
    if opset < GRID_SAMPLE_ND_OPSET and any(n.op_type == "GridSample" for n in model.graph.node):
        log.info(f"GridSample present: converting opset {opset} -> {GRID_SAMPLE_ND_OPSET} for 5D support")
        model = onnx.version_converter.convert_version(model, GRID_SAMPLE_ND_OPSET)
        changed = True
    if precision == "fp16":
        from modelopt.onnx.autocast import convert_to_mixed_precision
        if changed:
            tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
            onnx.save(model, tmp.name)
            onnx_path = tmp.name
        log.info("Converting to FP16 mixed precision with ModelOpt AutoCast (I/O kept FP32)")
        # data_max: AutoCast keeps any node whose reference activations exceed this in FP32.
        # Its default (512) kept much of warping_spade in FP32 (activations reach ~6.6k,
        # well inside FP16's 65504), making it 47% slower than TensorRT 8.6's all-FP16
        # engine; at FP16's own limit it's 17% faster with the same output (PSNR vs the
        # 8.6 engines unchanged: mean 48.9 dB, worst frame 45.5 dB).
        model = convert_to_mixed_precision(onnx_path=onnx_path, low_precision_type="fp16", keep_io_types=True,
                                           data_max=65504)
        changed = True
    if not changed:
        return onnx_path
    out = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    onnx.save(model, out)
    return out


def build(onnx_path, engine_path, precision, verbose=False):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # strongly typed (the only kind in TensorRT 11)
    parser = trt.OnnxParser(network, logger)
    prepared = prepare_onnx(os.path.realpath(onnx_path), precision)
    if not parser.parse_from_file(prepared):
        for i in range(parser.num_errors):
            log.error(parser.get_error(i))
        sys.exit(f"failed to parse {onnx_path}")
    for i in range(network.num_inputs):
        t = network.get_input(i)
        log.info(f"input  {t.name} {t.shape} {t.dtype}")
    for i in range(network.num_outputs):
        t = network.get_output(i)
        log.info(f"output {t.name} {t.shape} {t.dtype}")

    config = builder.create_builder_config()
    # TRT_WORKSPACE_GB caps the scratch memory tactics may use while building, e.g.
    # when building next to other GPU jobs; tactics that need more are skipped.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(os.environ.get("TRT_WORKSPACE_GB", "12")) << 30)
    log.info(f"building {precision} engine -> {engine_path}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        sys.exit(f"engine build failed for {onnx_path}")
    os.makedirs(os.path.dirname(os.path.realpath(engine_path)), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", required=True, help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16"],
                        help="fp16 (mixed precision via ModelOpt AutoCast) or fp32, default: fp16")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    args = parser.parse_args()
    if args.engine is None:
        args.engine = args.onnx.replace(".onnx", ".trt")
    build(args.onnx, args.engine, args.precision, args.verbose)
