# benchmark_trt.py
import os
import time
import logging

import torch
import torch_tensorrt
import numpy as np
from tqdm import tqdm

from network import GRAttNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ===================== Parameters =====================
PTH_WEIGHT   = './checkpoints/best_iou.pth'   # trained weights
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INPUT_SHAPE  = (1, 4, 224, 224)               # NCHW
NWARMUP      = 50
NRUNS        = 200
SAVE_TRT     = True                           # whether to save TensorRT models
# ======================================================

def build_model():
    """Build model, load weights, set eval mode"""
    model = GRAttNet().to(DEVICE)
    state_dict = torch.load(PTH_WEIGHT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f'GRAttNet weights loaded from {PTH_WEIGHT}')
    return model


def trace_model(model):
    """Trace model with torch.jit.trace"""
    example_input = torch.empty(INPUT_SHAPE, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    logger.info('Tracing finished')
    return traced


def compile_trt(traced, fp16=False):
    """Compile TensorRT engine"""
    logger.info(f'Compiling TensorRT model (FP16={fp16}) ...')
    trt_model = torch_tensorrt.compile(
        traced,
        inputs=[torch_tensorrt.Input(INPUT_SHAPE, dtype=torch.float32)],
        enabled_precisions={torch.float16 if fp16 else torch.float32},
        workspace_size=1 << 30,        # 1 GB
        truncate_long_and_double=True
    )
    logger.info('TensorRT compilation finished')
    return trt_model


def benchmark(model, name='model', nwarmup=NWARMUP, nruns=NRUNS):
    """Benchmark and return average latency in ms"""
    dummy = torch.randn(INPUT_SHAPE, dtype=torch.float32, device=DEVICE)

    # warmup
    logger.info(f'[{name}] Warmup ...')
    with torch.no_grad():
        for _ in tqdm(range(nwarmup), desc=f'[{name}] Warmup'):
            model(dummy)
    torch.cuda.synchronize()

    # real timing
    logger.info(f'[{name}] Timing ...')
    timings = []
    with torch.no_grad():
        for i in tqdm(range(1, nruns + 1), desc=f'[{name}] Timing'):
            start = time.time()
            model(dummy)
            torch.cuda.synchronize()
            timings.append(time.time() - start)

    avg_ms = np.mean(timings) * 1000
    logger.info(f'[{name}] Average inference time: {avg_ms:.2f} ms '
                f'(±{np.std(timings)*1000:.2f} ms)')
    return avg_ms


def main():
    # 1. Build model & trace
    base_model = build_model()
    traced_model = trace_model(base_model)

    # 2. Compile TensorRT
    trt_fp32 = compile_trt(traced_model, fp16=False)
    trt_fp16 = compile_trt(traced_model, fp16=True)

    # 3. Benchmark
    benchmark(base_model, name='PyTorch FP32')
    benchmark(trt_fp32,   name='TensorRT FP32')
    benchmark(trt_fp16,   name='TensorRT FP16')

    # 4. Save TensorRT models (optional)
    if SAVE_TRT:
        os.makedirs('./trt', exist_ok=True)
        torch.jit.save(trt_fp32, './trt/grattnet_trt_fp32.jit.pt')
        torch.jit.save(trt_fp16, './trt/grattnet_trt_fp16.jit.pt')
        logger.info('TensorRT models saved to ./trt/')


if __name__ == '__main__':
    main()